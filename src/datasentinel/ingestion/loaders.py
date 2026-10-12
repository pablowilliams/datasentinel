"""Ingestion: read a batch, record the attempt, load it once.

The run record is written before the load is attempted, so a crash leaves a
failed run rather than silence. Silence is the one outcome a monitoring platform
must never produce about itself.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from datasentinel.schema import BY_NAME

RAW_COLUMNS = tuple(BY_NAME)


@dataclass(frozen=True)
class LoadOutcome:
    run_id: uuid.UUID
    status: str
    rows: int
    message: str | None = None

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"


def payload_hash(frame: pd.DataFrame) -> str:
    """A stable fingerprint of a batch, used to make loading idempotent.

    Sorted columns so a reordered file is recognised as the same payload; the
    index is dropped because it is positional, not data.
    """
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    return hashlib.sha256(
        pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dataset(eng: Engine, name: str, *, sla_minutes: int = 1440) -> uuid.UUID:
    with eng.begin() as conn:
        found = conn.execute(
            text("SELECT dataset_id FROM datasets WHERE name = :name"), {"name": name}
        ).scalar()
        if found:
            return uuid.UUID(str(found))
        dataset_id = uuid.uuid4()
        conn.execute(
            text(
                """
                INSERT INTO datasets (dataset_id, name, owner, source_type,
                                      expected_frequency, freshness_sla_minutes)
                VALUES (:id, :name, :owner, 'csv', 'daily', :sla)
                """
            ),
            {"id": dataset_id, "name": name, "owner": "data-platform", "sla": sla_minutes},
        )
        return dataset_id


def already_loaded(eng: Engine, dataset_id: uuid.UUID, digest: str) -> uuid.UUID | None:
    with eng.begin() as conn:
        found = conn.execute(
            text(
                """
                SELECT run_id FROM pipeline_runs
                WHERE dataset_id = :dataset AND source_hash = :digest AND status = 'success'
                """
            ),
            {"dataset": dataset_id, "digest": digest},
        ).scalar()
    return uuid.UUID(str(found)) if found else None


def load_batch(
    eng: Engine,
    frame: pd.DataFrame,
    *,
    dataset_id: uuid.UUID,
    pipeline_name: str = "loan_applications_daily",
    scenario: str | None = None,
    source_uri: str | None = None,
) -> LoadOutcome:
    """Load one batch, exactly once.

    A batch already loaded successfully is skipped rather than loaded again: a
    retried delivery is the most ordinary cause of duplicate rows, and
    duplicated rows are the kind of fault that quietly doubles a revenue figure.
    """
    digest = payload_hash(frame)
    prior = already_loaded(eng, dataset_id, digest)
    if prior is not None:
        return LoadOutcome(prior, "skipped", 0, f"identical payload already loaded as {prior}")

    run_id = uuid.uuid4()
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pipeline_runs (run_id, dataset_id, pipeline_name, scenario,
                                           source_uri, source_hash, started_at, status)
                VALUES (:run, :dataset, :pipeline, :scenario, :uri, :digest, :started, 'running')
                """
            ),
            {
                "run": run_id,
                "dataset": dataset_id,
                "pipeline": pipeline_name,
                "scenario": scenario,
                "uri": source_uri,
                "digest": digest,
                "started": datetime.now(UTC),
            },
        )

    try:
        staged = frame.copy()
        for column in RAW_COLUMNS:
            if column not in staged.columns:
                staged[column] = None
        staged = staged[list(RAW_COLUMNS)]
        # The raw layer accepts what arrived. Coercion is the staging layer's
        # job, and a load that rejects malformed rows destroys the evidence the
        # quality checks exist to find.
        staged["credit_score"] = pd.to_numeric(staged["credit_score"], errors="coerce")
        staged["run_id"] = str(run_id)
        staged = staged.astype(object).where(pd.notna(staged), None)

        with eng.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO raw_loan_applications (run_id, {", ".join(RAW_COLUMNS)})
                    VALUES (:run_id, {", ".join(f":{c}" for c in RAW_COLUMNS)})
                    """
                ),
                staged.to_dict("records"),
            )
            conn.execute(
                text(
                    """
                    UPDATE pipeline_runs
                    SET status = 'success', completed_at = :done, rows_processed = :rows
                    WHERE run_id = :run
                    """
                ),
                {"done": datetime.now(UTC), "rows": len(staged), "run": run_id},
            )
        return LoadOutcome(run_id, "success", len(staged))
    except Exception as exc:  # the run record must outlive the failure
        with eng.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE pipeline_runs
                    SET status = 'failed', completed_at = :done, error_message = :err
                    WHERE run_id = :run
                    """
                ),
                {"done": datetime.now(UTC), "err": str(exc)[:500], "run": run_id},
            )
        return LoadOutcome(run_id, "failed", 0, str(exc)[:500])
