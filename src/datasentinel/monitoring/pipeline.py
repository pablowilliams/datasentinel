"""One run: load a batch, check it, compare it, record what was found."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
from sqlalchemy import Engine, text

from datasentinel.config import settings
from datasentinel.drift.detector import detect, load_config, to_incident_payloads
from datasentinel.generator import generate
from datasentinel.incidents.service import Incident, incidents_from
from datasentinel.ingestion.loaders import ensure_dataset, load_batch
from datasentinel.quality.checks import CheckResult, Context, run_checks
from datasentinel.quality.runner import load_specs

REFERENCE_DAYS = 90


@dataclass
class RunReport:
    run_id: uuid.UUID
    scenario: str
    day: date
    rows: int
    status: str
    quality: list[CheckResult]
    drift_raised: list[dict]
    incidents: list[Incident]

    @property
    def failed_checks(self) -> int:
        return sum(1 for r in self.quality if not r.passed)

    @property
    def open_incidents(self) -> int:
        return len(self.incidents)


def reference_window(eng: Engine, *, days: int = REFERENCE_DAYS) -> pd.DataFrame:
    """Healthy history from the warehouse.

    Only successful runs of healthy batches: a reference window that includes a
    broken day teaches the detector that broken is normal.
    """
    with eng.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT r.application_id, r.application_date, r.customer_age, r.annual_income,
                       r.employment_status, r.loan_amount, r.loan_term_months, r.credit_score,
                       r.debt_to_income_ratio, r.previous_defaults, r.approved, r.interest_rate
                FROM raw_loan_applications r
                JOIN pipeline_runs p USING (run_id)
                WHERE p.status = 'success'
                  AND p.scenario = 'healthy'
                  AND r.application_date >= (CURRENT_DATE - CAST(:days AS INTEGER))
                """
            ),
            {"days": days * 2},
        )
        frame = pd.DataFrame(rows.mappings().all())
    for column in ("annual_income", "loan_amount", "credit_score", "debt_to_income_ratio",
                   "interest_rate", "customer_age", "previous_defaults"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _persist(
    eng: Engine,
    *,
    run_id: uuid.UUID,
    dataset_id: uuid.UUID,
    quality: list[CheckResult],
    drift_findings: list,
    incidents: list[Incident],
) -> None:
    with eng.begin() as conn:
        for result in quality:
            conn.execute(
                text(
                    """
                    INSERT INTO quality_results (result_id, run_id, check_name, check_type,
                        column_name, passed, severity, observed, expected, message)
                    VALUES (:id, :run, :name, :type, :column, :passed, :severity,
                            CAST(:observed AS JSONB), CAST(:expected AS JSONB), :message)
                    """
                ),
                {
                    "id": uuid.uuid4(), "run": run_id, "name": result.spec.name,
                    "type": result.spec.type, "column": result.spec.column,
                    "passed": result.passed, "severity": result.spec.severity,
                    "observed": json.dumps(result.observed, default=str),
                    "expected": json.dumps(result.expected, default=str),
                    "message": result.message,
                },
            )
        for finding in drift_findings:
            conn.execute(
                text(
                    """
                    INSERT INTO drift_results (drift_id, run_id, column_name, test_name,
                        statistic, p_value, effect_size, threshold, drift_detected,
                        reference_rows, current_rows, explanation)
                    VALUES (:id, :run, :column, :test, :stat, :p, :effect, :gate,
                            :detected, :ref_rows, :cur_rows, :explanation)
                    """
                ),
                {
                    "id": uuid.uuid4(), "run": run_id, "column": finding.spec.column,
                    "test": finding.result.test, "stat": finding.result.statistic,
                    "p": finding.result.p_value, "effect": finding.result.effect_size,
                    "gate": finding.spec.min_effect, "detected": finding.detected,
                    "ref_rows": finding.result.reference_rows,
                    "cur_rows": finding.result.current_rows,
                    "explanation": finding.explanation,
                },
            )
        for incident in incidents:
            conn.execute(
                text(
                    """
                    INSERT INTO incidents (incident_id, dataset_id, run_id, category, severity,
                        status, title, description, column_name, evidence)
                    VALUES (:id, :dataset, :run, :category, :severity, 'open', :title,
                            :description, :column, CAST(:evidence AS JSONB))
                    """
                ),
                {
                    "id": incident.incident_id, "dataset": dataset_id, "run": run_id,
                    "category": incident.category, "severity": incident.severity,
                    "title": incident.title, "description": incident.description,
                    "column": incident.column_name,
                    "evidence": json.dumps(incident.evidence, default=str),
                },
            )


def run_once(
    eng: Engine,
    *,
    scenario: str,
    day: date,
    seed: int | None = None,
    as_of: datetime | None = None,
    frame: pd.DataFrame | None = None,
) -> RunReport:
    """The whole monitoring pass for one batch."""
    config = settings()
    seed = config.seed if seed is None else seed
    as_of = as_of or datetime.combine(day, datetime.min.time(), tzinfo=UTC).replace(hour=18)

    batch = generate(scenario, day, seed=seed) if frame is None else frame
    dataset_id = ensure_dataset(eng, "loan_applications")
    outcome = load_batch(eng, batch, dataset_id=dataset_id, scenario=scenario)
    if outcome.status == "failed":
        return RunReport(outcome.run_id, scenario, day, 0, "failed", [], [], [])

    reference = reference_window(eng)
    quality = run_checks(
        batch, load_specs(config.config_dir / "checks.yml"),
        Context(as_of=as_of, reference=reference if not reference.empty else None),
    )
    incidents = incidents_from(quality)

    drift_findings: list = []
    if not reference.empty:
        drift_findings = detect(reference, batch, load_config(config.config_dir / "drift.yml"))
        for payload in to_incident_payloads(drift_findings):
            incidents.append(
                Incident(
                    incident_id=uuid.uuid4(),
                    category=payload["category"],
                    severity=payload["severity"],
                    title=f"{payload['severity'].upper()}: drift in {payload['column_name']}",
                    description=payload["description"],
                    column_name=payload["column_name"],
                    evidence=payload["evidence"],
                )
            )

    _persist(eng, run_id=outcome.run_id, dataset_id=dataset_id, quality=quality,
             drift_findings=drift_findings, incidents=incidents)
    return RunReport(outcome.run_id, scenario, day, outcome.rows, outcome.status,
                     quality, [i for i in incidents if i.category == "drift"], incidents)


def seed_history(eng: Engine, *, days: int = REFERENCE_DAYS, until: date, seed: int) -> int:
    """Load a healthy run of history, one run per day, as the pipeline would have."""
    from datetime import timedelta

    dataset_id = ensure_dataset(eng, "loan_applications")
    loaded = 0
    for offset in range(days, 0, -1):
        day = until - timedelta(days=offset)
        batch = generate("healthy", day, seed=seed)
        outcome = load_batch(eng, batch, dataset_id=dataset_id, scenario="healthy")
        loaded += outcome.rows
    return loaded
