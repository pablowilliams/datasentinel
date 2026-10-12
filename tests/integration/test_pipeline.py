from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import text

from datasentinel.db import reset
from datasentinel.generator import SCENARIOS, generate
from datasentinel.ingestion.loaders import ensure_dataset, load_batch, payload_hash
from datasentinel.monitoring.pipeline import run_once, seed_history

pytestmark = pytest.mark.integration

SEED = 20261012
DAY = date(2026, 10, 9)


@pytest.fixture(scope="module")
def warehouse(engine):
    """A database seeded with healthy history, rebuilt once for this module."""
    reset(engine)
    seed_history(engine, days=45, until=DAY, seed=SEED)
    return engine


def count(engine, table: str, **where) -> int:
    clause = " AND ".join(f"{k} = :{k}" for k in where)
    sql = f"SELECT count(*) FROM {table}" + (f" WHERE {clause}" if clause else "")
    with engine.begin() as conn:
        return int(conn.execute(text(sql), where).scalar() or 0)


def test_history_seeded_one_run_per_day(warehouse):
    assert count(warehouse, "pipeline_runs", status="success", scenario="healthy") == 45
    assert count(warehouse, "raw_loan_applications") > 20_000


def test_the_same_payload_is_not_loaded_twice(warehouse):
    """A retried delivery is the most ordinary cause of duplicate rows."""
    dataset_id = ensure_dataset(warehouse, "loan_applications")
    frame = generate("healthy", DAY - timedelta(days=200), seed=SEED)

    first = load_batch(warehouse, frame, dataset_id=dataset_id, scenario="healthy")
    assert first.status == "success" and first.rows == len(frame)

    second = load_batch(warehouse, frame, dataset_id=dataset_id, scenario="healthy")
    assert second.skipped
    assert second.rows == 0
    assert second.run_id == first.run_id

    with warehouse.begin() as conn:
        loaded = conn.execute(
            text("SELECT count(*) FROM raw_loan_applications WHERE run_id = :r"),
            {"r": first.run_id},
        ).scalar()
    assert loaded == len(frame)


def test_payload_hash_ignores_column_order():
    frame = generate("healthy", DAY, seed=SEED)
    assert payload_hash(frame) == payload_hash(frame[list(reversed(frame.columns))])


def test_payload_hash_changes_with_the_data():
    a = generate("healthy", DAY, seed=SEED)
    b = generate("healthy", DAY, seed=SEED + 1)
    assert payload_hash(a) != payload_hash(b)


def test_a_failed_load_still_leaves_a_run_record(warehouse):
    """Silence is the one outcome a monitoring platform must never produce about
    itself."""
    import pandas as pd

    dataset_id = ensure_dataset(warehouse, "loan_applications")
    broken = pd.DataFrame({"application_id": ["x"], "loan_amount": ["not a number at all"]})
    outcome = load_batch(warehouse, broken, dataset_id=dataset_id, scenario="corrupt")
    assert outcome.status == "failed"
    with warehouse.begin() as conn:
        row = conn.execute(
            text("SELECT status, error_message FROM pipeline_runs WHERE run_id = :r"),
            {"r": outcome.run_id},
        ).one()
    assert row.status == "failed"
    assert row.error_message


def test_a_run_persists_results_incidents_and_drift(warehouse):
    report = run_once(warehouse, scenario="missing_income", day=DAY - timedelta(days=1))
    assert report.status == "success"
    assert count(warehouse, "quality_results", run_id=str(report.run_id)) > 0
    assert count(warehouse, "incidents", run_id=str(report.run_id)) > 0
    assert count(warehouse, "drift_results", run_id=str(report.run_id)) > 0


def test_healthy_run_records_checks_and_no_incidents(warehouse):
    report = run_once(warehouse, scenario="healthy", day=DAY - timedelta(days=2))
    assert report.failed_checks == 0
    assert report.open_incidents == 0
    assert count(warehouse, "quality_results", run_id=str(report.run_id)) > 0


def test_incidents_carry_jsonb_evidence(warehouse):
    report = run_once(warehouse, scenario="invalid_values", day=DAY - timedelta(days=3))
    with warehouse.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT severity, evidence FROM incidents WHERE run_id = :r ORDER BY severity"
            ),
            {"r": str(report.run_id)},
        ).all()
    assert rows
    for severity, evidence in rows:
        assert severity in {"critical", "high", "medium", "low"}
        assert evidence["observed"]
        assert evidence["expected"]


def test_reference_window_excludes_unhealthy_runs(warehouse):
    """A reference window containing a broken day teaches the detector that
    broken is normal."""
    from datasentinel.monitoring.pipeline import reference_window

    run_once(warehouse, scenario="population_shift", day=DAY - timedelta(days=4))
    reference = reference_window(warehouse)
    assert "gig_economy" not in set(reference["employment_status"].dropna())


@pytest.mark.parametrize(
    "scenario", [n for n, s in SCENARIOS.items() if s.kind == "failure"]
)
def test_every_failure_raises_an_incident_through_the_full_pipeline(warehouse, scenario):
    report = run_once(warehouse, scenario=scenario, day=DAY - timedelta(days=30))
    assert report.open_incidents > 0, f"{scenario} passed through the pipeline unnoticed"
