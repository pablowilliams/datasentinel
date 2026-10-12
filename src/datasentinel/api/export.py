"""Export a snapshot of the warehouse for the static dashboard.

GitHub Pages serves files, not processes, so the published dashboard reads a
committed snapshot rather than querying PostgreSQL. It is therefore a record of
a particular run of the pipeline, not a live console, and the page says so
rather than implying a service that is not there.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, text


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def _rows(eng: Engine, sql: str, **params: Any) -> list[dict[str, Any]]:
    with eng.begin() as conn:
        result = conn.execute(text(sql), params)
        return [{k: _plain(v) for k, v in row.items()} for row in result.mappings().all()]


def snapshot(eng: Engine, *, run_limit: int = 40) -> dict[str, Any]:
    runs = _rows(
        eng,
        """
        SELECT p.run_id::text, p.scenario, p.status, p.rows_processed, p.started_at,
               p.completed_at, p.error_message,
               -- The day the data describes, not the clock time it was loaded.
               -- Every run in a replayed week starts within seconds of the
               -- others, so started_at makes a week look like a minute.
               (SELECT max(r.application_date) FROM raw_loan_applications r
                 WHERE r.run_id = p.run_id) AS business_date,
               (SELECT count(*) FROM incidents i WHERE i.run_id = p.run_id) AS incidents,
               (SELECT count(*) FROM quality_results q
                 WHERE q.run_id = p.run_id AND NOT q.passed) AS failed_checks,
               (SELECT count(*) FROM quality_results q WHERE q.run_id = p.run_id) AS total_checks
        FROM pipeline_runs p
        ORDER BY p.started_at DESC
        LIMIT :limit
        """,
        limit=run_limit,
    )

    incidents = _rows(
        eng,
        """
        SELECT i.incident_id::text, i.run_id::text, i.category, i.severity, i.status,
               i.title, i.description, i.column_name, i.evidence, i.created_at,
               p.scenario
        FROM incidents i JOIN pipeline_runs p USING (run_id)
        ORDER BY
          CASE i.severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                          WHEN 'medium' THEN 2 ELSE 3 END,
          i.created_at DESC
        LIMIT 120
        """,
    )

    drift = _rows(
        eng,
        """
        SELECT d.column_name, d.test_name, d.statistic, d.p_value, d.effect_size,
               d.threshold, d.drift_detected, d.explanation, p.scenario
        FROM drift_results d JOIN pipeline_runs p USING (run_id)
        ORDER BY d.drift_detected DESC, d.effect_size DESC
        LIMIT 120
        """,
    )

    model = _rows(
        eng,
        """
        SELECT model_name, model_version, evaluation_window, metric_name,
               metric_value, sample_count
        FROM model_metrics
        ORDER BY evaluation_window, metric_name
        """,
    )

    checks = _rows(
        eng,
        """
        SELECT q.check_name, q.check_type, q.column_name, q.severity,
               count(*) FILTER (WHERE NOT q.passed) AS failures,
               count(*) AS runs
        FROM quality_results q
        GROUP BY q.check_name, q.check_type, q.column_name, q.severity
        ORDER BY failures DESC, q.check_name
        """,
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "caveat": (
            "A snapshot of one run of the pipeline on synthetic data, exported to "
            "static files. Not a live service."
        ),
        "runs": runs,
        "incidents": incidents,
        "drift": drift,
        "model_metrics": model,
        "checks": checks,
    }


def write(eng: Engine, directory: Path, *, results: Path | None = None) -> list[Path]:
    """Write the snapshot plus whichever measurement artefacts exist."""
    directory.mkdir(parents=True, exist_ok=True)
    written = [directory / "snapshot.json"]
    (directory / "snapshot.json").write_text(json.dumps(snapshot(eng), indent=2))

    if results is not None:
        for name in ("evaluation.json", "sensitivity.json", "model_decay.json"):
            source = results / name
            if source.exists():
                target = directory / name
                target.write_text(source.read_text())
                written.append(target)
    return written
