"""Database access: one engine, one place that knows the DDL."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, text

from datasentinel.config import settings

DDL = Path(__file__).with_name("schema.sql")


def engine(dsn: str | None = None) -> Engine:
    return create_engine(dsn or settings().dsn, future=True, pool_pre_ping=True)


def migrate(eng: Engine) -> None:
    """Apply the schema. Every statement is idempotent, so this is safe to rerun."""
    with eng.begin() as conn:
        conn.execute(text(DDL.read_text()))


def reset(eng: Engine) -> None:
    """Drop everything and rebuild. Used by the test suite and the demo reset."""
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                DROP TABLE IF EXISTS model_metrics, incidents, drift_results,
                    quality_results, raw_loan_applications, pipeline_runs, datasets CASCADE
                """
            )
        )
    migrate(eng)
