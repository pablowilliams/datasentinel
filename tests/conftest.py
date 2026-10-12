from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text

from datasentinel.config import ROOT
from datasentinel.generator import baseline
from datasentinel.quality.runner import load_specs

SEED = 20261012
TODAY = date(2026, 10, 9)


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return ROOT / "config"


@pytest.fixture(scope="session")
def specs(config_dir: Path):
    return load_specs(config_dir / "checks.yml")


@pytest.fixture(scope="session")
def reference():
    """A healthy reference window, built once; the suite never mutates it."""
    return baseline(90, TODAY - timedelta(days=1), seed=SEED)


@pytest.fixture(scope="session")
def engine():
    """A live Postgres engine, or a skip. Integration tests need a real database:
    the idempotency guard is a unique index, and an in-memory stub would be
    testing the stub."""
    dsn = os.environ.get(
        "DATASENTINEL_DSN", "postgresql+psycopg://localhost:5432/datasentinel"
    )
    from datasentinel.db import engine as make_engine

    eng = make_engine(dsn)
    try:
        with eng.begin() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no PostgreSQL available: {exc}")
    return eng
