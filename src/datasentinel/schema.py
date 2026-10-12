"""The loan application contract.

One declaration of the columns, their types and their plausible ranges, used by
the generator, the ingestion validator and the quality checks alike. Three
copies of a schema is how a schema drifts.
"""

from __future__ import annotations

from dataclasses import dataclass

CATEGORIES: tuple[str, ...] = (
    "full_time",
    "part_time",
    "self_employed",
    "retired",
    "unemployed",
)

TERMS: tuple[int, ...] = (12, 24, 36, 48, 60)


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    nullable: bool
    minimum: float | None = None
    maximum: float | None = None
    allowed: tuple[str, ...] | tuple[int, ...] | None = None
    critical: bool = False
    """A critical column carries the business meaning. Missingness in a critical
    column is an incident; missingness in a descriptive one is a note."""


COLUMNS: tuple[Column, ...] = (
    Column("application_id", "string", nullable=False, critical=True),
    Column("application_date", "date", nullable=False, critical=True),
    Column("customer_age", "int", nullable=True, minimum=18, maximum=100),
    Column("annual_income", "float", nullable=True, minimum=0, maximum=2_000_000, critical=True),
    Column("employment_status", "string", nullable=True, allowed=CATEGORIES),
    Column("loan_amount", "float", nullable=False, minimum=1, maximum=1_000_000, critical=True),
    Column("loan_term_months", "int", nullable=False, allowed=TERMS),
    Column("credit_score", "float", nullable=True, minimum=300, maximum=850, critical=True),
    Column("debt_to_income_ratio", "float", nullable=True, minimum=0, maximum=1),
    Column("previous_defaults", "int", nullable=True, minimum=0, maximum=20),
    Column("approved", "bool", nullable=False, critical=True),
    Column("interest_rate", "float", nullable=True, minimum=0, maximum=40),
)

BY_NAME: dict[str, Column] = {c.name: c for c in COLUMNS}
REQUIRED: tuple[str, ...] = tuple(c.name for c in COLUMNS if not c.nullable)
CRITICAL: tuple[str, ...] = tuple(c.name for c in COLUMNS if c.critical)
NUMERIC: tuple[str, ...] = tuple(c.name for c in COLUMNS if c.dtype in {"int", "float"})

# Real data is never perfectly complete. These are the rates the healthy
# generator produces, and the rates the quality thresholds are calibrated
# against: an alert on 1.2% missing income would be an alert on normality.
BASELINE_NULL_RATES: dict[str, float] = {
    "annual_income": 0.012,
    "credit_score": 0.020,
    "employment_status": 0.008,
    "debt_to_income_ratio": 0.015,
    "customer_age": 0.004,
    "previous_defaults": 0.010,
    "interest_rate": 0.006,
}
