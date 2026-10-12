"""What the approval model is allowed to look at.

The label is the underwriting decision, which is known the moment it is made, so
monitoring an approval model against it is realistic: you are asking whether the
model still agrees with the decisions actually being taken.

One column is deliberately excluded. `interest_rate` is priced after a loan is
approved, so it is only available for applications that got that far. A model
given it would score beautifully and be unusable, because at decision time the
field does not exist yet. That is the same shape of mistake as training on a
field whose value is determined by the outcome, and it is far easier to make
than to notice.
"""

from __future__ import annotations

from dataclasses import dataclass

LABEL = "approved"

NUMERIC: tuple[str, ...] = (
    "customer_age",
    "annual_income",
    "loan_amount",
    "loan_term_months",
    "credit_score",
    "debt_to_income_ratio",
    "previous_defaults",
)

CATEGORICAL: tuple[str, ...] = ("employment_status",)

FEATURES: tuple[str, ...] = NUMERIC + CATEGORICAL

POST_DECISION: tuple[str, ...] = ("interest_rate",)
"""Priced after approval. Available in the warehouse, unavailable at the moment
the model would have to run, and therefore not a feature."""

IDENTIFIERS: tuple[str, ...] = ("application_id", "application_date")


@dataclass(frozen=True)
class Derived:
    """Ratios an underwriter would actually reason about."""

    name: str
    description: str


DERIVED: tuple[Derived, ...] = (
    Derived("loan_to_income", "loan_amount / annual_income, the affordability question"),
)


def check_no_leakage(columns: list[str]) -> None:
    """Fail loudly if a post-decision field or the label reaches the feature set."""
    offenders = sorted(set(columns) & (set(POST_DECISION) | {LABEL}))
    if offenders:
        raise ValueError(
            f"these columns are not available at decision time: {', '.join(offenders)}"
        )
