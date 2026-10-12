"""Synthetic loan applications, healthy and otherwise.

Three kinds of scenario, and the distinction is the point of the project:

  healthy  the data a working pipeline delivers, including the untidiness real
           data always has
  failure  something a monitoring platform must catch, each one declaring the
           incident categories it ought to raise
  benign   something that changes without being wrong, which a monitoring
           platform must NOT page anyone about

Without the third kind a detector can score a perfect recall by alerting on
everything, and the precision figure that hides it is worth nothing. Benign
scenarios are what make precision mean something here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from datasentinel.schema import BASELINE_NULL_RATES, CATEGORIES, TERMS

Kind = Literal["healthy", "failure", "benign"]

WEEKDAY_VOLUME = 1_200
WEEKEND_VOLUME = 470


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: Kind
    description: str
    expected: tuple[str, ...] = field(default=())
    """Incident categories this scenario ought to produce. Empty for healthy and
    benign scenarios, where the correct behaviour is silence."""

    @property
    def should_alert(self) -> bool:
        return self.kind == "failure"


SCENARIOS: dict[str, Scenario] = {
    "healthy": Scenario(
        "healthy",
        "healthy",
        "A normal day. Realistic missingness, weekday volume, stable population.",
    ),
    "missing_income": Scenario(
        "missing_income",
        "failure",
        "An upstream join drops income for a fifth of applicants and credit "
        "score for a seventh.",
        ("completeness",),
    ),
    "schema_break": Scenario(
        "schema_break",
        "failure",
        "loan_amount renamed to requested_amount, credit_score arrives as text, "
        "approved disappears.",
        ("schema",),
    ),
    "population_shift": Scenario(
        "population_shift",
        "failure",
        "A different population: incomes up by half, loans much larger, approval "
        "rate moved, and a sixth employment category that has never been seen.",
        ("drift", "category"),
    ),
    "stale": Scenario(
        "stale",
        "failure",
        "Yesterday's file delivered again. Every timestamp is a day old.",
        ("freshness",),
    ),
    "duplicates": Scenario(
        "duplicates",
        "failure",
        "A retried load appends 18% of the batch a second time.",
        ("uniqueness",),
    ),
    "volume_drop": Scenario(
        "volume_drop",
        "failure",
        "A partition is missing upstream and only 40% of the rows arrive.",
        ("volume",),
    ),
    "invalid_values": Scenario(
        "invalid_values",
        "failure",
        "Credit scores of 9999 and negative loan amounts from a bad default.",
        ("validity",),
    ),
    # The three below must stay quiet. They are the reason the precision figure
    # in the evaluation report is worth reading.
    "minor_noise": Scenario(
        "minor_noise",
        "benign",
        "An ordinary day that happens to differ from the mean. Nothing is wrong.",
    ),
    "weekend": Scenario(
        "weekend",
        "benign",
        "Saturday. Volume is 40% of a weekday because it is a Saturday.",
    ),
    "trivial_income_tick": Scenario(
        "trivial_income_tick",
        "benign",
        "Incomes rise 1.5%, which a Kolmogorov-Smirnov test on 1,200 rows will "
        "happily call significant and no analyst would care about. The practical "
        "threshold exists for this day.",
    ),
}


def _rng(scenario: str, day: date, seed: int) -> np.random.Generator:
    """Deterministic per (scenario, day, seed), so any run is reproducible."""
    key = f"{scenario}:{day.isoformat()}:{seed}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def expected_volume(day: date) -> int:
    return WEEKEND_VOLUME if day.weekday() >= 5 else WEEKDAY_VOLUME


def _population(
    n: int,
    rng: np.random.Generator,
    *,
    income_multiplier: float = 1.0,
    loan_multiplier: float = 1.0,
    approval_shift: float = 0.0,
) -> pd.DataFrame:
    """A plausible population, with the correlations a real one has.

    Approval is a function of credit score, debt burden and prior defaults, so a
    classifier trained on this has something real to learn and label drift means
    something when the population moves.
    """
    age = np.clip(rng.normal(38, 12, n), 18, 100).round().astype("int64")

    # Income rises with age and then flattens, as careers do.
    income_base = np.exp(rng.normal(10.55, 0.52, n)) * (1 + 0.012 * np.minimum(age - 18, 25))
    income = np.clip(income_base * income_multiplier, 4_000, 2_000_000).round(2)

    employment = rng.choice(CATEGORIES, n, p=[0.62, 0.14, 0.12, 0.06, 0.06])

    # Credit score tracks income weakly; the rest is noise, as in life.
    income_z = (np.log(income) - np.log(income).mean()) / np.log(income).std()
    credit = np.clip(rng.normal(690, 80, n) + 22 * income_z, 300, 850).round(0)

    dti = np.clip(rng.beta(2.4, 6.0, n) + 0.04 * (-income_z), 0.0, 1.0).round(4)
    loan = np.clip(income * rng.beta(2.0, 5.5, n) * 2.4 * loan_multiplier, 1_000, 1_000_000).round(2)
    term = rng.choice(TERMS, n, p=[0.08, 0.17, 0.32, 0.23, 0.20])

    defaults = rng.poisson(np.clip(0.9 - credit / 850, 0.01, None) * 0.8, n).astype("int64")

    # Approval: the underwriting rule the monitoring platform is protecting.
    logit = (
        0.55
        + 0.0115 * (credit - 600)
        - 3.4 * dti
        - 0.62 * defaults
        - 1.25 * np.clip(loan / np.maximum(income, 1) - 0.5, 0, None)
        + approval_shift
    )
    approved = rng.random(n) < 1 / (1 + np.exp(-logit))

    rate = np.clip(
        14.5 - 0.011 * (credit - 600) + 5.5 * dti + rng.normal(0, 0.6, n), 2.5, 39.9
    ).round(2)

    return pd.DataFrame(
        {
            "customer_age": age,
            "annual_income": income,
            "employment_status": employment,
            "loan_amount": loan,
            "loan_term_months": term,
            "credit_score": credit,
            "debt_to_income_ratio": dti,
            "previous_defaults": defaults,
            "approved": approved,
            "interest_rate": rate,
        }
    )


def _blank(frame: pd.DataFrame, column: str, rate: float, rng: np.random.Generator) -> None:
    if rate <= 0 or column not in frame:
        return
    mask = rng.random(len(frame)) < rate
    frame.loc[mask, column] = None


def generate(scenario_name: str, day: date, *, seed: int, rows: int | None = None) -> pd.DataFrame:
    """One day's batch of applications under the named scenario."""
    scenario = SCENARIOS[scenario_name]
    rng = _rng(scenario_name, day, seed)

    if scenario_name == "weekend":
        day = day - timedelta(days=day.weekday() - 5) if day.weekday() < 5 else day
    n = rows if rows is not None else expected_volume(day)

    income_multiplier = 1.0
    loan_multiplier = 1.0
    approval_shift = 0.0
    if scenario_name == "population_shift":
        income_multiplier, loan_multiplier, approval_shift = 1.5, 1.9, -0.9
    elif scenario_name == "trivial_income_tick":
        income_multiplier = 1.015
    elif scenario_name == "volume_drop":
        n = int(n * 0.40)

    frame = _population(
        n,
        rng,
        income_multiplier=income_multiplier,
        loan_multiplier=loan_multiplier,
        approval_shift=approval_shift,
    )

    stamp = day - timedelta(days=1) if scenario_name == "stale" else day
    frame.insert(0, "application_date", pd.to_datetime(stamp).date())
    frame.insert(
        0,
        "application_id",
        [f"APP-{stamp:%Y%m%d}-{i:06d}" for i in range(len(frame))],
    )

    for column, rate in BASELINE_NULL_RATES.items():
        _blank(frame, column, rate, rng)

    if scenario_name == "missing_income":
        _blank(frame, "annual_income", 0.20, rng)
        _blank(frame, "credit_score", 0.15, rng)
    elif scenario_name == "population_shift":
        # A category nobody has configured a rule for.
        newcomers = rng.random(len(frame)) < 0.11
        frame.loc[newcomers, "employment_status"] = "gig_economy"
    elif scenario_name == "duplicates":
        repeat = frame.sample(frac=0.18, random_state=int(rng.integers(0, 2**31)))
        frame = pd.concat([frame, repeat], ignore_index=True)
    elif scenario_name == "invalid_values":
        bad_credit = rng.random(len(frame)) < 0.06
        frame.loc[bad_credit, "credit_score"] = 9999.0
        bad_loan = rng.random(len(frame)) < 0.03
        frame.loc[bad_loan, "loan_amount"] = -1.0
        bad_age = rng.random(len(frame)) < 0.02
        frame.loc[bad_age, "customer_age"] = 7
    elif scenario_name == "schema_break":
        frame = frame.rename(columns={"loan_amount": "requested_amount"})
        frame["credit_score"] = frame["credit_score"].astype("string")
        frame = frame.drop(columns=["approved"])

    return frame.reset_index(drop=True)


def baseline(days: int, until: date, *, seed: int) -> pd.DataFrame:
    """A healthy run of history, the reference window everything compares against."""
    frames = [
        generate("healthy", until - timedelta(days=offset), seed=seed)
        for offset in range(days, 0, -1)
    ]
    return pd.concat(frames, ignore_index=True)
