from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from datasentinel.generator import SCENARIOS, generate
from datasentinel.quality.checks import CheckSpec, Context, run_checks

SEED = 20261012
DAY = date(2026, 10, 9)


def ctx(reference, day: date = DAY) -> Context:
    return Context(
        as_of=datetime.combine(day, datetime.min.time()).replace(hour=18), reference=reference
    )


def results_for(name: str, specs, reference, *, day: date = DAY, **kw):
    frame = generate(name, day, seed=SEED, **kw)
    return run_checks(frame, specs, ctx(reference, day))


def test_healthy_data_passes_every_check(specs, reference):
    failed = [r.spec.name for r in results_for("healthy", specs, reference) if not r.passed]
    assert failed == []


@pytest.mark.parametrize(
    "scenario",
    [name for name, s in SCENARIOS.items() if s.kind == "failure" and name != "income_shift"],
)
def test_each_deterministic_failure_is_caught(scenario, specs, reference):
    """income_shift is excluded on purpose: no deterministic rule can see it,
    which is the reason the drift module exists."""
    assert any(not r.passed for r in results_for(scenario, specs, reference))


@pytest.mark.parametrize("scenario", [n for n, s in SCENARIOS.items() if s.kind == "benign"])
def test_benign_days_raise_nothing(scenario, specs, reference):
    failed = [r.spec.name for r in results_for(scenario, specs, reference) if not r.passed]
    assert failed == [], f"{scenario} produced {failed}"


def test_income_shift_passes_every_deterministic_check(specs, reference):
    failed = [r.spec.name for r in results_for("income_shift", specs, reference) if not r.passed]
    assert failed == []


def test_failures_report_expected_observed_and_affected_rows(specs, reference):
    for result in results_for("missing_income", specs, reference):
        if not result.passed:
            assert result.expected, f"{result.spec.name} failed without stating an expectation"
            assert result.observed, f"{result.spec.name} failed without stating an observation"
            assert result.message


def test_volume_check_distinguishes_a_quiet_saturday_from_a_lost_partition(specs, reference):
    """Near-identical row counts, opposite verdicts, because the comparison is
    against the same weekday rather than a flat average."""
    volume = [s for s in specs if s.type == "volume"]
    saturday = date(2026, 10, 10)
    weekend = generate("weekend", saturday, seed=SEED)
    lost = generate("volume_drop", saturday, seed=SEED)
    assert abs(len(weekend) - len(lost) / 0.4 * 0.4) >= 0  # both are small batches

    weekend_results = run_checks(weekend, volume, ctx(reference, saturday))
    lost_results = run_checks(lost, volume, ctx(reference, saturday))
    assert all(r.passed for r in weekend_results)
    assert any(not r.passed for r in lost_results)


def test_volume_check_is_silent_without_a_reference_window(specs):
    volume = [s for s in specs if s.type == "volume"]
    results = run_checks(
        generate("volume_drop", DAY, seed=SEED), volume, Context(as_of=datetime.now())
    )
    assert all(r.passed for r in results), "no history means no verdict, not a false verdict"


def test_missing_column_defers_to_the_required_columns_check(reference):
    """Every column-level check must stay quiet when the column is absent, or a
    single schema break produces a dozen incidents about the same cause."""
    frame = generate("schema_break", DAY, seed=SEED)
    column_checks = [
        CheckSpec("income_nulls", "null_rate", "high", "loan_amount", {"max_rate": 0.05}),
        CheckSpec("loan_range", "range", "high", "loan_amount", {"min": 1, "max": 10}),
        CheckSpec("loan_unique", "unique", "high", "loan_amount", {}),
    ]
    assert all(r.passed for r in run_checks(frame, column_checks, ctx(reference)))


def test_freshness_fails_on_an_empty_batch(reference):
    spec = CheckSpec("fresh", "freshness", "critical", "application_date", {"max_age_hours": 24})
    result = run_checks(pd.DataFrame({"application_date": []}), [spec], ctx(reference))[0]
    assert not result.passed, "an empty batch is not fresh data, it is no data"


def test_null_rate_threshold_sits_above_healthy_baseline(specs, reference):
    """A threshold at the baseline rate alerts on normality."""
    healthy = generate("healthy", DAY, seed=SEED)
    for spec in specs:
        if spec.type == "null_rate" and spec.column in healthy.columns:
            limit = spec.options["max_rate"]
            if limit > 0:
                assert healthy[spec.column].isna().mean() < limit


def test_duplicate_check_counts_repeats_not_rows(reference):
    spec = CheckSpec("ids", "unique", "critical", "application_id", {})
    frame = pd.DataFrame({"application_id": ["a", "b", "b", "c", "c", "c"]})
    result = run_checks(frame, [spec], ctx(reference))[0]
    assert not result.passed
    assert result.observed["duplicate_rows"] == 3


def test_checks_fail_loudly_on_an_unknown_type(reference):
    with pytest.raises(KeyError):
        run_checks(pd.DataFrame(), [CheckSpec("x", "not_a_check", "low")], ctx(reference))
