from __future__ import annotations

from datetime import date, timedelta

import pytest

from datasentinel.generator import (
    DEFAULT_INTENSITY,
    SCENARIOS,
    VOLUME_NOISE,
    expected_volume,
    generate,
)
from datasentinel.schema import BY_NAME, CATEGORIES

SEED = 20261012
DAY = date(2026, 10, 9)


def test_generation_is_deterministic():
    first = generate("healthy", DAY, seed=SEED)
    second = generate("healthy", DAY, seed=SEED)
    assert first.equals(second)


def test_different_seeds_give_different_data():
    assert not generate("healthy", DAY, seed=SEED).equals(generate("healthy", DAY, seed=SEED + 1))


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_scenario_produces_rows(name: str):
    assert len(generate(name, DAY, seed=SEED)) > 0


def test_healthy_batch_honours_the_contract():
    frame = generate("healthy", DAY, seed=SEED)
    assert set(frame.columns) == set(BY_NAME)
    assert frame["application_id"].is_unique
    assert set(frame["employment_status"].dropna()) <= set(CATEGORIES)
    for column in ("credit_score", "customer_age", "loan_amount", "debt_to_income_ratio"):
        declared = BY_NAME[column]
        values = frame[column].dropna()
        assert values.min() >= declared.minimum
        assert values.max() <= declared.maximum


def test_healthy_missingness_is_present_but_small():
    """Zero nulls would be unrealistic, and a threshold calibrated against zero
    would alert on the first ordinary day."""
    frame = generate("healthy", DAY, seed=SEED)
    rate = frame["annual_income"].isna().mean()
    assert 0 < rate < 0.05


def test_approval_rate_is_realistic_and_tracks_credit():
    frame = generate("healthy", DAY, seed=SEED)
    assert 0.40 < frame["approved"].mean() < 0.75
    scored = frame.dropna(subset=["credit_score"])
    low = scored[scored.credit_score < scored.credit_score.quantile(0.33)]
    high = scored[scored.credit_score > scored.credit_score.quantile(0.67)]
    assert high.approved.mean() > low.approved.mean() + 0.15


def test_volume_varies_between_days():
    """A generator with constant volume makes the volume check untestable and
    produced a threshold sweep that reported every floor as free of false
    alarms. That number was an artefact."""
    counts = {
        len(generate("healthy", DAY - timedelta(days=offset), seed=SEED))
        for offset in range(20)
        if (DAY - timedelta(days=offset)).weekday() < 5
    }
    assert len(counts) > 1
    nominal = expected_volume(date(2026, 10, 9))
    assert all(abs(c - nominal) / nominal < 5 * VOLUME_NOISE for c in counts)


def test_weekend_volume_is_lower_than_weekday():
    saturday, friday = date(2026, 10, 10), date(2026, 10, 9)
    assert expected_volume(saturday) < expected_volume(friday)


def test_intensity_scales_the_injected_fault():
    mild = generate("missing_income", DAY, seed=SEED, intensity=0.05)
    severe = generate("missing_income", DAY, seed=SEED, intensity=0.30)
    assert mild["annual_income"].isna().mean() < severe["annual_income"].isna().mean()


def test_unseen_category_scales_with_severity():
    """Held fixed, it answered for drift at every intensity and the sensitivity
    sweep measured the category check instead."""
    mild = generate("population_shift", DAY, seed=SEED, intensity=0.02)
    severe = generate("population_shift", DAY, seed=SEED, intensity=0.50)
    share = lambda f: (f["employment_status"] == "gig_economy").mean()  # noqa: E731
    assert share(mild) < share(severe)


def test_income_shift_breaks_no_deterministic_rule():
    """The fault only drift can see: every value stays legal."""
    frame = generate("income_shift", DAY, seed=SEED)
    assert set(frame.columns) == set(BY_NAME)
    assert frame["application_id"].is_unique
    assert set(frame["employment_status"].dropna()) <= set(CATEGORIES)
    assert frame["credit_score"].dropna().between(300, 850).all()
    assert frame["loan_amount"].dropna().gt(0).all()


def test_failure_scenarios_declare_what_they_expect():
    for name, scenario in SCENARIOS.items():
        if scenario.kind == "failure":
            assert scenario.expected, f"{name} must declare its incident categories"
        else:
            assert not scenario.expected, f"{name} must expect silence"


def test_every_swept_scenario_has_a_default_intensity():
    from datasentinel.monitoring.sensitivity import SWEEPS

    assert set(SWEEPS) <= set(DEFAULT_INTENSITY)
