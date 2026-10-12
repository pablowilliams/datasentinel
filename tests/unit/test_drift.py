from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from datasentinel.drift.detector import benjamini_hochberg, detect, load_config
from datasentinel.drift.distributions import (
    categorical_drift,
    missingness_drift,
    numeric_drift,
    population_stability_index,
)
from datasentinel.generator import generate

SEED = 20261012
DAY = date(2026, 10, 9)


@pytest.fixture(scope="module")
def config(request):
    from datasentinel.config import ROOT

    return load_config(ROOT / "config" / "drift.yml")


def test_psi_is_zero_for_the_same_distribution():
    rng = np.random.default_rng(1)
    values = pd.Series(rng.normal(size=20_000))
    assert population_stability_index(values, values) < 0.01


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(2)
    reference = pd.Series(rng.normal(size=20_000))
    small = pd.Series(rng.normal(0.05, 1, 5_000))
    large = pd.Series(rng.normal(1.50, 1, 5_000))
    assert population_stability_index(reference, small) < population_stability_index(
        reference, large
    )


def test_numeric_drift_reports_both_significance_and_magnitude():
    rng = np.random.default_rng(3)
    result = numeric_drift(pd.Series(rng.normal(size=5_000)), pd.Series(rng.normal(2, 1, 1_000)))
    assert result.p_value is not None and result.p_value < 0.01
    assert result.effect_size > 0.25
    assert result.effect_name == "psi"


def test_small_samples_return_no_verdict():
    rng = np.random.default_rng(4)
    result = numeric_drift(pd.Series(rng.normal(size=10)), pd.Series(rng.normal(size=10)))
    assert result.p_value is None, "ten rows cannot support a finding"


def test_categorical_drift_notices_a_new_level():
    reference = pd.Series(["a"] * 600 + ["b"] * 400)
    current = pd.Series(["a"] * 300 + ["b"] * 200 + ["z"] * 150)
    result = categorical_drift(reference, current)
    assert result.detail["unseen_levels"] == 1
    assert result.effect_size > 0.1


def test_missingness_drift_measures_the_rate_gap():
    reference = pd.Series([1.0] * 990 + [None] * 10)
    current = pd.Series([1.0] * 800 + [None] * 200)
    result = missingness_drift(reference, current)
    assert result.effect_size == pytest.approx(0.19, abs=0.01)
    assert result.p_value is not None and result.p_value < 0.001


def test_a_trivial_shift_is_significant_but_not_material():
    """The measured case the second gate exists for: at a hundred thousand rows a
    1.5% shift gives p around 1e-9 while PSI stays near 0.001."""
    rng = np.random.default_rng(5)
    reference = pd.Series(np.exp(rng.normal(10.55, 0.52, 90_000)))
    current = pd.Series(np.exp(rng.normal(10.55, 0.52, 100_000)) * 1.015)
    result = numeric_drift(reference, current)
    assert result.p_value < 0.05
    assert result.effect_size < 0.05


def test_both_gates_are_required(config, reference):
    current = generate("trivial_income_tick", DAY, seed=SEED)
    for finding in detect(reference, current, config):
        if finding.detected:
            assert finding.significant and finding.material


def test_income_shift_is_detected_by_drift(config, reference):
    """The fault no deterministic rule can see."""
    findings = detect(reference, generate("income_shift", DAY, seed=SEED), config)
    assert [f.spec.column for f in findings if f.detected]


def test_healthy_and_benign_days_produce_no_drift(config, reference):
    for scenario in ("healthy", "minor_noise", "trivial_income_tick"):
        detected = [
            f.spec.column
            for f in detect(reference, generate(scenario, DAY, seed=SEED), config)
            if f.detected
        ]
        assert detected == [], f"{scenario} drifted on {detected}"


def test_explanations_say_which_gate_closed(config, reference):
    for finding in detect(reference, generate("trivial_income_tick", DAY, seed=SEED), config):
        if finding.significant and not finding.material:
            assert "not materially" in finding.explanation


def test_benjamini_hochberg_rejects_nothing_when_all_p_values_are_large():
    assert benjamini_hochberg([0.4, 0.6, 0.9], 0.05) == [False, False, False]


def test_benjamini_hochberg_keeps_a_clearly_small_p_value():
    assert benjamini_hochberg([0.0001, 0.6, 0.9], 0.05)[0] is True


def test_benjamini_hochberg_is_stricter_than_an_uncorrected_threshold():
    """Ten tests, one at p=0.04: significant alone, not significant among ten."""
    p_values = [0.04] + [0.5] * 9
    assert p_values[0] < 0.05
    assert benjamini_hochberg(p_values, 0.05)[0] is False


def test_benjamini_hochberg_ignores_absent_p_values():
    assert benjamini_hochberg([None, None], 0.05) == [False, False]
