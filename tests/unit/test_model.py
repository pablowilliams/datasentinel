from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from datasentinel.generator import baseline, generate
from datasentinel.model.baseline import build_features, load, save, train
from datasentinel.model.features import FEATURES, LABEL, POST_DECISION, check_no_leakage
from datasentinel.model.monitor import evaluate_batch, expected_calibration_error

SEED = 20261012
TRAIN_END = date(2026, 8, 1)


@pytest.fixture(scope="module")
def trained():
    return train(baseline(90, TRAIN_END, seed=SEED), version="test")


def test_interest_rate_is_not_a_feature():
    """It is priced after approval, so at decision time it does not exist. A
    model given it scores beautifully and cannot be deployed."""
    assert "interest_rate" in POST_DECISION
    assert "interest_rate" not in FEATURES


def test_the_label_is_not_a_feature():
    assert LABEL not in FEATURES


def test_leakage_guard_rejects_post_decision_columns():
    with pytest.raises(ValueError, match="not available at decision time"):
        check_no_leakage(["credit_score", "interest_rate"])


def test_leakage_guard_rejects_the_label():
    with pytest.raises(ValueError, match="not available at decision time"):
        check_no_leakage(["credit_score", LABEL])


def test_built_features_contain_no_post_decision_column():
    designed = build_features(generate("healthy", TRAIN_END, seed=SEED))
    assert not set(designed.columns) & (set(POST_DECISION) | {LABEL})


def test_scoring_fails_loudly_when_a_feature_is_missing():
    """A schema break must stop the model, not be silently patched around."""
    with pytest.raises(ValueError, match="features absent"):
        build_features(generate("schema_break", TRAIN_END, seed=SEED))


def test_model_reaches_the_bayes_ceiling(trained):
    """The label is a Bernoulli draw from a known logistic function, so the
    generator's own logit is the best any model can do. Scoring materially above
    it would mean a leak, and materially below it would mean the pipeline is
    broken."""
    pipeline, _ = trained
    frame = generate("healthy", TRAIN_END + timedelta(days=10), seed=SEED)
    needed = [
        "credit_score",
        "debt_to_income_ratio",
        "previous_defaults",
        "loan_amount",
        "annual_income",
    ]
    complete = frame[needed].notna().all(axis=1)
    frame = frame[complete]

    credit = frame.credit_score
    income = frame.annual_income
    oracle = (
        0.55
        + 0.0115 * (credit - 600)
        - 3.4 * frame.debt_to_income_ratio
        - 0.62 * frame.previous_defaults
        - 1.25 * np.clip(frame.loan_amount / income.where(income > 0) - 0.5, 0, None)
    )
    truth = frame[LABEL].astype(bool)
    model_auc = roc_auc_score(truth, pipeline.predict_proba(build_features(frame))[:, 1])
    oracle_auc = roc_auc_score(truth, oracle)
    assert model_auc == pytest.approx(oracle_auc, abs=0.03)


def test_an_unseen_category_does_not_break_scoring(trained):
    """A new employment type is the drift module's finding to report, not an
    exception for the model to raise."""
    pipeline, _ = trained
    frame = generate("population_shift", TRAIN_END + timedelta(days=5), seed=SEED)
    assert "gig_economy" in set(frame.employment_status.dropna())
    result = evaluate_batch(pipeline, frame, window=TRAIN_END)
    assert result.scored


def test_a_batch_without_the_label_is_not_scoreable(trained):
    pipeline, _ = trained
    frame = generate("healthy", TRAIN_END, seed=SEED).drop(columns=[LABEL])
    result = evaluate_batch(pipeline, frame, window=TRAIN_END)
    assert not result.scored and result.reason


def test_a_schema_break_is_reported_not_rescued(trained):
    pipeline, _ = trained
    result = evaluate_batch(
        pipeline, generate("schema_break", TRAIN_END, seed=SEED), window=TRAIN_END
    )
    assert not result.scored
    assert "absent" in (result.reason or "")


def test_tiny_batches_are_not_scored(trained):
    pipeline, _ = trained
    frame = generate("healthy", TRAIN_END, seed=SEED).head(12)
    result = evaluate_batch(pipeline, frame, window=TRAIN_END)
    assert not result.scored


def test_calibration_error_is_zero_for_a_calibrated_forecast():
    rng = np.random.default_rng(11)
    probability = rng.uniform(0, 1, 50_000)
    truth = rng.random(50_000) < probability
    assert expected_calibration_error(truth, probability) < 0.01


def test_calibration_error_catches_a_biased_forecast():
    rng = np.random.default_rng(12)
    truth = rng.random(20_000) < 0.3
    assert expected_calibration_error(truth, np.full(20_000, 0.9)) > 0.5


def test_drift_damages_calibration_more_than_ranking(trained):
    """The finding worth tracking both for: under a large population shift the
    model keeps its ranking and loses its probabilities. A platform watching AUC
    alone would report everything as fine."""
    pipeline, _ = trained
    healthy = evaluate_batch(
        pipeline, generate("healthy", TRAIN_END + timedelta(days=3), seed=SEED), window=TRAIN_END
    )
    shifted = evaluate_batch(
        pipeline,
        generate("income_shift", TRAIN_END + timedelta(days=3), seed=SEED, intensity=0.50),
        window=TRAIN_END,
    )
    auc_change = abs(shifted.get("roc_auc") - healthy.get("roc_auc"))
    calibration_change = shifted.get("calibration_error") - healthy.get("calibration_error")
    assert calibration_change > 0.05
    assert calibration_change > auc_change


def test_approval_gap_widens_under_a_label_shift(trained):
    pipeline, _ = trained
    healthy = evaluate_batch(
        pipeline, generate("healthy", TRAIN_END + timedelta(days=4), seed=SEED), window=TRAIN_END
    )
    shifted = evaluate_batch(
        pipeline,
        generate("income_shift", TRAIN_END + timedelta(days=4), seed=SEED, intensity=0.50),
        window=TRAIN_END,
    )
    assert shifted.get("approval_rate_gap") > healthy.get("approval_rate_gap")


def test_model_card_records_what_was_excluded(trained):
    _, card = trained
    assert card.excluded_as_post_decision == ["interest_rate"]
    assert card.train_rows > 0
    assert "generator" in card.note


def test_a_saved_model_round_trips(trained, tmp_path):
    pipeline, card = trained
    save(pipeline, card, tmp_path)
    restored, restored_card = load(tmp_path, card.name, card.version)
    frame = generate("healthy", TRAIN_END, seed=SEED)
    first = pipeline.predict_proba(build_features(frame))[:, 1]
    second = restored.predict_proba(build_features(frame))[:, 1]
    assert np.allclose(first, second)
    assert restored_card == card


def test_degradation_thresholds_come_from_measured_noise(trained):
    """Fixed thresholds picked by eye sat 1.7 standard deviations out and
    reported ten silent failures, all of them ordinary days."""
    from datasentinel.model.decay import healthy_baseline

    pipeline, _ = trained
    base = healthy_baseline(pipeline, days=42, until=date(2026, 10, 9), seed=SEED)
    spread = base.spread["calibration_error"]
    headroom = base.calibration_ceiling - base.median["calibration_error"]
    assert headroom > 2 * spread, "the ceiling must sit clear of ordinary variation"
    assert base.auc_floor < base.median["roc_auc"]
    assert base.days >= 20


def test_a_noise_floor_refuses_to_form_from_too_few_days(trained):
    from datasentinel.model.decay import healthy_baseline

    pipeline, _ = trained
    with pytest.raises(ValueError, match="at least 20"):
        healthy_baseline(pipeline, days=5, until=date(2026, 10, 9), seed=SEED)
