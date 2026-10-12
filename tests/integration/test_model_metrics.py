from __future__ import annotations

from datetime import date, timedelta

import pytest

from datasentinel.db import reset
from datasentinel.generator import baseline, generate
from datasentinel.model.baseline import train
from datasentinel.model.monitor import evaluate_batch, history, record

pytestmark = pytest.mark.integration

SEED = 20261012
TRAIN_END = date(2026, 8, 1)


@pytest.fixture(scope="module")
def trained():
    return train(baseline(90, TRAIN_END, seed=SEED), version="itest")


@pytest.fixture(scope="module")
def clean(engine):
    reset(engine)
    return engine


def test_metrics_are_persisted_one_row_per_metric(clean, trained):
    pipeline, card = trained
    day = TRAIN_END + timedelta(days=1)
    result = evaluate_batch(pipeline, generate("healthy", day, seed=SEED), window=day)
    written = record(clean, result, model_name=card.name, model_version=card.version)
    assert written == len(result.metrics)

    stored = history(clean, model_name=card.name, metric="roc_auc")
    assert len(stored) == 1
    assert 0.5 < float(stored.metric_value.iloc[0]) < 1.0
    assert int(stored.sample_count.iloc[0]) == result.rows


def test_an_unscoreable_batch_writes_nothing(clean, trained):
    pipeline, card = trained
    before = len(history(clean, model_name=card.name, metric="roc_auc"))
    result = evaluate_batch(
        pipeline, generate("schema_break", TRAIN_END, seed=SEED), window=TRAIN_END
    )
    assert record(clean, result, model_name=card.name, model_version=card.version) == 0
    assert len(history(clean, model_name=card.name, metric="roc_auc")) == before


def test_history_comes_back_in_date_order(clean, trained):
    pipeline, card = trained
    for offset in (7, 3, 5):
        day = TRAIN_END + timedelta(days=offset)
        record(
            clean,
            evaluate_batch(pipeline, generate("healthy", day, seed=SEED), window=day),
            model_name=card.name,
            model_version=card.version,
        )
    stored = history(clean, model_name=card.name, metric="f1")
    windows = list(stored.evaluation_window)
    assert windows == sorted(windows)


def test_degradation_shows_up_in_the_recorded_series(clean, trained):
    """The point of persisting metrics: a shift is visible as a change over time
    rather than a single alarming number."""
    pipeline, card = trained
    for offset, scenario in ((20, "healthy"), (21, "healthy"), (22, "income_shift")):
        day = TRAIN_END + timedelta(days=offset)
        record(
            clean,
            evaluate_batch(pipeline, generate(scenario, day, seed=SEED), window=day),
            model_name=card.name,
            model_version="series",
        )
    stored = history(clean, model_name=card.name, metric="calibration_error")
    series = stored[stored.sample_count > 0].metric_value.astype(float).tolist()
    assert max(series) > min(series)
