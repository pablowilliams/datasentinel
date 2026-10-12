"""Scoring a batch and recording how the model did on it.

Calibration is tracked alongside discrimination because they fail separately. A
model can keep its ranking ability while its probabilities drift away from the
truth, and a credit decision made on a threshold cares about the probability,
not the ranking.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import Engine, text

from datasentinel.model.baseline import build_features
from datasentinel.model.features import LABEL

DECISION_THRESHOLD = 0.5


@dataclass
class BatchMetrics:
    window: date
    rows: int
    metrics: dict[str, float] = field(default_factory=dict)
    scored: bool = True
    reason: str | None = None

    def get(self, name: str) -> float | None:
        return self.metrics.get(name)


def expected_calibration_error(truth: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    """Average gap between predicted confidence and observed frequency.

    Reported because a threshold is applied to the probability. A model that
    ranks correctly but whose probabilities have drifted will approve the wrong
    number of people while its AUC looks untouched.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for bucket in range(bins):
        mask = index == bucket
        if not mask.any():
            continue
        error += mask.mean() * abs(truth[mask].mean() - probability[mask].mean())
    return float(error)


def evaluate_batch(pipeline: Any, frame: pd.DataFrame, *, window: date) -> BatchMetrics:
    """Metrics for one batch, or a stated reason why it could not be scored."""
    if LABEL not in frame.columns:
        return BatchMetrics(window, len(frame), scored=False,
                            reason="label absent; the batch cannot be evaluated")
    try:
        designed = build_features(frame)
    except ValueError as exc:
        # A schema break stops the model dead. That is the correct behaviour and
        # it is worth recording, because a silently rescued model is worse.
        return BatchMetrics(window, len(frame), scored=False, reason=str(exc))

    labelled = frame[LABEL].notna()
    if labelled.sum() < 30:
        return BatchMetrics(window, len(frame), scored=False, reason="fewer than 30 labelled rows")

    truth = frame.loc[labelled, LABEL].astype(bool).to_numpy()
    probability = pipeline.predict_proba(designed.loc[labelled])[:, 1]
    predicted = probability >= DECISION_THRESHOLD

    metrics = {
        "roc_auc": float(roc_auc_score(truth, probability)) if truth.std() else float("nan"),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "brier": float(brier_score_loss(truth, probability)),
        "calibration_error": expected_calibration_error(truth, probability),
        "predicted_approval_rate": float(predicted.mean()),
        "actual_approval_rate": float(truth.mean()),
        # The gap below is what an operator notices first: the model approving a
        # different share of applicants than the business is actually approving.
        "approval_rate_gap": float(predicted.mean() - truth.mean()),
    }
    return BatchMetrics(window, int(labelled.sum()), metrics=metrics)


def record(
    eng: Engine,
    result: BatchMetrics,
    *,
    model_name: str,
    model_version: str,
) -> int:
    """Persist a batch's metrics, one row per metric."""
    if not result.scored:
        return 0
    with eng.begin() as conn:
        for name, value in result.metrics.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            conn.execute(
                text(
                    """
                    INSERT INTO model_metrics (metric_id, model_name, model_version,
                        evaluation_window, metric_name, metric_value, sample_count)
                    VALUES (:id, :name, :version, :window, :metric, :value, :rows)
                    """
                ),
                {
                    "id": uuid.uuid4(), "name": model_name, "version": model_version,
                    "window": result.window, "metric": name, "value": value,
                    "rows": result.rows,
                },
            )
    return len(result.metrics)


def history(eng: Engine, *, model_name: str, metric: str) -> pd.DataFrame:
    with eng.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT evaluation_window, metric_value, sample_count
                FROM model_metrics
                WHERE model_name = :name AND metric_name = :metric
                ORDER BY evaluation_window
                """
            ),
            {"name": model_name, "metric": metric},
        )
        return pd.DataFrame(rows.mappings().all())
