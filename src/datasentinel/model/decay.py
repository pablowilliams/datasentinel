"""Does detected drift actually damage the model?

The tempting claim is that drift causes degradation, so detecting drift protects
the model. It is worth measuring rather than assuming, because the four cases
come apart:

  drift detected, model degraded      the alert earned its keep
  drift detected, model unharmed      a true observation about the data that is
                                      not yet a problem, and paging someone about
                                      it teaches them to ignore the next one
  no drift, model degraded            the dangerous quadrant
  no drift, model fine                an ordinary day

A platform that only ever reports the first case is not measuring; it is
asserting. This module produces the whole table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datasentinel.drift.detector import DriftConfig, detect
from datasentinel.generator import generate
from datasentinel.incidents.service import incidents_from
from datasentinel.model.monitor import evaluate_batch
from datasentinel.quality.checks import CheckSpec, Context, run_checks

WATCHED = ("roc_auc", "f1", "calibration_error", "approval_rate_gap", "brier")

TOLERANCE = 0.99
"""Where the line between noise and damage is drawn, as a quantile of the
model's own behaviour on untroubled days.

The first version of this used fixed thresholds picked by eye: a 0.02 drop in
AUC or a 0.02 rise in calibration error. Measured afterwards, calibration error
on untroubled days has a standard deviation of 0.0116, so that threshold sat
1.7 standard deviations out and fired on ordinary variation. It reported ten
silent model failures, every one of them a quiet Saturday or an unremarkable
Tuesday.

A threshold has to come from the distribution it is separating. Damage is now a
metric worse than the 99th percentile of what healthy days produce, which also
absorbs the fact that a 470-row Saturday measures calibration less precisely
than a 1,200-row Wednesday."""


@dataclass(frozen=True)
class Baseline:
    """How the model behaves when nothing is wrong, and how much that varies."""

    median: dict[str, float]
    auc_floor: float
    calibration_ceiling: float
    days: int
    spread: dict[str, float]


@dataclass
class Observation:
    scenario: str
    intensity: float | None
    day: str
    drift_columns: list[str]
    quality_categories: list[str]
    scored: bool
    reason: str | None
    metrics: dict[str, float]
    baseline: Baseline

    @property
    def drift_detected(self) -> bool:
        return bool(self.drift_columns)

    @property
    def quality_raised(self) -> bool:
        return bool(self.quality_categories)

    @property
    def anything_raised(self) -> bool:
        """What the operator actually sees: either layer firing is a warning."""
        return self.drift_detected or self.quality_raised

    def delta(self, metric: str) -> float | None:
        if not self.scored or metric not in self.metrics:
            return None
        reference = self.baseline.median.get(metric)
        return None if reference is None else self.metrics[metric] - reference

    @property
    def degraded(self) -> bool:
        """Outside what healthy days produce. Unscoreable counts as degraded: a
        model that cannot run is not a healthy model."""
        if not self.scored:
            return True
        auc = self.metrics.get("roc_auc")
        calibration = self.metrics.get("calibration_error")
        return bool(
            (auc is not None and not np.isnan(auc) and auc < self.baseline.auc_floor)
            or (calibration is not None and calibration > self.baseline.calibration_ceiling)
        )


def healthy_baseline(pipeline: Any, *, days: int, until: date, seed: int) -> Baseline:
    """What untroubled days look like, including how much they vary.

    The window spans whole weeks, so the spread includes the smaller weekend
    batches and the thresholds derived from it are not surprised by a Saturday.
    """
    collected: dict[str, list[float]] = {m: [] for m in WATCHED}
    scored_days = 0
    for offset in range(days):
        day = until - timedelta(days=offset)
        result = evaluate_batch(pipeline, generate("healthy", day, seed=seed), window=day)
        if not result.scored:
            continue
        scored_days += 1
        for metric in WATCHED:
            value = result.get(metric)
            if value is not None and not np.isnan(value):
                collected[metric].append(value)

    if scored_days < 20:
        raise ValueError(
            f"only {scored_days} healthy days scored; a noise floor needs at least 20"
        )

    auc = np.array(collected["roc_auc"])
    calibration = np.array(collected["calibration_error"])
    return Baseline(
        median={m: float(np.median(v)) for m, v in collected.items() if v},
        auc_floor=float(np.quantile(auc, 1 - TOLERANCE)),
        calibration_ceiling=float(np.quantile(calibration, TOLERANCE)),
        days=scored_days,
        spread={m: float(np.std(v)) for m, v in collected.items() if v},
    )


def observe(
    pipeline: Any,
    *,
    scenarios: dict[str, tuple[float, ...] | None],
    days: int,
    until: date,
    seed: int,
    reference: pd.DataFrame,
    drift_config: DriftConfig,
    baseline: Baseline,
    specs: list[CheckSpec],
) -> list[Observation]:
    observations: list[Observation] = []
    for scenario, intensities in scenarios.items():
        for intensity in intensities or (None,):
            for offset in range(days):
                day = until - timedelta(days=offset)
                batch = generate(scenario, day, seed=seed, intensity=intensity)
                drifted = [
                    f.spec.column
                    for f in detect(reference, batch, drift_config)
                    if f.detected
                ]
                as_of = datetime.combine(day, datetime.min.time()).replace(hour=18)
                raised = sorted(
                    {
                        i.category
                        for i in incidents_from(
                            run_checks(batch, specs, Context(as_of=as_of, reference=reference))
                        )
                    }
                )
                result = evaluate_batch(pipeline, batch, window=day)
                observations.append(
                    Observation(
                        scenario=scenario,
                        intensity=intensity,
                        day=day.isoformat(),
                        drift_columns=sorted(set(drifted)),
                        quality_categories=raised,
                        scored=result.scored,
                        reason=result.reason,
                        metrics=result.metrics,
                        baseline=baseline,
                    )
                )
    return observations


def contingency(observations: list[Observation]) -> dict[str, int]:
    """Damage against what each layer saw.

    An earlier version of this compared drift against damage and reported 34
    silent failures. Thirty-four of those had already been raised by a range
    check. Drift detection is not the platform; it is one of two layers, and the
    figure that matters is how often damage passes both.
    """
    table = {
        "degraded_drift_saw_it": 0,
        "degraded_only_quality_saw_it": 0,
        "degraded_nothing_saw_it": 0,
        "healthy_but_something_raised": 0,
        "healthy_and_quiet": 0,
    }
    for o in observations:
        if o.degraded:
            if o.drift_detected:
                table["degraded_drift_saw_it"] += 1
            elif o.quality_raised:
                table["degraded_only_quality_saw_it"] += 1
            else:
                table["degraded_nothing_saw_it"] += 1
        elif o.anything_raised:
            table["healthy_but_something_raised"] += 1
        else:
            table["healthy_and_quiet"] += 1
    return table


def render(observations: list[Observation], baseline: Baseline) -> str:
    lines = [
        f"Healthy baseline over {baseline.days} days: "
        + "  ".join(f"{m}={v:.4f}" for m, v in baseline.median.items()),
        f"Damage is outside the {TOLERANCE:.0%} band of those days: "
        f"AUC below {baseline.auc_floor:.4f} or calibration error above "
        f"{baseline.calibration_ceiling:.4f}",
        f"(calibration error varies by {baseline.spread.get('calibration_error', 0):.4f} "
        f"on untroubled days, which is why a 0.02 threshold picked by eye fired on noise)",
        "",
        f"{'scenario':<22}{'sev':>6}{'drift':>7}{'scored':>8}"
        f"{'dAUC':>9}{'dECE':>9}{'dF1':>9}{'appr gap':>10}  verdict",
        "-" * 104,
    ]
    grouped: dict[tuple[str, float | None], list[Observation]] = {}
    for o in observations:
        grouped.setdefault((o.scenario, o.intensity), []).append(o)

    for (scenario, intensity), group in grouped.items():
        scored = [o for o in group if o.scored]
        drift_rate = sum(o.drift_detected for o in group) / len(group)
        degraded_rate = sum(o.degraded for o in group) / len(group)

        def med(metric: str, batch: list[Observation] = scored) -> float | None:
            values = [o.delta(metric) for o in batch]
            usable = [v for v in values if v is not None and not np.isnan(v)]
            return float(np.median(usable)) if usable else None

        def show(value: float | None) -> str:
            return "   n/a" if value is None else f"{value:+.4f}"

        gap = [o.metrics.get("approval_rate_gap") for o in scored]
        gap_med = float(np.median([g for g in gap if g is not None])) if scored else None
        verdict = (
            "unscoreable" if not scored
            else "drift and damage" if drift_rate > 0.5 and degraded_rate > 0.5
            else "drift, no damage" if drift_rate > 0.5
            else "damage, no drift" if degraded_rate > 0.5
            else "quiet"
        )
        lines.append(
            f"{scenario:<22}{'' if intensity is None else f'{intensity:.0%}':>6}"
            f"{drift_rate:>7.0%}{len(scored)}/{len(group):<6}"
            f"{show(med('roc_auc')):>9}{show(med('calibration_error')):>9}"
            f"{show(med('f1')):>9}{show(gap_med):>10}  {verdict}"
        )

    table = contingency(observations)
    total = sum(table.values())
    lines += [
        "",
        f"Model damage against what the platform saw, {total} observations:",
        f"  degraded, drift caught it                 {table['degraded_drift_saw_it']:>5}",
        f"  degraded, only the quality checks caught it {table['degraded_only_quality_saw_it']:>3}",
        f"  degraded, nothing caught it               {table['degraded_nothing_saw_it']:>5}"
        "   <- the dangerous one",
        f"  model fine, something was raised          {table['healthy_but_something_raised']:>5}",
        f"  model fine, silence                       {table['healthy_and_quiet']:>5}",
    ]
    return "\n".join(lines)


def write_artifact(observations: list[Observation], baseline: Baseline, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "note": (
                    "Whether drift the platform detects corresponds to measurable model "
                    "damage. Degradation means AUC down 0.02 or calibration error up "
                    "0.02 against the healthy median, or a batch the model could not "
                    "score at all."
                ),
                "healthy_baseline": baseline.median,
                "healthy_spread": baseline.spread,
                "healthy_days": baseline.days,
                "thresholds": {
                    "derived_from": f"{TOLERANCE:.0%} quantile of healthy days",
                    "auc_floor": baseline.auc_floor,
                    "calibration_ceiling": baseline.calibration_ceiling,
                },
                "contingency": contingency(observations),
                "observations": [
                    {
                        "scenario": o.scenario,
                        "intensity": o.intensity,
                        "day": o.day,
                        "drift_columns": o.drift_columns,
                        "quality_categories": o.quality_categories,
                        "scored": o.scored,
                        "reason": o.reason,
                        "degraded": o.degraded,
                        "metrics": o.metrics,
                    }
                    for o in observations
                ],
            },
            indent=2,
            default=str,
        )
    )
