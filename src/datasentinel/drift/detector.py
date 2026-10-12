"""The drift decision rule.

Data quality asks whether the data is valid. Drift asks whether it is still
describing the same world. A batch can pass every range, null and uniqueness
check and still be a different population, which is the failure mode that
reaches a model without anyone noticing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from datasentinel.drift import distributions as dist

Kind = Literal["numeric", "categorical", "missingness"]

TESTS = {
    "numeric": dist.numeric_drift,
    "categorical": dist.categorical_drift,
    "missingness": dist.missingness_drift,
}


@dataclass(frozen=True)
class DriftSpec:
    column: str
    kind: Kind
    min_effect: float
    severity: str


@dataclass(frozen=True)
class DriftFinding:
    spec: DriftSpec
    result: dist.TestResult
    significant: bool
    material: bool
    explanation: str

    @property
    def detected(self) -> bool:
        """Both gates. Either alone produces an alert nobody thanks you for."""
        return self.significant and self.material


@dataclass(frozen=True)
class DriftConfig:
    alpha: float
    specs: list[DriftSpec]
    fdr_control: bool = True


def benjamini_hochberg(p_values: list[float | None], alpha: float) -> list[bool]:
    """Which p-values survive once you admit how many tests you ran.

    Ten columns are tested every run. At alpha = 0.05 that is a one-in-two
    chance per run of some column looking significant for no reason, and over a
    month of runs it is a certainty. Benjamini-Hochberg controls the share of
    raised findings that are false rather than the chance of any false finding,
    which is the right trade for monitoring: a little tolerance for noise in
    exchange for still being able to see a real shift.

    Measured here: an uncorrected run produced a false alarm on a 470-row
    Saturday, where the approval rate sat 6.7 points off a reference built from
    89,020 rows. Nothing was wrong with the data. There were simply three
    hundred opportunities to be unlucky.
    """
    usable = [(i, p) for i, p in enumerate(p_values) if p is not None]
    rejected = [False] * len(p_values)
    if not usable:
        return rejected

    order = sorted(usable, key=lambda pair: pair[1])
    m = len(order)
    cutoff: float | None = None
    for rank, (_, p) in enumerate(order, start=1):
        if p <= alpha * rank / m:
            cutoff = p
    if cutoff is None:
        return rejected
    for i, p in usable:
        rejected[i] = p <= cutoff
    return rejected


def load_config(path: Path) -> DriftConfig:
    document = yaml.safe_load(path.read_text())
    specs = [
        DriftSpec(
            column=entry["column"],
            kind=entry["kind"],
            min_effect=float(entry["min_effect"]),
            severity=entry["severity"],
        )
        for entry in document.get("columns", [])
    ]
    return DriftConfig(
        alpha=float(document.get("alpha", 0.05)),
        specs=specs,
        fdr_control=bool(document.get("fdr_control", True)),
    )


def _explain(spec: DriftSpec, result: dist.TestResult, significant: bool, material: bool) -> str:
    where = f"{spec.column} ({spec.kind})"
    effect = f"{result.effect_name} {result.effect_size:.3f} against a {spec.min_effect:.2f} gate"
    p = "p<0.001" if result.p_value is not None and result.p_value < 0.001 else (
        f"p={result.p_value:.3f}" if result.p_value is not None else "p unavailable"
    )
    if significant and material:
        detail = ""
        if spec.kind == "numeric":
            change = result.detail.get("relative_median_change", 0.0)
            detail = f"; median moved {change:+.1%}"
        elif spec.kind == "missingness":
            detail = (
                f"; null rate {result.detail.get('reference_rate', 0):.1%}"
                f" to {result.detail.get('current_rate', 0):.1%}"
            )
        elif result.detail.get("unseen_levels"):
            detail = f"; {int(result.detail['unseen_levels'])} category never seen before"
        return f"{where} has shifted: {effect}, {p}{detail}"
    if significant and not material:
        return (
            f"{where} differs detectably but not materially: {effect}, {p}. "
            "Reported, not raised."
        )
    if material and not significant:
        return (
            f"{where} shows a large {effect} that could still be chance, {p}. "
            "Reported, not raised."
        )
    return f"{where} is stable: {effect}, {p}"


def detect(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: DriftConfig,
) -> list[DriftFinding]:
    tested: list[tuple[DriftSpec, dist.TestResult]] = []
    for spec in config.specs:
        if spec.column not in reference.columns or spec.column not in current.columns:
            continue
        tested.append((spec, TESTS[spec.kind](reference[spec.column], current[spec.column])))

    # Significance is decided across the whole run, not per column, because the
    # run is what tests ten columns at once.
    if config.fdr_control:
        verdicts = benjamini_hochberg([r.p_value for _, r in tested], config.alpha)
    else:
        verdicts = [
            r.p_value is not None and r.p_value < config.alpha for _, r in tested
        ]

    findings: list[DriftFinding] = []
    for (spec, result), significant in zip(tested, verdicts, strict=True):
        material = result.effect_size >= spec.min_effect
        findings.append(
            DriftFinding(
                spec=spec,
                result=result,
                significant=significant,
                material=material,
                explanation=_explain(spec, result, significant, material),
            )
        )
    return findings


def to_incident_payloads(findings: list[DriftFinding]) -> list[dict[str, Any]]:
    return [
        {
            "category": "drift",
            "severity": f.spec.severity,
            "column_name": f.spec.column,
            "description": f.explanation,
            "evidence": {
                "test": f.result.test,
                "statistic": round(f.result.statistic, 5),
                "p_value": None if f.result.p_value is None else round(f.result.p_value, 6),
                f.result.effect_name: round(f.result.effect_size, 5),
                "gate": f.spec.min_effect,
                "reference_rows": f.result.reference_rows,
                "current_rows": f.result.current_rows,
                **f.result.detail,
            },
        }
        for f in findings
        if f.detected
    ]
