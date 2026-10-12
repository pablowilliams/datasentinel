"""Statistical primitives for comparing a current batch against a reference.

Each test returns both a significance and a magnitude, because on a batch of a
thousand rows significance is cheap. A 1.5% shift in median income will clear
p < 0.05 comfortably and means nothing to anyone. The decision rule in
`detector` requires both, and these functions exist to supply both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

EPSILON = 1e-6


@dataclass(frozen=True)
class TestResult:
    test: str
    statistic: float
    p_value: float | None
    effect_size: float
    effect_name: str
    reference_rows: int
    current_rows: int
    detail: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.detail is None:
            object.__setattr__(self, "detail", {})


def _clean(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.dropna().to_numpy(dtype=float)


def population_stability_index(
    reference: pd.Series, current: pd.Series, *, bins: int = 10
) -> float:
    """PSI, the credit-risk industry's measure of how far a population has moved.

    Bins come from the reference quantiles, so each reference bucket holds about
    a tenth of the history and the index measures where the current batch sits
    against it. Convention: below 0.1 is stable, 0.1 to 0.25 is worth a look,
    above 0.25 is a different population.
    """
    ref = _clean(reference)
    cur = _clean(current)
    if ref.size < bins or cur.size == 0:
        return 0.0

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_share = np.histogram(ref, bins=edges)[0] / ref.size
    cur_share = np.histogram(cur, bins=edges)[0] / cur.size
    ref_share = np.clip(ref_share, EPSILON, None)
    cur_share = np.clip(cur_share, EPSILON, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def numeric_drift(reference: pd.Series, current: pd.Series) -> TestResult:
    """Kolmogorov-Smirnov for significance, PSI for magnitude."""
    ref, cur = _clean(reference), _clean(current)
    if ref.size < 30 or cur.size < 30:
        return TestResult("kolmogorov_smirnov", 0.0, None, 0.0, "psi", ref.size, cur.size)

    result = stats.ks_2samp(ref, cur)
    psi = population_stability_index(reference, current)
    ref_median = float(np.median(ref))
    cur_median = float(np.median(cur))
    relative = (cur_median - ref_median) / ref_median if ref_median else 0.0
    return TestResult(
        test="kolmogorov_smirnov",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        effect_size=psi,
        effect_name="psi",
        reference_rows=int(ref.size),
        current_rows=int(cur.size),
        detail={
            "reference_median": round(ref_median, 4),
            "current_median": round(cur_median, 4),
            "relative_median_change": round(relative, 4),
        },
    )


def categorical_drift(reference: pd.Series, current: pd.Series) -> TestResult:
    """Chi-squared for significance, total variation distance for magnitude.

    Total variation is the largest share of the population that would have to be
    relabelled to turn one distribution into the other, which is a quantity a
    non-statistician can act on.
    """
    ref = reference.dropna().astype(str)
    cur = current.dropna().astype(str)
    if ref.empty or cur.empty:
        return TestResult("chi_squared", 0.0, None, 0.0, "total_variation", ref.size, cur.size)

    levels = sorted(set(ref.unique()) | set(cur.unique()))
    ref_counts = ref.value_counts().reindex(levels, fill_value=0)
    cur_counts = cur.value_counts().reindex(levels, fill_value=0)
    ref_share = (ref_counts / ref_counts.sum()).to_numpy()
    cur_share = (cur_counts / cur_counts.sum()).to_numpy()
    total_variation = float(0.5 * np.abs(cur_share - ref_share).sum())

    # Chi-squared compares the observed current counts against the counts the
    # reference shares predict. Sparse levels are pooled away by the clip.
    expected = np.clip(ref_share * cur_counts.sum(), EPSILON, None)
    statistic = float(np.sum((cur_counts.to_numpy() - expected) ** 2 / expected))
    degrees = max(len(levels) - 1, 1)
    p_value = float(stats.chi2.sf(statistic, degrees))

    unseen = sorted(set(cur.unique()) - set(ref.unique()))
    return TestResult(
        test="chi_squared",
        statistic=statistic,
        p_value=p_value,
        effect_size=total_variation,
        effect_name="total_variation",
        reference_rows=int(ref.size),
        current_rows=int(cur.size),
        detail={"unseen_levels": len(unseen), "degrees_of_freedom": degrees},
    )


def missingness_drift(reference: pd.Series, current: pd.Series) -> TestResult:
    """A two-proportion z-test on null rates, with the absolute gap as magnitude.

    Often the most actionable signal in the set: a column that quietly stops
    arriving does not change the distribution of the values that do arrive.
    """
    ref_rate = float(reference.isna().mean()) if len(reference) else 0.0
    cur_rate = float(current.isna().mean()) if len(current) else 0.0
    n_ref, n_cur = len(reference), len(current)
    if n_ref < 30 or n_cur < 30:
        return TestResult("two_proportion_z", 0.0, None, 0.0, "rate_difference", n_ref, n_cur)

    pooled = (reference.isna().sum() + current.isna().sum()) / (n_ref + n_cur)
    standard_error = np.sqrt(max(pooled * (1 - pooled) * (1 / n_ref + 1 / n_cur), EPSILON))
    z = (cur_rate - ref_rate) / standard_error
    return TestResult(
        test="two_proportion_z",
        statistic=float(z),
        p_value=float(2 * stats.norm.sf(abs(z))),
        effect_size=abs(cur_rate - ref_rate),
        effect_name="rate_difference",
        reference_rows=n_ref,
        current_rows=n_cur,
        detail={"reference_rate": round(ref_rate, 5), "current_rate": round(cur_rate, 5)},
    )
