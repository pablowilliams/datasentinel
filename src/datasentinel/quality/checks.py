"""Deterministic data quality checks.

These answer "is this data valid?" — a question with a knowable answer, unlike
the one the drift module asks. Every check is declared in YAML and resolved
here, so adding a rule never means editing the pipeline.

A failing check always reports four things: what was expected, what was
observed, how severe it is, and how much data is affected. An alert that does
not say that is a reason to go and read the SQL, which is not an alert.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from datasentinel.schema import BY_NAME, REQUIRED

Severity = str


@dataclass(frozen=True)
class CheckSpec:
    name: str
    type: str
    severity: Severity
    column: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckResult:
    spec: CheckSpec
    passed: bool
    message: str
    observed: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)

    @property
    def affected_rows(self) -> int:
        value = self.observed.get("affected_rows")
        return int(value) if value is not None else 0


@dataclass(frozen=True)
class Context:
    """Everything a check needs besides the data itself."""

    as_of: datetime
    reference: pd.DataFrame | None = None
    """A healthy window of history. Volume checks are meaningless without it."""


Checker = Callable[[pd.DataFrame, CheckSpec, Context], CheckResult]
REGISTRY: dict[str, Checker] = {}


def register(name: str) -> Callable[[Checker], Checker]:
    def wrap(fn: Checker) -> Checker:
        REGISTRY[name] = fn
        return fn

    return wrap


def _ok(spec: CheckSpec, message: str, **kw: Any) -> CheckResult:
    return CheckResult(spec=spec, passed=True, message=message, **kw)


def _fail(spec: CheckSpec, message: str, **kw: Any) -> CheckResult:
    return CheckResult(spec=spec, passed=False, message=message, **kw)


@register("schema_required")
def schema_required(frame: pd.DataFrame, spec: CheckSpec, ctx: Context) -> CheckResult:
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        return _fail(
            spec,
            f"required columns absent: {', '.join(missing)}",
            observed={"missing": missing, "present": list(frame.columns), "affected_rows": len(frame)},
            expected={"required": list(REQUIRED)},
        )
    return _ok(spec, "all required columns present", expected={"required": list(REQUIRED)})


@register("schema_unexpected")
def schema_unexpected(frame: pd.DataFrame, spec: CheckSpec, ctx: Context) -> CheckResult:
    unexpected = [c for c in frame.columns if c not in BY_NAME]
    if unexpected:
        return _fail(
            spec,
            f"columns not in the contract: {', '.join(unexpected)}",
            observed={"unexpected": unexpected, "affected_rows": len(frame)},
            expected={"contract": list(BY_NAME)},
        )
    return _ok(spec, "no unexpected columns")


@register("schema_type")
def schema_type(frame: pd.DataFrame, spec: CheckSpec, ctx: Context) -> CheckResult:
    column = spec.column or ""
    if column not in frame.columns:
        return _ok(spec, f"{column} absent; the required-columns check owns this")
    declared = BY_NAME[column].dtype
    series = frame[column]
    coerced = pd.to_numeric(series, errors="coerce") if declared in {"int", "float"} else series
    if declared in {"int", "float"}:
        broke = int((coerced.isna() & series.notna()).sum())
        if broke:
            return _fail(
                spec,
                f"{column} declared {declared} but {broke:,} values will not parse as a number",
                observed={"unparseable": broke, "dtype": str(series.dtype), "affected_rows": broke},
                expected={"dtype": declared},
            )
    return _ok(spec, f"{column} parses as {declared}", observed={"dtype": str(series.dtype)})


@register("null_rate")
def null_rate(frame: pd.DataFrame, spec: CheckSpec, ctx: Context) -> CheckResult:
    column = spec.column or ""
    limit = float(spec.options["max_rate"])
    if column not in frame.columns:
        return _ok(spec, f"{column} absent; the required-columns check owns this")
    rate = float(frame[column].isna().mean())
    affected = int(frame[column].isna().sum())
    observed = {"null_rate": round(rate, 5), "affected_rows": affected}
    expected = {"max_null_rate": limit}
    if rate > limit:
        return _fail(
            spec,
            f"{column} is {rate:.1%} null against a {limit:.1%} limit ({affected:,} rows)",
            observed=observed,
            expected=expected,
        )
    return _ok(spec, f"{column} is {rate:.1%} null", observed=observed, expected=expected)


@register("range")
def value_range(frame: pd.DataFrame, spec: CheckSpec, ctx: Context) -> CheckResult:
    column = spec.column or ""
    if column not in frame.columns:
        return _ok(spec, f"{column} absent; the required-columns check owns this")
    declared = BY_NAME[column]
    low = float(spec.options.get("min", declared.minimum if declared.minimum is not None else -1e18))
    high = float(spec.options.get("max", declared.maximum if declared.maximum is not None else 1e18))
    values = pd.to_numeric(frame[column], errors="coerce")
    outside = values.notna() & ((values < low) | (values > high))
    affected = int(outside.sum())
    observed = {
        "out_of_range": affected,
        "rate": round(affected / max(len(frame), 1), 5),
        "affected_rows": affected,
        "sample": [float(v) for v in values[outside].head(3)],
    }
    expected = {"min": low, "max": high}
    if affected:
        return _fail(
            spec,
            f"{column} has {affected:,} values outside [{low:g}, {high:g}]",
            observed=observed,
            expected=expected,
        )
    return _ok(spec, f"{column} within [{low:g}, {high:g}]", observed=observed, expected=expected)


@register("allowed_values")
def allowed_values(frame: pd.DataFrame, spec: CheckSpec, ctx: Context) -> CheckResult:
    column = spec.column or ""
    if column not in frame.columns:
        return _ok(spec, f"{column} absent; the required-columns check owns this")
    permitted = set(spec.options.get("allowed") or BY_NAME[column].allowed or ())
    seen = frame[column].dropna()
    unknown = sorted({str(v) for v in seen.unique()} - {str(v) for v in permitted})
    affected = int(seen.astype(str).isin(unknown).sum()) if unknown else 0
    if unknown:
        return _fail(
            spec,
            f"{column} contains unconfigured categories: {', '.join(unknown)} ({affected:,} rows)",
            observed={"unknown": unknown, "affected_rows": affected},
            expected={"allowed": sorted(str(v) for v in permitted)},
        )
    return _ok(spec, f"{column} categories all known")


@register("unique")
def unique(frame: pd.DataFrame, spec: CheckSpec, ctx: Context) -> CheckResult:
    column = spec.column or ""
    if column not in frame.columns:
        return _ok(spec, f"{column} absent; the required-columns check owns this")
    duplicated = frame[column].duplicated(keep="first")
    affected = int(duplicated.sum())
    rate = affected / max(len(frame), 1)
    observed = {"duplicate_rows": affected, "duplicate_rate": round(rate, 5), "affected_rows": affected}
    if affected:
        return _fail(
            spec,
            f"{column} repeats {affected:,} times ({rate:.1%} of the batch)",
            observed=observed,
            expected={"duplicate_rows": 0},
        )
    return _ok(spec, f"{column} is unique", observed=observed)


@register("freshness")
def freshness(frame: pd.DataFrame, spec: CheckSpec, ctx: Context) -> CheckResult:
    column = spec.column or ""
    limit = float(spec.options["max_age_hours"])
    if column not in frame.columns or frame.empty:
        return _fail(
            spec,
            f"{column} unavailable, so freshness cannot be established",
            observed={"affected_rows": len(frame)},
            expected={"max_age_hours": limit},
        )
    latest = pd.to_datetime(frame[column], errors="coerce").max()
    if pd.isna(latest):
        return _fail(spec, f"{column} holds no parseable dates", expected={"max_age_hours": limit})
    age = (ctx.as_of - latest.to_pydatetime().replace(tzinfo=ctx.as_of.tzinfo)).total_seconds() / 3600
    observed = {"latest": str(latest.date()), "age_hours": round(age, 1), "affected_rows": len(frame)}
    expected = {"max_age_hours": limit}
    if age > limit:
        return _fail(
            spec,
            f"newest record is {age:.0f}h old ({latest.date()}), past the {limit:.0f}h SLA",
            observed=observed,
            expected=expected,
        )
    return _ok(spec, f"newest record is {age:.0f}h old", observed=observed, expected=expected)


@register("volume")
def volume(frame: pd.DataFrame, spec: CheckSpec, ctx: Context) -> CheckResult:
    """Row count against the same weekday in the reference window.

    Comparing against a flat average would flag every Saturday and call it an
    incident. A pipeline that cries wolf each weekend gets muted, and then it
    catches nothing at all.
    """
    low = float(spec.options.get("min_ratio", 0.7))
    high = float(spec.options.get("max_ratio", 1.3))
    rows = len(frame)
    if ctx.reference is None or ctx.reference.empty:
        return _ok(spec, "no reference window yet, so volume is not assessable")

    reference = ctx.reference.copy()
    reference["_date"] = pd.to_datetime(reference["application_date"], errors="coerce")
    per_day = reference.groupby(reference["_date"].dt.date).size()
    if per_day.empty:
        return _ok(spec, "reference window holds no dated rows")

    weekday = pd.to_datetime(frame["application_date"], errors="coerce").dt.weekday.mode()
    same_weekday = per_day[[d.weekday() == int(weekday.iloc[0]) for d in per_day.index]] \
        if not weekday.empty else per_day
    baseline_rows = float((same_weekday if len(same_weekday) >= 3 else per_day).median())

    ratio = rows / baseline_rows if baseline_rows else 0.0
    observed = {
        "rows": rows,
        "weekday_baseline": round(baseline_rows, 1),
        "ratio": round(ratio, 3),
        "affected_rows": abs(rows - int(baseline_rows)),
    }
    expected = {"min_ratio": low, "max_ratio": high, "comparison": "same weekday median"}
    if ratio < low or ratio > high:
        return _fail(
            spec,
            f"{rows:,} rows is {ratio:.0%} of the {baseline_rows:,.0f} typical for this weekday",
            observed=observed,
            expected=expected,
        )
    return _ok(spec, f"{rows:,} rows, {ratio:.0%} of typical", observed=observed, expected=expected)


def run_checks(frame: pd.DataFrame, specs: list[CheckSpec], ctx: Context) -> list[CheckResult]:
    results: list[CheckResult] = []
    for spec in specs:
        checker = REGISTRY.get(spec.type)
        if checker is None:
            raise KeyError(f"no checker registered for type {spec.type!r}")
        results.append(checker(frame, spec, ctx))
    return results
