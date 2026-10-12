"""Where does detection actually start?

Recall of 100% against a fault chosen to be obvious measures the choice of
fault, not the detector. The number that characterises a monitoring platform is
the smallest fault it reliably catches, so this sweeps each injected fault from
negligible to severe and reports the detection rate at every step.

The resulting boundary is the honest headline. It is also the number an operator
needs: knowing that income missingness is caught from 6% upward tells you what
this platform will and will not protect you from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from datasentinel.drift.detector import detect, load_config, to_incident_payloads
from datasentinel.generator import SCENARIOS, generate
from datasentinel.incidents.service import incidents_from
from datasentinel.quality.checks import Context, run_checks
from datasentinel.quality.runner import load_specs

SWEEPS: dict[str, tuple[float, ...]] = {
    "missing_income": (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20),
    "duplicates": (0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.18),
    "invalid_values": (0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.06),
    "volume_drop": (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60),
    "population_shift": (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50),
    "income_shift": (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50),
}

UNITS = {
    "missing_income": "null rate forced into annual_income",
    "duplicates": "share of batch duplicated",
    "invalid_values": "share of rows corrupted",
    "volume_drop": "share of rows lost",
    "population_shift": "fractional uplift in income, plus a new category",
    "income_shift": "fractional uplift in income, nothing else wrong",
}


@dataclass
class Point:
    scenario: str
    intensity: float
    days: int
    detected: int

    @property
    def rate(self) -> float:
        return self.detected / self.days if self.days else 0.0


def sweep(
    *,
    reference: pd.DataFrame,
    days: int,
    until: date,
    seed: int,
    config_dir: Path,
) -> list[Point]:
    specs = load_specs(config_dir / "checks.yml")
    drift_config = load_config(config_dir / "drift.yml")
    points: list[Point] = []

    for scenario, intensities in SWEEPS.items():
        expected = set(SCENARIOS[scenario].expected)
        for intensity in intensities:
            detected = 0
            for offset in range(days):
                day = until - timedelta(days=offset)
                batch = generate(scenario, day, seed=seed, intensity=intensity)
                as_of = datetime.combine(day, datetime.min.time()).replace(hour=18)
                raised = {
                    i.category
                    for i in incidents_from(
                        run_checks(batch, specs, Context(as_of=as_of, reference=reference))
                    )
                }
                raised |= {
                    p["category"]
                    for p in to_incident_payloads(detect(reference, batch, drift_config))
                }
                detected += int(bool(expected & raised))
            points.append(Point(scenario, intensity, days, detected))
    return points


def detection_floor(points: list[Point], scenario: str, *, threshold: float = 0.9) -> float | None:
    """The smallest swept intensity caught on at least `threshold` of days."""
    for point in sorted((p for p in points if p.scenario == scenario), key=lambda p: p.intensity):
        if point.rate >= threshold:
            return point.intensity
    return None


def render(points: list[Point]) -> str:
    lines = ["Detection rate by fault severity", ""]
    for scenario in SWEEPS:
        rows = [p for p in points if p.scenario == scenario]
        floor = detection_floor(points, scenario)
        lines.append(f"{scenario}  ({UNITS[scenario]})")
        lines.append(
            "    " + "  ".join(f"{p.intensity:>6.1%}" for p in rows)
        )
        lines.append(
            "    " + "  ".join(f"{p.rate:>6.0%}" for p in rows)
        )
        lines.append(
            "    reliably caught (>=90% of days) from "
            + (f"{floor:.1%}" if floor is not None else "nowhere in the swept range")
        )
        lines.append("")
    return "\n".join(lines)


def write_artifact(points: list[Point], path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "note": (
                    "Detection rate against injected fault severity. The floor is the "
                    "smallest fault caught on at least 90% of days, and is a more "
                    "informative figure than recall against a fault chosen to be large."
                ),
                "units": UNITS,
                "floors": {s: detection_floor(points, s) for s in SWEEPS},
                "points": [
                    {
                        "scenario": p.scenario,
                        "intensity": p.intensity,
                        "days": p.days,
                        "detected": p.detected,
                        "rate": p.rate,
                    }
                    for p in points
                ],
            },
            indent=2,
        )
    )
