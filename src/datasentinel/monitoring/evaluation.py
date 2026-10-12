"""Does the platform actually detect what it claims to detect?

Every scenario is run over many days rather than once, because a single draw
measures luck. Detection is scored two ways:

  loose   something was raised
  strict  the incident categories the scenario declared were raised

The strict score is the honest one. A platform that notices a stale file by
complaining about row counts has not detected staleness; it has been lucky in a
way that will not survive the next outage.

What these numbers are: agreement between a generator that injects faults and a
detector configured by the same author, on synthetic data. That is a real test
of the wiring and a weak test of the world. The benign scenarios are what stop
it being circular -- a detector can score perfect recall by alerting always,
and only the false-positive count exposes it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from datasentinel.drift.detector import detect, load_config, to_incident_payloads
from datasentinel.generator import SCENARIOS, baseline, generate
from datasentinel.incidents.service import incidents_from
from datasentinel.quality.checks import Context, run_checks
from datasentinel.quality.runner import load_specs


@dataclass
class DayOutcome:
    scenario: str
    kind: str
    day: str
    expected: list[str]
    raised: list[str]
    incident_count: int

    @property
    def alerted(self) -> bool:
        return self.incident_count > 0

    @property
    def expected_hit(self) -> bool:
        return bool(self.expected) and set(self.expected).issubset(set(self.raised))


@dataclass
class Counts:
    true_positive: int = 0
    false_negative: int = 0
    false_positive: int = 0
    true_negative: int = 0

    @property
    def precision(self) -> float | None:
        flagged = self.true_positive + self.false_positive
        return self.true_positive / flagged if flagged else None

    @property
    def recall(self) -> float | None:
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r else None


@dataclass
class Report:
    days: int
    scenario_days: int
    loose: Counts
    strict: Counts
    per_scenario: dict = field(default_factory=dict)
    false_positive_rate_healthy: float | None = None
    outcomes: list[DayOutcome] = field(default_factory=list)


def evaluate(
    *,
    days: int,
    until: date,
    seed: int,
    config_dir: Path,
    reference_days: int = 90,
) -> Report:
    specs = load_specs(config_dir / "checks.yml")
    drift_config = load_config(config_dir / "drift.yml")
    reference = baseline(reference_days, until - timedelta(days=days), seed=seed)

    outcomes: list[DayOutcome] = []
    for offset in range(days):
        day = until - timedelta(days=offset)
        as_of = datetime.combine(day, datetime.min.time()).replace(hour=18)
        for name, scenario in SCENARIOS.items():
            batch = generate(name, day, seed=seed)
            quality = run_checks(batch, specs, Context(as_of=as_of, reference=reference))
            incidents = incidents_from(quality)
            raised = {i.category for i in incidents}

            drift = detect(reference, batch, drift_config)
            drift_payloads = to_incident_payloads(drift)
            raised |= {p["category"] for p in drift_payloads}

            outcomes.append(
                DayOutcome(
                    scenario=name,
                    kind=scenario.kind,
                    day=day.isoformat(),
                    expected=list(scenario.expected),
                    raised=sorted(raised),
                    incident_count=len(incidents) + len(drift_payloads),
                )
            )

    loose, strict = Counts(), Counts()
    per_scenario: dict[str, dict] = defaultdict(
        lambda: {"kind": "", "days": 0, "alerted": 0, "expected_hit": 0}
    )
    healthy_days = healthy_alerts = 0

    for outcome in outcomes:
        bucket = per_scenario[outcome.scenario]
        bucket["kind"] = outcome.kind
        bucket["days"] += 1
        bucket["alerted"] += int(outcome.alerted)
        bucket["expected_hit"] += int(outcome.expected_hit)

        if outcome.kind == "failure":
            loose.true_positive += int(outcome.alerted)
            loose.false_negative += int(not outcome.alerted)
            strict.true_positive += int(outcome.expected_hit)
            strict.false_negative += int(not outcome.expected_hit)
        else:
            loose.false_positive += int(outcome.alerted)
            loose.true_negative += int(not outcome.alerted)
            strict.false_positive += int(outcome.alerted)
            strict.true_negative += int(not outcome.alerted)
            if outcome.kind == "healthy":
                healthy_days += 1
                healthy_alerts += int(outcome.alerted)

    return Report(
        days=days,
        scenario_days=len(outcomes),
        loose=loose,
        strict=strict,
        per_scenario=dict(per_scenario),
        false_positive_rate_healthy=healthy_alerts / healthy_days if healthy_days else None,
        outcomes=outcomes,
    )


def render(report: Report) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    lines = [
        f"Evaluated {report.scenario_days} scenario-days "
        f"({len(SCENARIOS)} scenarios over {report.days} days)",
        "",
        f"{'':<26}{'loose':>12}{'strict':>12}",
        f"{'true positives':<26}{report.loose.true_positive:>12}{report.strict.true_positive:>12}",
        f"{'false negatives':<26}{report.loose.false_negative:>12}"
        f"{report.strict.false_negative:>12}",
        f"{'false positives':<26}{report.loose.false_positive:>12}"
        f"{report.strict.false_positive:>12}",
        f"{'true negatives':<26}{report.loose.true_negative:>12}{report.strict.true_negative:>12}",
        f"{'precision':<26}{pct(report.loose.precision):>12}{pct(report.strict.precision):>12}",
        f"{'recall':<26}{pct(report.loose.recall):>12}{pct(report.strict.recall):>12}",
        f"{'F1':<26}{pct(report.loose.f1):>12}{pct(report.strict.f1):>12}",
        "",
        f"false alarms on healthy days: {pct(report.false_positive_rate_healthy)}",
        "",
        f"{'scenario':<22}{'kind':<9}{'days':>6}{'alerted':>9}{'right category':>16}",
    ]
    for name, row in report.per_scenario.items():
        lines.append(
            f"{name:<22}{row['kind']:<9}{row['days']:>6}{row['alerted']:>9}"
            f"{row['expected_hit'] if row['kind'] == 'failure' else '-':>16}"
        )
    return "\n".join(lines)


def write_artifact(report: Report, path: Path) -> None:
    payload = {
        "days": report.days,
        "scenario_days": report.scenario_days,
        "caveat": (
            "Injected faults on synthetic data, detected by rules the same author "
            "configured. A test of the wiring, not of the world. The benign "
            "scenarios and the false-positive count are what keep it from being "
            "circular."
        ),
        "loose": asdict(report.loose)
        | {
            "precision": report.loose.precision,
            "recall": report.loose.recall,
            "f1": report.loose.f1,
        },
        "strict": asdict(report.strict)
        | {
            "precision": report.strict.precision,
            "recall": report.strict.recall,
            "f1": report.strict.f1,
        },
        "false_positive_rate_healthy": report.false_positive_rate_healthy,
        "per_scenario": report.per_scenario,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
