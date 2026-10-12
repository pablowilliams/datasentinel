"""The headline claim, pinned.

If a change makes the platform miss a failure or start alerting on an ordinary
day, this is the test that says so.
"""

from __future__ import annotations

from datetime import date

import pytest

from datasentinel.config import ROOT
from datasentinel.monitoring.evaluation import evaluate
from datasentinel.monitoring.sensitivity import detection_floor, sweep

SEED = 20261012
UNTIL = date(2026, 10, 9)


@pytest.fixture(scope="module")
def report():
    return evaluate(days=7, until=UNTIL, seed=SEED, config_dir=ROOT / "config")


def test_no_failure_goes_undetected(report):
    assert report.strict.false_negative == 0, [
        o.scenario for o in report.outcomes if o.kind == "failure" and not o.expected_hit
    ]


def test_detection_names_the_right_category(report):
    assert report.strict.recall == 1.0


def test_healthy_days_are_silent(report):
    assert report.false_positive_rate_healthy == 0.0


def test_false_alarms_stay_rare(report):
    """Not zero. One Saturday in the evaluation window sits far enough from the
    reference approval rate to be raised, which is what a 2.9 sigma day looks
    like rather than a bug."""
    negatives = report.loose.false_positive + report.loose.true_negative
    assert report.loose.false_positive / negatives <= 0.02


def test_benign_scenarios_outnumber_nothing(report):
    """A suite with no benign days cannot measure precision at all."""
    assert any(o.kind == "benign" for o in report.outcomes)


@pytest.mark.parametrize(
    ("scenario", "ceiling"),
    [("missing_income", 0.06), ("volume_drop", 0.30), ("duplicates", 0.01)],
)
def test_detection_floor_does_not_regress(scenario, ceiling):
    """The smallest fault reliably caught. A change that raises these numbers has
    made the platform blinder, whatever it did to the recall figure."""
    points = sweep(
        reference=__import__("datasentinel.generator", fromlist=["baseline"]).baseline(
            60, date(2026, 8, 1), seed=SEED
        ),
        days=5,
        until=UNTIL,
        seed=SEED,
        config_dir=ROOT / "config",
    )
    floor = detection_floor(points, scenario)
    assert floor is not None and floor <= ceiling, f"{scenario} floor is {floor}"
