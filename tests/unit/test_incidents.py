from __future__ import annotations

from datasentinel.incidents.service import (
    CATEGORY_BY_CHECK_TYPE,
    incident_from,
    incidents_from,
    severity_for,
)
from datasentinel.quality.checks import CheckResult, CheckSpec
from datasentinel.quality.runner import VALID_SEVERITIES


def failure(check_type: str, column: str | None, severity: str, affected: int = 10) -> CheckResult:
    return CheckResult(
        spec=CheckSpec(f"{check_type}_check", check_type, severity, column),
        passed=False,
        message="something is wrong",
        observed={"affected_rows": affected},
        expected={"limit": 1},
    )


def test_a_passing_check_is_not_an_incident():
    result = CheckResult(CheckSpec("ok", "null_rate", "high", "annual_income"), True, "fine")
    assert incident_from(result) is None


def test_critical_columns_escalate_one_step():
    """A 10% null rate in interest_rate is untidy. The same rate in credit_score
    changes who gets a loan."""
    assert severity_for(failure("null_rate", "credit_score", "high")) == "critical"
    assert severity_for(failure("null_rate", "interest_rate", "high")) == "high"


def test_escalation_needs_affected_rows():
    assert severity_for(failure("null_rate", "credit_score", "high", affected=0)) == "high"


def test_critical_does_not_escalate_past_itself():
    assert severity_for(failure("schema_required", "application_id", "critical")) == "critical"


def test_every_check_type_maps_to_a_category():
    from datasentinel.quality.checks import REGISTRY

    assert set(REGISTRY) <= set(CATEGORY_BY_CHECK_TYPE)


def test_incidents_are_ordered_most_severe_first():
    incidents = incidents_from(
        [
            failure("allowed_values", "employment_status", "low"),
            failure("schema_required", None, "critical"),
            failure("range", "debt_to_income_ratio", "medium"),
        ]
    )
    order = ["critical", "high", "medium", "low"]
    ranks = [order.index(i.severity) for i in incidents]
    assert ranks == sorted(ranks)


def test_an_incident_carries_the_evidence_for_its_claim():
    incident = incident_from(failure("null_rate", "annual_income", "high"))
    assert incident is not None
    assert incident.evidence["observed"] and incident.evidence["expected"]
    assert incident.evidence["check_type"] == "null_rate"
    assert incident.severity in VALID_SEVERITIES
