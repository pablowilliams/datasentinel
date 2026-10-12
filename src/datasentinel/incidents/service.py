"""Turning check failures into incidents a person can act on.

Not every failure deserves the same response. A missing required column means
the downstream model is about to be fed nonsense; an unfamiliar employment
category means someone should look when they get in. Severity here is about
consequence, not about how unusual the number is.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from datasentinel.quality.checks import CheckResult
from datasentinel.schema import CRITICAL

CATEGORY_BY_CHECK_TYPE = {
    "schema_required": "schema",
    "schema_unexpected": "schema",
    "schema_type": "schema",
    "null_rate": "completeness",
    "range": "validity",
    "allowed_values": "category",
    "unique": "uniqueness",
    "freshness": "freshness",
    "volume": "volume",
}

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_ESCALATE = {"low": "medium", "medium": "high", "high": "critical", "critical": "critical"}


@dataclass(frozen=True)
class Incident:
    incident_id: uuid.UUID
    category: str
    severity: str
    title: str
    description: str
    column_name: str | None
    evidence: dict[str, Any] = field(default_factory=dict)
    status: str = "open"


def severity_for(result: CheckResult) -> str:
    """Declared severity, escalated one step when a critical column is affected.

    A 10% null rate in interest_rate is untidy. The same rate in credit_score
    changes who gets a loan.
    """
    base = result.spec.severity
    column = result.spec.column
    if column in CRITICAL and result.affected_rows > 0:
        return _ESCALATE[base]
    return base


def incident_from(result: CheckResult) -> Incident | None:
    if result.passed:
        return None
    category = CATEGORY_BY_CHECK_TYPE.get(result.spec.type, "quality")
    severity = severity_for(result)
    subject = result.spec.column or "dataset"
    return Incident(
        incident_id=uuid.uuid4(),
        category=category,
        severity=severity,
        title=f"{severity.upper()}: {category} problem in {subject}",
        description=result.message,
        column_name=result.spec.column,
        evidence={
            "check": result.spec.name,
            "check_type": result.spec.type,
            "observed": result.observed,
            "expected": result.expected,
        },
    )


def incidents_from(results: list[CheckResult]) -> list[Incident]:
    found = [incident_from(r) for r in results]
    return sorted(
        (i for i in found if i is not None),
        key=lambda i: SEVERITY_ORDER[i.severity],
    )
