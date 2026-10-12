"""Load check definitions from YAML and run them."""

from __future__ import annotations

from pathlib import Path

import yaml

from datasentinel.quality.checks import CheckSpec

VALID_SEVERITIES = {"critical", "high", "medium", "low"}


def load_specs(path: Path) -> list[CheckSpec]:
    document = yaml.safe_load(path.read_text())
    specs: list[CheckSpec] = []
    seen: set[str] = set()
    for raw in document.get("checks", []):
        name = raw["name"]
        if name in seen:
            raise ValueError(f"duplicate check name {name!r}")
        seen.add(name)
        severity = raw["severity"]
        if severity not in VALID_SEVERITIES:
            raise ValueError(f"check {name!r} has unknown severity {severity!r}")
        specs.append(
            CheckSpec(
                name=name,
                type=raw["type"],
                severity=severity,
                column=raw.get("column"),
                options=raw.get("options") or {},
            )
        )
    return specs
