"""Command line entry points."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from datasentinel.config import settings
from datasentinel.db import engine, reset
from datasentinel.generator import SCENARIOS, baseline
from datasentinel.monitoring import sensitivity as sens
from datasentinel.monitoring.evaluation import evaluate, render, write_artifact
from datasentinel.monitoring.pipeline import run_once, seed_history

app = typer.Typer(add_completion=False, help="Data quality and ML monitoring for batch pipelines.")
console = Console()
RESULTS = Path("results")


@app.command()
def init(days: int = 90, until: str = "2026-10-09", wipe: bool = True) -> None:
    """Create the schema and load a healthy reference window."""
    eng = engine()
    if wipe:
        reset(eng)
    rows = seed_history(eng, days=days, until=date.fromisoformat(until), seed=settings().seed)
    console.print(f"seeded [bold]{rows:,}[/] healthy rows across {days} daily runs")


@app.command()
def run(scenario: str = "healthy", day: str = "2026-10-10") -> None:
    """Run one monitoring pass and print what it found."""
    if scenario not in SCENARIOS:
        raise typer.BadParameter(f"unknown scenario. choose from: {', '.join(SCENARIOS)}")
    report = run_once(engine(), scenario=scenario, day=date.fromisoformat(day))
    console.print(
        f"[bold]{scenario}[/] on {day}: {report.rows:,} rows, "
        f"{report.failed_checks} failed checks, {report.open_incidents} incidents"
    )
    if not report.incidents:
        console.print("[green]no incidents[/]")
        return
    table = Table("severity", "category", "column", "what was found")
    for incident in report.incidents:
        table.add_row(
            incident.severity, incident.category, incident.column_name or "-", incident.description
        )
    console.print(table)


@app.command("evaluate")
def evaluate_command(days: int = 60, until: str = "2026-10-09") -> None:
    """Score detection across every scenario, over many days."""
    report = evaluate(
        days=days,
        until=date.fromisoformat(until),
        seed=settings().seed,
        config_dir=settings().config_dir,
    )
    console.print(render(report))
    RESULTS.mkdir(exist_ok=True)
    write_artifact(report, RESULTS / "evaluation.json")


@app.command()
def sensitivity(days: int = 21, until: str = "2026-10-09") -> None:
    """Find the smallest fault of each kind that is reliably caught."""
    end = date.fromisoformat(until)
    reference = baseline(120, end - timedelta(days=days), seed=settings().seed)
    points = sens.sweep(
        reference=reference,
        days=days,
        until=end,
        seed=settings().seed,
        config_dir=settings().config_dir,
    )
    console.print(sens.render(points))
    RESULTS.mkdir(exist_ok=True)
    sens.write_artifact(points, RESULTS / "sensitivity.json")


@app.command()
def scenarios() -> None:
    """List the scenarios and what each one should produce."""
    table = Table("scenario", "kind", "expects", "description")
    for name, scenario in SCENARIOS.items():
        table.add_row(
            name, scenario.kind, ", ".join(scenario.expected) or "silence", scenario.description
        )
    console.print(table)


if __name__ == "__main__":
    app()
