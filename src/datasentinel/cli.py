"""Command line entry points."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from datasentinel.config import settings
from datasentinel.db import engine, reset
from datasentinel.generator import SCENARIOS, baseline, generate
from datasentinel.monitoring import sensitivity as sens
from datasentinel.monitoring.evaluation import evaluate, render, write_artifact
from datasentinel.monitoring.pipeline import run_once, seed_history

app = typer.Typer(add_completion=False, help="Data quality and ML monitoring for batch pipelines.")
model_app = typer.Typer(add_completion=False, help="The approval model being monitored.")
app.add_typer(model_app, name="model")
console = Console()
RESULTS = Path("results")
MODELS = Path("models")


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




@model_app.command("train")
def model_train(
    until: str = "2026-08-01", days: int = 90, version: str = "v1"
) -> None:
    """Train the approval classifier on healthy history and save it with a card."""
    from datasentinel.model.baseline import save, train

    frame = baseline(days, date.fromisoformat(until), seed=settings().seed)
    pipeline, card = train(frame, version=version)
    path = save(pipeline, card, MODELS)
    console.print(
        f"trained [bold]{card.name} {card.version}[/] on {card.train_rows:,} rows "
        f"({card.train_window[0]} to {card.train_window[1]})"
    )
    console.print(f"  excluded as post-decision: {', '.join(card.excluded_as_post_decision)}")
    console.print(f"  saved to {path}")


@model_app.command("monitor")
def model_monitor(
    scenario: str = "healthy", days: int = 14, until: str = "2026-10-09", version: str = "v1"
) -> None:
    """Score recent batches and record the metrics to model_metrics."""
    from datasentinel.model.baseline import load
    from datasentinel.model.monitor import evaluate_batch, record

    pipeline, card = load(MODELS, "approval_classifier", version)
    eng = engine()
    end = date.fromisoformat(until)
    table = Table("day", "rows", "AUC", "F1", "calib err", "approval gap", "note")
    written = 0
    for offset in range(days):
        day = end - timedelta(days=offset)
        result = evaluate_batch(pipeline, generate(scenario, day, seed=settings().seed), window=day)
        written += record(eng, result, model_name=card.name, model_version=card.version)
        if result.scored:
            table.add_row(
                str(day), f"{result.rows:,}", f"{result.get('roc_auc'):.3f}",
                f"{result.get('f1'):.3f}", f"{result.get('calibration_error'):.4f}",
                f"{result.get('approval_rate_gap'):+.3f}", "",
            )
        else:
            table.add_row(str(day), f"{result.rows:,}", "-", "-", "-", "-", result.reason or "")
    console.print(table)
    console.print(f"wrote {written} metric rows")


@model_app.command("decay")
def model_decay(days: int = 14, until: str = "2026-10-09", version: str = "v1") -> None:
    """Does detected drift correspond to measurable model damage?"""
    from datasentinel.drift.detector import load_config
    from datasentinel.model import decay as decay_module
    from datasentinel.model.baseline import load
    from datasentinel.quality.runner import load_specs

    pipeline, _ = load(MODELS, "approval_classifier", version)
    end = date.fromisoformat(until)
    reference = baseline(90, date(2026, 8, 1), seed=settings().seed)
    base = decay_module.healthy_baseline(pipeline, days=42, until=end, seed=settings().seed)
    observations = decay_module.observe(
        pipeline,
        scenarios={
            "healthy": None, "minor_noise": None, "weekend": None, "trivial_income_tick": None,
            "income_shift": (0.05, 0.10, 0.20, 0.50), "population_shift": (0.50,),
            "missing_income": (0.20,), "schema_break": None, "duplicates": (0.18,),
            "volume_drop": (0.60,), "invalid_values": (0.06,),
        },
        days=days,
        until=end,
        seed=settings().seed,
        reference=reference,
        drift_config=load_config(settings().config_dir / "drift.yml"),
        baseline=base,
        specs=load_specs(settings().config_dir / "checks.yml"),
    )
    console.print(decay_module.render(observations, base))
    RESULTS.mkdir(exist_ok=True)
    decay_module.write_artifact(observations, base, RESULTS / "model_decay.json")


if __name__ == "__main__":
    app()
