"""Build the data the published dashboard reads.

Runs a short narrative through the real pipeline rather than inventing numbers:
healthy days, then each failure, then a return to healthy. Everything the page
shows was produced by the code in src/.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasentinel.api.export import write
from datasentinel.config import ROOT, settings
from datasentinel.db import engine, reset
from datasentinel.generator import baseline, generate
from datasentinel.model.baseline import load, save, train
from datasentinel.model.monitor import evaluate_batch, record
from datasentinel.monitoring.pipeline import run_once, seed_history

HISTORY_DAYS = 90
TRAIN_END = date(2026, 8, 1)
START = date(2026, 10, 3)

# A week a data team would recognise: two quiet days, an upstream join that
# drops income, a recovery, a population that moves, a lost partition, a bad
# default writing impossible values, then quiet again.
NARRATIVE: tuple[tuple[int, str], ...] = (
    (0, "healthy"),
    (1, "healthy"),
    (2, "missing_income"),
    (3, "healthy"),
    (4, "income_shift"),
    (5, "volume_drop"),
    (6, "invalid_values"),
    (7, "healthy"),
)


def main() -> None:
    config = settings()
    eng = engine()

    print("resetting and seeding history")
    reset(eng)
    rows = seed_history(eng, days=HISTORY_DAYS, until=START, seed=config.seed)
    print(f"  {rows:,} healthy rows over {HISTORY_DAYS} days")

    print("training the approval model on healthy history")
    pipeline, card = train(baseline(HISTORY_DAYS, TRAIN_END, seed=config.seed), version="v1")
    save(pipeline, card, ROOT / "models")
    pipeline, card = load(ROOT / "models", card.name, card.version)

    print("running the narrative")
    for offset, scenario in NARRATIVE:
        day = START + timedelta(days=offset)
        report = run_once(eng, scenario=scenario, day=day)
        batch = generate(scenario, day, seed=config.seed)
        result = evaluate_batch(pipeline, batch, window=day)
        metrics = record(eng, result, model_name=card.name, model_version=card.version)
        note = "" if result.scored else f"  model unscoreable: {result.reason}"
        print(
            f"  {day}  {scenario:<16} {report.rows:>5} rows  "
            f"{report.failed_checks} failed checks  {report.open_incidents} incidents  "
            f"{metrics} metrics{note}"
        )

    print("exporting")
    for path in write(eng, ROOT / "site" / "data", results=ROOT / "results"):
        print(f"  {path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
