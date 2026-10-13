"""Regenerate the README chart from the exported snapshot.

The picture is the argument, so it is generated from the same data the dashboard
reads rather than screenshotted, and rerunning this after `make site` keeps it
honest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, LINE = "#15181d", "#5b6472", "#e3e6ec"
ACCENT, CRIT, HIGH = "#2f5fe0", "#b3261e", "#c2610a"


def main() -> None:
    snap = json.loads((ROOT / "site" / "data" / "snapshot.json").read_text())
    by_day: dict[str, dict[str, float]] = {}
    for m in snap["model_metrics"]:
        by_day.setdefault(m["evaluation_window"], {})[m["metric_name"]] = float(m["metric_value"])
    scenario = {r["business_date"]: r["scenario"] for r in snap["runs"] if r.get("business_date")}

    days = sorted(by_day)
    if len(days) < 2:
        sys.exit("not enough model metrics; run scripts/build_site.py first")
    auc = [by_day[d].get("roc_auc") for d in days]
    ece = [by_day[d].get("calibration_error") for d in days]

    fig, left = plt.subplots(figsize=(10, 4.2), dpi=160)
    right = left.twinx()

    left.plot(days, auc, color=ACCENT, lw=2.6, marker="o", ms=6, label="ROC AUC", zorder=3)
    right.plot(days, ece, color=CRIT, lw=2.6, marker="o", ms=6, label="Calibration error", zorder=3)

    shift = [i for i, d in enumerate(days) if scenario.get(d) == "income_shift"]
    for i in shift:
        left.axvspan(i - 0.35, i + 0.35, color=HIGH, alpha=0.10, zorder=0)
        left.text(i, 0.846, "income_shift", ha="center", fontsize=8.5, color=HIGH, weight="bold")

    left.set_ylabel("ROC AUC", color=ACCENT, fontsize=10)
    right.set_ylabel("Calibration error", color=CRIT, fontsize=10)
    left.set_ylim(0.68, 0.86)
    right.set_ylim(0, max(v for v in ece if v is not None) * 1.25)
    left.tick_params(axis="y", colors=ACCENT, labelsize=9)
    right.tick_params(axis="y", colors=CRIT, labelsize=9)
    left.tick_params(axis="x", colors=MUTED, labelsize=8.5, rotation=0)
    left.set_xticks(range(len(days)))
    left.set_xticklabels([d[5:] for d in days])

    left.grid(axis="y", color=LINE, lw=0.8, zorder=0)
    left.set_axisbelow(True)
    for spine in (*left.spines.values(), *right.spines.values()):
        spine.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.text(
        0.02, 0.975,
        "Under a population shift the ranking holds and the probabilities come loose",
        fontsize=12, color=INK, ha="left", va="top", weight="bold",
    )
    fig.text(
        0.02, 0.905,
        "AUC rises while calibration error jumps fivefold. A monitor watching "
        "discrimination alone would call this a good day.",
        fontsize=9.5, color=MUTED, ha="left", va="top",
    )
    out = ROOT / "docs" / "images" / "model-decay.png"
    fig.savefig(out, facecolor="white")
    print(f"wrote {out.relative_to(ROOT)}  {out.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
