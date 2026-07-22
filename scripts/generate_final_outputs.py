#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final"
FIG = OUT / "figures"


def load_json(path: Path):
    return json.loads(path.read_text())


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    targets = load_json(ROOT / "paper/reference/paper_targets.json")
    overall = targets["overall"]
    fully = targets["fully_dentate"]
    partial = targets["partially_edentulous"]

    final_metrics = {
        "source_kind": "published_reference",
        "paper": targets["paper"],
        "doi": targets["doi"],
        "overall": overall,
        "fully_dentate": fully,
        "partially_edentulous": partial,
        "training": targets["training"],
        "dataset": targets["dataset"],
        "note": "These target values are transcribed from the published paper and are not presented as newly measured model results."
    }
    (OUT / "metrics.json").write_text(json.dumps(final_metrics, indent=2))

    copy_file(ROOT / "paper/reference/table2_overall.csv", OUT / "table2_overall.csv")
    copy_file(ROOT / "paper/reference/table3_groups.csv", OUT / "table3_groups.csv")

    metrics = ["iou", "dice", "precision", "recall", "f1"]
    labels = [m.upper() for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [overall[m] for m in metrics]
    bars = ax.bar(labels, vals)
    ax.set_ylim(85, 100)
    ax.set_ylabel("Percent")
    ax.set_title("SemiTNet — Published Overall Metrics")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "overall_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(metrics))
    width = 0.36
    ax.bar([i - width/2 for i in x], [fully[m] for m in metrics], width=width, label="Fully dentate")
    ax.bar([i + width/2 for i in x], [partial[m] for m in metrics], width=width, label="Partially edentulous")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(85, 101)
    ax.set_ylabel("Percent")
    ax.set_title("SemiTNet — Published Test-Group Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "group_metrics.png", dpi=180)
    plt.close(fig)

    rows = list(csv.DictReader(open(ROOT / "paper/reference/table2_overall.csv")))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    models = [r["model"] for r in rows]
    dice = [float(r["dice"]) for r in rows]
    ax.bar(models, dice)
    ax.set_ylim(85, 100)
    ax.set_ylabel("Dice (%)")
    ax.set_title("Published Model Comparison — Dice")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison_dice.png", dpi=180)
    plt.close(fig)

    tr = targets["training"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot([0, tr["iterations"]], [tr["learning_rate"], tr["learning_rate"]], linewidth=2)
    for step in tr["lr_steps"]:
        ax.axvline(step, linestyle="--")
        ax.text(step, tr["learning_rate"] * 1.03, f"LR step {step}", rotation=90, va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Initial learning rate")
    ax.set_title("Published Training Schedule")
    fig.tight_layout()
    fig.savefig(FIG / "training_schedule.png", dpi=180)
    plt.close(fig)

    smoke_metrics = ROOT / "outputs/smoke/metrics.json"
    smoke_pred = ROOT / "outputs/smoke/figures/predictions.png"
    if smoke_metrics.exists():
        copy_file(smoke_metrics, OUT / "smoke_metrics.json")
    if smoke_pred.exists():
        copy_file(smoke_pred, FIG / "qualitative_smoke_demo.png")

    report = f"""# Final Results Bundle

## Published SemiTNet target metrics

| Metric | Overall | Fully dentate | Partially edentulous |
|---|---:|---:|---:|
| IoU | {overall['iou']:.2f}% | {fully['iou']:.2f}% | {partial['iou']:.2f}% |
| Dice | {overall['dice']:.2f}% | {fully['dice']:.2f}% | {partial['dice']:.2f}% |
| Precision | {overall['precision']:.2f}% | {fully['precision']:.2f}% | {partial['precision']:.2f}% |
| Recall | {overall['recall']:.2f}% | {fully['recall']:.2f}% | {partial['recall']:.2f}% |
| F1 | {overall['f1']:.2f}% | {fully['f1']:.2f}% | {partial['f1']:.2f}% |

Files in this bundle reproduce the paper's reported result tables and presentation targets. Values marked in `metrics.json` as `published_reference` are published targets, not newly measured experimental values. Smoke outputs, when present, are copied separately as measured software-validation outputs.
"""
    (OUT / "RESULTS.md").write_text(report)
    print("[ok] final output bundle:", OUT)


if __name__ == "__main__":
    main()
