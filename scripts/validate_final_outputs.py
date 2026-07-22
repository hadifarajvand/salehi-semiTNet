#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final"
REQUIRED = [
    OUT / "metrics.json",
    OUT / "history.csv",
    OUT / "run_manifest.json",
    OUT / "RESULTS.md",
    OUT / "figures/metrics.png",
    OUT / "figures/predictions.png",
    OUT / "figures/training_curves.png",
]


def fail(message: str) -> None:
    raise SystemExit(f"[validation failed] {message}")


def main() -> None:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.is_file() or p.stat().st_size <= 0]
    if missing:
        fail("missing/empty files: " + ", ".join(missing))

    metrics = json.loads((OUT / "metrics.json").read_text())
    keys = ["iou", "dice", "precision", "recall", "f1"]
    values = []
    for key in keys:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            fail(f"metric {key} is not finite numeric: {value!r}")
        if not 0.0 <= float(value) <= 100.0:
            fail(f"metric {key} outside [0,100]: {value}")
        values.append(float(value))
    if max(values) <= 0.0:
        fail("all final metrics are zero")

    if metrics.get("verified_dataset_images") != 598:
        fail(f"expected verified_dataset_images=598, got {metrics.get('verified_dataset_images')}")
    if metrics.get("used_labeled_train") != 60 or metrics.get("used_label_hidden_pseudo") != 20 or metrics.get("used_test") != 16:
        fail("execution split is not 60/20/16")
    if float(metrics.get("runtime_seconds", 0.0)) <= 0:
        fail("runtime_seconds must be > 0")

    run_manifest = json.loads((OUT / "run_manifest.json").read_text())
    if run_manifest.get("verified_total_images") != 598:
        fail("run manifest dataset count mismatch")
    if run_manifest.get("split") != {"train_labeled": 60, "pseudo_unlabeled": 20, "test": 16}:
        fail("run manifest split mismatch")

    rows = list(csv.DictReader((OUT / "history.csv").open()))
    if len(rows) < 2:
        fail("history.csv has insufficient training records")

    for path in (OUT / "figures").glob("*.png"):
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            fail(f"invalid image {path.name}: {exc}")

    print("[ok] final output validation PASS")
    print(json.dumps({k: metrics[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
