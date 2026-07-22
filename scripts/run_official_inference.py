#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-gpus", type=int, default=1)
    a = ap.parse_args()

    vendor = ROOT / "vendor/SemiT-SAM"
    if not (vendor / "PROJECT_PATCH_MANIFEST.json").exists():
        raise SystemExit("Project model setup is missing. Run: python project.py install")

    dataset = Path(a.dataset).resolve()
    ckpt = Path(a.checkpoint).resolve()
    if not (dataset / "labeled/test/annotations.json").exists():
        raise SystemExit("Prepared test annotations are missing.")
    if not ckpt.exists():
        raise SystemExit("Model checkpoint is missing. Run: python project.py download")

    env = os.environ.copy()
    env["TSI15K_ROOT"] = str(dataset)
    cmd = [
        sys.executable, "train_net.py",
        "--config-file", CONFIG,
        "--eval-only",
        "--num-gpus", str(a.num_gpus),
        "SSL.TRAIN_SSL", "True",
        "SSL.EVAL_WHO", "STUDENT",
        "MODEL.WEIGHTS", str(ckpt),
        "OUTPUT_DIR", "output/checkpoint_eval",
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(vendor), env=env, check=True)

    out = ROOT / "outputs/inference"
    out.mkdir(parents=True, exist_ok=True)
    eval_root = vendor / "output/checkpoint_eval"

    metric_logs = list(eval_root.rglob("metrics.json"))
    raw = []
    if metric_logs:
        for line in metric_logs[-1].read_text().splitlines():
            try:
                raw.append(json.loads(line))
            except Exception:
                pass
    (out / "raw_evaluator_output.json").write_text(json.dumps(raw, indent=2))

    predictions = next(iter(eval_root.rglob("coco_instances_results.json")), None)
    if predictions is None:
        candidates = list(eval_root.rglob("*instances*results*.json"))
        predictions = candidates[0] if candidates else None
    if predictions is None:
        raise SystemExit(f"Inference completed but COCO prediction JSON was not found under {eval_root}")

    gt = dataset / "labeled/test/annotations.json"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/compute_article_metrics.py"),
        "--ground-truth", str(gt),
        "--predictions", str(predictions),
        "--output", str(out / "metrics.json"),
    ], check=True)

    local_predictions = out / "coco_instances_results.json"
    shutil.copy2(predictions, local_predictions)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/visualize_predictions.py"),
        "--ground-truth", str(gt),
        "--predictions", str(local_predictions),
        "--images", str(dataset / "labeled/test/images"),
        "--output", str(out / "figures/qualitative_predictions.png"),
    ], check=True)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/compare_to_paper.py"),
        "--results", str(out / "metrics.json"),
    ], check=True)
    print("[ok] inference outputs:", out)


if __name__ == "__main__":
    main()
