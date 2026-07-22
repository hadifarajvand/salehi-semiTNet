#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"


def run_training(vendor: Path, args: list[str], env: dict[str, str]):
    cmd = [sys.executable, "train_net.py", "--config-file", CONFIG, *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(vendor), env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-gpus", type=int, default=int(os.getenv("NUM_GPUS", "1")))
    ap.add_argument("--burnin-iter", type=int, default=int(os.getenv("BURNIN_ITER", "3000")))
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "2")))
    ap.add_argument("--init-weights", default=os.getenv("INIT_WEIGHTS", ""))
    a = ap.parse_args()

    vendor = ROOT / "vendor/SemiT-SAM"
    dataset = Path(a.dataset).resolve()
    if not (vendor / "PROJECT_PATCH_MANIFEST.json").exists():
        raise SystemExit("Project model setup is missing. Run: python project.py install")

    env = os.environ.copy()
    env["TSI15K_ROOT"] = str(dataset)

    out = ROOT / "outputs/full"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_manifest.json").write_text(json.dumps({
        "dataset": str(dataset),
        "num_gpus": a.num_gpus,
        "burnin_iter": a.burnin_iter,
        "batch_size_total": a.batch_size,
        "init_weights": a.init_weights or None,
        "python": sys.executable,
        "platform": sys.platform,
    }, indent=2))

    common = ["--num-gpus", str(a.num_gpus)]
    run_training(vendor, common + [
        "SSL.TRAIN_SSL", "False",
        "MODEL.WEIGHTS", str(Path(a.init_weights).resolve()) if a.init_weights else "",
        "SOLVER.IMS_PER_BATCH", str(a.batch_size),
        "OUTPUT_DIR", "output/project_teacher",
    ], env)

    teacher_ckpt = vendor / "output/project_teacher/model_best.pth"
    if not teacher_ckpt.exists():
        candidates = list((vendor / "output/project_teacher").glob("*.pth"))
        if candidates:
            teacher_ckpt = candidates[-1]

    run_training(vendor, common + [
        "SSL.TRAIN_SSL", "True",
        "SSL.TEACHER_CKPT", str(teacher_ckpt),
        "SSL.BURNIN_ITER", str(a.burnin_iter),
        "SSL.EVAL_WHO", "STUDENT",
        "SOLVER.IMS_PER_BATCH", str(a.batch_size),
        "OUTPUT_DIR", "output/project_student",
    ], env)

    print("[ok] full training finished")


if __name__ == "__main__":
    main()
