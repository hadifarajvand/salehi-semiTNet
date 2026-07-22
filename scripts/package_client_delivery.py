#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERABLE = ROOT / "deliverables/SemiTNet-client-deliverable"
ZIP_PATH = ROOT / "SemiTNet-client-deliverable.zip"

SOURCE_FILES = [
    "project.py",
    "README.md",
    "README.en.md",
    "requirements-smoke.txt",
    "requirements-tools.txt",
    "docs/RUNBOOK.md",
    "docs/RUNBOOK.en.md",
    "scripts/download_quick_dataset.py",
    "scripts/prepare_quick_dataset.py",
    "scripts/run_quick_real_experiment.py",
    "scripts/validate_final_outputs.py",
    "scripts/package_client_delivery.py",
]


def copy_file(rel: str) -> None:
    src = ROOT / rel
    if not src.exists():
        return
    dst = DELIVERABLE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    shutil.rmtree(DELIVERABLE, ignore_errors=True)
    DELIVERABLE.mkdir(parents=True, exist_ok=True)

    for rel in SOURCE_FILES:
        copy_file(rel)

    for rel in [
        "data/raw/quick_teeth/download_manifest.json",
        "data/processed/quick_teeth/split_manifest.json",
    ]:
        copy_file(rel)

    out_src = ROOT / "outputs/final"
    out_dst = DELIVERABLE / "outputs/final"
    if out_src.exists():
        shutil.copytree(out_src, out_dst, dirs_exist_ok=True)

    metrics_path = out_src / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    report = f"""# Client Delivery Summary

This package contains the executable SemiTNet reduced-scale simulation code, verified dataset manifests, and the latest validated measured outputs.

## Dataset
- Dataset: {metrics.get('dataset', 'unknown')}
- Verified source images: {metrics.get('verified_dataset_images', 'unknown')}
- Classes: {metrics.get('classes', 'unknown')}
- Execution split: {metrics.get('used_labeled_train', '?')} labeled / {metrics.get('used_label_hidden_pseudo', '?')} pseudo-label / {metrics.get('used_test', '?')} test

## Measured outputs
- IoU: {metrics.get('iou', 'n/a')}
- Dice: {metrics.get('dice', 'n/a')}
- Precision: {metrics.get('precision', 'n/a')}
- Recall: {metrics.get('recall', 'n/a')}
- F1: {metrics.get('f1', 'n/a')}
- Selected checkpoint: {metrics.get('selected_checkpoint', 'n/a')}

## Run
```bash
python project.py install
python project.py download
python project.py full
```
"""
    (DELIVERABLE / "CLIENT_DELIVERY.md").write_text(report)

    ZIP_PATH.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(DELIVERABLE.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(DELIVERABLE.parent))

    with zipfile.ZipFile(ZIP_PATH) as zf:
        bad = zf.testzip()
        if bad:
            raise SystemExit(f"ZIP validation failed at {bad}")
        if not zf.namelist():
            raise SystemExit("ZIP is empty")

    print("[ok] client deliverable:", ZIP_PATH)
    print("[ok] zip bytes:", ZIP_PATH.stat().st_size)


if __name__ == "__main__":
    main()
