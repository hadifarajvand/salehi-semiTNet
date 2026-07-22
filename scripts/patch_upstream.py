#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(dest: Path):
    manifest = {"dependency_root": str(dest), "changes": [], "reference_settings": {}, "project_defaults": {}}

    p = dest / "train_net.py"
    s = p.read_text()
    marker = "# set by handdle for debug; add by hj"
    start = s.find(marker)
    if start >= 0:
        first = s.find("###########################################", start)
        second = s.find("###########################################", first + 1)
        if first < 0 or second < 0:
            raise RuntimeError("Malformed private debug override block")
        end = s.find("\n", second)
        end = len(s) if end < 0 else end + 1
        line_start = s.rfind("\n", 0, start) + 1
        s2 = s[:line_start] + "    # Project settings remain controlled by YAML/CLI.\n" + s[end:]
    else:
        s2 = s
    if "/root/paddlejob/" in s2:
        raise RuntimeError("Private path remains after patching train_net.py")

    hook = "from tsi15k_runtime import register_tsi15k\n"
    if hook not in s2:
        s2 = s2.replace("from config.add_cfg import add_ssl_config\n", "from config.add_cfg import add_ssl_config\n" + hook)
        s2 = s2.replace("def setup(args):\n", "def setup(args):\n    register_tsi15k()\n")
    s2 = s2.replace('(\"coco_2017_unlabel_train\",)', '(\"tsi15k_unlabeled\",)')
    p.write_text(s2)
    manifest["changes"].append("configured runtime settings and project dataset registration")

    runtime = '''from __future__ import annotations
import os
from pathlib import Path
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

_REGISTERED = False

def _unlabeled_loader(root):
    image_dir = Path(root) / "unlabeled/images"
    rows = []
    for p in sorted(image_dir.glob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        with Image.open(p) as im:
            w, h = im.size
        rows.append({"file_name": str(p), "image_id": p.stem, "height": h, "width": w})
    if not rows:
        raise RuntimeError(f"No unlabeled images found in {image_dir}")
    return rows


def register_tsi15k():
    global _REGISTERED
    if _REGISTERED:
        return
    root = Path(os.environ.get("TSI15K_ROOT", "data/processed/TSI15k")).resolve()
    register_coco_instances("tsi15k_train", {}, str(root / "labeled/train/annotations.json"), str(root / "labeled/train/images"))
    register_coco_instances("tsi15k_test", {}, str(root / "labeled/test/annotations.json"), str(root / "labeled/test/images"))
    DatasetCatalog.register("tsi15k_unlabeled", lambda r=root: _unlabeled_loader(r))
    MetadataCatalog.get("tsi15k_unlabeled").set(evaluator_type="")
    _REGISTERED = True
'''
    (dest / "tsi15k_runtime.py").write_text(runtime)
    manifest["changes"].append("added environment-driven TSI15k dataset registration")

    base = dest / "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
    cfg = base.read_text().replace("NUM_CLASSES: 52 # 32", "NUM_CLASSES: 32").replace("IMS_PER_BATCH: 32", "IMS_PER_BATCH: 16")
    base.write_text(cfg)
    manifest["changes"].append("configured 32 classes and total batch size 16")

    base2 = dest / "configs/coco/instance-segmentation/Base-COCO-InstanceSegmentation.yaml"
    cfg2 = base2.read_text().replace('TRAIN: ("coco_2017_train",)', 'TRAIN: ("tsi15k_train",)').replace('TEST: ("coco_2017_val",)', 'TEST: ("tsi15k_test",)')
    base2.write_text(cfg2)
    manifest["changes"].append("switched dataset registration to tsi15k_train/test")

    # Old shell launchers are intentionally removed. Python launchers work on Windows and Linux.
    for name in ("project_train_teacher.sh", "project_train_student.sh", "project_eval_released.sh"):
        q = dest / name
        if q.exists():
            q.unlink()
    manifest["changes"].append("enabled cross-platform Python launchers")

    manifest["reference_settings"] = {
        "classes": 32,
        "batch_total": 16,
        "max_iter": 26250,
        "lr": 1e-4,
        "steps": [24000, 25000],
        "input": 1024,
        "queries": 100,
        "decoder_layers": 9,
    }
    manifest["project_defaults"] = {"burnin_iter": 3000, "note": "configurable project default"}
    (dest / "PROJECT_PATCH_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("[ok] project dependency configuration applied")


if __name__ == "__main__":
    patch(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "vendor/SemiT-SAM")
