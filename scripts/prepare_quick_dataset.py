#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/quick_teeth"
OUT = ROOT / "data/processed/quick_teeth"
SEED = 2024
LABELED_N = 60
PSEUDO_N = 20
TEST_N = 16
EXPECTED_IMAGES = 598
EXPECTED_CLASSES = {str(i) for i in range(1, 33)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_pairs(root: Path):
    pairs = []
    all_images = []
    classes = set()
    object_images = 0

    for img_dir in root.rglob("img"):
        if not img_dir.is_dir():
            continue
        ann_dir = img_dir.parent / "ann"
        for image in sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            all_images.append(image)
            ann = ann_dir / f"{image.name}.json"
            if not ann.exists():
                continue
            try:
                data = json.loads(ann.read_text())
            except Exception as exc:
                raise SystemExit(f"Invalid annotation JSON: {ann}: {exc}") from exc
            objects = data.get("objects") or []
            if objects:
                object_images += 1
                for obj in objects:
                    title = str(obj.get("classTitle", "")).strip()
                    if title:
                        classes.add(title)
                pairs.append((image, ann))

    return all_images, pairs, classes, object_images


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def put_pair(split: str, index: int, image: Path, ann: Path | None) -> dict:
    safe_name = f"{index:03d}_{image.name}"
    img_dst = OUT / split / "img" / safe_name
    link_or_copy(image, img_dst)
    ann_dst = None
    if ann is not None:
        ann_dst = OUT / split / "ann" / f"{safe_name}.json"
        link_or_copy(ann, ann_dst)
    return {
        "image": str(img_dst.relative_to(ROOT)),
        "annotation": str(ann_dst.relative_to(ROOT)) if ann_dst else None,
        "source_image": str(image.relative_to(ROOT)),
        "source_annotation": str(ann.relative_to(ROOT)) if ann else None,
    }


def main() -> None:
    all_images, pairs, classes, object_images = find_pairs(RAW)

    if len(all_images) != EXPECTED_IMAGES:
        raise SystemExit(
            f"Dataset verification failed: expected exactly {EXPECTED_IMAGES} images, found {len(all_images)}. "
            "Delete data/raw/quick_teeth and rerun `python project.py download`."
        )
    if classes != EXPECTED_CLASSES:
        missing = sorted(EXPECTED_CLASSES - classes, key=int)
        extra = sorted(classes - EXPECTED_CLASSES)
        raise SystemExit(f"Dataset verification failed: expected classes 1..32. Missing={missing}, extra={extra}")
    if len(pairs) < LABELED_N + PSEUDO_N + TEST_N:
        raise SystemExit(f"Dataset verification failed: only {len(pairs)} annotated images are usable; need at least 96.")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    ordered = sorted(pairs, key=lambda x: str(x[0].relative_to(RAW)))
    random.Random(SEED).shuffle(ordered)
    selected = ordered[: LABELED_N + PSEUDO_N + TEST_N]
    labeled = selected[:LABELED_N]
    pseudo = selected[LABELED_N : LABELED_N + PSEUDO_N]
    test = selected[LABELED_N + PSEUDO_N :]

    manifest = {
        "dataset": "Teeth Segmentation on Dental X-ray Images",
        "publisher": "Humans in the Loop",
        "distribution": "Dataset Ninja / Supervisely format",
        "license": "CC0 1.0",
        "verified_total_images": len(all_images),
        "verified_images_with_objects": object_images,
        "verified_classes": sorted(classes, key=int),
        "seed": SEED,
        "split": {
            "train_labeled": LABELED_N,
            "pseudo_unlabeled": PSEUDO_N,
            "test": TEST_N,
        },
        "note": "Pseudo-label images are copied without annotations into the prepared simulation input so their labels are not consumed during training.",
        "files": {
            "train_labeled": [put_pair("train_labeled", i, image, ann) for i, (image, ann) in enumerate(labeled)],
            "pseudo_unlabeled": [put_pair("pseudo_unlabeled", i, image, None) for i, (image, ann) in enumerate(pseudo)],
            "test": [put_pair("test", i, image, ann) for i, (image, ann) in enumerate(test)],
        },
    }

    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    print("[ok] prepared simulation dataset:", OUT)


if __name__ == "__main__":
    main()
