#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tarfile
import shutil
from pathlib import Path

import requests
import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/raw/quick_teeth"
STAGING = ROOT / "data/raw/.quick_teeth_download"
DATASET_NAME = "Teeth Segmentation on Dental X-ray Images"
EXPECTED_IMAGES = 598
EXPECTED_CLASSES = {str(i) for i in range(1, 33)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DOWNLOAD_URL = "https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMTU5OF9UZWV0aCBTZWdtZW50YXRpb24gb24gRGVudGFsIFgtcmF5IEltYWdlcy90ZWV0aC1zZWdtZW50YXRpb24tb24tZGVudGFsLXgtcmF5LWltYWdlcy1EYXRhc2V0TmluamEudGFyIiwgInNpZyI6ICJsZzV2QjE5L0RzbkREam1wV2JwN3Y2Z01XT3U5c28yb0FhcllRZGV5OTZzPSJ9?response-content-disposition=attachment%3B%20filename%3D%22teeth-segmentation-on-dental-x-ray-images-DatasetNinja.tar%22"


def inspect_dataset(root: Path):
    images = []
    classes = set()
    annotated_with_objects = 0

    for img_dir in root.rglob("img"):
        if not img_dir.is_dir():
            continue
        ann_dir = img_dir.parent / "ann"
        for image in sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            images.append(image)
            ann = ann_dir / f"{image.name}.json"
            if not ann.exists():
                continue
            try:
                data = json.loads(ann.read_text())
            except Exception:
                continue
            objects = data.get("objects") or []
            if objects:
                annotated_with_objects += 1
                for obj in objects:
                    title = str(obj.get("classTitle", "")).strip()
                    if title:
                        classes.add(title)

    return images, classes, annotated_with_objects


def verify(root: Path) -> dict:
    images, classes, annotated = inspect_dataset(root)
    if len(images) != EXPECTED_IMAGES:
        raise SystemExit(
            f"Wrong or incomplete dataset: expected exactly {EXPECTED_IMAGES} images, found {len(images)} under {root}."
        )
    if classes != EXPECTED_CLASSES:
        missing = sorted(EXPECTED_CLASSES - classes, key=int)
        extra = sorted(classes - EXPECTED_CLASSES)
        raise SystemExit(
            f"Wrong or incomplete dataset: expected tooth classes 1..32. Missing={missing}, extra={extra}."
        )
    return {
        "dataset": DATASET_NAME,
        "source": "Dataset Ninja / Humans in the Loop",
        "license": "CC0 1.0",
        "verified_total_images": len(images),
        "verified_images_with_objects": annotated,
        "verified_classes": sorted(classes, key=int),
    }

def download_archive(url: str, target: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with target.open("wb") as file, tqdm.tqdm(
            total=total_size, unit="B", unit_scale=True, desc=f"Downloading '{DATASET_NAME}'"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                pbar.update(len(chunk))

def unpack_archive(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path) as archive:
        archive.extractall(destination)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if DEST.exists() and not args.force:
        try:
            info = verify(DEST)
            (DEST / "download_manifest.json").write_text(json.dumps(info, indent=2))
            print("[ok] exact dataset already downloaded and verified:", DEST)
            return
        except SystemExit:
            print("[info] Existing dataset is incomplete or incorrect; redownloading cleanly.")

    shutil.rmtree(STAGING, ignore_errors=True)
    STAGING.mkdir(parents=True, exist_ok=True)

    print(f"[download] exact dataset: {DATASET_NAME}")
    print("[expected] 598 panoramic X-rays, classes 1..32, pixel-level annotations, CC0 1.0")
    try:
        archive_path = STAGING / "dataset.tar"
        download_archive(DOWNLOAD_URL, archive_path)
        unpack_archive(archive_path, STAGING)
        archive_path.unlink(missing_ok=True)
    except Exception as exc:
        shutil.rmtree(STAGING, ignore_errors=True)
        raise SystemExit(
            "Exact dataset download failed. Official Dataset Ninja page: "
            "https://datasetninja.com/teeth-segmentation\n"
            f"Reason: {exc}"
        ) from exc

    info = verify(STAGING)

    shutil.rmtree(DEST, ignore_errors=True)
    STAGING.replace(DEST)
    (DEST / "download_manifest.json").write_text(json.dumps(info, indent=2))

    print(json.dumps(info, indent=2))
    print("[ok] exact dataset downloaded and verified:", DEST)


if __name__ == "__main__":
    main()
