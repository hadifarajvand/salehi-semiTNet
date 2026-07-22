#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/raw/quick_teeth"
DATASET_NAME = "Teeth Segmentation on Dental X-ray Images"


def has_images(root: Path) -> bool:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return root.exists() and any(p.is_file() and p.suffix.lower() in exts for p in root.rglob("*"))


def main() -> None:
    if has_images(DEST):
        print("[ok] replacement dataset already present:", DEST)
        return

    DEST.mkdir(parents=True, exist_ok=True)
    try:
        import dataset_tools as dtools
    except ImportError as exc:
        raise SystemExit("dataset-tools is missing. Run: python project.py install") from exc

    print("[download] 598-image / 32-class panoramic tooth segmentation dataset (~464 MB)")
    print("[source] Dataset Ninja / Humans in the Loop, CC0 1.0")
    try:
        dtools.download(dataset=DATASET_NAME, dst_dir=str(DEST))
    except Exception as exc:
        if DEST.exists() and not any(DEST.iterdir()):
            shutil.rmtree(DEST, ignore_errors=True)
        raise SystemExit(
            "Replacement dataset download failed. You can download the 463.67 MB Supervisely package from "
            "https://datasetninja.com/teeth-segmentation and extract it under data/raw/quick_teeth/.\n"
            f"Reason: {exc}"
        ) from exc

    if not has_images(DEST):
        raise SystemExit(
            "Dataset downloader finished but no images were found. Extract the downloaded Supervisely-format dataset "
            "under data/raw/quick_teeth/ and rerun python project.py full."
        )
    print("[ok] replacement dataset:", DEST)


if __name__ == "__main__":
    main()
