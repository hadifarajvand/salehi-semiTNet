#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reproduction/assets"
CHECKPOINT_DIR = ASSETS / "checkpoints"
DATASET_DIR = ASSETS / "dataset"
MANIFEST = ASSETS / "asset_manifest.json"

CHECKPOINT_URL = (
    "https://huggingface.co/Bryceee/SemiTNet/resolve/"
    "2d79fa571467b159fbcf279d8676bd07fd0dcc9a/"
    "SemiTNet_Tooth_Instance_Segmentation_32Classes.pth?download=true"
)
CHECKPOINT_NAME = "SemiTNet_Tooth_Instance_Segmentation_32Classes.pth"
CHECKPOINT_SHA256 = "8364853c7632a491fd66108e23a536fa68e9a9c9b416b21c69143a4d02a26c0a"
CHECKPOINT_SIZE = 338584301

# This is the exact dataset location cited by the 2024 paper. Never silently
# replace it with a different dental dataset if it becomes unavailable.
DATASET_URL = (
    "https://huggingface.co/datasets/Bryceee/TISI15k-Dataset/resolve/main/"
    "TISI15k-Dataset.tar?download=true"
)
DATASET_NAME = "TISI15k-Dataset.tar"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_exact(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "SemiTNet-reproduction/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"Exact published asset could not be downloaded from {url}\n"
            "Do not substitute a different dataset/checkpoint. Resolve the exact asset provenance first.\n"
            f"Reason: {exc}"
        ) from exc
    tmp.replace(target)


def checkpoint_record(download: bool) -> dict:
    path = CHECKPOINT_DIR / CHECKPOINT_NAME
    if download and not path.exists():
        print("[download] exact author checkpoint")
        download_exact(CHECKPOINT_URL, path)
    if not path.is_file():
        return {"status": "MISSING", "path": str(path.relative_to(ROOT))}
    digest = sha256(path)
    size = path.stat().st_size
    ok = digest == CHECKPOINT_SHA256 and size == CHECKPOINT_SIZE
    return {
        "status": "PASS" if ok else "FAIL",
        "path": str(path.relative_to(ROOT)),
        "size_bytes": size,
        "sha256": digest,
        "expected_size_bytes": CHECKPOINT_SIZE,
        "expected_sha256": CHECKPOINT_SHA256,
        "source_url": CHECKPOINT_URL,
    }


def dataset_record(download: bool) -> dict:
    path = DATASET_DIR / DATASET_NAME
    if download and not path.exists():
        print("[download] exact paper-cited TSI15k archive")
        download_exact(DATASET_URL, path)
    if not path.is_file():
        return {
            "status": "MISSING",
            "path": str(path.relative_to(ROOT)),
            "source_url": DATASET_URL,
            "warning": "Exact paper dataset absent. G1/G2/G3 remain blocked; no fallback dataset is permitted.",
        }
    return {
        "status": "PRESENT_UNVERIFIED_CONTENT",
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "source_url": DATASET_URL,
        "warning": "Archive presence alone does not prove the exact 1398/14728/191 identities. Build and hash an identity manifest before G1.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Acquire exact publication assets without silent substitutions")
    ap.add_argument("--checkpoint", action="store_true", help="Download/verify exact released SemiTNet checkpoint")
    ap.add_argument("--dataset", action="store_true", help="Download exact paper-cited TSI15k archive")
    ap.add_argument("--all", action="store_true", help="Download both exact assets")
    ap.add_argument("--verify-only", action="store_true", help="Do not download; only inspect local assets")
    args = ap.parse_args()

    want_checkpoint = args.all or args.checkpoint
    want_dataset = args.all or args.dataset
    if not (want_checkpoint or want_dataset or args.verify_only):
        ap.error("choose --checkpoint, --dataset, --all, or --verify-only")

    checkpoint = checkpoint_record(download=(want_checkpoint and not args.verify_only))
    dataset = dataset_record(download=(want_dataset and not args.verify_only))
    result = {
        "checkpoint": checkpoint,
        "dataset": dataset,
        "policy": "No substitute dataset or checkpoint may be used for paper-equivalence claims.",
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("[manifest]", MANIFEST.relative_to(ROOT))

    if want_checkpoint and checkpoint["status"] != "PASS":
        raise SystemExit(2)
    if want_dataset and dataset["status"] == "MISSING":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
