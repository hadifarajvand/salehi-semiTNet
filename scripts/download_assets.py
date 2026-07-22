#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Dataset location cited by the paper (accessed by the authors in Aug 2024).
DATA_REPO = "Bryceee/TISI15k-Dataset"
DATA_FILE = "TISI15k-Dataset.tar"
DATA_URL = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/main/{DATA_FILE}"

MODEL_REPO = "Bryceee/SemiTNet"
MODEL_FILE = "SemiTNet_Tooth_Instance_Segmentation_32Classes.pth"
MODEL_SHA256 = "8364853c7632a491fd66108e23a536fa68e9a9c9b416b21c69143a4d02a26c0a"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _available_token() -> str | None:
    # Explicit environment tokens take priority. A cached CLI token is only used
    # as a fallback after anonymous/public access has already failed.
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _copy_local_archive(source: Path, destination: Path) -> None:
    if not source.exists():
        raise SystemExit(f"Dataset archive does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def _download_file(url: str, destination: Path, token: str | None = None, force: bool = False) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")

    if destination.exists() and not force:
        print("[ok] dataset archive already exists:", destination)
        return

    if force:
        destination.unlink(missing_ok=True)
        part.unlink(missing_ok=True)

    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    existing = part.stat().st_size if part.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"

    auth_label = "authenticated" if token else "anonymous"
    print(f"[download] dataset ({auth_label}):", url)

    try:
        with requests.get(url, headers=headers, stream=True, timeout=(30, 300), allow_redirects=True) as response:
            if response.status_code in {401, 403, 404}:
                raise RuntimeError(f"HTTP {response.status_code}")
            response.raise_for_status()

            append = existing > 0 and response.status_code == 206
            mode = "ab" if append else "wb"
            if existing and not append:
                existing = 0

            total = response.headers.get("content-length")
            total_bytes = (int(total) + existing) if total and total.isdigit() else None
            downloaded = existing
            next_report = downloaded + 512 * 1024 * 1024

            with part.open(mode) as f:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        if total_bytes:
                            print(f"  {downloaded / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GiB")
                        else:
                            print(f"  {downloaded / 1024**3:.1f} GiB")
                        next_report += 512 * 1024 * 1024
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    part.replace(destination)


def dataset(force: bool = False) -> None:
    outdir = ROOT / "data/raw/TSI15k"
    archive = outdir / DATA_FILE
    outdir.mkdir(parents=True, exist_ok=True)

    if archive.exists() and not force:
        print("[ok] dataset archive:", archive)
        return

    local_archive = os.getenv("TSI15K_DATASET_ARCHIVE")
    if local_archive:
        _copy_local_archive(Path(local_archive).expanduser(), archive)
        print("[ok] dataset archive copied:", archive)
        return

    url = os.getenv("TSI15K_DATASET_URL", DATA_URL).strip()
    first_error: Exception | None = None

    # Public/anonymous first: stale cached Hugging Face credentials must not break
    # a public download.
    try:
        _download_file(url, archive, token=None, force=force)
    except RuntimeError as exc:
        first_error = exc
        token = _available_token()
        if token:
            try:
                _download_file(url, archive, token=token, force=force)
            except RuntimeError as auth_exc:
                first_error = auth_exc

    if not archive.exists():
        original = url == DATA_URL
        details = (
            "The original TSI15k dataset URL cited by the 2024 paper is not currently reachable as a public Hugging Face file."
            if original
            else "The configured TSI15k dataset URL could not be downloaded."
        )
        raise SystemExit(
            "\n[dataset unavailable]\n"
            f"{details}\n"
            f"Source attempted: {url}\n"
            f"Reason: {first_error or 'download failed'}\n\n"
            "Use one of these options and rerun `python project.py download`:\n"
            "  1. If you were granted access to the original dataset, run `hf auth login` or set HF_TOKEN.\n"
            "  2. Set TSI15K_DATASET_URL to a valid mirror of TISI15k-Dataset.tar.\n"
            "  3. Set TSI15K_DATASET_ARCHIVE=/path/to/TISI15k-Dataset.tar for a local copy.\n"
        )

    import tarfile

    if not tarfile.is_tarfile(archive):
        archive.unlink(missing_ok=True)
        raise SystemExit(
            "Downloaded dataset file is not a valid tar archive. "
            "The source likely returned an access/login page instead of TISI15k-Dataset.tar."
        )

    print("[ok] dataset archive:", archive)


def checkpoint(force: bool = False) -> None:
    from huggingface_hub import hf_hub_download

    outdir = ROOT / "assets/checkpoints"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / MODEL_FILE

    if out.exists() and not force and sha256(out) == MODEL_SHA256:
        print("[ok] checkpoint already verified:", out)
        return

    # Public model: deliberately try without credentials first so a stale token
    # cannot turn a public download into a 401.
    try:
        cached = Path(
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                force_download=force,
                token=False,
            )
        )
    except Exception as public_exc:
        token = _available_token()
        if not token:
            raise SystemExit(f"Checkpoint download failed: {public_exc}") from public_exc
        try:
            cached = Path(
                hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=MODEL_FILE,
                    force_download=force,
                    token=token,
                )
            )
        except Exception as auth_exc:
            raise SystemExit(f"Checkpoint download failed: {auth_exc}") from auth_exc

    shutil.copy2(cached, out)
    got = sha256(out)
    if got != MODEL_SHA256:
        raise SystemExit(f"Checkpoint SHA256 mismatch: {got}")
    (outdir / "CHECKSUMS.json").write_text(json.dumps({MODEL_FILE: got}, indent=2))
    print("[ok] checkpoint verified:", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="store_true")
    ap.add_argument("--checkpoint", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not (args.dataset or args.checkpoint or args.all):
        args.all = True

    if args.checkpoint or args.all:
        checkpoint(args.force)
    if args.dataset or args.all:
        dataset(args.force)


if __name__ == "__main__":
    main()
