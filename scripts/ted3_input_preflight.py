#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs/ted3_reproduction/preflight/input_directories.json"
PROCESSED_ROOT = ROOT / "data/processed/ted3"

EXPECTED_DIRS = {
    "train": DATA / "train",
    "test": DATA / "test",
    "unlabeled": DATA / "TED3-unlabeled-data-15k-pseudo-mask",
}
OPTIONAL_ARCHIVES = (
    DATA / "TED3-train.tar",
    DATA / "TED3-test.tar",
    DATA / "TED3-unlabeled-data-15k-pseudo-mask.tar",
)


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(path: Path) -> dict:
    rel = str(path.relative_to(ROOT))
    ignored = run_git("check-ignore", "-v", "--", rel)
    tracked = run_git("ls-files", "--", rel)
    return {
        "relative_path": rel,
        "ignored": ignored.returncode == 0,
        "ignore_rule": ignored.stdout.strip() if ignored.returncode == 0 else None,
        "tracked_entries": [x for x in tracked.stdout.splitlines() if x.strip()],
    }


def inspect_directory(path: Path) -> dict:
    files = 0
    dirs = 0
    total_bytes = 0
    extensions: dict[str, int] = {}
    empty_files: list[str] = []

    for item in path.rglob("*"):
        if item.is_dir():
            dirs += 1
            continue
        if not item.is_file():
            continue
        files += 1
        size = item.stat().st_size
        total_bytes += size
        suffix = item.suffix.lower() or "<none>"
        extensions[suffix] = extensions.get(suffix, 0) + 1
        if size == 0 and len(empty_files) < 100:
            empty_files.append(str(item.relative_to(path)))

    if files == 0:
        raise SystemExit(f"[preflight failed] extracted dataset folder is empty: {path}")

    return {
        "file_count": files,
        "directory_count": dirs,
        "total_bytes": total_bytes,
        "extensions": dict(sorted(extensions.items(), key=lambda kv: (-kv[1], kv[0]))),
        "empty_files_sample": empty_files,
    }


def optional_archive_records() -> list[dict]:
    records = []
    for path in OPTIONAL_ARCHIVES:
        if not path.is_file():
            continue
        records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "note": "Archive retained only for provenance. Do not extract again; use the already-extracted folders under data/.",
            }
        )
    return records


def main() -> None:
    if run_git("rev-parse", "--show-toplevel").returncode != 0:
        raise SystemExit("[preflight failed] repository is not a Git working tree")

    records = {}
    for role, path in EXPECTED_DIRS.items():
        if not path.is_dir():
            raise SystemExit(
                f"[preflight failed] missing already-extracted TED3 folder for {role}: "
                f"{path.relative_to(ROOT)}"
            )
        state = git_state(path)
        if state["tracked_entries"]:
            raise SystemExit(
                f"[preflight failed] dataset content under {state['relative_path']} is tracked by Git. "
                "Remove it from the Git index without deleting local files before continuing."
            )
        info = inspect_directory(path)
        records[role] = {**state, **info}

    head = run_git("rev-parse", "HEAD")
    status = run_git("status", "--short")
    manifest = {
        "status": "PASS",
        "git_head": head.stdout.strip() if head.returncode == 0 else None,
        "git_status_short": status.stdout.splitlines(),
        "authoritative_inputs": {
            role: str(path.relative_to(ROOT)) for role, path in EXPECTED_DIRS.items()
        },
        "policy": (
            "Use the already-extracted TED3 folders under data/. Do not re-extract TAR archives and do not use "
            "the legacy quick_teeth dataset for this campaign."
        ),
        "processed_dataset_root": str(PROCESSED_ROOT.relative_to(ROOT)),
        "directories": records,
        "optional_archives_present_for_provenance_only": optional_archive_records(),
        "next_stage": (
            "Forensic-audit the extracted directories in place, then build TED3-specific indexes/manifests/loaders "
            "under data/processed/ted3 without re-unzipping/re-extracting archives."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"[ok] TED3 extracted-folder preflight PASS: {OUT.relative_to(ROOT)}")
    print("[important] Do NOT extract the TED3 TAR archives again.")


if __name__ == "__main__":
    main()
