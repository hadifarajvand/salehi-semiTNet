#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ted3_reproduction/preflight/input_archives.json"
EXPECTED = (
    "TED3-train.tar",
    "TED3-test.tar",
    "TED3-unlabeled-data-15k-pseudo-mask.tar",
)

# The user-facing contract is that downloaded archives are placed under data/.
# Search these locations in order before falling back to a whole-repo scan.
PREFERRED_DIRS = (
    ROOT / "data",
    ROOT / "data/TED3",
    ROOT / "data/ted3",
    ROOT / "data/incoming",
    ROOT / "reproduction/assets/dataset/incoming",
)
PROCESSED_ROOT = ROOT / "data/processed/ted3"


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def discover(name: str) -> Path:
    for directory in PREFERRED_DIRS:
        candidate = directory / name
        if candidate.is_file():
            return candidate

    # Also search recursively under data/ before searching the rest of the repo.
    data_root = ROOT / "data"
    if data_root.exists():
        data_matches = [p for p in data_root.rglob(name) if p.is_file()]
        if len(data_matches) == 1:
            return data_matches[0]
        if len(data_matches) > 1:
            shown = "\n  - ".join(str(p.relative_to(ROOT)) for p in data_matches)
            raise SystemExit(
                f"[preflight failed] multiple copies of {name} found under data/:\n  - {shown}\n"
                "Keep one authoritative copy under data/ and remove or rename duplicates."
            )

    matches = []
    for path in ROOT.rglob(name):
        if ".git" in path.parts:
            continue
        if path.is_file():
            matches.append(path)
    if not matches:
        preferred_text = ", ".join(str(p.relative_to(ROOT)) for p in PREFERRED_DIRS[:4])
        raise SystemExit(
            f"[preflight failed] missing required archive: {name}. "
            f"Place it under the repository data folder, preferably one of: {preferred_text}."
        )
    if len(matches) > 1:
        shown = "\n  - ".join(str(p.relative_to(ROOT)) for p in matches)
        raise SystemExit(f"[preflight failed] multiple copies of {name} found:\n  - {shown}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(path: Path) -> dict:
    rel = str(path.relative_to(ROOT))
    ignored = run_git("check-ignore", "-v", "--", rel)
    tracked = run_git("ls-files", "--error-unmatch", "--", rel)
    return {
        "relative_path": rel,
        "ignored": ignored.returncode == 0,
        "ignore_rule": ignored.stdout.strip() if ignored.returncode == 0 else None,
        "tracked": tracked.returncode == 0,
    }


def safe_member_name(name: str) -> bool:
    p = Path(name)
    return not p.is_absolute() and ".." not in p.parts


def inspect_tar(path: Path) -> dict:
    files = 0
    dirs = 0
    symlinks = 0
    unsafe = []
    extensions: dict[str, int] = {}
    total_member_bytes = 0

    try:
        with tarfile.open(path, "r:*") as archive:
            members = archive.getmembers()
            for member in members:
                if not safe_member_name(member.name):
                    unsafe.append(member.name)
                if member.issym() or member.islnk():
                    symlinks += 1
                    unsafe.append(member.name)
                if member.isfile():
                    files += 1
                    total_member_bytes += max(0, int(member.size))
                    suffix = Path(member.name).suffix.lower() or "<none>"
                    extensions[suffix] = extensions.get(suffix, 0) + 1
                elif member.isdir():
                    dirs += 1
    except tarfile.TarError as exc:
        raise SystemExit(f"[preflight failed] unreadable tar archive {path.name}: {exc}") from exc

    return {
        "member_files": files,
        "member_directories": dirs,
        "member_symlinks_or_hardlinks": symlinks,
        "declared_uncompressed_file_bytes": total_member_bytes,
        "extensions": dict(sorted(extensions.items(), key=lambda kv: (-kv[1], kv[0]))),
        "unsafe_members": unsafe[:100],
        "unsafe_member_count": len(unsafe),
    }


def main() -> None:
    git_root = run_git("rev-parse", "--show-toplevel")
    if git_root.returncode != 0:
        raise SystemExit("[preflight failed] repository is not a Git working tree")

    records = []
    for name in EXPECTED:
        path = discover(name)
        state = git_state(path)
        if not state["ignored"]:
            raise SystemExit(f"[preflight failed] {state['relative_path']} is not ignored by Git")
        if state["tracked"]:
            raise SystemExit(
                f"[preflight failed] {state['relative_path']} is tracked by Git. "
                "Remove it from the index with `git rm --cached -- <path>` without deleting the local file."
            )

        tar_info = inspect_tar(path)
        if tar_info["unsafe_member_count"]:
            raise SystemExit(
                f"[preflight failed] {state['relative_path']} contains "
                f"{tar_info['unsafe_member_count']} unsafe/link members; do not extract automatically."
            )

        records.append(
            {
                "filename": name,
                **state,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "tar": tar_info,
            }
        )

    head = run_git("rev-parse", "HEAD")
    status = run_git("status", "--short")
    manifest = {
        "status": "PASS",
        "git_head": head.stdout.strip() if head.returncode == 0 else None,
        "git_status_short": status.stdout.splitlines(),
        "authoritative_input_policy": "Use the three discovered TED3 archives from data/; do not use the legacy 598-image quick dataset for this campaign.",
        "processed_dataset_root": str(PROCESSED_ROOT.relative_to(ROOT)),
        "archives": records,
        "next_stage": "safe extraction into data/processed/ted3 followed by dataset forensic audit",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"[ok] TED3 input preflight PASS: {OUT.relative_to(ROOT)}")
    print(f"[next] Extract/process only these verified archives into {PROCESSED_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
