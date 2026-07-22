#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PREPARED_MANIFEST = ROOT / "data/processed/quick_teeth/split_manifest.json"


def run(cmd, *, cwd=ROOT, env=None, check=True):
    print("+", " ".join(map(str, cmd)))
    return subprocess.run([str(x) for x in cmd], cwd=str(cwd), env=env, check=check)


def python310_command():
    if sys.version_info[:2] == (3, 10):
        return [sys.executable]
    if os.name == "nt":
        py = shutil.which("py")
        if py:
            probe = subprocess.run([py, "-3.10", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True)
            if probe.returncode == 0:
                return [py, "-3.10"]
    candidate = shutil.which("python3.10")
    if candidate:
        return [candidate]
    raise SystemExit("Python 3.10 is required. Install Python 3.10 and run this command again.")


def venv_is_python310() -> bool:
    if not VENV_PY.exists():
        return False
    probe = subprocess.run(
        [str(VENV_PY), "-c", "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return probe.returncode == 0


def install():
    if VENV.exists() and not venv_is_python310():
        print("[info] Recreating .venv with Python 3.10")
        shutil.rmtree(VENV)
    if not VENV_PY.exists():
        run([*python310_command(), "-m", "venv", str(VENV)])
    run([VENV_PY, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    has_nvidia = shutil.which("nvidia-smi") is not None
    if has_nvidia and platform.system() in {"Windows", "Linux"}:
        run([VENV_PY, "-m", "pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/cu117", "torch==1.13.1+cu117", "torchvision==0.14.1+cu117"])
    elif platform.system() == "Linux":
        run([VENV_PY, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu", "torch==1.13.1+cpu", "torchvision==0.14.1+cpu"])
    else:
        run([VENV_PY, "-m", "pip", "install", "torch==1.13.1", "torchvision==0.14.1"])

    # The reduced-scale simulation uses only the lightweight runtime dependencies.
    # requirements-tools.txt is retained for optional legacy/full-research utilities.
    run([VENV_PY, "-m", "pip", "install", "-r", ROOT / "requirements-smoke.txt"])
    print(f"[ok] .venv ready: {VENV_PY}")


def require_venv():
    if not VENV_PY.exists() or not venv_is_python310():
        raise SystemExit("Valid Python 3.10 .venv is missing. Run: python project.py install")


def relaunch_in_venv(command: str):
    require_venv()
    if Path(sys.executable).resolve() != VENV_PY.resolve():
        result = subprocess.run([str(VENV_PY), str(Path(__file__).resolve()), command])
        raise SystemExit(result.returncode)


def download():
    run([sys.executable, ROOT / "scripts/download_quick_dataset.py"])
    run([sys.executable, ROOT / "scripts/prepare_quick_dataset.py"])
    print("[ok] reduced-simulation dataset downloaded, verified, split, and ready")
    print("[manifest]", PREPARED_MANIFEST)


def smoke():
    run([sys.executable, ROOT / "scripts/run_smoke.py"])
    run([sys.executable, ROOT / "scripts/acceptance_check.py", "--mode", "smoke"])


def full():
    if not PREPARED_MANIFEST.exists():
        print("[info] Prepared reduced-simulation dataset is missing; running download/setup first.")
        download()
    run([sys.executable, ROOT / "scripts/run_quick_real_experiment.py"])
    run([sys.executable, ROOT / "scripts/validate_final_outputs.py"])
    run([sys.executable, ROOT / "scripts/package_client_delivery.py"])
    print("[ok] reduced measured simulation outputs are ready under outputs/final")
    print("[important] This command does NOT reproduce the paper-scale numerical experiment.")
    print("[ok] client package: SemiTNet-client-deliverable.zip")


def audit():
    run([sys.executable, ROOT / "scripts/reproducibility_gate.py"])


def paper_asset(kind: str):
    flag = "--checkpoint" if kind == "checkpoint" else "--dataset"
    run([sys.executable, ROOT / "scripts/download_paper_assets.py", flag])


def main():
    parser = argparse.ArgumentParser(description="SemiTNet project launcher")
    parser.add_argument(
        "command",
        choices=[
            "install",
            "download",
            "smoke",
            "full",
            "audit",
            "paper-assets-checkpoint",
            "paper-assets-dataset",
        ],
    )
    args = parser.parse_args()

    if args.command == "install":
        install()
        return
    if args.command == "audit":
        audit()
        return
    if args.command == "paper-assets-checkpoint":
        paper_asset("checkpoint")
        return
    if args.command == "paper-assets-dataset":
        paper_asset("dataset")
        return

    relaunch_in_venv(args.command)
    {"download": download, "smoke": smoke, "full": full}[args.command]()


if __name__ == "__main__":
    main()
