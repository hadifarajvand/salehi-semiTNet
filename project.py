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


def run(cmd, *, cwd=ROOT, env=None):
    print("+", " ".join(map(str, cmd)))
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), env=env, check=True)


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
        run([
            VENV_PY, "-m", "pip", "install",
            "--extra-index-url", "https://download.pytorch.org/whl/cu117",
            "torch==1.13.1+cu117", "torchvision==0.14.1+cu117",
        ])
    elif platform.system() == "Linux":
        run([
            VENV_PY, "-m", "pip", "install",
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "torch==1.13.1+cpu", "torchvision==0.14.1+cpu",
        ])
    else:
        run([VENV_PY, "-m", "pip", "install", "torch==1.13.1", "torchvision==0.14.1"])

    run([VENV_PY, "-m", "pip", "install", "-r", ROOT / "requirements-smoke.txt"])
    run([VENV_PY, "-m", "pip", "install", "-r", ROOT / "requirements-tools.txt"])
    run([VENV_PY, ROOT / "scripts/bootstrap_upstream.py"])
    print(f"[ok] .venv ready: {VENV_PY}")


def require_venv():
    if not VENV_PY.exists() or not venv_is_python310():
        raise SystemExit("Valid Python 3.10 .venv is missing. Run: python project.py install")


def relaunch_in_venv(command: str):
    require_venv()
    current = Path(sys.executable).resolve()
    target = VENV_PY.resolve()
    if current != target:
        result = subprocess.run([str(VENV_PY), str(Path(__file__).resolve()), command])
        raise SystemExit(result.returncode)


def download():
    run([sys.executable, ROOT / "scripts/download_assets.py", "--all"])


def smoke():
    run([sys.executable, ROOT / "scripts/run_smoke.py"])
    run([sys.executable, ROOT / "scripts/acceptance_check.py", "--mode", "smoke"])


def full():
    if not (ROOT / "data/raw/TSI15k").exists():
        raise SystemExit("Dataset is missing. Run: python project.py download")

    run([sys.executable, ROOT / "scripts/bootstrap_upstream.py"])
    run([sys.executable, ROOT / "scripts/setup_full_env.py"])
    run([sys.executable, ROOT / "scripts/inspect_dataset.py", "--root", ROOT / "data/raw/TSI15k"])
    run([
        sys.executable, ROOT / "scripts/prepare_dataset.py",
        "--source", ROOT / "data/raw/TSI15k",
        "--dest", ROOT / "data/processed/TSI15k",
    ])

    code = "import torch; print(torch.cuda.device_count())"
    gpu_count = int(subprocess.check_output([sys.executable, "-c", code], text=True).strip() or "0")
    if gpu_count < 1:
        raise SystemExit("No CUDA GPU detected. Full run requires an NVIDIA CUDA GPU.")

    default_gpus = 1 if os.name == "nt" else gpu_count
    num_gpus = int(os.environ.get("NUM_GPUS", default_gpus))
    batch_size = int(os.environ.get("BATCH_SIZE", 16 if num_gpus >= 8 else max(1, num_gpus * 2)))

    run([
        sys.executable, ROOT / "scripts/run_full.py",
        "--dataset", ROOT / "data/processed/TSI15k",
        "--num-gpus", str(num_gpus),
        "--batch-size", str(batch_size),
    ])


def main():
    parser = argparse.ArgumentParser(description="SemiTNet project launcher")
    parser.add_argument("command", choices=["install", "download", "smoke", "full"])
    args = parser.parse_args()

    if args.command == "install":
        install()
        return

    relaunch_in_venv(args.command)
    {"download": download, "smoke": smoke, "full": full}[args.command]()


if __name__ == "__main__":
    main()
