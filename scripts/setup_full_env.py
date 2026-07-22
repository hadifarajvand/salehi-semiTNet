#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/SemiT-SAM"
OPS = UPSTREAM / "mask2former/modeling/pixel_decoder/ops"


def run(args, cwd=ROOT, env=None):
    print("+", " ".join(map(str, args)))
    subprocess.run([str(x) for x in args], cwd=str(cwd), env=env, check=True)


def pip(*args):
    run([sys.executable, "-m", "pip", *args])


def ensure_cuda_torch():
    import torch

    if torch.cuda.is_available() and torch.version.cuda:
        return

    if shutil.which("nvidia-smi") is None:
        raise SystemExit("NVIDIA GPU/driver not detected. Full run requires CUDA.")

    pip(
        "install", "--force-reinstall",
        "--extra-index-url", "https://download.pytorch.org/whl/cu117",
        "torch==1.13.1+cu117", "torchvision==0.14.1+cu117",
    )


def main():
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Full setup requires the project's Python 3.10 .venv.")
    if not UPSTREAM.exists():
        raise SystemExit("Model source is missing. Run: python project.py install")

    pip("install", "--upgrade", "pip", "setuptools", "wheel")
    ensure_cuda_torch()
    pip("install", "-r", str(ROOT / "requirements-tools.txt"))

    detectron2_local = UPSTREAM / "detectron2-main"
    if detectron2_local.exists():
        pip("install", "-e", str(detectron2_local))
    else:
        pip("install", "git+https://github.com/facebookresearch/detectron2.git@v0.6")

    upstream_requirements = UPSTREAM / "requirements.txt"
    if upstream_requirements.exists():
        pip("install", "-r", str(upstream_requirements))

    pip("install", "git+https://github.com/cocodataset/panopticapi.git")

    if not OPS.exists():
        raise SystemExit(f"Missing Mask2Former CUDA ops directory: {OPS}")

    env = os.environ.copy()
    env["FORCE_CUDA"] = "1"
    run([sys.executable, "setup.py", "build", "install"], cwd=OPS, env=env)

    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        hint = (
            "Install CUDA Toolkit 11.7 and Visual Studio C++ Build Tools, then reopen the terminal."
            if platform.system() == "Windows"
            else "Install CUDA Toolkit 11.7 with nvcc and a compatible C/C++ compiler."
        )
        raise SystemExit("CUDA toolkit compiler was not detected. " + hint)

    import detectron2  # noqa: F401
    print("[ok] full environment ready")
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA_HOME:", CUDA_HOME)


if __name__ == "__main__":
    main()
