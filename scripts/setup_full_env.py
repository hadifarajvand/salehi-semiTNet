#!/usr/bin/env python3
from __future__ import annotations

import json
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


def torch_info():
    code = (
        "import json,torch; from torch.utils.cpp_extension import CUDA_HOME; "
        "print(json.dumps({'version':torch.__version__,'cuda':torch.version.cuda,"
        "'available':torch.cuda.is_available(),'devices':torch.cuda.device_count(),'cuda_home':CUDA_HOME}))"
    )
    return json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))


def ensure_cuda_torch():
    info = torch_info()
    if info["available"] and info["cuda"]:
        return info

    if shutil.which("nvidia-smi") is None:
        raise SystemExit("NVIDIA GPU/driver not detected. Full run requires CUDA.")

    pip(
        "install", "--force-reinstall",
        "--extra-index-url", "https://download.pytorch.org/whl/cu117",
        "torch==1.13.1+cu117", "torchvision==0.14.1+cu117",
    )
    info = torch_info()
    if not info["available"]:
        raise SystemExit("CUDA PyTorch was installed but no CUDA GPU is available to the .venv.")
    return info


def main():
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Full setup requires the project's Python 3.10 .venv.")
    if not UPSTREAM.exists():
        raise SystemExit("Model source is missing. Run: python project.py install")

    pip("install", "--upgrade", "pip", "setuptools", "wheel")
    info = ensure_cuda_torch()

    if not info.get("cuda_home"):
        hint = (
            "Install CUDA Toolkit 11.7 and Visual Studio C++ Build Tools, then reopen the terminal."
            if platform.system() == "Windows"
            else "Install CUDA Toolkit 11.7 with nvcc and a compatible C/C++ compiler."
        )
        raise SystemExit("CUDA toolkit compiler was not detected. " + hint)

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

    subprocess.run([sys.executable, "-c", "import detectron2, MultiScaleDeformableAttention"], check=True)
    info = torch_info()
    print("[ok] full environment ready")
    print("Python:", sys.executable)
    print("PyTorch:", info["version"])
    print("CUDA runtime:", info["cuda"])
    print("CUDA available:", info["available"])
    print("CUDA devices:", info["devices"])
    print("CUDA_HOME:", info["cuda_home"])


if __name__ == "__main__":
    main()
