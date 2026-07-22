#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"

if [[ ! -x "$PY" ]]; then
  echo "Project .venv not found. Run: make install" >&2
  exit 1
fi

"$PY" -c 'import sys; assert sys.version_info[:2] == (3,10), "The .venv must use Python 3.10"'
"$PY" -m pip install --upgrade pip setuptools wheel
"$PIP" install -r "$ROOT/requirements-tools.txt"

UP="$ROOT/vendor/SemiT-SAM"
if [[ -d "$UP/detectron2-main" ]]; then
  "$PIP" install -e "$UP/detectron2-main"
else
  "$PIP" install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
fi

if [[ -f "$UP/requirements.txt" ]]; then
  "$PIP" install -r "$UP/requirements.txt"
fi

OPS="$UP/mask2former/modeling/pixel_decoder/ops"
if [[ -f "$OPS/make.sh" ]]; then
  (cd "$OPS" && chmod +x make.sh && PATH="$ROOT/.venv/bin:$PATH" ./make.sh)
fi

"$PIP" install 'git+https://github.com/cocodataset/panopticapi.git'

"$PY" - <<'PY'
import torch
print('Python environment: .venv')
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit('Full SemiTNet run needs an NVIDIA CUDA GPU.')
PY
