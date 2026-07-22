#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -d "$ROOT/vendor/SemiT-SAM" ]]; then
  echo "Run 'make bootstrap' first." >&2; exit 1
fi
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT/requirements-tools.txt"
UP="$ROOT/vendor/SemiT-SAM"
if [[ -d "$UP/detectron2-main" ]]; then python -m pip install -e "$UP/detectron2-main"; else python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'; fi
if [[ -f "$UP/requirements.txt" ]]; then python -m pip install -r "$UP/requirements.txt"; fi
OPS="$UP/mask2former/modeling/pixel_decoder/ops"
if [[ -f "$OPS/make.sh" ]]; then (cd "$OPS" && chmod +x make.sh && ./make.sh); fi
python -m pip install git+https://github.com/cocodataset/panopticapi.git
python - <<'PY'
import torch
print('PyTorch:',torch.__version__)
print('CUDA:',torch.cuda.is_available(), 'devices:',torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit('Full SemiTNet setup needs a CUDA-enabled PyTorch installation.')
PY
