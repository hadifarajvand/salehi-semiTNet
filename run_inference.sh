#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then echo "Run: make install" >&2; exit 1; fi
"$PY" scripts/run_official_inference.py \
  --dataset data/processed/TSI15k \
  --checkpoint assets/checkpoints/SemiTNet_Tooth_Instance_Segmentation_32Classes.pth
