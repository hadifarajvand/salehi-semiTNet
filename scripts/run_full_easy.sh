#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Project .venv not found. Run: make install" >&2
  exit 1
fi

if [[ ! -d data/raw/TSI15k ]]; then
  echo "Dataset not found. Run: make download" >&2
  exit 1
fi

export PATH="$ROOT/.venv/bin:$PATH"
"$PY" scripts/bootstrap_upstream.py

SETUP_MARKER="vendor/SemiT-SAM/.full_setup_done"
if [[ ! -f "$SETUP_MARKER" ]] || ! "$PY" -c 'import detectron2' >/dev/null 2>&1; then
  bash scripts/setup_full_env.sh
  touch "$SETUP_MARKER"
fi

"$PY" scripts/inspect_dataset.py --root data/raw/TSI15k
"$PY" scripts/prepare_dataset.py --source data/raw/TSI15k --dest data/processed/TSI15k

GPU_COUNT="$("$PY" -c 'import torch; print(torch.cuda.device_count())' | tail -n 1 | tr -d '[:space:]')"
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" == "0" ]]; then
  echo "No CUDA GPU detected. Full run requires an NVIDIA GPU." >&2
  exit 1
fi

export NUM_GPUS="${NUM_GPUS:-$GPU_COUNT}"
if [[ -z "${BATCH_SIZE:-}" ]]; then
  if (( NUM_GPUS >= 8 )); then export BATCH_SIZE=16; else export BATCH_SIZE=$((NUM_GPUS * 2)); fi
fi

bash run_full.sh
