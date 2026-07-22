#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d data/raw/TSI15k ]]; then
  echo "Dataset not found. Run: make download" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda/Miniconda is required for the full GPU run." >&2
  exit 1
fi

ENV_NAME="semitnet-paper"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env create -f environment-full.yml
fi

conda run -n "$ENV_NAME" python scripts/bootstrap_upstream.py

SETUP_MARKER="vendor/SemiT-SAM/.full_setup_done"
if [[ ! -f "$SETUP_MARKER" ]]; then
  conda run -n "$ENV_NAME" bash scripts/setup_full_env.sh
  touch "$SETUP_MARKER"
fi

conda run -n "$ENV_NAME" python scripts/inspect_dataset.py --root data/raw/TSI15k
conda run -n "$ENV_NAME" python scripts/prepare_dataset.py --source data/raw/TSI15k --dest data/processed/TSI15k

GPU_COUNT="$(conda run -n "$ENV_NAME" python -c 'import torch; print(torch.cuda.device_count())' | tail -n 1 | tr -d '[:space:]')"
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" == "0" ]]; then
  echo "No CUDA GPU detected. Full run requires an NVIDIA GPU." >&2
  exit 1
fi

export NUM_GPUS="${NUM_GPUS:-$GPU_COUNT}"
if [[ -z "${BATCH_SIZE:-}" ]]; then
  if (( NUM_GPUS >= 8 )); then export BATCH_SIZE=16; else export BATCH_SIZE=$((NUM_GPUS * 2)); fi
fi

conda run -n "$ENV_NAME" bash run_full.sh
