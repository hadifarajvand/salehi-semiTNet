#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/run_official_inference.py \
  --dataset data/processed/TSI15k \
  --checkpoint assets/checkpoints/SemiTNet_Tooth_Instance_Segmentation_32Classes.pth
