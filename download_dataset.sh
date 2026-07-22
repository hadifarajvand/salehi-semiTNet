#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/download_assets.py --dataset
python scripts/prepare_dataset.py --source data/raw/TSI15k --dest data/processed/TSI15k
