#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then echo "Run: make install" >&2; exit 1; fi
"$PY" scripts/download_assets.py --dataset
"$PY" scripts/prepare_dataset.py --source data/raw/TSI15k --dest data/processed/TSI15k
