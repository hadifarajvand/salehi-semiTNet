#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-smoke.txt
python -m pip install huggingface_hub requests pyyaml
python scripts/bootstrap_upstream.py
printf '\n[ok] Installation complete.\n'
