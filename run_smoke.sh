#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/run_smoke.py
python scripts/acceptance_check.py --mode smoke
