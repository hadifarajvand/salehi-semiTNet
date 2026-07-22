#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then echo "Run: make install" >&2; exit 1; fi
"$PY" scripts/run_smoke.py
"$PY" scripts/acceptance_check.py --mode smoke
