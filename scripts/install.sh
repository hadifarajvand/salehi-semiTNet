#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.10 is required. Install it and run: make install" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

PY="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"

"$PY" -c 'import sys; assert sys.version_info[:2] == (3,10), "The .venv must use Python 3.10"'
"$PY" -m pip install --upgrade pip setuptools wheel

if [[ "$(uname -s)" == "Linux" ]]; then
  "$PIP" install --extra-index-url https://download.pytorch.org/whl/cu117 \
    'torch==1.13.1+cu117' 'torchvision==0.14.1+cu117'
else
  "$PIP" install 'torch==1.13.1' 'torchvision==0.14.1'
fi

"$PIP" install -r requirements-smoke.txt
"$PIP" install -r requirements-tools.txt
"$PY" scripts/bootstrap_upstream.py

printf '\n[ok] .venv is ready: %s\n' "$PY"
