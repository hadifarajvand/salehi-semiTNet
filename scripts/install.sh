#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_CMD="python3.10"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "Python 3.10 is required." >&2
  exit 1
fi

"$PYTHON_CMD" -c 'import sys; assert sys.version_info[:2] == (3,10), "Python 3.10 is required"'

if [[ ! -d .venv ]]; then
  "$PYTHON_CMD" -m venv .venv
fi

PY="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3,10), "The .venv must use Python 3.10"'
"$PY" -m pip install --upgrade pip setuptools wheel

if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  "$PIP" install --extra-index-url https://download.pytorch.org/whl/cu117 \
    'torch==1.13.1+cu117' 'torchvision==0.14.1+cu117'
elif [[ "$(uname -s)" == "Linux" ]]; then
  "$PIP" install --index-url https://download.pytorch.org/whl/cpu \
    'torch==1.13.1+cpu' 'torchvision==0.14.1+cpu'
else
  "$PIP" install 'torch==1.13.1' 'torchvision==0.14.1'
fi

"$PIP" install -r requirements-smoke.txt
"$PIP" install -r requirements-tools.txt
"$PY" scripts/bootstrap_upstream.py

printf '\n[ok] Project environment ready: %s\n' "$PY"
