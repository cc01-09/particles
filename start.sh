#!/usr/bin/env bash
# One-command start (Linux / macOS / WSL / UGen300 host).
# Creates .venv on first run, installs dependencies, then launches run.py.
# Usage: ./start.sh [--backend cpu|hailo|mock] [--camera N] [...]
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  # MediaPipe 0.10.21 ships wheels for Python 3.9 - 3.12.
  PY=""
  for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1 &&
       "$cand" -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)'; then
      PY="$cand"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.9-3.12 is required (MediaPipe compatibility). Please install Python 3.11 or 3.12." >&2
    exit 1
  fi
  echo "Creating virtual environment with $PY ..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

exec .venv/bin/python run.py "$@"
