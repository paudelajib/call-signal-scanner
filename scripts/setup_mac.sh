#!/usr/bin/env bash
# One-time setup + first data download, then starts the app.
#   cd ~/Desktop/call-signal-scanner2 && bash scripts/setup_mac.sh
set -e
cd "$(dirname "$0")/.."
PY=python3
if command -v conda >/dev/null 2>&1; then
  if ! conda env list | grep -q "^scanner "; then
    echo ">> creating conda env 'scanner' (python 3.12)"
    conda create -n scanner python=3.12 -y >/dev/null
  fi
  PY="$(conda run -n scanner which python)"
else
  [ -d .venv ] || python3 -m venv .venv
  PY=".venv/bin/python"
fi
echo ">> installing requirements"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements.txt
echo ">> running tests"
"$PY" -m pytest -q
echo ">> first data refresh (10-20 min the first time)"
"$PY" -m src.cli refresh 2>&1 | tee data_refresh.log
echo ">> backtest on real history"
"$PY" -m src.cli backtest 2>&1 | tee backtest.log
echo ">> starting app at http://127.0.0.1:8050"
open http://127.0.0.1:8050 2>/dev/null || true
"$PY" app.py
