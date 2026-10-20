#!/usr/bin/env bash
# Start the web app:  ./scripts/run.sh   then open http://127.0.0.1:8050
cd "$(dirname "$0")/.." && python app.py "$@"
