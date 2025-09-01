#!/usr/bin/env bash
python -m src.pipeline --tickers data/tickers_50.txt --outdir results "$@"
