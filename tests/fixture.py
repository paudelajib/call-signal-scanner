"""Synthetic data fixture - FOR CODE-PATH TESTS ONLY.

These random series exist purely to check that the pipeline runs end to end without errors.
They are never written into the real data/ folder and never produce scores you would look at;
real scores and backtests always come from downloaded market data.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Paths, load_config

SECTORS = ["Information Technology", "Financials", "Health Care", "Energy"]


def make_fixture(tmp: Path, n_tickers: int = 60, start: str = "2015-01-01", seed: int = 0):
    rng = np.random.default_rng(seed)
    cfg = copy.deepcopy(load_config())
    cfg["data_dir"] = str(tmp)
    cfg["backtest"]["start"] = "2016-06-01"
    paths = Paths(cfg)
    dates = pd.bdate_range(start, "2026-10-16")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    uni = pd.DataFrame({"ticker": tickers, "name": [f"Test Co {i}" for i in range(n_tickers)],
                        "sector": [SECTORS[i % 4] for i in range(n_tickers)],
                        "sub_industry": "x", "date_added": "2000-01-01"})
    uni.to_csv(paths.universe, index=False)
    mkt = rng.normal(0.0004, 0.01, len(dates))
    rows = []
    for t in tickers + ["SPY"]:
        b = 1.0 if t == "SPY" else rng.uniform(0.6, 1.4)
        r = b * mkt + (0 if t == "SPY" else rng.normal(0, 0.015, len(dates)))
        c = 50 * np.exp(np.cumsum(r))
        o = c * (1 + rng.normal(0, 0.003, len(dates)))
        df = pd.DataFrame({"date": dates, "ticker": t, "open": o, "high": np.maximum(o, c) * 1.01,
                           "low": np.minimum(o, c) * 0.99, "close": c,
                           "volume": rng.lognormal(14, 0.4, len(dates))})
        if t == "T005":  # a late listing
            df = df[df["date"] >= "2020-01-01"]
        rows.append(df)
    pd.concat(rows).to_parquet(paths.prices, index=False)
    mac = pd.DataFrame({"DGS10": 2 + np.cumsum(rng.normal(0, 0.03, len(dates))),
                        "DGS2": 1 + np.cumsum(rng.normal(0, 0.03, len(dates))),
                        "T10YIE": 2 + np.cumsum(rng.normal(0, 0.01, len(dates))),
                        "DFF": 1.0, "VIX": 18 + rng.normal(0, 2, len(dates))}, index=dates)
    mac.index.name = "date"
    mac.to_parquet(paths.macro)
    q_ends = pd.date_range("2021-03-31", "2026-06-30", freq="QE")
    val = pd.DataFrame([{"ticker": t, "period_end": d, "trailing_pe": rng.uniform(8, 40),
                         "forward_pe": rng.uniform(8, 35), "pb": np.nan, "ps": np.nan,
                         "ev_ebitda": np.nan} for t in tickers for d in q_ends])
    val.loc[val.sample(frac=0.05, random_state=1).index, "trailing_pe"] = np.nan
    val.to_parquet(paths.valuation, index=False)
    info = pd.DataFrame({"ticker": tickers, "shortName": uni["name"],
                         "trailingEps": rng.uniform(-1, 6, n_tickers),
                         "forwardEps": rng.uniform(0.5, 7, n_tickers),
                         "marketCap": rng.uniform(1e10, 1e12, n_tickers),
                         "fetched_at": pd.Timestamp("2026-10-15")})
    info.to_parquet(paths.info_latest, index=False)
    ns = pd.DataFrame({"date": pd.Timestamp("2026-10-16"), "ticker": tickers,
                       "sentiment": rng.uniform(-0.5, 0.5, n_tickers), "n_headlines": 5,
                       "n_sources": 3, "n_48h": 2})
    ns.to_parquet(paths.news_sent, index=False)
    return cfg, paths
