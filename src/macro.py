"""Macro data: Treasury yields, breakeven inflation, Fed funds, CPI (FRED) and VIX (Yahoo).

Stored as one wide daily parquet. Monthly CPI is kept for display only (it is published with a
lag, so using it in scores would leak future information); the scoring uses the daily
10-year breakeven inflation rate (T10YIE) as the market's real-time inflation expectation.
"""
from __future__ import annotations

import io
from typing import Dict

import pandas as pd
import requests
import yfinance as yf

from .config import Paths

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
# Yahoo fallbacks if FRED is unreachable (yields in percent)
YAHOO_FALLBACK = {"DGS10": "^TNX"}


def _fred(series_id: str, start: str) -> pd.Series:
    r = requests.get(FRED_URL.format(sid=series_id), timeout=30,
                     headers={"User-Agent": "call-signal-scanner/2.0"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    date_col = df.columns[0]
    s = pd.to_numeric(df[series_id], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return s[s.index >= pd.Timestamp(start)].dropna().rename(series_id)


def _yahoo_close(symbol: str, start: str) -> pd.Series:
    raw = yf.download(symbol, start=start, interval="1d", auto_adjust=False, progress=False,
                      multi_level_index=False, timeout=30)
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    s = raw["Close"].astype(float)
    idx = pd.DatetimeIndex(s.index)
    s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return s


def update_macro(cfg: Dict, paths: Paths) -> Dict[str, str]:
    start = cfg["history"]["start"]
    cols, status = {}, {}
    for sid in cfg["macro"]["fred_series"]:
        try:
            cols[sid] = _fred(sid, start)
            status[sid] = "fred"
        except Exception as e:
            fb = YAHOO_FALLBACK.get(sid)
            if fb:
                try:
                    cols[sid] = _yahoo_close(fb, start).rename(sid)
                    status[sid] = f"yahoo {fb}"
                    continue
                except Exception:
                    pass
            status[sid] = f"failed: {e}"
    for sym, name in cfg["macro"].get("yahoo_series", {}).items():
        try:
            cols[name] = _yahoo_close(sym, start).rename(name)
            status[name] = "yahoo"
        except Exception as e:
            status[name] = f"failed: {e}"

    if not cols:
        return status
    new = pd.concat(cols.values(), axis=1).sort_index()
    if paths.macro.exists():  # keep any history we had if a source failed today
        old = pd.read_parquet(paths.macro)
        new = new.combine_first(old)
    new.index.name = "date"
    new.to_parquet(paths.macro)
    return status


def load_macro(paths: Paths) -> pd.DataFrame:
    if not paths.macro.exists():
        return pd.DataFrame()
    df = pd.read_parquet(paths.macro)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def macro_snapshot(macro: pd.DataFrame, trend_days: int = 63) -> Dict[str, Dict[str, float]]:
    """Latest level + change over the trend window, for the UI header."""
    out = {}
    for c in macro.columns:
        s = macro[c].dropna()
        if s.empty:
            continue
        last = float(s.iloc[-1])
        prev_date = s.index[-1] - pd.Timedelta(days=int(trend_days * 365 / 252))
        prev = s[s.index <= prev_date]
        chg = float(last - prev.iloc[-1]) if not prev.empty else None
        if c == "CPIAUCSL":  # show CPI as year-over-year inflation
            yago = s[s.index <= s.index[-1] - pd.Timedelta(days=365)]
            if not yago.empty:
                out["CPI_YOY"] = {"value": (last / float(yago.iloc[-1]) - 1) * 100,
                                  "change": None, "asof": str(s.index[-1].date())}
            continue
        out[c] = {"value": last, "change": chg, "asof": str(s.index[-1].date())}
    return out
