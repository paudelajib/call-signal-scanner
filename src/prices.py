"""Daily OHLCV store with incremental refresh.

Storage: one long-format parquet (date, ticker, open, high, low, close, volume), split/dividend
adjusted (yfinance auto_adjust=True).

Each refresh re-downloads the last `history.overlap_days` days for every ticker. If a ticker's
overlapping closes disagree with what we stored (a dividend or split re-adjusted Yahoo's whole
history), that ticker's full history is re-downloaded so the series stays internally consistent.
Unfinished intraday bars (refresh before ~16:15 New York time) are dropped: this is a daily-close
system.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from .config import Paths, ns

FIELDS = ["open", "high", "low", "close", "volume"]
NY = ZoneInfo("America/New_York")
CHUNK = 80


def load_prices(paths: Paths) -> pd.DataFrame:
    if paths.prices.exists():
        df = pd.read_parquet(paths.prices)
        df["date"] = ns(df["date"])
        return df
    return pd.DataFrame(columns=["date", "ticker"] + FIELDS)


def to_panel(long: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Long -> {field: wide DataFrame (index=date, columns=ticker)}."""
    out = {}
    for f in FIELDS:
        out[f] = long.pivot(index="date", columns="ticker", values=f).sort_index()
    return out


def _last_complete_session() -> pd.Timestamp:
    """Latest date whose daily bar is final (after 16:15 New York time)."""
    now = datetime.now(NY)
    d = now.date()
    if now.hour * 60 + now.minute < 16 * 60 + 15:
        d = d - timedelta(days=1)
    return pd.Timestamp(d)


def _download(tickers: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Download a chunk; returns long df."""
    if not tickers:
        return pd.DataFrame(columns=["date", "ticker"] + FIELDS)
    raw = yf.download(tickers, start=start, end=end, interval="1d", group_by="ticker",
                      auto_adjust=True, actions=False, threads=True, progress=False,
                      multi_level_index=True, timeout=30)
    frames = []
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "ticker"] + FIELDS)
    top = raw.columns.get_level_values(0).unique()
    for t in tickers:
        if t not in top:
            continue
        sub = raw[t].copy()
        sub.columns = [str(c).strip().lower() for c in sub.columns]
        if "close" not in sub.columns:
            continue
        sub = sub[[c for c in FIELDS if c in sub.columns]].dropna(subset=["close"])
        if sub.empty:
            continue
        idx = pd.DatetimeIndex(pd.to_datetime(sub.index))
        if idx.tz is not None:
            idx = idx.tz_localize(None)  # keep exchange-local calendar date
        sub.index = idx.normalize()
        sub.index.name = "date"
        sub = sub.reset_index()
        sub["ticker"] = t
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker"] + FIELDS)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = ns(out["date"])
    for f in FIELDS:
        if f not in out.columns:
            out[f] = np.nan
    return out[["date", "ticker"] + FIELDS]


def _download_chunked(tickers: List[str], start: str, progress: Optional[Callable] = None,
                      label: str = "prices") -> pd.DataFrame:
    parts = []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        try:
            parts.append(_download(chunk, start))
        except Exception as e:  # one bad chunk should not kill the refresh
            print(f"[prices] chunk {i // CHUNK} failed: {e}; retrying one by one")
            for t in chunk:
                try:
                    parts.append(_download([t], start))
                except Exception as e2:
                    print(f"[prices] {t} failed: {e2}")
        if progress:
            progress(label, min(i + CHUNK, len(tickers)), len(tickers))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["date", "ticker"] + FIELDS)


def update_prices(tickers: Iterable[str], cfg: Dict, paths: Paths,
                  progress: Optional[Callable] = None) -> Dict[str, int]:
    """Bring the store up to the latest completed session. Returns a small report."""
    tickers = sorted(set(tickers))
    store = load_prices(paths)
    cutoff = _last_complete_session()
    overlap = int(cfg["history"].get("overlap_days", 10))
    hist_start = cfg["history"]["start"]

    have = set(store["ticker"].unique()) if not store.empty else set()
    new_tickers = [t for t in tickers if t not in have]
    old_tickers = [t for t in tickers if t in have]

    report = {"new_tickers": len(new_tickers), "readjusted": 0, "rows_added": 0}

    recent = pd.DataFrame(columns=["date", "ticker"] + FIELDS)
    if old_tickers:
        last = store.groupby("ticker")["date"].max()
        win_start = (last.loc[old_tickers].min() - pd.Timedelta(days=overlap * 2)).strftime("%Y-%m-%d")
        recent = _download_chunked(old_tickers, win_start, progress, "prices (update)")

    # detect history re-adjustments (dividends / splits) on the overlapping window
    readjust: List[str] = []
    if not recent.empty and not store.empty:
        m = recent.merge(store[["date", "ticker", "close"]], on=["date", "ticker"],
                         suffixes=("", "_old"))
        if not m.empty:
            dev = (m["close"] / m["close_old"] - 1).abs().groupby(m["ticker"]).max()
            readjust = dev[dev > 0.002].index.tolist()
    report["readjusted"] = len(readjust)

    full = _download_chunked(new_tickers + readjust, hist_start, progress, "prices (full history)")

    # merge: drop superseded rows, append fresh ones
    keep = store
    if readjust:
        keep = keep[~keep["ticker"].isin(readjust)]
    if not recent.empty:
        rr = recent[~recent["ticker"].isin(readjust)]
        first_new = rr.groupby("ticker")["date"].min()
        k = keep.merge(first_new.rename("first_new"), left_on="ticker", right_index=True, how="left")
        keep = keep[~(k["first_new"].notna() & (k["date"] >= k["first_new"])).values]
        recent = rr
    before = len(store)
    merged = pd.concat([x for x in [keep, recent, full] if not x.empty], ignore_index=True)
    merged["date"] = ns(merged["date"])
    merged = merged[merged["date"] <= cutoff]
    merged = merged.drop_duplicates(["date", "ticker"], keep="last")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    for f in FIELDS:
        merged[f] = pd.to_numeric(merged[f], errors="coerce").astype("float64")
    merged["ticker"] = merged["ticker"].astype(str)
    merged.to_parquet(paths.prices, index=False)
    report["rows_added"] = len(merged) - before
    report["last_date"] = str(merged["date"].max().date()) if not merged.empty else None
    report["tickers"] = int(merged["ticker"].nunique())
    return report
