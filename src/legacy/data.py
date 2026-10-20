# src/data.py
from __future__ import annotations
import time, os
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# ---------- Helpers ----------

def _today_str(tz: str | None = None) -> str:
    # Keep it simple: local date for snapshot key
    return datetime.now().strftime("%Y-%m-%d")

def _normalize_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return single-index OHLCV frame with columns:
    ['Open','High','Low','Close','Adj Close','Volume'].
    """
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(list(range(df.columns.nlevels - 1)), axis=1)

    # Normalize labels
    ren = {c: c.strip().title() for c in df.columns}
    df = df.rename(columns=ren)

    cols = ["Open","High","Low","Close","Adj Close","Volume"]
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return None
    df = df[keep]

    # Coerce numeric & datetime index
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return None
    df = df.sort_index()
    df = df.dropna(how="all")
    if len(df) < 60:
        return None
    return df

def _snapshot_path(root: Path, date_str: str, ticker: str) -> Path:
    return root / date_str / f"{ticker.upper()}.parquet"

def _read_parquet(p: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(p)
    except Exception:
        # Fallback to CSV if parquet missing
        csv_p = p.with_suffix(".csv")
        if csv_p.exists():
            try:
                return pd.read_csv(csv_p, index_col=0, parse_dates=True)
            except Exception:
                return None
        return None

def _write_parquet(p: Path, df: pd.DataFrame) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p)  # requires pyarrow or fastparquet
    except Exception:
        df.to_csv(p.with_suffix(".csv"))

# ---------- Public API ----------

def fetch_one_yf(ticker: str, period: str = "2y", interval: str = "1d",
                 max_retries: int = 2, sleep_sec: float = 0.2) -> Optional[pd.DataFrame]:
    tries, last_err = 0, None
    while tries <= max_retries:
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval,
                                           auto_adjust=False, actions=False)
            df = _normalize_ohlcv(df)
            return df
        except Exception as e:
            last_err = e
            tries += 1
            time.sleep(sleep_sec * (tries + 1))
    print(f"[data] Skipped {ticker}: {last_err}")
    return None

def fetch_prices_yf(
    tickers: List[str],
    period: str = "2y",
    interval: str = "1d",
    *,
    snapshot_dir: str | Path = "cache/ohlcv_daily",
    force_new_snapshot: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Daily snapshot cache:
       - First call of the day: downloads and writes cache/<YYYY-MM-DD>/<T>.parquet
       - Later calls the same day: read from that folder (no network).
    """
    root = Path(snapshot_dir)
    today = _today_str()
    out: Dict[str, pd.DataFrame] = {}

    # If snapshot exists and not forcing refresh, just load from disk
    if not force_new_snapshot and (root / today).exists():
        for t in tickers:
            p = _snapshot_path(root, today, t)
            df = _read_parquet(p)
            if df is not None:
                out[t] = df
        if out:
            return out  # done

    # Otherwise, (re)create today snapshot
    for t in tickers:
        df = fetch_one_yf(t, period=period, interval=interval)
        if df is not None:
            out[t] = df
            _write_parquet(_snapshot_path(root, today, t), df)
        time.sleep(0.12)  # be gentle to Yahoo
    return out