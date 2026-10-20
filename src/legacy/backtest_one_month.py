# src/backtest_one_month.py
from __future__ import annotations
import os, argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .signals import compute_features, compute_signals

CACHE_ROOT = Path("cache/ohlcv_daily")

def load_tickers(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def _list_snapshot_days(root: Path) -> list[str]:
    if not root.exists():
        return []
    days = []
    for p in root.iterdir():
        if p.is_dir():
            try:
                datetime.strptime(p.name, "%Y-%m-%d")
                days.append(p.name)
            except Exception:
                pass
    return sorted(days)

def _ensure_dtindex_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex exists, sorted, and tz-naive."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()

def _read_one_ticker(day_dir: Path, ticker: str) -> pd.DataFrame | None:
    p = day_dir / f"{ticker.upper()}.parquet"
    if p.exists():
        try:
            df = pd.read_parquet(p)
            return _ensure_dtindex_naive(df)
        except Exception:
            pass
    csv_p = (day_dir / f"{ticker.upper()}").with_suffix(".csv")
    if csv_p.exists():
        try:
            df = pd.read_csv(csv_p, index_col=0, parse_dates=True)
            return _ensure_dtindex_naive(df)
        except Exception:
            pass
    return None

def _safe_last_le_date(df: pd.DataFrame, cutoff: pd.Timestamp):
    """Return last index <= cutoff, handling tz-naive consistently."""
    df = _ensure_dtindex_naive(df)
    cutoff_naive = pd.Timestamp(cutoff).tz_localize(None)
    idx = df.index[df.index <= cutoff_naive]
    if len(idx) == 0:
        return None
    return idx[-1]

def _latest_index_in_snapshot(day_dir: Path, tickers: list[str]):
    # Peek up to 20 tickers to infer the latest trading date (tz-naive)
    for t in tickers[:20]:
        df = _read_one_ticker(day_dir, t)
        if df is not None and not df.empty:
            return df.index.max()
    # Fallback: scan any file
    for f in sorted(day_dir.iterdir()):
        if f.suffix not in (".parquet", ".csv"):
            continue
        try:
            tmp = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, index_col=0, parse_dates=True)
            tmp = _ensure_dtindex_naive(tmp)
            if not tmp.empty:
                return tmp.index.max()
        except Exception:
            continue
    return None

def _ensure_spy_in_snapshot(day_dir: Path, last_idx_snapshot: pd.Timestamp) -> pd.DataFrame | None:
    """Load SPY from cache; if missing, fetch via yfinance and cache it (tz-naive)."""
    spy = _read_one_ticker(day_dir, "SPY")
    if spy is not None and not spy.empty:
        return spy

    # Try to fetch SPY only if missing
    try:
        import yfinance as yf
    except Exception:
        print("[backtest] yfinance not installed; cannot fetch SPY. Relative returns will be NaN.")
        return None

    try:
        end = (pd.Timestamp(last_idx_snapshot).tz_localize(None) + pd.Timedelta(days=1)).date().isoformat()
        start = (pd.Timestamp(last_idx_snapshot).tz_localize(None) - pd.Timedelta(days=800)).date().isoformat()
        print(f"[backtest] Fetching SPY via yfinance for {start} → {end} ...")
        # Make behavior explicit and consistent
        df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            print("[backtest] SPY fetch returned empty; skipping relative returns.")
            return None
        df = _ensure_dtindex_naive(df)

        # Save to snapshot folder (prefer parquet)
        out_parq = day_dir / "SPY.parquet"
        try:
            df.to_parquet(out_parq)
            print(f"[backtest] Cached SPY to {out_parq}")
        except Exception:
            out_csv = day_dir / "SPY.csv"
            df.to_csv(out_csv)
            print(f"[backtest] Cached SPY to {out_csv} (csv fallback)")
        return df
    except Exception as e:
        print(f"[backtest] SPY fetch failed: {e}; continuing without relative returns.")
        return None

def main():
    ap = argparse.ArgumentParser(description="Backtest: BuyScore vs forward returns (absolute & vs SPY)")
    ap.add_argument("--tickers", required=True, help="Path to tickers file")
    ap.add_argument("--outdir", default="results", help="Output folder")
    ap.add_argument("--snapshot_date", default="", help="Snapshot date YYYY-MM-DD (default: latest)")
    ap.add_argument("--lookback_days", type=int, default=30, help="Lookback days (default 30)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("[backtest] start")
    days = _list_snapshot_days(CACHE_ROOT)
    if not days:
        print("[backtest] No snapshots found, run pipeline first.")
        return

    snap_day = args.snapshot_date if args.snapshot_date else days[-1]
    day_dir = CACHE_ROOT / snap_day
    print(f"[backtest] Using snapshot: {snap_day}")

    tickers = load_tickers(args.tickers)
    last_idx_snapshot = _latest_index_in_snapshot(day_dir, tickers)
    if last_idx_snapshot is None:
        print("[backtest] No usable data inside snapshot.")
        return
    # Ensure tz-naive reference dates
    last_idx_snapshot = pd.Timestamp(last_idx_snapshot).tz_localize(None)
    asof_cutoff = last_idx_snapshot - pd.Timedelta(days=args.lookback_days)
    print(f"[backtest] Snapshot last date: {last_idx_snapshot.date()}, as-of cutoff: {asof_cutoff.date()}")

    # --- SPY baseline (from cache or fetch) ---
    spy_df = _ensure_spy_in_snapshot(day_dir, last_idx_snapshot)
    if spy_df is None or spy_df.empty:
        spy_ret = None
        print("[backtest] SPY unavailable; RelFwdRetPct will be NaN.")
    else:
        cutoff_idx_spy = _safe_last_le_date(spy_df, asof_cutoff)
        if cutoff_idx_spy is None:
            spy_ret = None
            print("[backtest] SPY has no bar <= cutoff; RelFwdRetPct will be NaN.")
        else:
            close_asof_spy = float(spy_df.loc[cutoff_idx_spy, "Close"])
            close_last_spy = float(spy_df.loc[spy_df.index[-1], "Close"])
            spy_ret = (close_last_spy / close_asof_spy - 1.0) * 100.0
            print(f"[backtest] SPY baseline return = {round(spy_ret, 2)} %")

    rows = []
    for t in tickers:
        df = _read_one_ticker(day_dir, t)
        if df is None or df.empty:
            continue
        if len(df) < 60:
            continue

        try:
            feat = compute_features(df)
            sigs = compute_signals(feat)
        except Exception:
            continue

        cutoff_idx = _safe_last_le_date(sigs, asof_cutoff)
        if cutoff_idx is None:
            continue

        try:
            row = sigs.loc[cutoff_idx]
            buy_score = float(row.get("BuyScore", np.nan))
            close_asof = float(row.get("Close", np.nan))
            last_idx = sigs.index[-1]
            close_last = float(sigs.loc[last_idx, "Close"])
        except Exception:
            continue

        if np.isnan(buy_score) or np.isnan(close_asof) or np.isnan(close_last):
            continue

        fwd_ret_pct = (close_last / close_asof - 1.0) * 100.0
        rel_fwd_ret_pct = fwd_ret_pct - spy_ret if spy_ret is not None else np.nan

        rows.append({
            "Ticker": t,
            "BuyScore": round(buy_score, 2),
            "FwdRetPct": round(fwd_ret_pct, 2),
            "RelFwdRetPct": round(rel_fwd_ret_pct, 2) if not np.isnan(rel_fwd_ret_pct) else np.nan
        })

    out = pd.DataFrame(rows).sort_values("BuyScore", ascending=False)

    csv_path = os.path.join(args.outdir, f"backtest_simple_{snap_day}.csv")
    out.to_csv(csv_path, index=False)
    print(f"[backtest] Saved CSV → {csv_path}, rows={len(out)}")

    if not out.empty:
        # Absolute scatter
        plt.figure()
        plt.scatter(out["BuyScore"], out["FwdRetPct"], alpha=0.6)
        plt.title("BuyScore vs Forward Return (Absolute)")
        plt.xlabel("BuyScore")
        plt.ylabel("Forward Return (%)")
        plt.axvline(60, linestyle="--")
        plt.axhline(0, linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"backtest_abs_scatter_{snap_day}.png"))
        plt.close()
        print(f"[backtest] Saved absolute scatter plot")

        # Relative scatter
        if "RelFwdRetPct" in out.columns and out["RelFwdRetPct"].notna().any():
            plt.figure()
            plt.scatter(out["BuyScore"], out["RelFwdRetPct"], alpha=0.6)
            plt.title("BuyScore vs Forward Return (Relative to SPY)")
            plt.xlabel("BuyScore")
            plt.ylabel("Excess Return vs SPY (%)")
            plt.axvline(60, linestyle="--")
            plt.axhline(0, linestyle="--")
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"backtest_rel_scatter_{snap_day}.png"))
            plt.close()
            print(f"[backtest] Saved relative scatter plot")

if __name__ == "__main__":
    main()