"""Fundamentals from Yahoo: P/E history (quarterly) + current company info.

Valuation is used as *earnings yield* (EPS / price) rather than raw P/E: it stays meaningful for
loss-making companies (negative yield = bad) and avoids the P/E blow-up near zero earnings.

Point-in-time handling for backtests: a quarter-end P/E is only treated as known
`fundamentals.report_lag_days` after the quarter ends (earnings are reported weeks later).
Each fetch of company info is also appended to info_history, so over time the project builds
its own point-in-time fundamentals record.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .config import Paths, ns

INFO_FIELDS = [
    "shortName", "longName", "sector", "industry", "marketCap", "currentPrice",
    "trailingPE", "forwardPE", "trailingEps", "forwardEps", "priceToBook",
    "priceToSalesTrailing12Months", "enterpriseToEbitda", "pegRatio", "dividendYield", "beta",
    "profitMargins", "returnOnEquity", "revenueGrowth", "earningsGrowth", "debtToEquity",
    "recommendationMean", "recommendationKey", "numberOfAnalystOpinions", "targetMeanPrice",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "earningsTimestamp", "website", "longBusinessSummary",
]

VAL_ROWS = {
    "trailing_pe": ["trailing p/e"],
    "forward_pe": ["forward p/e"],
    "pb": ["price/book"],
    "ps": ["price/sales"],
    "ev_ebitda": ["enterprise value/ebitda"],
}


def _pick_row(df: pd.DataFrame, keys: List[str]) -> Optional[pd.Series]:
    for lab in df.index:
        low = str(lab).lower()
        if any(k in low for k in keys):
            return df.loc[lab]
    return None


def _fetch_one(ticker: str) -> Dict:
    tk = yf.Ticker(ticker)
    out: Dict = {"ticker": ticker, "val": None, "info": None, "error": None}
    try:
        vm = tk.get_valuation_measures(freq="quarterly", periods=None)
        if vm is not None and not vm.empty:
            rows = []
            date_cols = [c for c in vm.columns if c != "Current"]
            picked = {k: _pick_row(vm, v) for k, v in VAL_ROWS.items()}
            for c in date_cols:
                try:
                    d = pd.to_datetime(c, format="%m/%d/%Y")
                except Exception:
                    d = pd.to_datetime(c, errors="coerce")
                if pd.isna(d):
                    continue
                row = {"ticker": ticker, "period_end": d}
                for k, s in picked.items():
                    row[k] = float(s[c]) if s is not None and pd.notna(s[c]) else np.nan
                rows.append(row)
            out["val"] = pd.DataFrame(rows)
    except Exception as e:
        out["error"] = f"valuation: {e}"
    try:
        info = tk.get_info() or {}
        rec = {k: info.get(k) for k in INFO_FIELDS}
        if rec.get("longBusinessSummary"):
            rec["longBusinessSummary"] = str(rec["longBusinessSummary"])[:1200]
        rec["ticker"] = ticker
        rec["fetched_at"] = pd.Timestamp(datetime.now().date())
        out["info"] = rec
    except Exception as e:
        out["error"] = (out["error"] or "") + f" info: {e}"
    return out


def update_fundamentals(tickers: Iterable[str], cfg: Dict, paths: Paths,
                        progress: Optional[Callable] = None, force: bool = False) -> Dict:
    tickers = list(tickers)
    info_latest = pd.read_parquet(paths.info_latest) if paths.info_latest.exists() else pd.DataFrame()
    max_age = pd.Timedelta(days=int(cfg["fundamentals"].get("refresh_days", 7)))
    today = pd.Timestamp(datetime.now().date())
    if not force and not info_latest.empty:
        age = today - ns(info_latest.set_index("ticker")["fetched_at"])
        fresh = set(age[age < max_age].index)
        todo = [t for t in tickers if t not in fresh]
    else:
        todo = tickers

    vals, infos, errors = [], [], 0
    workers = int(cfg["fundamentals"].get("workers", 4))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, t): t for t in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r["val"] is not None and not r["val"].empty:
                vals.append(r["val"])
            if r["info"]:
                infos.append(r["info"])
            if r["error"]:
                errors += 1
            if progress:
                progress("fundamentals", i, len(todo))
            time.sleep(0.05)

    if vals:
        new_val = pd.concat(vals, ignore_index=True)
        if paths.valuation.exists():
            old = pd.read_parquet(paths.valuation)
            new_val = pd.concat([old, new_val]).drop_duplicates(["ticker", "period_end"], keep="last")
        new_val.sort_values(["ticker", "period_end"]).to_parquet(paths.valuation, index=False)
    if infos:
        new_info = pd.DataFrame(infos)
        for c in new_info.columns:  # parquet-friendly types
            if new_info[c].dtype == object and c not in ("ticker",):
                if all(isinstance(v, (int, float, type(None))) for v in new_info[c]):
                    new_info[c] = pd.to_numeric(new_info[c], errors="coerce")
                else:
                    new_info[c] = new_info[c].astype("string")
        merged = pd.concat([info_latest, new_info]) if not info_latest.empty else new_info
        merged = merged.drop_duplicates("ticker", keep="last")
        merged.to_parquet(paths.info_latest, index=False)
        hist = pd.read_parquet(paths.info_history) if paths.info_history.exists() else pd.DataFrame()
        num_cols = ["ticker", "fetched_at", "trailingPE", "forwardPE", "trailingEps", "forwardEps",
                    "priceToBook", "marketCap", "recommendationMean", "targetMeanPrice"]
        snap = new_info[[c for c in num_cols if c in new_info.columns]]
        hist = pd.concat([hist, snap]).drop_duplicates(["ticker", "fetched_at"], keep="last")
        hist.to_parquet(paths.info_history, index=False)
    return {"fetched": len(todo), "errors": errors}


def load_info(paths: Paths) -> pd.DataFrame:
    if not paths.info_latest.exists():
        return pd.DataFrame(columns=["ticker"])
    return pd.read_parquet(paths.info_latest)


def earnings_yield_panels(close: pd.DataFrame, paths: Paths, lag_days: int) -> Dict[str, pd.DataFrame]:
    """Daily point-in-time trailing & forward earnings-yield panels (index=date, cols=ticker).

    EPS at each quarter end = quarter-end price / quarter-end P/E, made available `lag_days`
    later and forward-filled; the latest company info EPS is added from its fetch date. Daily
    earnings yield = EPS / today's close, so it moves with price like a live P/E would.
    """
    out = {}
    pts: List[pd.DataFrame] = []
    if paths.valuation.exists():
        v = pd.read_parquet(paths.valuation)
        v["period_end"] = ns(v["period_end"])
        v = v[v["ticker"].isin(close.columns)]
        # price at (or just before) each quarter end
        px_long = close.stack().rename("px").reset_index()
        px_long.columns = ["date", "ticker", "px"]
        px_long["date"] = ns(px_long["date"])
        px_long = px_long.dropna(subset=["px"])
        px_long = px_long.sort_values("date")
        v = v.sort_values("period_end")
        v = pd.merge_asof(v, px_long, left_on="period_end", right_on="date", by="ticker",
                          direction="backward")
        v["avail"] = v["period_end"] + pd.Timedelta(days=lag_days)
        v["eps_t"] = v["px"] / v["trailing_pe"]
        v["eps_f"] = v["px"] / v["forward_pe"]
        pts.append(v[["ticker", "avail", "eps_t", "eps_f"]])
    info = load_info(paths)
    if not info.empty and "fetched_at" in info.columns:
        i2 = pd.DataFrame({
            "ticker": info["ticker"],
            "avail": ns(info["fetched_at"]),
            "eps_t": pd.to_numeric(info.get("trailingEps"), errors="coerce"),
            "eps_f": pd.to_numeric(info.get("forwardEps"), errors="coerce"),
        })
        pts.append(i2)
    if not pts:
        nan = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        return {"ey_trailing": nan, "ey_forward": nan.copy()}
    allp = pd.concat(pts, ignore_index=True).dropna(subset=["avail"])
    allp["avail"] = ns(allp["avail"])
    for col, name in [("eps_t", "ey_trailing"), ("eps_f", "ey_forward")]:
        w = allp.dropna(subset=[col]).pivot_table(index="avail", columns="ticker", values=col,
                                                  aggfunc="last")
        w = w.reindex(columns=close.columns)
        # EPS older than ~1.5 years (about 380 trading rows) is treated as unknown
        eps = w.reindex(w.index.union(close.index)).sort_index().ffill(limit=380).reindex(close.index)
        out[name] = eps / close
    return out
