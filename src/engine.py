"""Build panels -> features -> scores, and persist what the web app needs."""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from . import fundamentals, macro, news, prices, universe
from .config import Paths, load_config
from .features import FEATURE_INFO, compute_features
from .scoring import load_weights, score_horizon, zscore_features

RECENT_ROWS = 260  # ~1 year of daily scores kept for charts / instant re-scoring
SCORED = list(FEATURE_INFO.keys())


def build_inputs(cfg: Dict, paths: Paths, with_news: bool = True) -> Dict:
    uni = universe.load_universe(cfg, paths)
    long = prices.load_prices(paths)
    if long.empty:
        raise RuntimeError("No price data yet - run a refresh first.")
    panel = prices.to_panel(long)
    bench_t = cfg["universe"].get("benchmark", "SPY")
    if bench_t not in panel["close"].columns:
        raise RuntimeError(f"Benchmark {bench_t} missing from price store.")
    bench = panel["close"][bench_t]
    cols = [t for t in uni["ticker"] if t in panel["close"].columns]
    # drop days where the market was closed for (almost) everyone, e.g. partial-data days
    panel = {k: v[cols] for k, v in panel.items()}
    good = panel["close"].notna().mean(axis=1) > 0.5
    panel = {k: v[good] for k, v in panel.items()}
    bench = bench.reindex(panel["close"].index).ffill()
    sectors = uni.set_index("ticker")["sector"].reindex(cols)
    mac = macro.load_macro(paths)
    ey = fundamentals.earnings_yield_panels(panel["close"], paths,
                                            int(cfg["fundamentals"].get("report_lag_days", 45)))
    nw = news.load_sentiment_panel(paths, panel["close"].index, cols) if with_news else None
    return {"universe": uni, "panel": panel, "bench": bench, "sectors": sectors,
            "macro": mac, "ey": ey, "news": nw}


def compute_live(cfg: Optional[Dict] = None, paths: Optional[Paths] = None) -> Dict:
    cfg = cfg or load_config()
    paths = paths or Paths(cfg)
    inp = build_inputs(cfg, paths)
    feats = compute_features(inp["panel"], inp["bench"], inp["macro"], inp["sectors"],
                             inp["ey"], inp["news"], cfg)
    idx = inp["panel"]["close"].index[-RECENT_ROWS:]
    winsor = float(cfg["score"].get("winsor_pct", 0.02))
    z = zscore_features(feats, SCORED, winsor, rows=idx)

    # persist recent z-scores (long) for instant re-scoring and score-history charts
    zl = pd.concat({k: v.stack(future_stack=True) if _has_future_stack() else v.stack()
                    for k, v in z.items()}, axis=1)
    zl.index.names = ["date", "ticker"]
    zl.reset_index().to_parquet(paths.features_recent, index=False)

    # latest raw values (for display)
    last = idx[-1]
    raw = pd.DataFrame({k: v.loc[last] for k, v in feats.items()
                        if k in SCORED or k.startswith("_")})
    raw.index.name = "ticker"
    close = inp["panel"]["close"]
    raw["close"] = close.loc[last]
    prev = close.loc[:last].iloc[-2] if len(close.loc[:last]) > 1 else np.nan
    raw["chg_1d"] = close.loc[last] / prev - 1
    raw.reset_index().to_parquet(paths.features_latest, index=False)

    # small per-stock extras for the detail page (EMA lines need history)
    paths.write_state(last_bar=str(last.date()))
    out = rescore(cfg, paths, append_history=True)
    return out


def _has_future_stack() -> bool:
    import inspect
    return "future_stack" in inspect.signature(pd.DataFrame.stack).parameters


def load_recent_z(paths: Paths) -> Dict[str, pd.DataFrame]:
    zl = pd.read_parquet(paths.features_recent)
    zl["date"] = pd.to_datetime(zl["date"])
    zl = zl.set_index(["date", "ticker"])
    return {c: zl[c].unstack("ticker") for c in zl.columns}


def recent_scores(cfg: Dict, paths: Paths, weights: Optional[Dict] = None) -> Dict:
    weights = weights or load_weights()
    z = load_recent_z(paths)
    scale = float(cfg["score"].get("tanh_scale", 0.9))
    out = {}
    for h in cfg["horizons"]:
        s, parts = score_horizon(z, weights.get(h, {}), scale, return_parts=True)
        out[h] = {"score": s, "parts": parts}
    return out


def rescore(cfg: Optional[Dict] = None, paths: Optional[Paths] = None,
            weights: Optional[Dict] = None, append_history: bool = False) -> Dict:
    """Recompute today's ranking from stored z-scores (no downloads) and save it."""
    cfg = cfg or load_config()
    paths = paths or Paths(cfg)
    rs = recent_scores(cfg, paths, weights)
    raw = pd.read_parquet(paths.features_latest).set_index("ticker")
    uni = universe.load_universe(cfg, paths).set_index("ticker")
    info = fundamentals.load_info(paths)
    info = info.set_index("ticker") if not info.empty else pd.DataFrame()
    some = next(iter(rs.values()))["score"]
    last = some.index[-1]

    t = pd.DataFrame(index=raw.index)
    t["name"] = uni["name"].reindex(t.index)
    t["sector"] = uni["sector"].reindex(t.index)
    t["sub_industry"] = uni["sub_industry"].reindex(t.index)
    t["close"] = raw["close"]
    t["chg_1d"] = raw["chg_1d"]
    ey_t, ey_f = raw.get("_ey_trailing"), raw.get("_ey_forward")
    t["pe"] = np.where(ey_t > 0, 1 / ey_t, np.nan) if ey_t is not None else np.nan
    t["fwd_pe"] = np.where(ey_f > 0, 1 / ey_f, np.nan) if ey_f is not None else np.nan
    t["sector_pe"] = t.groupby("sector")["pe"].transform("median")
    t["pe_rel"] = t["pe"] / t["sector_pe"] - 1
    for h in cfg["horizons"]:
        t[f"score_{h}"] = rs[h]["score"].loc[last].reindex(t.index)
    t["news_sentiment"] = raw.get("news_sentiment")
    t["beta_mkt"] = raw.get("_beta_mkt")
    t["beta_rate"] = raw.get("_beta_rate")
    t["beta_infl"] = raw.get("_beta_infl")
    t["mom_3m"] = raw.get("mom_3m")
    t["rsi14"] = raw.get("rsi14")
    if not info.empty:
        t["market_cap"] = pd.to_numeric(info.get("marketCap"), errors="coerce").reindex(t.index)
    t["date"] = last
    t = t.reset_index().rename(columns={"index": "ticker"})
    t.to_parquet(paths.scores_latest, index=False)

    if append_history:
        hist_cols = ["date", "ticker"] + [f"score_{h}" for h in cfg["horizons"]] + ["close"]
        new = t[hist_cols].copy()
        new["recorded_at"] = datetime.now().isoformat(timespec="seconds")
        if paths.scores_history.exists():
            old = pd.read_parquet(paths.scores_history)
            new = pd.concat([old, new]).drop_duplicates(["date", "ticker"], keep="last")
        new.to_parquet(paths.scores_history, index=False)
    return {"date": str(pd.Timestamp(last).date()), "n": int(len(t))}


def load_ranking(paths: Paths) -> pd.DataFrame:
    if not paths.scores_latest.exists():
        return pd.DataFrame()
    return pd.read_parquet(paths.scores_latest)


# --------------------------------------------------------------------------- full refresh
def run_refresh(progress: Optional[Callable[[str, int, int], None]] = None,
                fundamentals_force: bool = False, skip_news: bool = False) -> Dict:
    cfg = load_config()
    paths = Paths(cfg)
    log = {}
    uni = universe.load_universe(cfg, paths)
    tickers = list(uni["ticker"]) + [cfg["universe"].get("benchmark", "SPY")]
    log["prices"] = prices.update_prices(tickers, cfg, paths, progress)
    if progress:
        progress("macro", 0, 1)
    log["macro"] = macro.update_macro(cfg, paths)
    log["fundamentals"] = fundamentals.update_fundamentals(uni["ticker"], cfg, paths, progress,
                                                           force=fundamentals_force)
    if progress:
        progress("scoring", 0, 1)
    compute_live(cfg, paths)
    if not skip_news:
        # NewsAPI budget goes to the names nearest the top and bottom of the medium-term list
        rk = load_ranking(paths).sort_values("score_medium", ascending=False)
        pri = list(rk["ticker"].head(60)) + list(rk["ticker"].tail(30))
        log["news"] = news.update_news(uni, cfg, paths, pri, progress)
        if progress:
            progress("scoring", 0, 1)
        compute_live(cfg, paths)
    paths.write_state(last_refresh=datetime.now().isoformat(timespec="seconds"), refresh_log=log)
    return log
