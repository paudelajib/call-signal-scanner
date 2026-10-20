"""Walk-forward backtest on real history.

Every `rebalance_every` trading days from `backtest.start`, score the S&P 500 using only data
available at that close (same code as the live app), then measure what actually happened:
entry at the next day's open, exit at the close `h` trading days later (h = 5, 21, 126).

Reported per horizon:
  * IC  - Spearman rank correlation between score and forward return, per date (the standard
          measure of a cross-sectional signal; ~0.03-0.05 is useful, >0.08 is strong)
  * t-stat of the mean IC on NON-overlapping dates only (overlapping windows inflate it)
  * quintile returns - average forward return of each score quintile minus the universe average
  * top-quintile hit rate and the cumulative excess return of the top quintile
  * per-feature IC, so weights can be tuned against evidence
  * out-of-sample check: weights learned from feature ICs on the training years vs your current
    weights, both evaluated on the held-out last `oos_years` years

Caveats (also shown in the UI): the universe is today's S&P 500 members (survivorship bias makes
everything look a bit better); P/E history only goes back as far as Yahoo provides; news
sentiment has no history until this app has been collecting it for a while.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from .config import Paths, load_config
from .engine import SCORED, build_inputs
from .features import compute_features
from .scoring import load_weights, score_horizon, xs_z


def rowwise_spearman(a: pd.DataFrame, b: pd.DataFrame, min_n: int = 50) -> pd.Series:
    a, b = a.align(b, join="inner")
    mask = a.notna() & b.notna()
    ra = a.where(mask).rank(axis=1)
    rb = b.where(mask).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    return ic[mask.sum(axis=1) >= min_n]


def _ic_stats(ic: pd.Series, step: int) -> Dict:
    ic = ic.dropna()
    if ic.empty:
        return {"mean": None, "std": None, "t": None, "n": 0, "pos_frac": None}
    non = ic.iloc[::max(1, step)]
    t = non.mean() / (non.std() / math.sqrt(len(non))) if len(non) > 2 and non.std() > 0 else None
    return {"mean": float(ic.mean()), "std": float(ic.std()), "t": None if t is None else float(t),
            "n": int(len(non)), "pos_frac": float((ic > 0).mean())}


def _quantile_returns(score: pd.DataFrame, fwd_ex: pd.DataFrame, q: int) -> pd.DataFrame:
    s, f = score.align(fwd_ex, join="inner")
    rk = s.rank(axis=1, pct=True)
    bucket = np.ceil(rk * q).clip(1, q)
    res = {}
    for k in range(1, q + 1):
        res[k] = f.where(bucket == k).mean(axis=1)
    return pd.DataFrame(res)


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def learned_weights(feat_ic: Dict[str, pd.Series], shrink: float = 0.5) -> Dict[str, float]:
    """Weights proportional to each feature's information ratio (mean IC / std IC) on the
    training window, shrunk toward zero and normalised to sum(|w|) = 1."""
    w = {}
    for k, ic in feat_ic.items():
        ic = ic.dropna()
        if len(ic) < 20 or ic.std() == 0:
            continue
        ir = ic.mean() / ic.std()
        if abs(ir) < 0.05:  # ignore features with essentially no signal
            continue
        w[k] = ir * shrink
    s = sum(abs(v) for v in w.values())
    return {k: round(v / s, 4) for k, v in w.items()} if s else {}


def run_backtest(progress: Optional[Callable[[str, int, int], None]] = None,
                 cfg: Optional[Dict] = None, weights: Optional[Dict] = None) -> Dict:
    cfg = cfg or load_config()
    paths = Paths(cfg)
    bt = cfg["backtest"]
    weights = weights or load_weights()
    if progress:
        progress("backtest: loading data", 0, 1)
    inp = build_inputs(cfg, paths, with_news=True)
    if progress:
        progress("backtest: computing features", 0, 1)
    feats = compute_features(inp["panel"], inp["bench"], inp["macro"], inp["sectors"],
                             inp["ey"], inp["news"], cfg)
    close, open_ = inp["panel"]["close"], inp["panel"]["open"]
    idx = close.index
    start = pd.Timestamp(bt.get("start", "2016-01-01"))
    step = int(bt.get("rebalance_every", 5))
    maxh = max(int(h["days"]) for h in cfg["horizons"].values())
    pos = np.arange(idx.searchsorted(start), len(idx) - 1, step)
    rdates = idx[pos]
    winsor = float(cfg["score"].get("winsor_pct", 0.02))
    scale = float(cfg["score"].get("tanh_scale", 0.9))
    q = int(bt.get("quantiles", 5))

    # Point-in-time membership: a stock only enters the backtest from the date it joined the
    # S&P 500. Without this, stocks that were added *because* they rallied are scored years
    # before anyone knew they would be index members (a strong look-ahead bias).
    member = pd.DataFrame(True, index=rdates, columns=close.columns)
    if bt.get("point_in_time_membership", True) and "date_added" in inp["universe"]:
        added = pd.to_datetime(inp["universe"].set_index("ticker")["date_added"], errors="coerce")
        added = added.reindex(close.columns)
        for t, d in added.items():
            if pd.notna(d):
                member.loc[member.index < d, t] = False
    results_member = float(member.values.mean())

    z = {}
    for i, n in enumerate(SCORED):
        if n in feats:
            z[n] = xs_z(feats[n].reindex(rdates).astype(float).where(member), winsor)
        if progress:
            progress("backtest: z-scoring", i + 1, len(SCORED))

    coverage = {n: float(z[n].notna().mean().mean()) for n in z}
    oos_start = idx[-1] - pd.DateOffset(years=int(bt.get("oos_years", 2)))
    results = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "first_date": str(rdates[0].date()), "last_date": str(rdates[-1].date()),
               "n_dates": int(len(rdates)), "n_tickers": int(close.shape[1]),
               "rebalance_every": step, "oos_start": str(oos_start.date()),
               "coverage": coverage, "horizons": {},
               "point_in_time_membership": bool(bt.get("point_in_time_membership", True)),
               "avg_members_frac": results_member,
               "caveats": [
                   "Stocks enter the test only from the date they joined the S&P 500, but companies "
                   "that were later removed are still missing (some survivorship bias remains).",
                   "P/E history only reaches back as far as Yahoo's quarterly valuation data; "
                   "earlier dates treat valuation as neutral.",
                   "News sentiment has history only from the day this app started collecting it.",
                   "Returns ignore trading costs; entry is next day's open, exit the close h days later.",
               ]}

    for hi, (h, spec) in enumerate(cfg["horizons"].items()):
        H = int(spec["days"])
        fwd_all = close.shift(-H) / open_.shift(-1) - 1
        fwd = fwd_all.reindex(rdates).where(member)
        valid_until = idx[max(0, len(idx) - 1 - H)]
        fwd = fwd[fwd.index <= valid_until]
        fwd_ex = fwd.sub(fwd.mean(axis=1), axis=0)
        nonover = max(1, math.ceil(H / step))

        score = score_horizon(z, weights.get(h, {}), scale)
        score = score.reindex(fwd.index)
        ic = rowwise_spearman(score, fwd)
        qr = _quantile_returns(score, fwd_ex, q)

        # top quintile cumulative excess return on non-overlapping periods
        top = qr[q].iloc[::nonover].dropna()
        curve = (1 + top).cumprod()
        roll = ic.rolling(52, min_periods=20).mean()  # ~1-year rolling mean IC

        feat_ic = {}
        for n in z:
            fi = rowwise_spearman(z[n].reindex(fwd.index), fwd)
            if len(fi.dropna()):
                feat_ic[n] = fi

        # out-of-sample: learn on train (with an embargo of H days), test on the hold-out
        embargo_end = oos_start - pd.tseries.offsets.BDay(H)
        train_ic = {n: s[s.index < embargo_end] for n, s in feat_ic.items()}
        lw = learned_weights(train_ic)
        test_rows = fwd.index[fwd.index >= oos_start]
        oos = {}
        for label, w in [("current", weights.get(h, {})), ("learned", lw)]:
            if not w:
                continue
            sc = score_horizon({k: v.reindex(test_rows) for k, v in z.items()}, w, scale)
            if sc is None:
                continue
            oic = rowwise_spearman(sc, fwd.reindex(test_rows))
            oqr = _quantile_returns(sc, fwd_ex.reindex(test_rows), q)
            oos[label] = {"ic": _ic_stats(oic, nonover),
                          "top_minus_bottom": float((oqr[q] - oqr[1]).mean()) if len(oqr) else None}

        results["horizons"][h] = {
            "label": spec["label"], "days": H,
            "ic": _ic_stats(ic, nonover),
            "quantile_excess": [float(qr[k].mean()) for k in range(1, q + 1)],
            "top_minus_bottom": float((qr[q] - qr[1]).mean()),
            "top_minus_bottom_annualised": float((qr[q] - qr[1]).mean() * 252 / H),
            "top_hit_rate": float((qr[q].dropna() > 0).mean()),
            "ic_series": {"dates": [str(d.date()) for d in roll.dropna().index],
                          "values": [float(v) for v in roll.dropna().values]},
            "top_curve": {"dates": [str(d.date()) for d in curve.index],
                          "values": [float(v) for v in curve.values]},
            "feature_ic": {n: _ic_stats(s, nonover) for n, s in feat_ic.items()},
            "oos": oos, "learned_weights": lw,
        }
        if progress:
            progress("backtest: evaluating horizons", hi + 1, len(cfg["horizons"]))

    results = _clean(results)
    paths.backtest.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


def load_backtest(paths: Paths) -> Optional[Dict]:
    if paths.backtest.exists():
        return json.loads(paths.backtest.read_text(encoding="utf-8"))
    return None


if __name__ == "__main__":
    def _p(stage, i, n):
        print(f"\r{stage}: {i}/{n}", end="", flush=True)
    r = run_backtest(_p)
    print()
    for h, v in r["horizons"].items():
        print(h, "IC", v["ic"], "Q5-Q1", v["top_minus_bottom"])
