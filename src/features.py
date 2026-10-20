"""Feature engine on wide panels (index = trading date, columns = ticker).

Every feature on date d uses only information available at the close of d, so the same panels
drive live scoring and the walk-forward backtest.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# Human-readable descriptions (shown in the UI next to each contribution)
FEATURE_INFO: Dict[str, str] = {
    "ret_1w": "Return over the last 5 trading days",
    "mom_1m": "Return over the last month (21d)",
    "mom_3m": "Return over the last 3 months (63d)",
    "mom_6m": "Return over the last 6 months (126d)",
    "mom_12_1": "12-month return skipping the most recent month",
    "trend_50": "Price vs its 50-day EMA",
    "trend_50_200": "50-day EMA vs 200-day EMA (golden/death cross distance)",
    "rsi14": "14-day RSI",
    "macd_hist": "MACD histogram relative to price",
    "near_52w_high": "Distance from the 52-week high",
    "vol_surge": "5-day volume vs 3-month average volume (log)",
    "volatility_3m": "Annualised volatility over 3 months",
    "rel_sector_3m": "3-month return vs its sector's median",
    "pe_vs_sector": "Earnings yield vs sector (trailing + forward; high = cheap vs peers)",
    "rate_tilt": "Rate sensitivity x current 10Y-yield trend",
    "infl_tilt": "Inflation sensitivity x current breakeven-inflation trend",
    "news_sentiment": "Recency-weighted headline sentiment",
}


def _ema(df: pd.DataFrame, span: int) -> pd.DataFrame:
    return df.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def sector_z(df: pd.DataFrame, sectors: pd.Series, winsor: float = 0.02) -> pd.DataFrame:
    """Z-score each row within each sector (min 5 names per sector)."""
    out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for sec, cols in sectors.groupby(sectors).groups.items():
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 5:
            continue
        sub = df[cols]
        lo, hi = sub.quantile(winsor, axis=1), sub.quantile(1 - winsor, axis=1)
        sub = sub.clip(lo, hi, axis=0)
        mu, sd = sub.mean(axis=1), sub.std(axis=1)
        out[cols] = sub.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return out


def macro_betas(close: pd.DataFrame, bench: pd.Series, macro: pd.DataFrame,
                window_weeks: int = 104) -> Dict[str, pd.DataFrame]:
    """Rolling weekly sensitivities of each stock's market-adjusted return to
    changes in the 10Y yield (beta_rate, % return per +1pp) and in 10Y breakeven inflation
    (beta_infl). Two-step: remove market beta, then jointly regress on both macro changes.
    Returned on the daily index (each week's estimate applies from that Friday's close)."""
    wk = close.resample("W-FRI").last()
    r = wk.pct_change(fill_method=None) * 100
    m = bench.resample("W-FRI").last().pct_change(fill_method=None) * 100
    mac = macro.resample("W-FRI").last()
    dy = mac["DGS10"].diff() if "DGS10" in mac else pd.Series(np.nan, index=wk.index)
    di = mac["T10YIE"].diff() if "T10YIE" in mac else pd.Series(np.nan, index=wk.index)
    dy, di, m = dy.reindex(wk.index), di.reindex(wk.index), m.reindex(wk.index)
    W, mp = window_weeks, window_weeks // 2

    def rmean(x):
        return x.rolling(W, min_periods=mp).mean()

    # market beta
    cov_rm = rmean(r.mul(m, axis=0)) - rmean(r).mul(rmean(m), axis=0)
    var_m = rmean(m * m) - rmean(m) ** 2
    bmk = cov_rm.div(var_m, axis=0)
    e = r - bmk.mul(m, axis=0)

    # joint regression of e on (dy, di): solve 2x2 normal equations per date
    vy = rmean(dy * dy) - rmean(dy) ** 2
    vi = rmean(di * di) - rmean(di) ** 2
    cyi = rmean(dy * di) - rmean(dy) * rmean(di)
    cey = rmean(e.mul(dy, axis=0)) - rmean(e).mul(rmean(dy), axis=0)
    cei = rmean(e.mul(di, axis=0)) - rmean(e).mul(rmean(di), axis=0)
    det = (vy * vi - cyi ** 2).replace(0, np.nan)
    have_infl = di.notna().rolling(W, min_periods=mp).sum() >= mp
    b_rate = cey.mul(vi, axis=0).sub(cei.mul(cyi, axis=0)).div(det, axis=0)
    b_infl = cei.mul(vy, axis=0).sub(cey.mul(cyi, axis=0)).div(det, axis=0)
    # if no inflation data, fall back to single-regressor rate beta
    b_rate_1 = cey.div(vy.replace(0, np.nan), axis=0)
    b_rate = b_rate.where(have_infl, b_rate_1, axis=0)
    corr_rate = cey.div(np.sqrt((rmean(e * e) - rmean(e) ** 2)).mul(np.sqrt(vy), axis=0))

    def daily(x):
        return x.reindex(x.index.union(close.index)).sort_index().ffill(limit=7).reindex(close.index)

    return {"beta_mkt": daily(bmk), "beta_rate": daily(b_rate), "beta_infl": daily(b_infl),
            "corr_rate": daily(corr_rate)}


def compute_features(panel: Dict[str, pd.DataFrame], bench: pd.Series, macro: pd.DataFrame,
                     sectors: pd.Series, ey: Dict[str, pd.DataFrame],
                     news: Optional[pd.DataFrame], cfg: Dict) -> Dict[str, pd.DataFrame]:
    c, v = panel["close"], panel["volume"]
    f: Dict[str, pd.DataFrame] = {}
    f["ret_1w"] = c / c.shift(5) - 1
    f["mom_1m"] = c / c.shift(21) - 1
    f["mom_3m"] = c / c.shift(63) - 1
    f["mom_6m"] = c / c.shift(126) - 1
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    e20, e50, e200 = _ema(c, 20), _ema(c, 50), _ema(c, 200)
    f["trend_50"] = c / e50 - 1
    f["trend_50_200"] = e50 / e200 - 1
    f["rsi14"] = _rsi(c, 14)
    macd = _ema(c, 12) - _ema(c, 26)
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    f["near_52w_high"] = c / c.rolling(252, min_periods=126).max() - 1
    f["vol_surge"] = np.log(v.rolling(5).mean() / v.rolling(63, min_periods=40).mean())
    daily_ret = c.pct_change(fill_method=None)
    f["volatility_3m"] = daily_ret.rolling(63, min_periods=40).std() * np.sqrt(252)
    sec = sectors.reindex(c.columns)
    med = f["mom_3m"].T.groupby(sec).transform("median").T
    f["rel_sector_3m"] = f["mom_3m"] - med

    winsor = float(cfg["score"].get("winsor_pct", 0.02))
    zt = sector_z(ey["ey_trailing"], sec, winsor)
    zf = sector_z(ey["ey_forward"], sec, winsor)
    # average of trailing and forward sector-z; if only one exists, use it
    f["pe_vs_sector"] = (zt.fillna(zf) + zf.fillna(zt)) / 2

    betas = macro_betas(c, bench, macro, int(cfg["macro"].get("sensitivity_window_weeks", 104)))
    td = int(cfg["macro"].get("trend_window_days", 63))
    mac = macro.reindex(macro.index.union(c.index)).sort_index().ffill().reindex(c.index)
    if "DGS10" in mac:
        rate_trend = mac["DGS10"] - mac["DGS10"].shift(td)
        f["rate_tilt"] = betas["beta_rate"].mul(rate_trend, axis=0)
    if "T10YIE" in mac:
        infl_trend = mac["T10YIE"] - mac["T10YIE"].shift(td)
        f["infl_tilt"] = betas["beta_infl"].mul(infl_trend, axis=0)

    if news is not None:
        f["news_sentiment"] = news.reindex(index=c.index, columns=c.columns)

    # display-only extras (not scored)
    f["_beta_mkt"] = betas["beta_mkt"]
    f["_beta_rate"] = betas["beta_rate"]
    f["_beta_infl"] = betas["beta_infl"]
    f["_corr_rate"] = betas["corr_rate"]
    f["_ey_trailing"] = ey["ey_trailing"]
    f["_ey_forward"] = ey["ey_forward"]
    f["_ema50"], f["_ema200"], f["_ema20"] = e50, e200, e20
    return f
