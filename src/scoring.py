"""Turn features into three buy scores in [-1, +1] (1 week, 1 month, 6 months).

For each date:
  1. every feature is winsorised and z-scored across the S&P 500 (cross-sectional), so a +1
     means "one standard deviation better than the average stock today";
  2. composite = sum(w_f * z_f) / sum(|w_f|)   (missing feature -> 0 = neutral);
  3. composite is re-standardised across stocks and squashed: score = tanh(scale * z).

So a score is *relative*: +0.8 means "among the most attractive S&P 500 names for this
horizon by this model", 0 means average, -0.8 among the least attractive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from .config import PROJECT_ROOT

WEIGHTS_PATH = PROJECT_ROOT / "weights.yaml"
HEADER = """# Signed weights per horizon. Positive = higher value is bullish, negative = bearish.
# Each feature is z-scored across the S&P 500 on the same day before weighting, so weights are
# comparable. Only relative size matters (they are normalised by the sum of |w|).
# Edit here or in the web UI (Weights tab). The backtest's feature-IC table shows which
# features actually predicted returns on real history, per horizon.
"""


def load_weights(path: Path = WEIGHTS_PATH) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["weights"]


def save_weights(w: Dict[str, Dict[str, float]], path: Path = WEIGHTS_PATH) -> None:
    clean = {h: {k: round(float(v), 4) for k, v in fw.items() if abs(float(v)) > 1e-9}
             for h, fw in w.items()}
    path.write_text(HEADER + yaml.safe_dump({"weights": clean}, sort_keys=False), encoding="utf-8")


def xs_z(df: pd.DataFrame, winsor: float = 0.02) -> pd.DataFrame:
    """Row-wise (per date) winsorised z-score."""
    lo, hi = df.quantile(winsor, axis=1), df.quantile(1 - winsor, axis=1)
    x = df.clip(lo, hi, axis=0)
    sd = x.std(axis=1).replace(0, np.nan)
    return x.sub(x.mean(axis=1), axis=0).div(sd, axis=0)


def zscore_features(feats: Dict[str, pd.DataFrame], names: Iterable[str],
                    winsor: float, rows: Optional[pd.Index] = None) -> Dict[str, pd.DataFrame]:
    out = {}
    for n in names:
        if n not in feats:
            continue
        df = feats[n] if rows is None else feats[n].reindex(rows)
        out[n] = xs_z(df.astype(float), winsor)
    return out


def score_horizon(z: Dict[str, pd.DataFrame], weights: Dict[str, float], scale: float,
                  return_parts: bool = False):
    used = {k: float(w) for k, w in weights.items() if k in z and abs(float(w)) > 0}
    if not used:
        return (None, {}) if return_parts else None
    denom = sum(abs(w) for w in used.values())
    parts = {k: z[k].fillna(0.0) * (w / denom) for k, w in used.items()}
    comp = sum(parts.values())
    # a stock with no usable features at all on a date stays NaN
    any_data = sum(z[k].notna().astype(int) for k in used) > 0
    comp = comp.where(any_data)
    sd = comp.std(axis=1).replace(0, np.nan)
    cz = comp.sub(comp.mean(axis=1), axis=0).div(sd, axis=0)
    score = np.tanh(scale * cz)
    if return_parts:
        # contributions expressed on the final composite-z scale
        parts = {k: p.div(sd, axis=0) for k, p in parts.items()}
        return score, parts
    return score


def all_feature_names(weights: Dict[str, Dict[str, float]]) -> list:
    names = []
    for fw in weights.values():
        for k in fw:
            if k not in names:
                names.append(k)
    return names
