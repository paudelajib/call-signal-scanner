from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

def fit_weights_on_forward_returns(frames: Dict[str, pd.DataFrame], horizon: int = 20) -> Dict[str, float]:
    """
    Train simple logistic regression on (S1..S4) to predict forward +horizon return > 7%.
    Returns dict of weights for signals.
    """
    rows = []
    for t, df in frames.items():
        df = df.dropna(subset=['S1_BREAKOUT_VOL','S2_RSI_TREND','S3_MACD_MOM','S4_SQUEEZE_BO','Close']).copy()
        # HH/HL may be missing for early rows; fill with 0 then safe-drop
        if 'S5_HH_HL' in df.columns:
            df['S5_HH_HL'] = df['S5_HH_HL'].fillna(0)
        if df.empty:
            continue
        fwd = df['Close'].shift(-horizon) / df['Close'] - 1.0
        y = (fwd > 0.07).astype(int)  # 7% in ~1 month
        cols = ['S1_BREAKOUT_VOL','S2_RSI_TREND','S3_MACD_MOM','S4_SQUEEZE_BO']
        if 'S5_HH_HL' in df.columns:
            cols.append('S5_HH_HL')
        X = df[cols].values
        msk = (~y.isna()).values
        X, y = X[msk], y[msk]
        for i in range(len(y)):
            rows.append(np.concatenate([[y[i]], X[i]]))
    if not rows:
        return {}
    arr = np.array(rows)
    y = arr[:,0].astype(int)
    X = arr[:,1:]
    tscv = TimeSeriesSplit(n_splits=5)
    best_auc = -1.0
    best_coef = None
    # Since X has no datetime ordering across tickers, do a simple split
    # (for a quick calibration, not production-grade)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    coefs = clf.coef_.ravel()
    names = ['S1_BREAKOUT_VOL','S2_RSI_TREND','S3_MACD_MOM','S4_SQUEEZE_BO']
    if X.shape[1] == 5:
        names.append('S5_HH_HL')
    weights = {n: float(max(0.0, c)) for n, c in zip(names, clf.coef_.ravel())}
    s = sum(weights.values())
    if s > 0:
        for k in weights:
            weights[k] = weights[k] / s  # normalize to 1.0
    else:
        # fallback equal weights
        weights = {k: 0.25 for k in weights}
    return weights
