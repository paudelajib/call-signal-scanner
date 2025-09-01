from typing import Dict, Tuple
import numpy as np
import pandas as pd

# -----------------------------
# Core indicators / features
# -----------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, window: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    # Bandwidth relative to MA magnitude; protect against divide-by-zero
    bandwidth = (upper - lower) / (ma.replace(0, np.nan).abs())
    return ma, upper, lower, bandwidth

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given OHLCV dataframe (Date index; columns: Open, High, Low, Close, AdjClose, Volume),
    compute standard features used by our signals.
    """
    out = df.copy()
    # EMAs
    out['EMA20']  = _ema(out['Close'], 20)
    out['EMA50']  = _ema(out['Close'], 50)
    out['EMA200'] = _ema(out['Close'], 200)

    # RSI
    out['RSI14'] = rsi(out['Close'], 14)

    # MACD
    macd_line, signal_line, hist = macd(out['Close'], 12, 26, 9)
    out['MACD']        = macd_line
    out['MACD_SIGNAL'] = signal_line
    out['MACD_HIST']   = hist

    # Bollinger
    ma, ub, lb, bw = bollinger(out['Close'], 20, 2.0)
    out['BB_MA'] = ma
    out['BB_UP'] = ub
    out['BB_LO'] = lb
    out['BB_BW'] = bw

    # Volume / Highs
    out['VOL20']   = out['Volume'].rolling(20).mean()
    out['MAX_63']  = out['Close'].rolling(63).max()
    out['MAX_252'] = out['Close'].rolling(252).max()
    out['NEAR_52W'] = out['Close'] / out['MAX_252'] - 1.0  # 0 at 52w high, negative below

    return out

# -----------------------------
# HH/HL pivots
# -----------------------------

def _pivots(df: pd.DataFrame, k: int = 2) -> Tuple[pd.Series, pd.Series]:
    """
    Fractal pivots:
      - swing high at i if High[i] is the max in window [i-k, i+k]
      - swing low  at i if Low[i]  is the min in window [i-k, i+k]
    """
    H = df['High']
    L = df['Low']
    swing_high = (H == H.rolling(window=2 * k + 1, center=True).max())
    swing_low  = (L == L.rolling(window=2 * k + 1, center=True).min())
    # Remove edges (incomplete window)
    swing_high.iloc[:k] = False
    swing_high.iloc[-k:] = False
    swing_low.iloc[:k] = False
    swing_low.iloc[-k:] = False
    return swing_high.fillna(False), swing_low.fillna(False)

def _hh_hl_flag(df: pd.DataFrame, lookback: int = 90, k: int = 2) -> pd.Series:
    """
    Marks 1 when the *most recent* swing high is greater than the previous swing high
    AND the *most recent* swing low is greater than the previous swing low (within lookback bars).
    Otherwise 0.
    """
    sh, sl = _pivots(df, k=k)
    idx = df.index
    HHHL = pd.Series(0, index=idx, dtype=int)

    # Index position map for quick lookback slicing
    pos = {ts: i for i, ts in enumerate(idx)}
    high_pts = [ts for ts, v in sh.items() if v]
    low_pts  = [ts for ts, v in sl.items() if v]

    def last_two_in_window(points, i_now):
        i_lo = max(0, i_now - lookback)
        # keep points whose index position is within [i_lo, i_now]
        in_win = [p for p in points if i_lo <= pos[p] <= i_now]
        return in_win[-2:]  # last two

    for ts in idx:
        i = pos[ts]
        hs = last_two_in_window(high_pts, i)
        ls = last_two_in_window(low_pts, i)
        if len(hs) >= 2 and len(ls) >= 2:
            h_prev, h_last = hs[-2], hs[-1]
            l_prev, l_last = ls[-2], ls[-1]
            if df.at[h_last, 'High'] > df.at[h_prev, 'High'] and df.at[l_last, 'Low'] > df.at[l_prev, 'Low']:
                HHHL.at[ts] = 1
    return HHHL

# -----------------------------
# Signals & composite score
# -----------------------------

def compute_signals(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Build signal flags:
      S1: Breakout + Volume near 52w high
      S2: RSI trend filter
      S3: MACD momentum + recent positive flip
      S4: Volatility squeeze breakout
      S5: Higher Highs / Higher Lows (structure)
    And compute BuyScore (0..100).
    """
    f = df_feat

    # S1: Breakout + Volume (+ near 52w high)
    cond_breakout = (f['Close'] >= f['MAX_63']) & (f['Volume'] > 1.5 * f['VOL20'])
    near_52w_ok = (f['NEAR_52W'] > -0.02)  # within 2% of 52w high
    s1 = (cond_breakout & near_52w_ok).astype(int)

    # S2: RSI trend filter
    rsi_up = f['RSI14'] > 55
    rsi_rising = f['RSI14'] > f['RSI14'].rolling(3).mean()
    s2 = (rsi_up & rsi_rising).astype(int)

    # S3: MACD momentum (above signal + recent hist flip positive)
    macd_above = f['MACD'] > f['MACD_SIGNAL']
    hist_flip_pos = (f['MACD_HIST'] > 0) & (f['MACD_HIST'].shift(1) <= 0)
    recent_flip = hist_flip_pos.rolling(5).max().fillna(0) > 0
    s3 = (macd_above & recent_flip).astype(int)

    # S4: Volatility squeeze breakout (bandwidth 6m low + close above upper band)
    bw_6m_low = f['BB_BW'] <= f['BB_BW'].rolling(126).min()
    s4 = (bw_6m_low & (f['Close'] > f['BB_UP'])).astype(int)

    # S5: Higher Highs / Higher Lows (structure)
    s5 = _hh_hl_flag(f, lookback=90, k=2)

    out = f.copy()
    out['S1_BREAKOUT_VOL'] = s1
    out['S2_RSI_TREND']    = s2
    out['S3_MACD_MOM']     = s3
    out['S4_SQUEEZE_BO']   = s4
    out['S5_HH_HL']        = s5

    # Composite weights (tweakable; must sum to 1.0)
    weights = {
        'S1_BREAKOUT_VOL': 0.30,
        'S2_RSI_TREND':    0.18,
        'S3_MACD_MOM':     0.22,
        'S4_SQUEEZE_BO':   0.15,
        'S5_HH_HL':        0.15,
    }
    score = (
        weights['S1_BREAKOUT_VOL'] * out['S1_BREAKOUT_VOL'] +
        weights['S2_RSI_TREND']    * out['S2_RSI_TREND'] +
        weights['S3_MACD_MOM']     * out['S3_MACD_MOM'] +
        weights['S4_SQUEEZE_BO']   * out['S4_SQUEEZE_BO'] +
        weights['S5_HH_HL']        * out['S5_HH_HL']
    ) * 100.0
    out['BuyScore'] = score

    return out

# -----------------------------
# Row summarizer for table/CSV
# -----------------------------

def latest_row_summary(ticker: str, df_signals: pd.DataFrame) -> Dict[str, object]:
    last = df_signals.dropna().iloc[-1]
    return {
        "Ticker": ticker,
        "Date": df_signals.index[-1].date().isoformat(),
        "Close": float(last['Close']),
        "BuyScore": float(last['BuyScore']),
        "S1_BREAKOUT_VOL": int(last['S1_BREAKOUT_VOL']),
        "S2_RSI_TREND":    int(last['S2_RSI_TREND']),
        "S3_MACD_MOM":     int(last['S3_MACD_MOM']),
        "S4_SQUEEZE_BO":   int(last['S4_SQUEEZE_BO']),
        "S5_HH_HL":        int(last.get('S5_HH_HL', 0)),
        "NEAR_52W":        float(last['NEAR_52W']),
    }