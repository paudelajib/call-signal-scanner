# src/viz_interactive.py
from __future__ import annotations
import os
from typing import Dict, List
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def _range_buttons():
    return [
        dict(count=3, step="month", stepmode="backward", label="3M"),
        dict(count=6, step="month", stepmode="backward", label="6M"),
        dict(count=1, step="year",  stepmode="backward", label="1Y"),
        dict(count=2, step="year",  stepmode="backward", label="2Y"),
        dict(step="all", label="All")
    ]

def _plotly_chart_for_ticker(ticker: str, df: pd.DataFrame) -> go.Figure:
    """Interactive price chart with S1..S5 overlays."""
    f = df.copy()
    fig = go.Figure()

    # Price
    fig.add_trace(go.Scatter(x=f.index, y=f["Close"], name="Close", mode="lines"))

    # EMAs if present
    for col, nm in [("EMA20","EMA20"),("EMA50","EMA50"),("EMA200","EMA200")]:
        if col in f.columns:
            fig.add_trace(go.Scatter(x=f.index, y=f[col], name=nm, mode="lines", opacity=0.6))

    # Bollinger if present
    if {"BB_UP","BB_MA","BB_LO"}.issubset(f.columns):
        fig.add_trace(go.Scatter(x=f.index, y=f["BB_UP"], name="BB_UP", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=f.index, y=f["BB_MA"], name="BB_MA", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=f.index, y=f["BB_LO"], name="BB_LO", line=dict(dash="dot")))

    # Signals S1..S5: mark last 120 bars where flag == 1
    last_n = 250
    xx = f.index[-last_n:] if len(f) > last_n else f.index
    for col, nm, sym in [
        ("S1_BREAKOUT_VOL","S1", "triangle-up"),
        ("S2_RSI_TREND",   "S2", "circle"),
        ("S3_MACD_MOM",    "S3", "diamond"),
        ("S4_SQUEEZE_BO",  "S4", "x"),
        ("S5_HH_HL",       "S5", "star")
    ]:
        if col in f.columns:
            mask = (f[col] == 1) & f.index.isin(xx)
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=f.index[mask], y=f.loc[mask, "Close"],
                    mode="markers", name=nm,
                    marker_symbol=sym, marker_size=9
                ))

    fig.update_layout(
        title=f"{ticker} — Price & Signals",
        xaxis=dict(rangeslider=dict(visible=True),
                   rangeselector=dict(buttons=_range_buttons())),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def make_top10_dashboard(frames: Dict[str, pd.DataFrame], top_tickers: List[str], outdir: str):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    html_path = outdir / "top10_dashboard.html"

    figs = []
    for t in top_tickers:
        df = frames.get(t)
        if df is None or df.empty:
            continue
        figs.append(_plotly_chart_for_ticker(t, df))

    # save a simple multi-figure HTML (stacked vertically)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<meta charset='utf-8'><h2>Top 10 — Interactive</h2>\n")
        for i, fig in enumerate(figs, 1):
            f.write(f"<h3>{top_tickers[i-1]}</h3>")
            f.write(pio.to_html(fig, include_plotlyjs='cdn', full_html=False))
            f.write("<hr/>")
    print(f"Saved interactive dashboard → {html_path}")

def export_top10_pdf(frames: Dict[str, pd.DataFrame], top_tickers: List[str], out_pdf: str):
    """Static PDF with one page per ticker (matplotlib)."""
    with PdfPages(out_pdf) as pdf:
        for t in top_tickers:
            df = frames.get(t)
            if df is None or df.empty:
                continue
            f = df.copy()
            plt.figure()
            plt.plot(f.index, f["Close"], label="Close")
            for col, nm, ls in [("EMA20","EMA20","-."),("EMA50","EMA50","--"),("EMA200","EMA200",":")]:
                if col in f.columns:
                    plt.plot(f.index, f[col], ls, label=nm)
            if {"BB_UP","BB_MA","BB_LO"}.issubset(f.columns):
                plt.plot(f.index, f["BB_UP"], label="BB_UP")
                plt.plot(f.index, f["BB_MA"], label="BB_MA")
                plt.plot(f.index, f["BB_LO"], label="BB_LO")
            # mark signals on last 250 bars
            last_n = 250
            xx = f.index[-last_n:] if len(f) > last_n else f.index
            for col, nm in [("S1_BREAKOUT_VOL","S1"),("S2_RSI_TREND","S2"),
                            ("S3_MACD_MOM","S3"),("S4_SQUEEZE_BO","S4"),("S5_HH_HL","S5")]:
                if col in f.columns:
                    mask = (f[col] == 1) & f.index.isin(xx)
                    if mask.any():
                        plt.scatter(f.index[mask], f.loc[mask, "Close"], label=nm, s=12)
            plt.title(f"{t} — Price & Signals")
            plt.xlabel("Date"); plt.ylabel("Price")
            plt.legend(loc="upper left", fontsize=8)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"Saved PDF → {out_pdf}")
