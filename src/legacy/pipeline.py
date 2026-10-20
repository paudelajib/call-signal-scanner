# src/pipeline.py
from __future__ import annotations
import os, argparse, json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv

# ---- Project modules ----
from .data import fetch_prices_yf
from .signals import compute_features, compute_signals, latest_row_summary
from .news import NewsSentiment
from .learn import fit_weights_on_forward_returns

# Optional pretty outputs (HTML table + extra charts)
try:
    from .viz import make_all_viz  # produces top_buys.html + extra static charts
except Exception:
    make_all_viz = None

# Interactive dashboard + PDF for top-N
try:
    from .viz_interactive import make_top10_dashboard, export_top10_pdf
except Exception:
    make_top10_dashboard = None
    export_top10_pdf = None

# ---- .env ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def load_tickers(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def apply_weights(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Recompute BuyScore with custom weights (e.g., learned). Missing signals default to 0.
    """
    s = (
        weights.get("S1_BREAKOUT_VOL", 0) * df.get("S1_BREAKOUT_VOL", 0) +
        weights.get("S2_RSI_TREND",   0) * df.get("S2_RSI_TREND", 0) +
        weights.get("S3_MACD_MOM",    0) * df.get("S3_MACD_MOM", 0) +
        weights.get("S4_SQUEEZE_BO",  0) * df.get("S4_SQUEEZE_BO", 0) +
        weights.get("S5_HH_HL",       0) * df.get("S5_HH_HL", 0)
    ) * 100.0
    out = df.copy()
    out["BuyScore"] = s
    return out


def main():
    ap = argparse.ArgumentParser(description="Call Signal Scanner (daily-cached, robust, with viz)")
    ap.add_argument("--tickers", required=True, help="Path to file with tickers (one per line)")
    ap.add_argument("--outdir", default="results", help="Output directory")
    ap.add_argument("--learn", action="store_true", help="Fit weights via logistic regression (slower)")

    # Daily snapshot controls (in data.py this writes cache/ohlcv_daily/YYYY-MM-DD/<T>.parquet)
    ap.add_argument("--force_new_snapshot", action="store_true",
                    help="Ignore today's cached prices and pull fresh from Yahoo once")
    # Top-N interactive/PDF viz
    ap.add_argument("--top_n", type=int, default=10,
                    help="How many top tickers to visualize in interactive dashboard & PDF (default 10)")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- Load tickers ----------
    tickers = load_tickers(args.tickers)
    if not tickers:
        print("No tickers found in file.")
        return

    # ---------- Prices (daily snapshot cache) ----------
    prices = fetch_prices_yf(
        tickers,
        period="2y",
        interval="1d",
        snapshot_dir="cache/ohlcv_daily",
        force_new_snapshot=args.force_new_snapshot,
    )

    # ---------- Compute features/signals per ticker ----------
    frames: Dict[str, pd.DataFrame] = {}
    rows = []
    for t in tqdm(tickers, desc="Processing"):
        df = prices.get(t)
        if df is None or df.empty:
            continue
        feat = compute_features(df)          # builds indicators (EMA/RSI/MACD/BB/52w etc.)
        sigs = compute_signals(feat)         # builds S1..S5 & BuyScore
        frames[t] = sigs
        rows.append(latest_row_summary(t, sigs))

    table = pd.DataFrame(rows)
    if table.empty:
        print("No usable data for any tickers. Exiting.")
        return
    table = table.sort_values(["BuyScore", "Close"], ascending=[False, False]).reset_index(drop=True)

    # ---------- Optional: learn weights across the 2y window ----------
    if args.learn and frames:
        learned = fit_weights_on_forward_returns(frames, horizon=20)
        if learned:
            new_rows = []
            for t, df in frames.items():
                df2 = apply_weights(df, learned)
                new_rows.append(latest_row_summary(t, df2))
            table = pd.DataFrame(new_rows).sort_values(
                ["BuyScore", "Close"], ascending=[False, False]
            ).reset_index(drop=True)
            with open(os.path.join(args.outdir, "learned_weights.json"), "w", encoding="utf-8") as f:
                json.dump(learned, f, indent=2)

    # ---------- News sentiment (daily-cached inside news.py) ----------
    ns = NewsSentiment()  # provider/keys via .env; daily cache in cache/news_daily/YYYY-MM-DD
    sentiments = []
    for t in tqdm(table["Ticker"], desc="News"):
        s = ns.sentiment_score(t)  # returns None if < min articles or provider empty
        sentiments.append(s)
    table["Sentiment"] = sentiments

    # Relative (cross-sectional) sentiment for fair comparison across 500 names
    if table["Sentiment"].notna().any():
        mu = table["Sentiment"].mean()
        sigma = table["Sentiment"].std(ddof=0)
        table["SentimentZ"] = (table["Sentiment"] - mu) / (sigma if sigma else 1.0)

    # ---------- Save CSVs ----------
    out_csv = os.path.join(args.outdir, "buy_signals.csv")
    table.to_csv(out_csv, index=False)

    topN = table.head(max(1, args.top_n))
    topN_csv = os.path.join(args.outdir, f"top_{len(topN)}.csv")
    topN.to_csv(topN_csv, index=False)

    # ---------- Histograms (robust to empty/None) ----------
    buy = pd.to_numeric(table["BuyScore"], errors="coerce")
    if buy.notna().any():
        plt.figure()
        buy.dropna().plot(kind="hist", bins=20)
        plt.title("BuyScore Distribution")
        plt.xlabel("BuyScore")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "buyscore_hist.png"))
        plt.close()
    else:
        print("No BuyScore data; skipping buyscore_hist.png")

    sent = pd.to_numeric(table["Sentiment"], errors="coerce")
    if sent.notna().any():
        plt.figure()
        sent.dropna().plot(kind="hist", bins=20)
        plt.title("News Sentiment Distribution")
        plt.xlabel("VADER Compound")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "sentiment_hist.png"))
        plt.close()
    else:
        print("No sentiment data (provider returned none or <min headlines); skipping sentiment_hist.png")

    # ---------- Pretty HTML + extra static charts (if viz.py is present) ----------
    if make_all_viz is not None:
        try:
            make_all_viz(out_csv, args.outdir)
        except Exception as e:
            print(f"[viz] Skipped viz due to: {e}")

    # ---------- Top-N interactive dashboard + PDF (if viz_interactive.py is present) ----------
    top_tickers = topN["Ticker"].tolist()
    if make_top10_dashboard is not None and export_top10_pdf is not None:
        try:
            make_top10_dashboard(frames, top_tickers, args.outdir)
            export_top10_pdf(frames, top_tickers, os.path.join(args.outdir, "top10.pdf"))
        except Exception as e:
            print(f"[viz10] Skipped top-N interactive/PDF due to: {e}")

    # ---------- Markdown report ----------
    md = ["# Call Signal Scanner Report\n"]
    md.append(f"Tickers scanned: {len(tickers)}\n")
    md.append(f"Top list size: {len(topN)}\n")
    if "SentimentZ" in table.columns:
        md.append("SentimentZ included (cross-sectional z-score).\n")
    md.append("\n## Top by BuyScore\n")
    try:
        md.append(topN.to_markdown(index=False))
    except Exception:
        md.append(topN.to_string(index=False))
    md.append("\n\n---\nGenerated by Call Signal Scanner.\n")
    with open(os.path.join(args.outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # Also print to console
    try:
        print(topN.to_string(index=False))
    except Exception:
        print(topN)


if __name__ == "__main__":
    main()