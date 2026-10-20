# src/viz.py
import os
import pandas as pd
import matplotlib.pyplot as plt

def styled_table_html(df: pd.DataFrame, out_html: str, top_n: int = 100):
    """
    Save a pretty, ranked HTML table with color bars and emoji flags for S1..S5.
    """
    table = df.copy().reset_index(drop=True)
    # Limit columns and order
    cols = ["Ticker","Date","Close","BuyScore",
            "S1_BREAKOUT_VOL","S2_RSI_TREND","S3_MACD_MOM","S4_SQUEEZE_BO","S5_HH_HL",
            "NEAR_52W","Sentiment"]
    cols = [c for c in cols if c in table.columns]
    table = table[cols].head(top_n)

    # emoji for signals
    def flag(x):
        return "✅" if x == 1 else "—"
    for c in ["S1_BREAKOUT_VOL","S2_RSI_TREND","S3_MACD_MOM","S4_SQUEEZE_BO","S5_HH_HL"]:
        if c in table.columns:
            table[c] = table[c].map(flag)

    # format some numbers
    if "Close" in table.columns:
        table["Close"] = table["Close"].map(lambda v: f"${v:,.2f}")
    if "BuyScore" in table.columns:
        table["BuyScore"] = table["BuyScore"].map(lambda v: f"{v:.0f}")
    if "NEAR_52W" in table.columns:
        table["NEAR_52W"] = table["NEAR_52W"].map(lambda v: f"{100*v:.1f}%")
    if "Sentiment" in table.columns:
        table["Sentiment"] = table["Sentiment"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")

    # style (BuyScore gradient; neutral header)
    sty = (table.style
        .hide(axis="index")
        .set_table_styles([
            {"selector":"th","props":[("text-align","center"),("font-weight","600"),("padding","8px")]},
            {"selector":"td","props":[("padding","6px 10px")]}
        ])
    )

    if "BuyScore" in table.columns:
        # color bar in BuyScore column
        sty = sty.bar(color=None, subset=["BuyScore"], vmin=0, vmax=100)  # default colormap

    html = sty.to_html()
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("<meta charset='utf-8'>\n")
        f.write("<h2>Top Buys (ranked by BuyScore)</h2>\n")
        f.write(html)
    print(f"Saved styled table → {out_html}")

def plot_top_buys(df: pd.DataFrame, out_png: str, top_n: int = 25):
    top = df.sort_values("BuyScore", ascending=False).head(top_n)
    plt.figure()
    plt.barh(top["Ticker"], top["BuyScore"])
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} BuyScores")
    plt.xlabel("BuyScore")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved → {out_png}")

def plot_signal_counts(df: pd.DataFrame, out_png: str):
    cols = [c for c in ["S1_BREAKOUT_VOL","S2_RSI_TREND","S3_MACD_MOM","S4_SQUEEZE_BO","S5_HH_HL"] if c in df.columns]
    counts = {c: int(df[c].sum()) for c in cols}
    if not counts:
        return
    plt.figure()
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.title("Signal Counts (across universe)")
    plt.ylabel("Tickers with signal = 1")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved → {out_png}")

def plot_score_vs_sentiment(df: pd.DataFrame, out_png: str):
    if "Sentiment" not in df.columns or df["Sentiment"].dropna().empty:
        return
    x = df["BuyScore"]
    y = df["Sentiment"]
    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    plt.title("BuyScore vs News Sentiment")
    plt.xlabel("BuyScore")
    plt.ylabel("Sentiment (VADER)")
    plt.axhline(0.2)       # loose "positive" bar
    plt.axvline(60)        # example BuyScore cut
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved → {out_png}")

def plot_score_vs_near52w(df: pd.DataFrame, out_png: str):
    if "NEAR_52W" not in df.columns:
        return
    x = df["BuyScore"]
    y = 100*df["NEAR_52W"]  # percent
    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    plt.title("BuyScore vs Proximity to 52W High")
    plt.xlabel("BuyScore")
    plt.ylabel("NEAR_52W (%)")
    plt.axhline(0)         # 52w high line
    plt.axvline(60)        # sample BuyScore cut
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved → {out_png}")

def make_all_viz(df_csv: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(df_csv)
    # styled table
    styled_table_html(df, os.path.join(outdir, "top_buys.html"), top_n=100)
    # charts
    plot_top_buys(df, os.path.join(outdir, "top_buys_bar.png"), top_n=25)
    plot_signal_counts(df, os.path.join(outdir, "signal_counts.png"))
    plot_score_vs_sentiment(df, os.path.join(outdir, "score_vs_sentiment.png"))
    plot_score_vs_near52w(df, os.path.join(outdir, "score_vs_near52w.png"))