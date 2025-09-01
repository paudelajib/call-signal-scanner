# 📈 Call Signal Scanner

A repeatable, end-to-end scanner for **S&P 500 tickers** using rule-based technical signals, optional news sentiment, and interactive visualization.  

The system is designed for **daily use**:  
- Pulls OHLCV data (cached locally so you don’t burn API calls)  
- Computes signals and a **BuyScore (0–100)**  
- Optionally integrates **news sentiment** (cached daily as well)  
- Produces **CSVs + charts + an interactive Top-10 dashboard + a PDF report**  
- Includes a **backtest** module to check “what if we had traded this a month ago?”  

---

## 🚀 Features

- **Signals (S1–S5)**
  - **S1**: Breakout on volume  
  - **S2**: RSI trend confirmation  
  - **S3**: MACD momentum  
  - **S4**: Squeeze breakout (Bollinger/Keltner)  
  - **S5**: Higher-High / Higher-Low structure  
- **BuyScore**: weighted sum (0–100)  
- **Sentiment (optional)**:
  - Raw VADER compound score (`Sentiment`)  
  - Cross-sectional z-score (`SentimentZ`) for ranking tickers  
- **Caching**:
  - Prices → `cache/ohlcv_daily/YYYY-MM-DD/<TICKER>.parquet`  
  - News → `cache/news_daily/YYYY-MM-DD/<TICKER>.json`  
- **Outputs**:
  - `buy_signals.csv` and `top_N.csv`  
  - Histograms (BuyScore, sentiment)  
  - Interactive **Top-10 dashboard** (`top10_dashboard.html`)  
  - PDF (`top10.pdf`)  
- **Backtest (new)**:  
  - Simulate “as of 30 days ago” signals vs actual returns today  
  - Produce scatterplots + grouped histograms

---

## ⚙️ Setup

### 1. Clone and install dependencies
```bash
git clone https://github.com/yourusername/call-signal-scanner.git
cd call-signal-scanner
pip install -r requirements.txt
```

### 2. (Optional) Configure News API
Create a `.env` file in the project root:
```
NEWS_PROVIDER=newsapi
NEWSAPI_KEY=YOUR_NEWSAPI_KEY
```
If you skip this, the scanner will run with **empty sentiment columns**.

### 3. Provide tickers
The scanner expects a text file with one ticker per line:
```
data/tickers_sp500.txt
```
Example:
```
NVDA
MSFT
AAPL
AMZN
META
```

---

## 📊 Running the Scanner

### Normal run (reuses cached data if available)
```bash
python -m src.pipeline --tickers data/tickers_sp500.txt --outdir results
```

### Force a fresh snapshot (ignore cache)
```bash
python -m src.pipeline --tickers data/tickers_sp500.txt --outdir results --force_new_snapshot
```

### Change number of top tickers in dashboard/PDF
```bash
python -m src.pipeline --tickers data/tickers_sp500.txt --outdir results --top_n 20
```

### Train signal weights from past performance
```bash
python -m src.pipeline --tickers data/tickers_sp500.txt --outdir results --learn
```

---

## 📂 Outputs

- **Signals**
  - `results/buy_signals.csv` → all tickers with signals + BuyScore (+ sentiment if enabled)  
  - `results/top_10.csv` → top N tickers sorted by BuyScore  
- **Charts**
  - `results/buyscore_hist.png` → distribution of BuyScores  
  - `results/sentiment_hist.png` → distribution of sentiment scores (if enabled)  
- **Interactive dashboard**
  - `results/top10_dashboard.html` → open in any browser for zoomable charts  
- **PDF**
  - `results/top10.pdf` → static summary of top N tickers  
- **Cache**
  - `cache/ohlcv_daily/YYYY-MM-DD/` → daily OHLCV snapshots  
  - `cache/news_daily/YYYY-MM-DD/` → cached news JSON (if enabled)

---

## 🔎 Backtest (One-Month Snapshot)

**Goal:** Check “If we had only data up to 30 days ago, what BuyScores would we have assigned, and what actually happened since then?”

### Run backtest
```bash
python -m src.backtest_one_month --tickers data/tickers_sp500.txt --outdir results

## modified backtest code

```bash
python -u -m src.backtest_one_month \
  --tickers data/tickers_sp500.txt \
  --outdir results \
  --snapshot_date 2025-09-01 \
  --lookback_days 60
```

Options:
- `--lookback_days 30` → how far back the “as of” date should be  
- `--snapshot_date YYYY-MM-DD` → use a specific cached snapshot

### Outputs
- `results/backtest_1m_<SNAPDAY>.csv` → per-ticker BuyScore (as of) + actual forward % return  
- `results/backtest_1m_scatter_<SNAPDAY>.png` → scatter: BuyScore vs forward % return  
- `results/backtest_1m_bins_<SNAPDAY>.png` → average forward % return by BuyScore bin  

---

## 🛠️ Troubleshooting

- Run once with `--force_new_snapshot` to populate caches.  
- Ignore `viz.py` errors if dashboard + PDF are generated.  
- News sentiment slows after ~100 calls; cached thereafter.

---

## 🧠 Understanding Sentiment

- **Sentiment** = raw VADER compound (−1 to +1)  
- **SentimentZ** = z-score vs all tickers that day  

Use `SentimentZ` when **ranking across many tickers**.

---

## 📌 Example Workflow

```bash
# Morning scan with new snapshot
python -m src.pipeline --tickers data/tickers_sp500.txt --outdir results --force_new_snapshot

# Open interactive dashboard
open results/top10_dashboard.html   # macOS
start results\top10_dashboard.html # Windows

# Optional backtest
python -m src.backtest_one_month --tickers data/tickers_sp500.txt --outdir results
```

---

## 📈 Roadmap (AI/ML Upgrade)

- Add ML models (LightGBM, logistic regression) for 40/60/120-day forecasts  
- Meta-labeling: filter rule-based entries  
- Replace VADER with FinBERT sentiment  
- Time-aware cross-validation (purged/embargoed CV)  
- Portfolio risk-adjusted allocation

---

## 📜 License

MIT License — free to use and modify.
