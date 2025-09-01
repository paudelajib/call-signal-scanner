# src/news.py
from __future__ import annotations
import os, time, json
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

class NewsSentiment:
    def __init__(self, provider: Optional[str] = None, window_days: int = 7,
                 cache_dir: str | Path = "cache/news_daily",
                 min_articles: int = 3, title_only: bool = True):
        self.provider = (provider or os.getenv("NEWS_PROVIDER") or "none").lower()
        self.window_days = window_days
        self.min_articles = min_articles
        self.title_only = title_only
        self.session = requests.Session()
        self.analyzer = SentimentIntensityAnalyzer()
        self.cache_root = Path(cache_dir)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    # ---------- caching ----------
    def _cache_path(self, ticker: str) -> Path:
        return self.cache_root / _today_str() / f"{ticker.upper()}.json"

    def _read_cache(self, ticker: str) -> Optional[List[str]]:
        p = self._cache_path(ticker)
        if not p.exists(): return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, ticker: str, texts: List[str]) -> None:
        p = self._cache_path(ticker)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # ---------- providers ----------
    def _newsapi_fetch(self, q: str) -> List[str]:
        key = os.getenv("NEWSAPI_KEY", "").strip()
        if not key:
            return []
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": q,
            "pageSize": 20,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": key,
        }
        try:
            r = self.session.get(url, params=params, timeout=15)
            # Handle rate limiting with brief backoff
            if r.status_code == 429:
                time.sleep(1.5)
                r = self.session.get(url, params=params, timeout=15)
            if r.status_code != 200:
                return []
            data = r.json()
            texts = []
            for a in data.get("articles", []):
                if self.title_only:
                    t = (a.get("title") or "").strip()
                else:
                    t = " ".join(filter(None, [a.get("title",""), a.get("description","")])).strip()
                if t:
                    texts.append(t)
            return texts
        except Exception:
            return []

    def _finnhub_fetch(self, ticker: str) -> List[str]:
        key = os.getenv("FINNHUB_KEY", "").strip()
        if not key:
            return []
        from datetime import timedelta
        to = datetime.utcnow().date()
        fr = (datetime.utcnow() - timedelta(days=self.window_days)).date()
        url = "https://finnhub.io/api/v1/company-news"
        params = {"symbol": ticker, "from": fr.isoformat(), "to": to.isoformat(), "token": key}
        try:
            r = self.session.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(1.5)
                r = self.session.get(url, params=params, timeout=15)
            if r.status_code != 200:
                return []
            data = r.json()
            texts = []
            for a in data:
                t = (a.get("headline") or "").strip() if self.title_only \
                    else " ".join(filter(None, [a.get("headline",""), a.get("summary","")])).strip()
                if t:
                    texts.append(t)
            return texts
        except Exception:
            return []

    def headlines_for(self, ticker: str) -> List[str]:
        # daily cache first
        cached = self._read_cache(ticker)
        if cached is not None:
            return cached

        texts: List[str] = []
        if self.provider == "newsapi":
            # try ticker query; if weak, try "ticker stock" to disambiguate
            texts = self._newsapi_fetch(f'{ticker} stock OR {ticker} shares')
            if len(texts) < self.min_articles:
                texts = self._newsapi_fetch(f'"{ticker}" company OR "{ticker}" stock')
        elif self.provider == "finnhub":
            texts = self._finnhub_fetch(ticker)
        else:
            texts = []

        # write daily cache even if empty to avoid hammering provider
        self._write_cache(ticker, texts)
        return texts

    def sentiment_score(self, ticker: str) -> Optional[float]:
        texts = self.headlines_for(ticker)
        if len(texts) < self.min_articles:
            return None
        scores = [self.analyzer.polarity_scores(t)["compound"] for t in texts]
        if not scores:
            return None
        # mild clipping to reduce outliers
        avg = sum(scores) / len(scores)
        return float(max(-0.9, min(0.9, avg)))