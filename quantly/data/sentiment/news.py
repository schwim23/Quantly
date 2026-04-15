"""Alpha Vantage news sentiment — AI-scored headlines per ticker.

Free tier: unlimited calls (soft throttle ~5 req/min).
Returns a rolling 7-day average sentiment score in [-1, +1].

Ref: https://www.alphavantage.co/documentation/#news-sentiment
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import httpx

from quantly.config import get_config
from quantly.data.cache import Cache, TTL_SENTIMENT

logger = logging.getLogger(__name__)

_BASE = "https://www.alphavantage.co/query"


def _av_get(params: dict) -> Any:
    cfg = get_config()
    try:
        resp = httpx.get(
            _BASE,
            params={"apikey": cfg.alpha_vantage_api_key, **params},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Alpha Vantage error: %s", exc)
        return None


def get_news_sentiment(ticker: str, cache: Cache) -> dict[str, float]:
    """Return 7-day rolling average news sentiment for *ticker*.

    Returns dict with keys:
        avg_sentiment   - average relevance-weighted sentiment score (-1 to +1)
        article_count   - number of scored articles in the window
        bullish_pct     - fraction of articles with positive sentiment
    """
    key = f"av:news_sentiment:{ticker}"

    def _fetch() -> dict[str, float]:
        today = date.today()
        week_ago = today - timedelta(days=7)
        time_from = week_ago.strftime("%Y%m%dT0000")

        data = _av_get({
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": time_from,
            "limit": 50,
        })

        if not data or "feed" not in data:
            return {"avg_sentiment": 0.0, "article_count": 0, "bullish_pct": 0.0}

        scores: list[float] = []
        bullish = 0

        for article in data["feed"]:
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker") != ticker:
                    continue
                try:
                    score = float(ts.get("ticker_sentiment_score", 0))
                    relevance = float(ts.get("relevance_score", 0.5))
                    # Weight by relevance
                    scores.append(score * relevance)
                    if score > 0.15:
                        bullish += 1
                except (ValueError, TypeError):
                    continue

        if not scores:
            return {"avg_sentiment": 0.0, "article_count": 0, "bullish_pct": 0.0}

        avg = sum(scores) / len(scores)
        return {
            "avg_sentiment": round(avg, 4),
            "article_count": len(scores),
            "bullish_pct": round(bullish / len(scores), 4),
        }

    return cache.get_or_fetch(key, _fetch, ttl_seconds=TTL_SENTIMENT)
