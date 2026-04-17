"""News headline sentiment via yfinance (free, no API key required).

yfinance exposes recent news articles for each ticker from Yahoo Finance.
Sentiment scored with VADER (rule-based, no model download needed).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from quantly.data.cache import cached

logger = logging.getLogger(__name__)

_analyzer = SentimentIntensityAnalyzer()


@cached("news_raw", ttl=3_600)  # 1-hour TTL — news goes stale fast
def get_news_headlines(ticker: str) -> pd.DataFrame:
    """Fetch recent news for a ticker from Yahoo Finance via yfinance.

    Returns:
        DataFrame with columns: published_at, title, publisher, url
        Sorted by published_at descending (most recent first)
    """
    t = yf.Ticker(ticker)
    articles = t.news or []

    rows = []
    for a in articles:
        rows.append({
            "published_at": datetime.fromtimestamp(
                a.get("providerPublishTime", 0), tz=timezone.utc
            ),
            "title": a.get("title", ""),
            "publisher": a.get("publisher", ""),
            "url": a.get("link", ""),
        })

    if not rows:
        logger.debug("No news found for %s", ticker)
        return pd.DataFrame(columns=["published_at", "title", "publisher", "url"])

    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)
    return df


def score_headlines(headlines: pd.DataFrame) -> pd.DataFrame:
    """Add VADER sentiment scores to a headlines DataFrame.

    Returns:
        Input DataFrame with added column: vader_compound (-1 to 1)
    """
    if headlines.empty:
        return headlines.assign(vader_compound=float("nan"))
    df = headlines.copy()
    df["vader_compound"] = df["title"].apply(
        lambda t: _analyzer.polarity_scores(str(t))["compound"]
    )
    return df


def get_scored_news(ticker: str) -> pd.DataFrame:
    """Convenience wrapper: fetch + score in one call."""
    headlines = get_news_headlines(ticker)
    return score_headlines(headlines)


def rolling_sentiment_avg(scored: pd.DataFrame, days: int = 7) -> float:
    """Mean VADER compound score over the last N days.

    Returns:
        Float in [-1, 1], or 0.0 if no articles found.
    """
    if scored.empty or "vader_compound" not in scored.columns:
        return 0.0
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    recent = scored[scored["published_at"] >= cutoff]
    if recent.empty:
        return 0.0
    return float(recent["vader_compound"].mean())
