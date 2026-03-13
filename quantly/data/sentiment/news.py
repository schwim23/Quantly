"""Fetch and score financial news headlines.

Sources: Polygon.io news endpoint, Bloomberg News API.
Sentiment scoring: VADER (fast) + FinBERT (accurate, slower).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from quantly.data.cache import cached

logger = logging.getLogger(__name__)


@cached("news")
def get_news_headlines(
    ticker: str,
    start: date,
    end: date,
    source: str = "polygon",
) -> pd.DataFrame:
    """Fetch news headlines for a ticker over a date range.

    Returns:
        DataFrame with columns: published_at, title, summary, source, url
    """
    if source == "polygon":
        return _fetch_polygon_news(ticker, start, end)
    raise ValueError(f"Unknown source: {source}")


def score_headlines(headlines: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment scores to a headlines DataFrame.

    Uses VADER for speed; FinBERT for higher accuracy if available.

    Returns:
        Input DataFrame with added columns: vader_compound, finbert_label, finbert_score
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    headlines = headlines.copy()
    headlines["vader_compound"] = headlines["title"].apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )
    return headlines


def rolling_sentiment(
    scored: pd.DataFrame,
    window_days: int = 7,
) -> pd.Series:
    """Compute rolling average sentiment score over a time window.

    Returns:
        Series indexed by date with the rolling mean vader_compound score
    """
    daily = scored.set_index("published_at")["vader_compound"].resample("D").mean()
    return daily.rolling(window=window_days, min_periods=1).mean()


def _fetch_polygon_news(ticker: str, start: date, end: date) -> pd.DataFrame:
    # TODO: implement GET /v2/reference/news?ticker={ticker}&published_utc.gte={start}
    raise NotImplementedError
