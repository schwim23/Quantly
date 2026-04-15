"""Aggregate all sentiment sources into a flat feature vector for the ML model.

Each source is normalised to [-1, +1] before aggregation.
Missing / errored sources default to 0.0 (neutral) — never block the pipeline.
"""
from __future__ import annotations

from typing import Any

from quantly.data.cache import Cache
from quantly.data.sentiment.news import get_news_sentiment
from quantly.data.sentiment.gdelt import get_gdelt_tone
from quantly.data.sentiment.stocktwits import get_stocktwits_sentiment
from quantly.data.sentiment.congress import get_congress_sentiment
from quantly.data.sentiment.options_flow import get_options_flow


def compute_sentiment_features(ticker: str, cache: Cache) -> dict[str, float]:
    """Return a flat dict of sentiment features for the given ticker.

    Features:
        news_sentiment          - Alpha Vantage 7-day avg sentiment [-1, +1]
        news_bullish_pct        - fraction of bullish articles [0, 1]
        gdelt_tone              - GDELT 7-day avg normalised tone [-1, +1]
        stocktwits_sentiment    - StockTwits bull/bear ratio mapped [-1, +1]
        congress_net            - congressional buy - sell count (raw int)
        congress_latest_buy_days - days since last congressional buy (99 if none)
        options_unusual_calls   - call volume / 30d avg (>1.5 = unusual)
        options_put_call_ratio  - put/call ratio (<0.7 = bullish)
        options_net_score       - composite options score
        sentiment_composite     - equal-weighted average of normalised sources
    """
    news = get_news_sentiment(ticker, cache)
    gdelt = get_gdelt_tone(ticker, cache)
    st = get_stocktwits_sentiment(ticker, cache)
    congress = get_congress_sentiment(ticker, cache)
    options = get_options_flow(ticker, cache)

    # Normalise congress net: cap at ±5 buys to keep scale similar
    congress_norm = max(-1.0, min(1.0, congress["net_congress"] / 5.0))

    # Composite: simple equal-weight of the three direct sentiment scores
    direct_scores = [
        news["avg_sentiment"],
        gdelt["tone"],
        st["sentiment"],
        congress_norm,
    ]
    composite = sum(direct_scores) / len(direct_scores)

    return {
        "news_sentiment": news["avg_sentiment"],
        "news_bullish_pct": news["bullish_pct"],
        "gdelt_tone": gdelt["tone"],
        "stocktwits_sentiment": st["sentiment"],
        "congress_net": float(congress["net_congress"]),
        "congress_latest_buy_days": float(congress["latest_buy_days"]),
        "options_unusual_calls": options["unusual_call_ratio"],
        "options_put_call_ratio": options["put_call_ratio"],
        "options_net_score": options["net_options_score"],
        "sentiment_composite": round(composite, 4),
    }
