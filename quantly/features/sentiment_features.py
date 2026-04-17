"""Aggregate raw sentiment data into numeric model features."""

from __future__ import annotations

import numpy as np


def news_sentiment_features(ticker: str) -> dict[str, float]:
    """Fetch and aggregate news sentiment for a ticker."""
    from quantly.data.sentiment.news import get_scored_news, rolling_sentiment_avg

    try:
        scored = get_scored_news(ticker)
        return {
            "news_sentiment_7d_avg": rolling_sentiment_avg(scored, days=7),
            "news_headline_count": float(len(scored)),
        }
    except Exception:
        return {"news_sentiment_7d_avg": float("nan"), "news_headline_count": float("nan")}


def earnings_tone_features(ticker: str) -> dict[str, float]:
    """Fetch and score most recent earnings press release from SEC EDGAR."""
    from quantly.data.sentiment.earnings_calls import get_latest_earnings_release, score_earnings_tone

    try:
        filing = get_latest_earnings_release(ticker)
        return score_earnings_tone(filing)
    except Exception:
        return {
            "ec_overall_sentiment": float("nan"),
            "ec_guidance_positive_ratio": float("nan"),
            "ec_guidance_negative_ratio": float("nan"),
            "ec_uncertainty_ratio": float("nan"),
        }


def sec_filing_features(ticker: str) -> dict[str, float]:
    """Fetch and score MD&A section from most recent 10-Q/10-K."""
    from quantly.data.sentiment.sec_filings import get_latest_filing, score_mda_sentiment

    try:
        filing = get_latest_filing(ticker)
        return score_mda_sentiment(filing)
    except Exception:
        return {
            "mda_vader_score": float("nan"),
            "mda_positive_ratio": float("nan"),
            "mda_negative_ratio": float("nan"),
            "mda_uncertainty_ratio": float("nan"),
        }


def reddit_sentiment_features(ticker: str) -> dict[str, float]:
    """Reddit mention velocity and sentiment (skipped gracefully if no credentials)."""
    from quantly.data.sentiment.reddit import compute_reddit_features

    try:
        return compute_reddit_features(ticker)
    except Exception:
        return {
            "reddit_mention_count_7d": float("nan"),
            "reddit_avg_sentiment": float("nan"),
            "reddit_velocity_ratio": float("nan"),
            "reddit_available": 0.0,
        }
