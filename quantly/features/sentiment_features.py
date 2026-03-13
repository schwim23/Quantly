"""Aggregate raw sentiment data into numeric model features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def news_sentiment_feature(scored_headlines: pd.DataFrame) -> dict[str, float]:
    """Rolling 7-day average sentiment from news headlines.

    Returns:
        Dict with keys: news_sentiment_7d_avg, news_headline_count_7d
    """
    if scored_headlines.empty:
        return {"news_sentiment_7d_avg": np.nan, "news_headline_count_7d": 0}
    recent = scored_headlines.tail(7)
    return {
        "news_sentiment_7d_avg": recent["vader_compound"].mean(),
        "news_headline_count_7d": len(recent),
    }


def earnings_call_features(tone_scores: dict) -> dict[str, float]:
    """Features derived from earnings call tone analysis.

    Returns:
        Dict with keys: ec_prepared_sentiment, ec_guidance_positive, ec_uncertainty,
        ec_qa_sentiment
    """
    if not tone_scores:
        return {
            "ec_prepared_sentiment": np.nan,
            "ec_guidance_positive": np.nan,
            "ec_uncertainty": np.nan,
            "ec_qa_sentiment": np.nan,
        }
    return {
        "ec_prepared_sentiment": tone_scores.get("prepared_sentiment", np.nan),
        "ec_guidance_positive": tone_scores.get("guidance_positive", np.nan),
        "ec_uncertainty": tone_scores.get("uncertainty_score", np.nan),
        "ec_qa_sentiment": tone_scores.get("qa_sentiment", np.nan),
    }


def sec_filing_features(mda_scores: dict, risk_delta: dict) -> dict[str, float]:
    """Features from SEC filing analysis.

    Returns:
        Dict with keys: mda_positive, mda_negative, mda_uncertainty,
        risk_factors_net_change
    """
    return {
        "mda_positive": mda_scores.get("positive_ratio", np.nan),
        "mda_negative": mda_scores.get("negative_ratio", np.nan),
        "mda_uncertainty": mda_scores.get("uncertainty_ratio", np.nan),
        "risk_factors_net_change": risk_delta.get("net_change", np.nan),
    }


def reddit_features(mention_stats: dict) -> dict[str, float]:
    """Features from Reddit mention analysis.

    Returns:
        Dict with keys: reddit_mention_velocity, reddit_avg_sentiment
    """
    if not mention_stats:
        return {"reddit_mention_velocity": np.nan, "reddit_avg_sentiment": np.nan}
    return {
        "reddit_mention_velocity": mention_stats.get("velocity_ratio", np.nan),
        "reddit_avg_sentiment": mention_stats.get("avg_sentiment", np.nan),
    }
