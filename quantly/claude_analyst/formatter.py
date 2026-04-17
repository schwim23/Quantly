"""Assemble the data package sent to Claude for each candidate."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


def format_candidate(
    ticker: str,
    ml_rank: int,
    ensemble_prob: float,
    shap_features: list[dict],
    signal_date: date,
) -> dict:
    """Build the data dict for one candidate."""
    from quantly.data.sentiment import earnings_calls, news, sec_filings

    candidate: dict = {
        "ticker": ticker,
        "ml_rank": ml_rank,
        "ensemble_prob": ensemble_prob,
        "top_shap_features": shap_features,
    }

    # Recent news
    try:
        scored = news.get_scored_news(ticker)
        candidate["recent_headlines"] = scored["title"].head(5).tolist()
    except Exception:
        candidate["recent_headlines"] = []

    # SEC 8-K earnings press release excerpt
    try:
        filing = earnings_calls.get_latest_earnings_release(ticker)
        text = filing.get("excerpt", "")
        candidate["earnings_excerpt"] = text[:600] if text else ""
    except Exception:
        candidate["earnings_excerpt"] = ""

    # SEC 10-Q MD&A excerpt
    try:
        filing = sec_filings.get_latest_filing(ticker)
        mda = filing.get("mda_text", "")
        candidate["mda_excerpt"] = mda[:600] if mda else ""
    except Exception:
        candidate["mda_excerpt"] = ""

    # Reddit summary
    try:
        from quantly.data.sentiment.reddit import compute_reddit_features
        stats = compute_reddit_features(ticker)
        if stats.get("reddit_available"):
            candidate["reddit_summary"] = (
                f"mentions (7d): {stats['reddit_mention_count_7d']:.0f}, "
                f"velocity: {stats['reddit_velocity_ratio']:.1f}x, "
                f"sentiment: {stats['reddit_avg_sentiment']:.2f}"
            )
        else:
            candidate["reddit_summary"] = "Reddit data not available"
    except Exception:
        candidate["reddit_summary"] = ""

    return candidate


def format_all_candidates(
    scored_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    signal_date: date,
    top_n: int = 20,
) -> list[dict]:
    """Format top-N ML candidates into the Claude prompt payload."""
    from quantly.models.predict import get_top_shap_features

    candidates = []
    for rank, row in enumerate(scored_df.head(top_n).itertuples(), 1):
        ticker = row.ticker
        shap_feats = get_top_shap_features(shap_df, ticker, top_n=5)
        candidate = format_candidate(
            ticker=ticker,
            ml_rank=rank,
            ensemble_prob=row.ensemble_prob,
            shap_features=shap_feats,
            signal_date=signal_date,
        )
        candidates.append(candidate)

    return candidates
