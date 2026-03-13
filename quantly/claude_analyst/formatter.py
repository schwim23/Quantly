"""Format ML pipeline output for Claude consumption.

Prepares the data package sent to Claude by assembling:
- SHAP feature explanations
- Recent news headlines
- Earnings call excerpts
- SEC MD&A excerpts
- Reddit sentiment summaries
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


def format_candidate_for_claude(
    ticker: str,
    ml_rank: int,
    ensemble_prob: float,
    shap_features: list[dict],
    signal_date: date,
) -> dict:
    """Build the data package for a single candidate.

    Fetches supporting data (news, filings, transcripts, reddit) and assembles
    a dict ready for the Claude prompt builder.
    """
    from quantly.data.sentiment import earnings_calls, news, reddit, sec_filings

    candidate = {
        "ticker": ticker,
        "ml_rank": ml_rank,
        "ensemble_prob": ensemble_prob,
        "top_shap_features": shap_features,
    }

    # Recent news headlines
    try:
        from datetime import timedelta
        headlines_df = news.get_news_headlines(
            ticker,
            signal_date - timedelta(days=7),
            signal_date,
        )
        scored = news.score_headlines(headlines_df)
        candidate["recent_headlines"] = scored["title"].head(5).tolist()
    except Exception:
        candidate["recent_headlines"] = []
        logger.debug("No news headlines for %s", ticker)

    # SEC filing MD&A excerpt
    try:
        filing = sec_filings.get_latest_filing(ticker)
        mda = filing.get("mda_text", "")
        candidate["mda_excerpt"] = mda[-600:] if len(mda) > 600 else mda
    except Exception:
        candidate["mda_excerpt"] = ""

    # Earnings call excerpt
    try:
        transcript = earnings_calls.get_transcript(ticker, "latest")
        remarks = transcript.get("prepared_remarks", "")
        candidate["earnings_call_excerpt"] = remarks[-600:] if len(remarks) > 600 else remarks
    except Exception:
        candidate["earnings_call_excerpt"] = ""

    # Reddit summary
    try:
        from datetime import timedelta
        mentions = reddit.get_ticker_mentions(
            ticker,
            signal_date - timedelta(days=7),
            signal_date,
        )
        stats = reddit.compute_mention_velocity(mentions)
        candidate["reddit_summary"] = (
            f"Mentions (7d): {stats.get('mentions_last_7d', 'N/A')}, "
            f"velocity ratio: {stats.get('velocity_ratio', 'N/A'):.1f}x, "
            f"avg sentiment: {stats.get('avg_sentiment', 0):.2f}"
        )
    except Exception:
        candidate["reddit_summary"] = ""

    return candidate


def format_all_candidates(
    scored_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    signal_date: date,
    top_n: int = 20,
) -> list[dict]:
    """Format top-N ML candidates for Claude analysis.

    Args:
        scored_df: Output from predict.score_candidates()
        shap_df: Output from predict.compute_shap_values()
        signal_date: Date of the signal
        top_n: Number of candidates to send to Claude

    Returns:
        List of candidate dicts ready for build_candidate_prompt()
    """
    from quantly.models.predict import get_top_shap_features

    candidates = []
    for rank, row in enumerate(scored_df.head(top_n).itertuples(), 1):
        ticker = row.ticker
        shap_feats = get_top_shap_features(shap_df, ticker, top_n=5)
        candidate = format_candidate_for_claude(
            ticker=ticker,
            ml_rank=rank,
            ensemble_prob=row.ensemble_prob,
            shap_features=shap_feats,
            signal_date=signal_date,
        )
        candidates.append(candidate)

    return candidates
