"""Master feature pipeline.

Orchestrates all data fetches and feature computations for a list of tickers,
producing a model-ready DataFrame.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quantly.features.kpi import compute_all_kpi_features
from quantly.features.sentiment_features import (
    earnings_call_features,
    news_sentiment_feature,
    reddit_features,
    sec_filing_features,
)
from quantly.features.technical import compute_all_technical_features

logger = logging.getLogger(__name__)


def build_features_for_ticker(
    ticker: str,
    signal_date: date,
) -> dict[str, Any]:
    """Compute the full feature vector for a single ticker at a given signal date.

    All data fetched is strictly from before signal_date (no lookahead).

    Returns:
        Dict mapping feature_name -> float (or NaN if data unavailable)
    """
    from quantly.data import fundamentals, prices
    from quantly.data.sentiment import earnings_calls, news, reddit, sec_filings

    features: dict[str, Any] = {"ticker": ticker, "signal_date": signal_date}

    # -- Technical features --
    try:
        ohlcv = prices.get_ohlcv(ticker, date(signal_date.year - 1, signal_date.month, signal_date.day), signal_date)
        features.update(compute_all_technical_features(ohlcv))
    except Exception:
        logger.warning("Technical features failed for %s", ticker)

    # -- KPI features --
    try:
        income = fundamentals.get_income_statement(ticker)
        balance = fundamentals.get_balance_sheet(ticker)
        cashflow = fundamentals.get_cash_flow(ticker)
        market_cap = ohlcv["close"].iloc[-1] * ohlcv.get("shares_outstanding", pd.Series([1e9])).iloc[-1]
        features.update(compute_all_kpi_features(income, balance, cashflow, market_cap))
    except Exception:
        logger.warning("KPI features failed for %s", ticker)

    # -- Sentiment features --
    try:
        headlines = news.get_news_headlines(ticker, date(signal_date.year, signal_date.month, signal_date.day), signal_date)
        scored = news.score_headlines(headlines)
        features.update(news_sentiment_feature(scored))
    except Exception:
        logger.warning("News sentiment failed for %s", ticker)

    try:
        filing = sec_filings.get_latest_filing(ticker)
        mda_scores = sec_filings.score_mda_sentiment(filing["mda_text"])
        # Compare with prior filing if available
        risk_delta = {"net_change": np.nan}
        features.update(sec_filing_features(mda_scores, risk_delta))
    except Exception:
        logger.warning("SEC filing features failed for %s", ticker)

    try:
        transcript = earnings_calls.get_transcript(ticker, "latest")
        tone = earnings_calls.score_transcript_tone(transcript)
        features.update(earnings_call_features(tone))
    except Exception:
        logger.warning("Earnings call features failed for %s", ticker)

    try:
        from datetime import timedelta
        mention_start = date(signal_date.year, signal_date.month, signal_date.day) - timedelta(days=14)
        mentions = reddit.get_ticker_mentions(ticker, mention_start, signal_date)
        stats = reddit.compute_mention_velocity(mentions)
        features.update(reddit_features(stats))
    except Exception:
        logger.warning("Reddit features failed for %s", ticker)

    return features


def build_feature_matrix(
    tickers: list[str],
    signal_date: date,
) -> pd.DataFrame:
    """Build the full feature matrix for a list of tickers.

    Returns:
        DataFrame where each row is a ticker and columns are features
    """
    rows = []
    for ticker in tickers:
        logger.info("Computing features for %s", ticker)
        row = build_features_for_ticker(ticker, signal_date)
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")
