"""Feature pipeline — assembles the full feature matrix for the ML model.

For each ticker in the candidate universe, computes all features and
returns a single DataFrame with one row per ticker and one column per feature.

Feature groups and approximate model weights:
  - Sentiment + options (high weight)   ~40%
  - Catalyst / earnings (high weight)   ~30%
  - Technical / price   (medium weight) ~20%
  - Fundamental         (low weight)    ~10%
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from quantly.data.cache import Cache
from quantly.data.catalysts import get_upcoming_earnings
from quantly.data.prices import get_ohlcv
from quantly.features.technical import compute_all_technical
from quantly.features.sentiment import compute_sentiment_features
from quantly.features.catalyst import compute_catalyst_features
from quantly.features.fundamental import compute_fundamental_features

logger = logging.getLogger(__name__)


def build_feature_row(
    ticker: str,
    cache: Cache,
    price_df: pd.DataFrame,
    upcoming_earnings: dict,
) -> dict[str, float]:
    """Build a single feature row for one ticker.

    Returns a flat dict mapping feature_name → float value.
    Never raises — missing sources default to neutral values.
    """
    row: dict[str, float] = {"ticker": ticker}

    try:
        technical = compute_all_technical(price_df)
        row.update(technical)
    except Exception as exc:
        logger.warning("%s: technical features failed: %s", ticker, exc)

    try:
        sentiment = compute_sentiment_features(ticker, cache)
        row.update(sentiment)
    except Exception as exc:
        logger.warning("%s: sentiment features failed: %s", ticker, exc)

    try:
        catalyst = compute_catalyst_features(ticker, cache, upcoming_earnings)
        row.update(catalyst)
    except Exception as exc:
        logger.warning("%s: catalyst features failed: %s", ticker, exc)

    try:
        fundamental = compute_fundamental_features(ticker, cache)
        row.update(fundamental)
    except Exception as exc:
        logger.warning("%s: fundamental features failed: %s", ticker, exc)

    return row


def build_feature_matrix(
    tickers: list[str],
    cache: Cache,
    as_of_date: Optional[date] = None,
) -> pd.DataFrame:
    """Build the full feature matrix for a list of tickers.

    Args:
        tickers: Candidate tickers (post-universe-screen).
        cache: Shared cache instance.
        as_of_date: Signal date (default: today). All features use data
                    strictly available on or before this date.

    Returns:
        DataFrame with tickers as index and features as columns.
        Rows with all-zero features are retained (model handles them).
    """
    if not tickers:
        return pd.DataFrame()

    as_of = as_of_date or date.today()
    price_start = as_of - timedelta(days=365)  # 1 year of price history

    # Pre-fetch earnings calendar once for all tickers
    upcoming = get_upcoming_earnings(tickers, cache)

    rows: list[dict] = []
    for ticker in tickers:
        price_df = get_ohlcv(ticker, price_start, as_of, cache)
        row = build_feature_row(ticker, cache, price_df, upcoming)
        rows.append(row)
        logger.debug("Features built for %s (%d features)", ticker, len(row) - 1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ticker")
    # Fill NaN with 0 (neutral) — model is robust to this
    df = df.fillna(0.0)
    return df


# ── Feature column names (for model training consistency) ────────────────────

FEATURE_COLUMNS = [
    # Technical
    "rsi_14", "macd_crossover", "macd_histogram",
    "momentum_5d", "momentum_10d", "momentum_20d",
    "volume_spike_ratio", "dist_from_52w_high", "dist_from_52w_low",
    "bollinger_pct_b",
    # Sentiment
    "news_sentiment", "news_bullish_pct", "gdelt_tone",
    "stocktwits_sentiment", "congress_net", "congress_latest_buy_days",
    "options_unusual_calls", "options_put_call_ratio", "options_net_score",
    "sentiment_composite",
    # Catalyst
    "days_to_earnings", "earnings_proximity", "eps_beat_streak",
    "eps_surprise_last", "eps_surprise_prev",
    "analyst_upgrades_10d", "analyst_downgrades_10d", "analyst_net_10d",
    # Fundamental
    "revenue_growth_yoy", "revenue_growth_qoq", "gross_margin",
    "gross_margin_trend", "fcf_yield", "eps_growth_slope",
    "debt_to_equity", "eps_surprise_last", "has_positive_fcf",
]
