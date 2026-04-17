"""Master feature pipeline.

Orchestrates all data fetches and feature computations for a list of tickers,
producing a model-ready DataFrame. Every feature is computed strictly from
data available at or before the signal_date (no lookahead).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from quantly.features.kpi import compute_all_kpi_features
from quantly.features.sentiment_features import (
    earnings_tone_features,
    news_sentiment_features,
    reddit_sentiment_features,
    sec_filing_features,
)
from quantly.features.technical import compute_all_technical_features

logger = logging.getLogger(__name__)


def build_features_for_ticker(ticker: str, signal_date: date) -> dict:
    """Compute the full feature vector for one ticker at the signal date.

    Returns:
        Dict of feature_name -> float (NaN where data unavailable)
    """
    from quantly.data import fundamentals, prices

    features: dict = {"ticker": ticker, "signal_date": signal_date}

    # ── Technical features (from 1 year of OHLCV) ──
    try:
        start = signal_date - timedelta(days=365)
        ohlcv = prices.get_ohlcv(ticker, start, signal_date)
        features.update(compute_all_technical_features(ohlcv))
    except Exception as e:
        logger.debug("Technical features failed for %s: %s", ticker, e)

    # ── KPI features (from quarterly fundamentals) ──
    try:
        income = fundamentals.get_income_statement(ticker)
        balance = fundamentals.get_balance_sheet(ticker)
        cashflow = fundamentals.get_cash_flow(ticker)
        info = fundamentals.get_company_info(ticker)
        market_cap = info.get("market_cap") or 0
        features.update(compute_all_kpi_features(income, balance, cashflow, market_cap))
        # Company metadata features
        features["beta"] = info.get("beta") or float("nan")
        features["short_ratio"] = info.get("short_ratio") or float("nan")
        features["forward_pe"] = info.get("forward_pe") or float("nan")
    except Exception as e:
        logger.debug("KPI features failed for %s: %s", ticker, e)

    # ── Sentiment features ──
    features.update(news_sentiment_features(ticker))
    features.update(earnings_tone_features(ticker))
    features.update(sec_filing_features(ticker))
    features.update(reddit_sentiment_features(ticker))

    return features


def build_feature_matrix(
    tickers: list[str],
    signal_date: date,
) -> pd.DataFrame:
    """Build the full feature matrix for a list of tickers.

    Returns:
        DataFrame indexed by ticker, columns are features (floats / NaN)
    """
    rows = []
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0 or i == total:
            logger.info("Features: %d / %d tickers processed", i, total)
        row = build_features_for_ticker(ticker, signal_date)
        rows.append(row)

    df = pd.DataFrame(rows)
    if "ticker" in df.columns:
        df = df.set_index("ticker")
    if "signal_date" in df.columns:
        df = df.drop(columns=["signal_date"])

    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")
    return df
