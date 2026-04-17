"""OHLCV price data via yfinance (free, no API key required).

yfinance wraps the Yahoo Finance API. No key needed, but subject to
Yahoo's unofficial rate limits — use the cache layer aggressively.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from quantly.data.cache import cached

logger = logging.getLogger(__name__)


@cached("prices_daily")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_ohlcv(
    ticker: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch daily OHLCV for a single ticker via yfinance.

    Args:
        ticker: Stock symbol (e.g. "AAPL")
        start: Start date (inclusive)
        end: End date (inclusive)

    Returns:
        DataFrame indexed by date with columns: open, high, low, close, volume
    """
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}")
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).normalize()
    return df


def get_spy_returns(start: date, end: date) -> pd.Series:
    """Daily SPY returns for benchmark comparison."""
    df = get_ohlcv("SPY", start, end)
    return df["close"].pct_change().dropna()


def get_bulk_close(
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch adjusted close prices for multiple tickers in one call.

    Much faster than calling get_ohlcv() per ticker.

    Returns:
        DataFrame with tickers as columns, dates as index
    """
    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    close.index = pd.to_datetime(close.index).normalize()
    return close
