"""Fetch OHLCV price data from Polygon.io or Bloomberg.

Used for:
- Technical feature computation
- Label construction (forward returns)
- Backtesting
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from quantly.data.cache import cached

logger = logging.getLogger(__name__)


@cached("prices")
def get_ohlcv(
    ticker: str,
    start: date,
    end: date,
    source: str = "polygon",
) -> pd.DataFrame:
    """Fetch daily OHLCV for a single ticker.

    Args:
        ticker: Stock symbol (e.g., "AAPL")
        start: Start date (inclusive)
        end: End date (inclusive)
        source: Data source — "polygon" | "bloomberg" | "yfinance"

    Returns:
        DataFrame with columns: date, open, high, low, close, volume, vwap
        Index: date (DatetimeIndex)
    """
    if source == "polygon":
        return _fetch_polygon(ticker, start, end)
    if source == "yfinance":
        return _fetch_yfinance(ticker, start, end)
    raise ValueError(f"Unknown source: {source}")


def _fetch_polygon(ticker: str, start: date, end: date) -> pd.DataFrame:
    # TODO: implement Polygon.io /v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}
    raise NotImplementedError


def _fetch_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.index = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]
    return df


def get_spy_returns(start: date, end: date) -> pd.Series:
    """Return daily SPY returns for benchmark comparison."""
    df = get_ohlcv("SPY", start, end, source="yfinance")
    return df["close"].pct_change().dropna()
