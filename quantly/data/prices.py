"""OHLCV price data via Tiingo REST API.

Tiingo free tier: 100 requests/day, 50 symbols/hour, 30+ years history.
All responses are cached to stay well within limits.

Ref: https://api.tiingo.com/docs/tiingo/daily
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import httpx
import pandas as pd

from quantly.config import get_config
from quantly.data.cache import Cache, TTL_PRICE

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.tiingo.com/tiingo/daily"


def _tiingo_headers() -> dict[str, str]:
    cfg = get_config()
    return {
        "Authorization": f"Token {cfg.tiingo_api_key}",
        "Content-Type": "application/json",
    }


def get_ohlcv(
    ticker: str,
    start: date,
    end: date,
    cache: Cache,
) -> pd.DataFrame:
    """Fetch daily OHLCV for *ticker* between *start* and *end* (inclusive).

    Returns a DataFrame with columns:
        date, open, high, low, close, volume, adj_close, dollar_volume

    Indexed by ``date`` (datetime.date). Empty DataFrame if unavailable.
    """
    cache_key = f"tiingo:ohlcv:{ticker}:{start}:{end}"

    def _fetch() -> pd.DataFrame:
        url = f"{_BASE_URL}/{ticker}/prices"
        params = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "resampleFreq": "daily",
            "token": get_config().tiingo_api_key,
        }
        try:
            resp = httpx.get(url, params=params, headers=_tiingo_headers(), timeout=15)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("Tiingo %s HTTP %s: %s", ticker, exc.response.status_code, exc)
            return pd.DataFrame()
        except httpx.RequestError as exc:
            logger.warning("Tiingo %s request error: %s", ticker, exc)
            return pd.DataFrame()

        data = resp.json()
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.rename(
            columns={
                "adjClose": "adj_close",
                "adjOpen": "adj_open",
                "adjHigh": "adj_high",
                "adjLow": "adj_low",
                "adjVolume": "adj_volume",
            }
        )
        # Keep only the columns we use
        keep = [c for c in ["date", "open", "high", "low", "close", "volume",
                             "adj_close"] if c in df.columns]
        df = df[keep].set_index("date").sort_index()
        df["dollar_volume"] = df["close"] * df["volume"]
        return df

    return cache.get_or_fetch(cache_key, _fetch, ttl_seconds=TTL_PRICE)


def get_current_price(ticker: str, cache: Cache) -> Optional[float]:
    """Return the latest closing price for *ticker*, or None if unavailable."""
    today = date.today()
    start = today - timedelta(days=5)  # buffer for weekends / holidays
    df = get_ohlcv(ticker, start, today, cache)
    if df.empty:
        return None
    return float(df["close"].iloc[-1])


def get_latest_snapshot(tickers: list[str], cache: Cache) -> pd.DataFrame:
    """Return a single-row-per-ticker DataFrame with the most recent OHLCV.

    Columns: ticker (index), close, volume, dollar_volume, 20d_avg_dollar_volume.
    Used by the universe screener.
    """
    today = date.today()
    start = today - timedelta(days=35)  # ~20 trading days + buffer

    rows: list[dict] = []
    for ticker in tickers:
        df = get_ohlcv(ticker, start, today, cache)
        if df.empty or len(df) < 5:
            continue
        last = df.iloc[-1]
        avg_dv = df["dollar_volume"].tail(20).mean()
        rows.append(
            {
                "ticker": ticker,
                "close": float(last["close"]),
                "volume": float(last["volume"]),
                "dollar_volume": float(last["dollar_volume"]),
                "avg_dollar_volume_20d": float(avg_dv),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["ticker", "close", "volume", "dollar_volume",
                     "avg_dollar_volume_20d"]
        ).set_index("ticker")

    result = pd.DataFrame(rows).set_index("ticker")
    # Screener uses the 20-day average for the dollar_volume filter
    result["dollar_volume"] = result["avg_dollar_volume_20d"]
    return result


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range(period) for a price DataFrame.

    Args:
        df: DataFrame with ``high``, ``low``, ``close`` columns, indexed by date.
        period: Look-back window (default 14).

    Returns:
        Series of ATR values, same index as *df*.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()
