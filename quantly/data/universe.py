"""US equity universe — fetched entirely from free sources.

Sources used (no API key required):
  - SEC EDGAR company_tickers.json  → full list of all SEC-registered US companies
  - Wikipedia                        → S&P 500 and NASDAQ 100 constituent lists
  - yfinance                         → volume / market-cap metadata for screening
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

from quantly.data.cache import cached

logger = logging.getLogger(__name__)

_EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_HEADERS = {"User-Agent": "quantly research@quantly.local"}


@cached("universe_edgar", ttl=86_400 * 7)  # refresh weekly
def get_edgar_universe() -> pd.DataFrame:
    """Fetch all SEC-registered US companies from EDGAR (free, no key).

    Returns:
        DataFrame with columns: ticker, cik, name
    """
    resp = requests.get(_EDGAR_TICKERS_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rows = [
        {"ticker": v["ticker"], "cik": str(v["cik_str"]).zfill(10), "name": v["title"]}
        for v in data.values()
        if v.get("ticker")
    ]
    df = pd.DataFrame(rows)
    logger.info("EDGAR universe: %d tickers", len(df))
    return df


@cached("universe_sp500", ttl=86_400 * 7)
def get_sp500_tickers() -> list[str]:
    """Return current S&P 500 tickers from Wikipedia (free, no key)."""
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        attrs={"id": "constituents"},
    )
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    logger.info("S&P 500: %d tickers", len(tickers))
    return tickers


@cached("universe_nasdaq100", ttl=86_400 * 7)
def get_nasdaq100_tickers() -> list[str]:
    """Return current NASDAQ 100 tickers from Wikipedia (free, no key)."""
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        attrs={"id": "constituents"},
    )
    tickers = tables[0]["Ticker"].str.replace(".", "-", regex=False).tolist()
    logger.info("NASDAQ 100: %d tickers", len(tickers))
    return tickers


def get_universe_tickers(universe: str = "sp500") -> list[str]:
    """Return tickers for the requested universe.

    Args:
        universe: "sp500" | "nasdaq100" | "all_us"

    Returns:
        List of ticker strings
    """
    if universe == "sp500":
        return get_sp500_tickers()
    if universe == "nasdaq100":
        return get_nasdaq100_tickers()
    if universe == "all_us":
        df = get_edgar_universe()
        return df["ticker"].tolist()
    raise ValueError(f"Unknown universe: {universe!r}. Use sp500 | nasdaq100 | all_us")


def screen_by_liquidity(
    tickers: list[str],
    min_volume: int = 500_000,
    min_price: float = 5.0,
    batch_size: int = 100,
) -> list[str]:
    """Filter tickers by average daily volume and minimum price using yfinance.

    Fetches 5-day summary data in batches to stay within yfinance rate limits.

    Args:
        tickers: Input ticker list
        min_volume: Minimum average daily volume
        min_price: Minimum share price (filters out penny stocks)
        batch_size: Tickers per yfinance batch call

    Returns:
        Filtered list of tickers passing liquidity criteria
    """
    import yfinance as yf

    passing: list[str] = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            data = yf.download(
                batch,
                period="5d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if data.empty:
                continue
            close = data["Close"] if "Close" in data else data.get("close", pd.DataFrame())
            volume = data["Volume"] if "Volume" in data else data.get("volume", pd.DataFrame())
            if close.empty or volume.empty:
                continue
            avg_close = close.mean()
            avg_volume = volume.mean()
            for ticker in batch:
                try:
                    price = avg_close.get(ticker, 0) if hasattr(avg_close, "get") else avg_close
                    vol = avg_volume.get(ticker, 0) if hasattr(avg_volume, "get") else avg_volume
                    if price >= min_price and vol >= min_volume:
                        passing.append(ticker)
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Batch screen failed for batch starting at %d: %s", i, e)

    logger.info("Liquidity screen: %d / %d tickers passed", len(passing), len(tickers))
    return passing
