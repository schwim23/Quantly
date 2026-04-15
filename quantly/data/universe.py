"""Universe screener — fetches the S&P 500 + Russell 1000 ticker list and applies
liquidity / price filters to produce the daily candidate universe (~300–500 tickers).

Data source: Wikipedia S&P 500 table + iShares IWB holdings CSV (Russell 1000).
Both are free and require no API key.
Results are cached for 24 hours.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx
import pandas as pd

from quantly.config import get_config
from quantly.data.cache import Cache, TTL_UNIVERSE

logger = logging.getLogger(__name__)

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_IWB_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
)


def _fetch_sp500_tickers() -> list[str]:
    """Scrape S&P 500 constituent tickers from Wikipedia."""
    try:
        tables = pd.read_html(_SP500_URL)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        return [t.strip() for t in tickers if isinstance(t, str)]
    except Exception as exc:
        logger.warning("Failed to fetch S&P 500 list: %s", exc)
        return []


def _fetch_russell1000_tickers() -> list[str]:
    """Fetch Russell 1000 tickers from iShares ETF holdings CSV."""
    try:
        resp = httpx.get(_IWB_HOLDINGS_URL, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        # iShares CSV has metadata rows at the top; find the header row
        lines = resp.text.splitlines()
        header_idx = next(
            (i for i, line in enumerate(lines) if "Ticker" in line), None
        )
        if header_idx is None:
            return []
        csv_text = "\n".join(lines[header_idx:])
        df = pd.read_csv(pd.io.common.StringIO(csv_text))
        col = next((c for c in df.columns if "Ticker" in c), None)
        if col is None:
            return []
        tickers = df[col].dropna().tolist()
        return [
            str(t).strip().replace(".", "-")
            for t in tickers
            if str(t).strip() not in ("", "-", "nan")
        ]
    except Exception as exc:
        logger.warning("Failed to fetch Russell 1000 list: %s", exc)
        return []


def get_raw_universe(cache: Cache) -> list[str]:
    """Return deduplicated S&P 500 ∪ Russell 1000 ticker list (cached 24 h)."""
    def _fetch() -> list[str]:
        sp500 = _fetch_sp500_tickers()
        r1000 = _fetch_russell1000_tickers()
        combined = sorted(set(sp500) | set(r1000))
        logger.info(
            "Universe: %d S&P500 + %d Russell1000 = %d unique tickers",
            len(sp500),
            len(r1000),
            len(combined),
        )
        return combined

    return cache.get_or_fetch("universe:raw", _fetch, ttl_seconds=TTL_UNIVERSE)


def screen_universe(
    tickers: list[str],
    prices_df: pd.DataFrame,
    *,
    min_price: Optional[float] = None,
    min_dollar_volume: Optional[float] = None,
) -> list[str]:
    """Apply liquidity and price filters to a list of tickers.

    Args:
        tickers: Candidate ticker list.
        prices_df: DataFrame indexed by ticker with columns
                   ``close`` and ``dollar_volume`` (20-day avg).
        min_price: Minimum close price (default from Config).
        min_dollar_volume: Minimum avg daily dollar volume (default from Config).

    Returns:
        Filtered list of tickers that pass all screens.
    """
    cfg = get_config()
    min_price = min_price if min_price is not None else cfg.min_price
    min_dollar_volume = (
        min_dollar_volume if min_dollar_volume is not None else cfg.min_dollar_volume
    )

    if prices_df.empty:
        return []

    available = prices_df.index.intersection(tickers)
    df = prices_df.loc[available].copy()

    passed = df[
        (df["close"] >= min_price) & (df["dollar_volume"] >= min_dollar_volume)
    ].index.tolist()

    logger.info(
        "Screen: %d → %d tickers (price ≥ $%.0f, vol ≥ $%.0fM)",
        len(tickers),
        len(passed),
        min_price,
        min_dollar_volume / 1_000_000,
    )
    return passed
