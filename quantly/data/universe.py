"""Fetch and screen the US equity universe.

Stage 1 of the pipeline: returns a list of tickers passing
liquidity and volume thresholds.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from quantly.config import config
from quantly.data.cache import cached

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@cached("universe")
def get_us_equity_universe() -> pd.DataFrame:
    """Return all tradeable US equities from Polygon.io.

    Returns:
        DataFrame with columns: ticker, name, market_cap, avg_volume, sector, industry
    """
    # TODO: implement Polygon.io /v3/reference/tickers pagination
    raise NotImplementedError


def screen_universe(
    universe: pd.DataFrame,
    min_volume: int = config.MIN_VOLUME,
    min_market_cap: float = 100e6,
    exclude_otc: bool = True,
) -> pd.DataFrame:
    """Apply liquidity and quality filters to the full universe.

    Args:
        universe: Raw universe DataFrame from get_us_equity_universe()
        min_volume: Minimum average daily volume
        min_market_cap: Minimum market cap in dollars
        exclude_otc: Whether to exclude OTC-traded securities

    Returns:
        Filtered DataFrame of candidate tickers
    """
    df = universe.copy()
    df = df[df["avg_volume"] >= min_volume]
    df = df[df["market_cap"] >= min_market_cap]
    if exclude_otc:
        df = df[df["exchange"].isin(["NYSE", "NASDAQ", "AMEX"])]
    logger.info("Universe screened: %d candidates (from %d)", len(df), len(universe))
    return df


def get_sp500_tickers() -> list[str]:
    """Return current S&P 500 tickers (from Wikipedia, as a fast alternative)."""
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return tables[0]["Symbol"].tolist()
