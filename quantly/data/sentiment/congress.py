"""Congressional trading disclosures via Quiver Quant API.

Congressional trades are a documented alpha source — Members of Congress
historically outperform the market. Recent buys in a stock are a bullish signal.

Quiver Quant free tier: congressional trades endpoint is publicly accessible.

Ref: https://api.quiverquant.com/beta/live/congresstrading
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import httpx

from quantly.config import get_config
from quantly.data.cache import Cache, TTL_SENTIMENT

logger = logging.getLogger(__name__)

_BASE = "https://api.quiverquant.com/beta"


def _quiver_get(endpoint: str) -> Any:
    cfg = get_config()
    headers = {}
    if cfg.quiver_api_key:
        headers["Authorization"] = f"Token {cfg.quiver_api_key}"
    try:
        resp = httpx.get(
            f"{_BASE}/{endpoint}",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Quiver Quant %s error: %s", endpoint, exc)
        return None


def get_congress_sentiment(ticker: str, cache: Cache, lookback_days: int = 90) -> dict[str, Any]:
    """Return recent congressional trade sentiment for *ticker*.

    Returns dict with keys:
        recent_buys      - count of buy transactions in last *lookback_days*
        recent_sells     - count of sell transactions
        net_congress     - buys - sells (positive = congressional bullish)
        latest_buy_days  - days since most recent buy (99 if none)
    """
    key = f"quiver:congress:{ticker}:{lookback_days}"

    def _fetch() -> dict[str, Any]:
        data = _quiver_get(f"live/congresstrading/{ticker}")

        if not isinstance(data, list):
            return {
                "recent_buys": 0, "recent_sells": 0,
                "net_congress": 0, "latest_buy_days": 99,
            }

        cutoff = date.today() - timedelta(days=lookback_days)
        buys, sells = 0, 0
        latest_buy_date: date | None = None

        for trade in data:
            try:
                trade_date = date.fromisoformat(trade.get("Date", "")[:10])
            except (ValueError, TypeError):
                continue
            if trade_date < cutoff:
                continue

            tx_type = (trade.get("Transaction") or "").lower()
            if "purchase" in tx_type or "buy" in tx_type:
                buys += 1
                if latest_buy_date is None or trade_date > latest_buy_date:
                    latest_buy_date = trade_date
            elif "sale" in tx_type or "sell" in tx_type:
                sells += 1

        latest_buy_days = (
            (date.today() - latest_buy_date).days
            if latest_buy_date else 99
        )

        return {
            "recent_buys": buys,
            "recent_sells": sells,
            "net_congress": buys - sells,
            "latest_buy_days": latest_buy_days,
        }

    return cache.get_or_fetch(key, _fetch, ttl_seconds=TTL_SENTIMENT)
