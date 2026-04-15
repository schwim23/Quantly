"""StockTwits sentiment — bull/bear ratio per ticker.

StockTwits provides a free public sentiment API for any ticker.
We scrape the sentiment meter (bullish_percent) from the stream endpoint.

Ref: https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from quantly.data.cache import Cache, TTL_SENTIMENT

logger = logging.getLogger(__name__)

_BASE = "https://api.stocktwits.com/api/2/streams/symbol"


def get_stocktwits_sentiment(ticker: str, cache: Cache) -> dict[str, float]:
    """Return StockTwits bull/bear sentiment for *ticker*.

    Returns dict with keys:
        bullish_pct   - fraction of messages that are bullish (0.0–1.0)
        message_count - number of messages in the stream
        sentiment     - bullish_pct mapped to [-1, +1]:  (2 * bullish_pct - 1)
    """
    key = f"stocktwits:sentiment:{ticker}"

    def _fetch() -> dict[str, float]:
        try:
            resp = httpx.get(
                f"{_BASE}/{ticker}.json",
                headers={"User-Agent": "Quantly/1.0"},
                timeout=10,
            )
            if resp.status_code == 429:
                logger.warning("StockTwits rate limit for %s", ticker)
                return {"bullish_pct": 0.5, "message_count": 0, "sentiment": 0.0}
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("StockTwits fetch for %s: %s", ticker, exc)
            return {"bullish_pct": 0.5, "message_count": 0, "sentiment": 0.0}

        messages = data.get("messages", [])
        if not messages:
            return {"bullish_pct": 0.5, "message_count": 0, "sentiment": 0.0}

        bullish = sum(
            1 for m in messages
            if (m.get("entities", {}).get("sentiment") or {}).get("basic") == "Bullish"
        )
        total = len(messages)
        bullish_pct = bullish / total if total > 0 else 0.5
        sentiment = round(2 * bullish_pct - 1, 4)  # maps [0,1] → [-1,+1]

        return {
            "bullish_pct": round(bullish_pct, 4),
            "message_count": total,
            "sentiment": sentiment,
        }

    return cache.get_or_fetch(key, _fetch, ttl_seconds=TTL_SENTIMENT)
