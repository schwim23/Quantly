"""Unusual options flow via Quiver Quant (free tier).

Unusual call activity relative to 30-day average is a leading indicator
of institutional positioning — one of our highest-weight sentiment signals.

Falls back gracefully if API key is not set.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import httpx

from quantly.config import get_config
from quantly.data.cache import Cache, TTL_SENTIMENT

logger = logging.getLogger(__name__)

_QUIVER_BASE = "https://api.quiverquant.com/beta"


def _quiver_get(endpoint: str) -> Any:
    cfg = get_config()
    headers = {}
    if cfg.quiver_api_key:
        headers["Authorization"] = f"Token {cfg.quiver_api_key}"
    try:
        resp = httpx.get(f"{_QUIVER_BASE}/{endpoint}", headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Quiver options flow %s: %s", endpoint, exc)
        return None


def get_options_flow(ticker: str, cache: Cache, lookback_days: int = 10) -> dict[str, Any]:
    """Return unusual options flow metrics for *ticker*.

    Returns dict with keys:
        unusual_call_ratio  - call volume today vs 30-day avg (>1.5 = unusual)
        put_call_ratio      - puts / calls (< 0.7 = bullish skew)
        net_options_score   - composite score: high calls + low P/C = positive
    """
    key = f"quiver:options:{ticker}:{lookback_days}"

    def _fetch() -> dict[str, Any]:
        data = _quiver_get(f"live/options/{ticker}")

        if not isinstance(data, list) or not data:
            return {
                "unusual_call_ratio": 1.0,
                "put_call_ratio": 1.0,
                "net_options_score": 0.0,
            }

        cutoff = date.today() - timedelta(days=lookback_days)
        recent_calls, recent_puts = [], []
        all_calls, all_puts = [], []

        for row in data:
            try:
                row_date = date.fromisoformat(str(row.get("Date", ""))[:10])
            except (ValueError, TypeError):
                continue

            calls = float(row.get("CallVolume", 0) or 0)
            puts = float(row.get("PutVolume", 0) or 0)
            all_calls.append(calls)
            all_puts.append(puts)

            if row_date >= cutoff:
                recent_calls.append(calls)
                recent_puts.append(puts)

        if not recent_calls:
            return {
                "unusual_call_ratio": 1.0,
                "put_call_ratio": 1.0,
                "net_options_score": 0.0,
            }

        avg_calls_30d = sum(all_calls) / len(all_calls) if all_calls else 1.0
        recent_avg_calls = sum(recent_calls) / len(recent_calls)
        recent_avg_puts = sum(recent_puts) / len(recent_puts) if recent_puts else 0.0

        unusual_call_ratio = recent_avg_calls / avg_calls_30d if avg_calls_30d > 0 else 1.0
        put_call_ratio = recent_avg_puts / recent_avg_calls if recent_avg_calls > 0 else 1.0

        # Net score: unusual call activity and low P/C = bullish
        net_options_score = (unusual_call_ratio - 1.0) - (put_call_ratio - 1.0)

        return {
            "unusual_call_ratio": round(unusual_call_ratio, 3),
            "put_call_ratio": round(put_call_ratio, 3),
            "net_options_score": round(net_options_score, 3),
        }

    return cache.get_or_fetch(key, _fetch, ttl_seconds=TTL_SENTIMENT)
