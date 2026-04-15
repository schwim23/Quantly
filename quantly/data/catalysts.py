"""Catalyst data — earnings dates, EPS calendars, analyst actions.

Sources:
  - Finnhub (free tier): earnings calendar, upcoming events
  - FMP (free tier): earnings surprises, analyst upgrades/downgrades

Ref:
  https://finnhub.io/docs/api/earnings-calendar
  https://site.financialmodelingprep.com/developer/docs
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Optional

import httpx

from quantly.config import get_config
from quantly.data.cache import Cache, TTL_CATALYST

logger = logging.getLogger(__name__)

_FINNHUB_BASE = "https://finnhub.io/api/v1"
_FMP_BASE = "https://financialmodelingprep.com/api/v3"


# ── Finnhub helpers ───────────────────────────────────────────────────────────

def _finnhub_get(endpoint: str, params: Optional[dict] = None) -> Any:
    cfg = get_config()
    url = f"{_FINNHUB_BASE}/{endpoint}"
    p = {"token": cfg.finnhub_api_key}
    if params:
        p.update(params)
    try:
        resp = httpx.get(url, params=p, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Finnhub %s error: %s", endpoint, exc)
        return None


def _fmp_get(endpoint: str, params: Optional[dict] = None) -> Any:
    cfg = get_config()
    url = f"{_FMP_BASE}/{endpoint}"
    p = {"apikey": cfg.fmp_api_key}
    if params:
        p.update(params)
    try:
        resp = httpx.get(url, params=p, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("FMP %s error: %s", endpoint, exc)
        return None


# ── Earnings calendar ─────────────────────────────────────────────────────────

def get_upcoming_earnings(
    tickers: list[str],
    cache: Cache,
    lookahead_days: int = 30,
) -> dict[str, date]:
    """Return a mapping of ticker → next earnings date for the given tickers.

    Uses Finnhub earnings calendar endpoint.
    """
    today = date.today()
    end = today + timedelta(days=lookahead_days)
    cache_key = f"finnhub:earnings_calendar:{today}:{lookahead_days}"

    def _fetch() -> dict[str, str]:
        data = _finnhub_get(
            "calendar/earnings",
            {"from": today.isoformat(), "to": end.isoformat()},
        )
        if not data or "earningsCalendar" not in data:
            return {}
        result: dict[str, str] = {}
        for item in data["earningsCalendar"]:
            sym = item.get("symbol", "")
            dt = item.get("date", "")
            if sym and dt and sym not in result:
                result[sym] = dt
        return result

    raw: dict[str, str] = cache.get_or_fetch(
        cache_key, _fetch, ttl_seconds=TTL_CATALYST
    )

    out: dict[str, date] = {}
    for ticker in tickers:
        if ticker in raw:
            try:
                out[ticker] = date.fromisoformat(raw[ticker])
            except ValueError:
                pass
    return out


def get_days_to_earnings(ticker: str, upcoming: dict[str, date]) -> int:
    """Return trading days until next earnings, or 99 if none scheduled in window."""
    if ticker not in upcoming:
        return 99
    delta = (upcoming[ticker] - date.today()).days
    return max(0, delta)


# ── Analyst upgrades / downgrades ─────────────────────────────────────────────

def get_analyst_actions(ticker: str, cache: Cache, lookback_days: int = 10) -> dict[str, Any]:
    """Return analyst upgrade/downgrade summary for *ticker* in last *lookback_days*.

    Returns dict with:
        upgrades   - count of upgrades
        downgrades - count of downgrades
        net        - upgrades - downgrades  (+ve = bullish analyst sentiment)
    """
    key = f"fmp:analyst_actions:{ticker}:{lookback_days}"

    def _fetch() -> dict[str, Any]:
        data = _fmp_get(f"upgrades-downgrades/{ticker}")
        if not isinstance(data, list):
            return {"upgrades": 0, "downgrades": 0, "net": 0}

        cutoff = date.today() - timedelta(days=lookback_days)
        upgrades = 0
        downgrades = 0
        for item in data:
            try:
                action_date = date.fromisoformat(item.get("publishedDate", "")[:10])
            except (ValueError, TypeError):
                continue
            if action_date < cutoff:
                continue
            action = (item.get("newGrade") or item.get("action") or "").lower()
            if any(w in action for w in ("upgrade", "buy", "outperform", "overweight")):
                upgrades += 1
            elif any(w in action for w in ("downgrade", "sell", "underperform", "underweight")):
                downgrades += 1

        return {"upgrades": upgrades, "downgrades": downgrades, "net": upgrades - downgrades}

    return cache.get_or_fetch(key, _fetch, ttl_seconds=TTL_CATALYST)


# ── Earnings surprise streak ──────────────────────────────────────────────────

def get_eps_surprise_streak(earnings_history: list[dict]) -> int:
    """Return count of consecutive quarters with positive EPS surprise (beat).

    Args:
        earnings_history: List of dicts from fundamentals.get_earnings_history(),
                          sorted newest first.

    Returns:
        Integer ≥ 0. E.g. 3 means last 3 quarters all beat estimates.
    """
    streak = 0
    for q in earnings_history:
        actual = q.get("actualEarningResult") or q.get("actual") or 0
        est = q.get("estimatedEarning") or q.get("estimate") or 0
        if actual is None or est is None:
            break
        if float(actual) > float(est):
            streak += 1
        else:
            break
    return streak
