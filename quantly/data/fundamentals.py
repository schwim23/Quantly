"""Fundamental financial data via Financial Modeling Prep (FMP) API.

FMP free tier: 250 requests/day. We cache aggressively (24 h) to stay within limits.
Provides: income statement, balance sheet, cash flow, key metrics, earnings history.

Ref: https://site.financialmodelingprep.com/developer/docs
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
import pandas as pd

from quantly.config import get_config
from quantly.data.cache import Cache, TTL_FUNDAMENTAL

logger = logging.getLogger(__name__)

_BASE = "https://financialmodelingprep.com/api/v3"


def _get(endpoint: str, params: Optional[dict] = None) -> Any:
    """Make a single FMP GET request. Returns parsed JSON or None on error."""
    cfg = get_config()
    url = f"{_BASE}/{endpoint}"
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


# ── Income Statement ──────────────────────────────────────────────────────────

def get_income_statements(ticker: str, cache: Cache, limit: int = 8) -> list[dict]:
    """Return up to *limit* quarters of income statement data, newest first."""
    key = f"fmp:income:{ticker}:{limit}"
    def _fetch() -> list[dict]:
        data = _get(f"income-statement/{ticker}", {"period": "quarter", "limit": limit})
        return data if isinstance(data, list) else []
    return cache.get_or_fetch(key, _fetch, ttl_seconds=TTL_FUNDAMENTAL)


# ── Balance Sheet ─────────────────────────────────────────────────────────────

def get_balance_sheets(ticker: str, cache: Cache, limit: int = 4) -> list[dict]:
    """Return up to *limit* quarters of balance sheet data, newest first."""
    key = f"fmp:balance:{ticker}:{limit}"
    def _fetch() -> list[dict]:
        data = _get(f"balance-sheet-statement/{ticker}", {"period": "quarter", "limit": limit})
        return data if isinstance(data, list) else []
    return cache.get_or_fetch(key, _fetch, ttl_seconds=TTL_FUNDAMENTAL)


# ── Cash Flow ─────────────────────────────────────────────────────────────────

def get_cash_flows(ticker: str, cache: Cache, limit: int = 4) -> list[dict]:
    """Return up to *limit* quarters of cash flow data, newest first."""
    key = f"fmp:cashflow:{ticker}:{limit}"
    def _fetch() -> list[dict]:
        data = _get(f"cash-flow-statement/{ticker}", {"period": "quarter", "limit": limit})
        return data if isinstance(data, list) else []
    return cache.get_or_fetch(key, _fetch, ttl_seconds=TTL_FUNDAMENTAL)


# ── Key Metrics ───────────────────────────────────────────────────────────────

def get_key_metrics(ticker: str, cache: Cache, limit: int = 4) -> list[dict]:
    """Return up to *limit* quarters of key metrics (FCF yield, D/E, ROE, etc.)."""
    key = f"fmp:metrics:{ticker}:{limit}"
    def _fetch() -> list[dict]:
        data = _get(f"key-metrics/{ticker}", {"period": "quarter", "limit": limit})
        return data if isinstance(data, list) else []
    return cache.get_or_fetch(key, _fetch, ttl_seconds=TTL_FUNDAMENTAL)


# ── Earnings History ──────────────────────────────────────────────────────────

def get_earnings_history(ticker: str, cache: Cache, limit: int = 8) -> list[dict]:
    """Return historical EPS actuals vs. estimates, newest first."""
    key = f"fmp:earnings:{ticker}:{limit}"
    def _fetch() -> list[dict]:
        data = _get(f"historical/earning_calendar/{ticker}", {"limit": limit})
        return data if isinstance(data, list) else []
    return cache.get_or_fetch(key, _fetch, ttl_seconds=TTL_FUNDAMENTAL)


# ── Analyst Estimates ─────────────────────────────────────────────────────────

def get_analyst_estimates(ticker: str, cache: Cache, limit: int = 4) -> list[dict]:
    """Return analyst EPS/revenue estimates for upcoming quarters."""
    key = f"fmp:estimates:{ticker}:{limit}"
    def _fetch() -> list[dict]:
        data = _get(f"analyst-estimates/{ticker}", {"period": "quarter", "limit": limit})
        return data if isinstance(data, list) else []
    return cache.get_or_fetch(key, _fetch, ttl_seconds=TTL_FUNDAMENTAL)


# ── Computed fundamentals summary ────────────────────────────────────────────

def get_fundamentals_summary(ticker: str, cache: Cache) -> dict[str, Any]:
    """Return a flat dict of fundamental metrics ready for the feature pipeline.

    Metrics returned:
        revenue_growth_yoy      - YoY quarterly revenue growth (latest vs. year-ago)
        revenue_growth_qoq      - QoQ quarterly revenue growth
        gross_margin            - Latest gross margin
        gross_margin_trend      - +1 expanding, -1 contracting, 0 flat
        fcf_yield               - Free cash flow / market cap (approx via FCF / revenue)
        eps_growth_slope        - Linear slope of last 4 quarters EPS (normalised)
        debt_to_equity          - Latest D/E ratio
        eps_surprise_last       - Last EPS surprise %
        eps_surprise_prev       - Prior EPS surprise %
        has_positive_fcf        - 1 if TTM FCF > 0 else 0
        revenue_growth_gt10     - 1 if YoY revenue growth > 10% else 0
    """
    key = f"fmp:summary:{ticker}"

    def _build() -> dict[str, Any]:
        income = get_income_statements(ticker, cache, limit=8)
        cashflows = get_cash_flows(ticker, cache, limit=4)
        metrics = get_key_metrics(ticker, cache, limit=4)
        earnings = get_earnings_history(ticker, cache, limit=4)

        result: dict[str, Any] = {
            "revenue_growth_yoy": 0.0,
            "revenue_growth_qoq": 0.0,
            "gross_margin": 0.0,
            "gross_margin_trend": 0,
            "fcf_yield": 0.0,
            "eps_growth_slope": 0.0,
            "debt_to_equity": 0.0,
            "eps_surprise_last": 0.0,
            "eps_surprise_prev": 0.0,
            "has_positive_fcf": 0,
            "revenue_growth_gt10": 0,
        }

        if len(income) >= 5:
            rev = [q.get("revenue", 0) or 0 for q in income]
            result["revenue_growth_yoy"] = (
                (rev[0] - rev[4]) / abs(rev[4]) if rev[4] != 0 else 0.0
            )
            result["revenue_growth_qoq"] = (
                (rev[0] - rev[1]) / abs(rev[1]) if rev[1] != 0 else 0.0
            )
            gm = [
                (q.get("grossProfit", 0) or 0) / (q.get("revenue", 1) or 1)
                for q in income[:4]
            ]
            result["gross_margin"] = gm[0]
            result["gross_margin_trend"] = (
                1 if gm[0] > gm[-1] else (-1 if gm[0] < gm[-1] else 0)
            )
            eps = [q.get("eps", 0) or 0 for q in income[:4]]
            if len(eps) == 4:
                import numpy as np
                xs = range(len(eps))
                slope = float(np.polyfit(list(xs), eps[::-1], 1)[0])
                result["eps_growth_slope"] = slope

        if cashflows:
            fcf_vals = [q.get("freeCashFlow", 0) or 0 for q in cashflows]
            ttm_fcf = sum(fcf_vals[:4])
            result["has_positive_fcf"] = 1 if ttm_fcf > 0 else 0
            rev0 = income[0].get("revenue", 1) or 1 if income else 1
            result["fcf_yield"] = ttm_fcf / (rev0 * 4) if rev0 else 0.0

        if metrics:
            result["debt_to_equity"] = float(metrics[0].get("debtToEquity", 0) or 0)

        if len(earnings) >= 2:
            def _surprise(e: dict) -> float:
                actual = e.get("actualEarningResult") or e.get("actual", 0) or 0
                est = e.get("estimatedEarning") or e.get("estimate", 0) or 0
                return (actual - est) / abs(est) if est else 0.0

            result["eps_surprise_last"] = _surprise(earnings[0])
            result["eps_surprise_prev"] = _surprise(earnings[1])

        result["revenue_growth_gt10"] = (
            1 if result["revenue_growth_yoy"] > 0.10 else 0
        )
        return result

    return cache.get_or_fetch(key, _build, ttl_seconds=TTL_FUNDAMENTAL)


def passes_fundamentals_gate(summary: dict[str, Any]) -> bool:
    """Return True if ticker clears the minimum quality bar for final picks.

    Rules (from strategy spec):
    - Positive TTM FCF OR revenue growth > 10% YoY (growth exception)
    - D/E ratio not extreme (we use > 10 as a heuristic red flag)
    """
    if not summary:
        return False
    fcf_ok = summary.get("has_positive_fcf", 0) == 1
    growth_ok = summary.get("revenue_growth_gt10", 0) == 1
    de_ok = summary.get("debt_to_equity", 0) <= 10.0
    return (fcf_ok or growth_ok) and de_ok
