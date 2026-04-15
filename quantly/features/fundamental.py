"""Fundamental features — used as a gate signal and weak ML feature.

These are intentionally low-weight in the model; their primary role is
the fundamentals gate (passes_fundamentals_gate) not ranking.
"""
from __future__ import annotations

from quantly.data.cache import Cache
from quantly.data.fundamentals import get_fundamentals_summary


def compute_fundamental_features(ticker: str, cache: Cache) -> dict[str, float]:
    """Return fundamental features for *ticker* from the cached FMP summary.

    Features:
        revenue_growth_yoy   - YoY quarterly revenue growth (fraction)
        revenue_growth_qoq   - QoQ quarterly revenue growth
        gross_margin         - latest gross margin (fraction)
        gross_margin_trend   - +1 expanding, -1 contracting, 0 flat
        fcf_yield            - free cash flow / revenue proxy
        eps_growth_slope     - linear slope of last 4 quarters EPS
        debt_to_equity       - D/E ratio
        eps_surprise_last    - last EPS surprise fraction
        eps_surprise_prev    - prior EPS surprise fraction
        has_positive_fcf     - 1 if TTM FCF > 0
    """
    summary = get_fundamentals_summary(ticker, cache)

    return {
        "revenue_growth_yoy": float(summary.get("revenue_growth_yoy", 0.0)),
        "revenue_growth_qoq": float(summary.get("revenue_growth_qoq", 0.0)),
        "gross_margin": float(summary.get("gross_margin", 0.0)),
        "gross_margin_trend": float(summary.get("gross_margin_trend", 0)),
        "fcf_yield": float(summary.get("fcf_yield", 0.0)),
        "eps_growth_slope": float(summary.get("eps_growth_slope", 0.0)),
        "debt_to_equity": float(summary.get("debt_to_equity", 0.0)),
        "eps_surprise_last": float(summary.get("eps_surprise_last", 0.0)),
        "eps_surprise_prev": float(summary.get("eps_surprise_prev", 0.0)),
        "has_positive_fcf": float(summary.get("has_positive_fcf", 0)),
    }
