"""Catalyst features — earnings proximity, surprise history, analyst actions.

These are high-weight signals in the ML model alongside sentiment.
"""
from __future__ import annotations

from typing import Any

from quantly.data.cache import Cache
from quantly.data.catalysts import (
    get_analyst_actions,
    get_days_to_earnings,
    get_eps_surprise_streak,
)
from quantly.data.fundamentals import get_earnings_history


def compute_catalyst_features(
    ticker: str,
    cache: Cache,
    upcoming_earnings: dict,  # dict[str, date] from get_upcoming_earnings()
) -> dict[str, float]:
    """Return catalyst features for *ticker*.

    Features:
        days_to_earnings        - trading days until next earnings (99 if not scheduled)
        earnings_proximity      - 1 / (days_to_earnings + 1), peaks at announcement
        eps_beat_streak         - consecutive quarters with positive EPS surprise
        eps_surprise_last       - last quarter EPS surprise as fraction
        eps_surprise_prev       - prior quarter EPS surprise
        analyst_upgrades_10d    - upgrade count in last 10 days
        analyst_downgrades_10d  - downgrade count in last 10 days
        analyst_net_10d         - net analyst sentiment (upgrades - downgrades)
    """
    days = get_days_to_earnings(ticker, upcoming_earnings)
    earnings_hist = get_earnings_history(ticker, cache, limit=8)
    streak = get_eps_surprise_streak(earnings_hist)
    analyst = get_analyst_actions(ticker, cache, lookback_days=10)

    # EPS surprises from history
    def _surprise(entry: dict) -> float:
        actual = entry.get("actualEarningResult") or entry.get("actual") or 0
        est = entry.get("estimatedEarning") or entry.get("estimate") or 0
        try:
            return (float(actual) - float(est)) / abs(float(est)) if est else 0.0
        except (TypeError, ZeroDivisionError):
            return 0.0

    eps_last = _surprise(earnings_hist[0]) if len(earnings_hist) > 0 else 0.0
    eps_prev = _surprise(earnings_hist[1]) if len(earnings_hist) > 1 else 0.0

    return {
        "days_to_earnings": float(days),
        "earnings_proximity": round(1.0 / (days + 1), 4),
        "eps_beat_streak": float(streak),
        "eps_surprise_last": round(eps_last, 4),
        "eps_surprise_prev": round(eps_prev, 4),
        "analyst_upgrades_10d": float(analyst["upgrades"]),
        "analyst_downgrades_10d": float(analyst["downgrades"]),
        "analyst_net_10d": float(analyst["net"]),
    }
