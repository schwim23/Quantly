"""Volatility-scaled position sizing.

Target: each position risks at most max_loss_per_position_pct of portfolio.
Position size = min(target_size, max_size) where target is derived from ATR stop.

Formula:
  dollar_risk_per_share = entry_price - stop_price  (= 2 × ATR)
  shares = (portfolio × max_loss_pct) / dollar_risk_per_share
  position_value = shares × entry_price

Then cap at max_position_pct of portfolio.
Scale down so max_position_pct ≤ target ≤ min_position_pct.
"""
from __future__ import annotations

import math

from quantly.config import get_config


def compute_position_size(
    entry_price: float,
    stop_price: float,
    portfolio_value: float,
    *,
    max_loss_pct: float | None = None,
    min_position_pct: float | None = None,
    max_position_pct: float | None = None,
) -> dict[str, float]:
    """Return recommended position size for a single trade.

    Args:
        entry_price: Planned entry price per share.
        stop_price: Initial ATR-based stop loss price.
        portfolio_value: Current total portfolio value (cash + positions).
        max_loss_pct: Max portfolio fraction to risk on this trade (default from Config).
        min_position_pct: Minimum position size as fraction (default from Config).
        max_position_pct: Maximum position size as fraction (default from Config).

    Returns:
        Dict with:
            shares          - number of whole shares to buy
            position_value  - total dollar value of position
            position_pct    - position_value / portfolio_value
            dollar_risk     - maximum dollar loss if stop is hit
    """
    cfg = get_config()
    max_loss_pct = max_loss_pct if max_loss_pct is not None else cfg.max_loss_per_position_pct
    min_pct = min_position_pct if min_position_pct is not None else cfg.min_position_pct
    max_pct = max_position_pct if max_position_pct is not None else cfg.max_position_pct

    if entry_price <= 0 or stop_price <= 0 or portfolio_value <= 0:
        return {"shares": 0, "position_value": 0.0, "position_pct": 0.0, "dollar_risk": 0.0}

    dollar_risk_per_share = abs(entry_price - stop_price)
    if dollar_risk_per_share < 0.01:
        dollar_risk_per_share = 0.01  # floor to avoid division by near-zero

    max_dollar_risk = portfolio_value * max_loss_pct
    shares_from_risk = max_dollar_risk / dollar_risk_per_share
    position_value_from_risk = shares_from_risk * entry_price

    # Apply min/max caps
    min_value = portfolio_value * min_pct
    max_value = portfolio_value * max_pct

    position_value = max(min_value, min(max_value, position_value_from_risk))
    shares = max(1, math.floor(position_value / entry_price))
    actual_value = shares * entry_price
    actual_risk = shares * dollar_risk_per_share

    return {
        "shares": shares,
        "position_value": round(actual_value, 2),
        "position_pct": round(actual_value / portfolio_value, 4),
        "dollar_risk": round(actual_risk, 2),
    }


def max_concurrent_positions_reached(open_count: int) -> bool:
    """Return True if we are at the maximum allowed concurrent positions."""
    return open_count >= get_config().max_positions
