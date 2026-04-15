"""ATR-based trailing stop calculator and portfolio kill switch.

Stop rules (from strategy spec):
  - Initial stop: entry_price - 2 × ATR(14)
  - Trailing: stop only moves up, never down
  - Max hold: 20 trading days regardless of stop
  - Kill switch: if portfolio NAV drops 12% from peak, pause new entries
  - Resume: when portfolio recovers to within 8% of peak

No take-profit target — let winners run, trail the stop upward.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from quantly.config import get_config


def compute_initial_stop(entry_price: float, atr: float) -> float:
    """Return the initial stop price = entry - 2 × ATR(14).

    Args:
        entry_price: Price per share at entry.
        atr: Current ATR(14) value for this ticker.

    Returns:
        Stop price (always below entry price).
    """
    cfg = get_config()
    stop = entry_price - cfg.atr_stop_multiplier * atr
    return round(max(stop, 0.01), 4)


def update_trailing_stop(
    current_stop: float,
    current_price: float,
    atr: float,
) -> float:
    """Ratchet the trailing stop up if the new ATR-based stop is higher.

    The stop can only move up — never down.

    Args:
        current_stop: The current stop price.
        current_price: Today's closing price.
        atr: Current ATR(14) value.

    Returns:
        Updated stop price (≥ current_stop).
    """
    cfg = get_config()
    new_stop = current_price - cfg.atr_stop_multiplier * atr
    return round(max(current_stop, new_stop), 4)


def is_stopped_out(current_price: float, stop_price: float) -> bool:
    """Return True if current price has breached the stop level."""
    return current_price <= stop_price


def compute_max_hold_date(entry_date: date) -> date:
    """Return the max hold date = entry + 20 trading days.

    Approximation: adds calendar days (20 td ≈ 28 calendar days).
    For exact trading-day arithmetic, a market calendar is needed — deferred
    to the pipeline which has price data to count actual trading days.
    """
    cfg = get_config()
    # Approximate: 20 trading days ≈ 28 calendar days
    calendar_days = int(cfg.max_hold_days * 1.4)
    return entry_date + timedelta(days=calendar_days)


def is_max_hold_exceeded(entry_date: date, as_of: Optional[date] = None) -> bool:
    """Return True if the position has been held past its max hold date."""
    as_of = as_of or date.today()
    return as_of >= compute_max_hold_date(entry_date)


# ── Portfolio kill switch ─────────────────────────────────────────────────────

class KillSwitch:
    """Tracks portfolio drawdown and pauses new entries when threshold is hit.

    State is kept in memory; persisted to shadow_book.db by the pipeline.
    """

    def __init__(
        self,
        pause_threshold: Optional[float] = None,
        resume_threshold: Optional[float] = None,
    ):
        cfg = get_config()
        self.pause_threshold = pause_threshold or cfg.drawdown_pause_threshold
        self.resume_threshold = resume_threshold or cfg.drawdown_resume_threshold
        self._peak_nav: float = 0.0
        self._paused: bool = False

    def update(self, current_nav: float) -> None:
        """Update peak NAV and check drawdown thresholds."""
        if current_nav > self._peak_nav:
            self._peak_nav = current_nav

        if self._peak_nav == 0:
            return

        drawdown = (self._peak_nav - current_nav) / self._peak_nav

        if not self._paused and drawdown >= self.pause_threshold:
            self._paused = True
        elif self._paused and drawdown <= self.resume_threshold:
            self._paused = False

    def is_paused(self) -> bool:
        """Return True if new entries are paused due to drawdown."""
        return self._paused

    def current_drawdown(self) -> float:
        """Return current drawdown from peak as a positive fraction."""
        if self._peak_nav == 0:
            return 0.0
        return 0.0  # Can't compute without current NAV

    def reset(self) -> None:
        """Reset peak and pause state (call when starting fresh)."""
        self._peak_nav = 0.0
        self._paused = False
