"""Tests for risk management — position sizing, ATR stops, kill switch."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from quantly.risk.sizing import compute_position_size, max_concurrent_positions_reached
from quantly.risk.stops import (
    KillSwitch,
    compute_initial_stop,
    compute_max_hold_date,
    is_max_hold_exceeded,
    is_stopped_out,
    update_trailing_stop,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Position sizing
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputePositionSize:
    def test_basic_sizing(self):
        result = compute_position_size(
            entry_price=100.0,
            stop_price=95.0,   # $5 risk per share
            portfolio_value=10_000.0,
            max_loss_pct=0.02,   # 2% = $200 max risk
            min_position_pct=0.07,
            max_position_pct=0.10,
        )
        # $200 risk / $5 per share = 40 shares = $4000 position
        # But that's 40% of portfolio — exceeds max 10%, so capped at $1000
        assert result["shares"] >= 1
        assert result["position_pct"] <= 0.10 + 0.001  # within max cap

    def test_dollar_risk_within_max(self):
        result = compute_position_size(
            entry_price=50.0,
            stop_price=45.0,
            portfolio_value=10_000.0,
            max_loss_pct=0.02,
        )
        # Max dollar risk = $200; actual risk should be ≤ $200 + rounding
        assert result["dollar_risk"] <= 210.0  # allow $10 rounding buffer

    def test_zero_entry_price_returns_zeros(self):
        result = compute_position_size(0.0, 0.0, 10_000.0)
        assert result["shares"] == 0
        assert result["position_value"] == 0.0

    def test_position_at_least_min_pct(self):
        result = compute_position_size(
            entry_price=100.0,
            stop_price=99.9,    # tiny risk per share → huge share count → capped at max
            portfolio_value=10_000.0,
            max_loss_pct=0.02,
            min_position_pct=0.07,
            max_position_pct=0.10,
        )
        assert result["position_pct"] >= 0.065  # at least near min (floor applied)

    def test_high_beta_stock_scaled_down(self):
        """High-vol stock (wide ATR stop) gets smaller position."""
        low_vol = compute_position_size(100.0, 98.0, 10_000.0,
                                         max_loss_pct=0.02, min_position_pct=0.07,
                                         max_position_pct=0.10)
        high_vol = compute_position_size(100.0, 85.0, 10_000.0,
                                          max_loss_pct=0.02, min_position_pct=0.07,
                                          max_position_pct=0.10)
        # Wide stop → fewer shares needed to keep dollar risk constant
        # But min_pct floor may equalise them in small portfolio
        assert low_vol["shares"] >= high_vol["shares"]

    def test_max_positions_check(self):
        assert max_concurrent_positions_reached(15) is True
        assert max_concurrent_positions_reached(14) is False
        assert max_concurrent_positions_reached(0) is False


# ═══════════════════════════════════════════════════════════════════════════════
# ATR stops
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeInitialStop:
    def test_stop_below_entry(self):
        stop = compute_initial_stop(entry_price=100.0, atr=3.0)
        assert stop < 100.0

    def test_stop_at_2x_atr(self):
        stop = compute_initial_stop(entry_price=100.0, atr=3.0)
        assert stop == pytest.approx(94.0, abs=0.01)

    def test_stop_never_goes_negative(self):
        stop = compute_initial_stop(entry_price=5.0, atr=10.0)
        assert stop > 0


class TestUpdateTrailingStop:
    def test_stop_moves_up_when_price_rises(self):
        initial = compute_initial_stop(100.0, 3.0)  # 94.0
        updated = update_trailing_stop(initial, current_price=110.0, atr=3.0)
        # New stop = 110 - 6 = 104 > 94 → moves up
        assert updated > initial

    def test_stop_does_not_move_down(self):
        current_stop = 95.0
        # New ATR-based stop would be 88 — below current → stays at 95
        updated = update_trailing_stop(current_stop, current_price=97.0, atr=4.5)
        assert updated >= current_stop

    def test_stop_stays_put_when_equal(self):
        stop = 95.0
        updated = update_trailing_stop(stop, current_price=101.0, atr=3.0)
        assert updated == pytest.approx(95.0, abs=0.01)


class TestIsStoppedOut:
    def test_price_below_stop(self):
        assert is_stopped_out(93.0, 94.0) is True

    def test_price_at_stop(self):
        assert is_stopped_out(94.0, 94.0) is True

    def test_price_above_stop(self):
        assert is_stopped_out(95.0, 94.0) is False


class TestMaxHoldDate:
    def test_hold_date_is_after_entry(self):
        entry = date(2024, 1, 2)
        max_hold = compute_max_hold_date(entry)
        assert max_hold > entry

    def test_is_exceeded_after_period(self):
        entry = date(2024, 1, 2)
        well_after = compute_max_hold_date(entry) + timedelta(days=1)
        assert is_max_hold_exceeded(entry, as_of=well_after) is True

    def test_not_exceeded_on_entry_day(self):
        entry = date(2024, 1, 2)
        assert is_max_hold_exceeded(entry, as_of=entry) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Kill switch
# ═══════════════════════════════════════════════════════════════════════════════

class TestKillSwitch:
    def test_not_paused_initially(self):
        ks = KillSwitch(pause_threshold=0.12, resume_threshold=0.08)
        ks.update(10_000.0)
        assert ks.is_paused() is False

    def test_pauses_at_threshold(self):
        ks = KillSwitch(pause_threshold=0.12, resume_threshold=0.08)
        ks.update(10_000.0)   # peak = 10000
        ks.update(8_700.0)    # drawdown = 13% → pause
        assert ks.is_paused() is True

    def test_does_not_pause_below_threshold(self):
        ks = KillSwitch(pause_threshold=0.12, resume_threshold=0.08)
        ks.update(10_000.0)
        ks.update(9_000.0)    # drawdown = 10% → not yet paused
        assert ks.is_paused() is False

    def test_resumes_after_recovery(self):
        ks = KillSwitch(pause_threshold=0.12, resume_threshold=0.08)
        ks.update(10_000.0)
        ks.update(8_700.0)    # pause
        assert ks.is_paused() is True
        ks.update(9_250.0)    # drawdown = 7.5% < resume threshold → resume
        assert ks.is_paused() is False

    def test_does_not_resume_if_not_recovered_enough(self):
        ks = KillSwitch(pause_threshold=0.12, resume_threshold=0.08)
        ks.update(10_000.0)
        ks.update(8_700.0)    # pause
        ks.update(9_100.0)    # drawdown = 9% → still paused (> 8% threshold)
        assert ks.is_paused() is True

    def test_peak_only_moves_up(self):
        ks = KillSwitch(pause_threshold=0.12, resume_threshold=0.08)
        ks.update(10_000.0)
        ks.update(12_000.0)   # new peak
        ks.update(10_500.0)   # below 12k peak → drawdown = 12.5% → pause
        assert ks.is_paused() is True

    def test_reset_clears_state(self):
        ks = KillSwitch(pause_threshold=0.12, resume_threshold=0.08)
        ks.update(10_000.0)
        ks.update(8_700.0)
        assert ks.is_paused() is True
        ks.reset()
        ks.update(9_000.0)
        assert ks.is_paused() is False
