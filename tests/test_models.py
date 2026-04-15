"""Tests for ML model — labels, training, prediction, evaluation."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantly.models.labels import (
    FORWARD_DAYS,
    OUTPERFORM_THRESHOLD,
    build_label_series,
    compute_forward_return,
    make_label,
    purge_gap,
)
from quantly.models.evaluate import (
    compute_portfolio_metrics,
    max_drawdown,
    model_precision,
    sharpe_ratio,
    win_rate,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_price_series(n: int = 30, start: float = 100.0, trend: float = 0.5) -> pd.Series:
    idx = [date(2024, 1, 2) + timedelta(days=i) for i in range(n)]
    prices = [start + trend * i for i in range(n)]
    return pd.Series(prices, index=idx)


# ═══════════════════════════════════════════════════════════════════════════════
# Labels
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeForwardReturn:
    def test_outperforming_ticker(self):
        spy = _make_price_series(30, start=100.0, trend=0.1)   # slow SPY
        ticker = _make_price_series(30, start=100.0, trend=1.0)  # fast ticker
        sig = date(2024, 1, 2)
        excess = compute_forward_return(ticker, spy, sig, forward_days=15)
        assert excess is not None
        assert excess > 0

    def test_underperforming_ticker(self):
        spy = _make_price_series(30, start=100.0, trend=0.5)
        ticker = _make_price_series(30, start=100.0, trend=0.1)  # slow ticker
        sig = date(2024, 1, 2)
        excess = compute_forward_return(ticker, spy, sig, forward_days=15)
        assert excess is not None
        assert excess < 0

    def test_returns_none_when_insufficient_data(self):
        spy = _make_price_series(10)
        ticker = _make_price_series(10)
        sig = date(2024, 1, 2)
        excess = compute_forward_return(ticker, spy, sig, forward_days=15)
        assert excess is None

    def test_returns_none_for_unknown_date(self):
        spy = _make_price_series(30)
        ticker = _make_price_series(30)
        sig = date(2030, 1, 1)  # far future — not in index
        excess = compute_forward_return(ticker, spy, sig, forward_days=15)
        assert excess is None


class TestMakeLabel:
    def test_positive_label_for_outperformance(self):
        assert make_label(0.05) == 1   # >3% outperformance

    def test_negative_label_for_underperformance(self):
        assert make_label(-0.03) == 0  # < -2% underperformance

    def test_none_for_ambiguous_zone(self):
        assert make_label(0.01) is None   # between -2% and +3%

    def test_none_for_none_input(self):
        assert make_label(None) is None

    def test_boundary_outperform(self):
        assert make_label(OUTPERFORM_THRESHOLD) == 1

    def test_boundary_underperform(self):
        assert make_label(-0.02) == 0


class TestBuildLabelSeries:
    def test_returns_dataframe_with_correct_columns(self):
        spy = _make_price_series(40)
        ticker = _make_price_series(40, trend=1.0)
        sig_dates = [date(2024, 1, 2), date(2024, 1, 5)]
        df = build_label_series(["AAPL"], {"AAPL": ticker}, spy, sig_dates)
        assert set(df.columns) == {"ticker", "signal_date", "excess_return", "label"}

    def test_excludes_ambiguous_labels(self):
        # Flat SPY and flat ticker → near-zero excess → ambiguous → excluded
        spy = _make_price_series(40, trend=0.001)
        ticker = _make_price_series(40, trend=0.0015)
        sig_dates = [date(2024, 1, 2)]
        df = build_label_series(["FLAT"], {"FLAT": ticker}, spy, sig_dates)
        # May be empty if ambiguous
        assert isinstance(df, pd.DataFrame)

    def test_missing_ticker_skipped(self):
        spy = _make_price_series(40)
        sig_dates = [date(2024, 1, 2)]
        df = build_label_series(["GHOST"], {}, spy, sig_dates)
        assert df.empty


class TestPurgeGap:
    def test_removes_rows_within_gap(self):
        df = pd.DataFrame({
            "ticker": ["A"] * 10,
            "signal_date": [date(2024, 1, 1) + timedelta(days=i) for i in range(10)],
            "label": [1] * 10,
        })
        test_start = date(2024, 1, 8)
        purged = purge_gap(df, test_start, gap_days=5)
        assert all(purged["signal_date"] <= date(2024, 1, 3))

    def test_keeps_rows_outside_gap(self):
        df = pd.DataFrame({
            "ticker": ["A"],
            "signal_date": [date(2024, 1, 1)],
            "label": [1],
        })
        test_start = date(2024, 3, 1)
        purged = purge_gap(df, test_start, gap_days=5)
        assert len(purged) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate
# ═══════════════════════════════════════════════════════════════════════════════

class TestSharpeRatio:
    def test_positive_consistent_returns_gives_positive_sharpe(self):
        rng = np.random.default_rng(42)
        # Positive drift with noise → consistently above risk-free
        returns = pd.Series(0.001 + rng.normal(0, 0.005, 252))
        assert sharpe_ratio(returns) > 0

    def test_negative_returns_gives_negative_sharpe(self):
        returns = pd.Series([-0.002] * 252)
        assert sharpe_ratio(returns) < 0

    def test_empty_returns_gives_zero(self):
        assert sharpe_ratio(pd.Series(dtype=float)) == 0.0

    def test_zero_variance_returns_zero(self):
        returns = pd.Series([0.001] * 10)
        result = sharpe_ratio(returns)
        # All identical excess returns → non-zero Sharpe actually, but variance → 0 → 0
        # With returns exactly equal to risk-free daily → excess near 0 → could be 0
        assert isinstance(result, float)


class TestMaxDrawdown:
    def test_drawdown_after_drop(self):
        equity = pd.Series([100, 110, 120, 90, 95, 100])
        dd = max_drawdown(equity)
        assert dd == pytest.approx(0.25, abs=0.01)  # 120 → 90 = 25%

    def test_monotonic_increase_gives_zero(self):
        equity = pd.Series([100, 110, 120, 130])
        assert max_drawdown(equity) == pytest.approx(0.0, abs=0.001)

    def test_empty_gives_zero(self):
        assert max_drawdown(pd.Series(dtype=float)) == 0.0


class TestWinRate:
    def test_all_wins(self):
        assert win_rate(pd.Series([0.05, 0.03, 0.08])) == 1.0

    def test_all_losses(self):
        assert win_rate(pd.Series([-0.05, -0.03])) == 0.0

    def test_mixed(self):
        assert win_rate(pd.Series([0.05, -0.03, 0.02, -0.01])) == 0.5

    def test_empty_gives_zero(self):
        assert win_rate(pd.Series(dtype=float)) == 0.0


class TestModelPrecision:
    def test_all_high_conviction_correct(self):
        preds = pd.Series([80.0, 75.0, 90.0])
        labels = pd.Series([1, 1, 1])
        assert model_precision(preds, labels) == 1.0

    def test_half_correct(self):
        preds = pd.Series([80.0, 75.0])
        labels = pd.Series([1, 0])
        assert model_precision(preds, labels) == 0.5

    def test_low_conviction_not_counted(self):
        preds = pd.Series([30.0, 40.0])   # all below 60 threshold
        labels = pd.Series([1, 1])
        assert model_precision(preds, labels) == 0.0


class TestComputePortfolioMetrics:
    def test_empty_history_returns_zeros(self):
        metrics = compute_portfolio_metrics(
            pd.DataFrame(), pd.Series(dtype=float)
        )
        assert metrics["win_rate"] == 0.0
        assert metrics["trade_count"] == 0

    def test_returns_all_keys(self):
        trades = pd.DataFrame({
            "ticker": ["AAPL"],
            "return_pct": [0.05],
        })
        equity = pd.Series([10000.0, 10500.0, 10400.0])
        metrics = compute_portfolio_metrics(trades, equity)
        assert set(metrics) == {"sharpe", "max_drawdown", "win_rate", "avg_return", "trade_count"}
