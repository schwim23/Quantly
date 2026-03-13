"""Tests for label construction logic."""

import numpy as np
import pandas as pd
import pytest

from quantly.models.evaluate import compute_sharpe, compute_max_drawdown, compute_win_rate, compute_summary_metrics


class TestSharpe:
    def test_positive_sharpe_for_positive_returns(self):
        returns = pd.Series([0.01] * 100)
        assert compute_sharpe(returns, risk_free_rate=0.0) > 0

    def test_zero_variance_returns_zero(self):
        returns = pd.Series([0.0] * 50)
        assert compute_sharpe(returns) == 0.0

    def test_negative_sharpe_for_negative_returns(self):
        returns = pd.Series([-0.01] * 100)
        assert compute_sharpe(returns, risk_free_rate=0.0) < 0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        returns = pd.Series([0.01] * 50)
        dd = compute_max_drawdown(returns)
        assert dd == pytest.approx(0.0, abs=1e-6)

    def test_drawdown_after_peak(self):
        returns = pd.Series([0.1, 0.1, -0.2, -0.2, 0.1])
        dd = compute_max_drawdown(returns)
        assert dd < -0.1


class TestWinRate:
    def test_all_wins(self):
        assert compute_win_rate(pd.Series([1, 1, 1, 1])) == 1.0

    def test_half_wins(self):
        assert compute_win_rate(pd.Series([1, 0, 1, 0])) == 0.5

    def test_all_losses(self):
        assert compute_win_rate(pd.Series([0, 0, 0, 0])) == 0.0


class TestSummaryMetrics:
    def make_picks_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "NVDA", "GOOG", "META"],
            "signal_date": ["2024-01-01"] * 5,
            "outperformance": [0.05, -0.01, 0.08, 0.03, -0.04],
            "label": [1, 0, 1, 1, 0],
        })

    def test_returns_all_keys(self):
        picks = self.make_picks_df()
        metrics = compute_summary_metrics(picks)
        expected_keys = {"sharpe", "max_drawdown", "win_rate", "avg_outperformance", "total_picks", "annualized_return"}
        assert expected_keys.issubset(metrics.keys())

    def test_win_rate_correct(self):
        picks = self.make_picks_df()
        metrics = compute_summary_metrics(picks)
        assert metrics["win_rate"] == pytest.approx(0.6, abs=0.01)
