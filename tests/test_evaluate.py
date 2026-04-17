"""Unit tests for backtest evaluation metrics."""

import pandas as pd
import pytest

from quantly.models.evaluate import compute_max_drawdown, compute_sharpe, compute_summary_metrics


class TestSharpe:
    def test_positive_returns_positive_sharpe(self):
        returns = pd.Series([0.02] * 50)
        assert compute_sharpe(returns, risk_free_rate=0.0) > 0

    def test_zero_variance(self):
        assert compute_sharpe(pd.Series([0.0] * 20)) == 0.0

    def test_negative_returns_negative_sharpe(self):
        returns = pd.Series([-0.02] * 50)
        assert compute_sharpe(returns, risk_free_rate=0.0) < 0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        returns = pd.Series([0.01] * 30)
        assert compute_max_drawdown(returns) == pytest.approx(0.0, abs=1e-6)

    def test_drawdown_after_peak(self):
        returns = pd.Series([0.1, 0.1, -0.3, -0.1])
        assert compute_max_drawdown(returns) < -0.1


class TestSummaryMetrics:
    def make_picks(self):
        return pd.DataFrame({
            "ticker": ["A", "B", "C", "D", "E"],
            "outperformance": [0.05, -0.01, 0.08, 0.03, -0.04],
            "label": [1.0, 0.0, 1.0, 1.0, 0.0],
        })

    def test_all_keys_present(self):
        m = compute_summary_metrics(self.make_picks())
        assert {"sharpe", "max_drawdown", "win_rate", "avg_outperformance", "total_picks", "annualized_return"}.issubset(m)

    def test_win_rate(self):
        m = compute_summary_metrics(self.make_picks())
        assert m["win_rate"] == pytest.approx(0.6, abs=0.01)

    def test_total_picks(self):
        m = compute_summary_metrics(self.make_picks())
        assert m["total_picks"] == 5
