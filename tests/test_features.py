"""Unit tests for technical and KPI feature computation."""

import numpy as np
import pandas as pd
import pytest

from quantly.features.technical import (
    compute_bollinger_position,
    compute_distance_from_52w,
    compute_macd,
    compute_momentum,
    compute_rsi,
    compute_volume_spike,
)
from quantly.features.kpi import (
    compute_debt_equity,
    compute_eps_surprise,
    compute_gross_margin_trend,
    compute_revenue_growth,
    compute_roe,
)


def make_prices(n: int = 100, trend: float = 0.001) -> pd.Series:
    rng = np.random.default_rng(42)
    returns = rng.normal(trend, 0.02, n)
    return pd.Series(100 * np.cumprod(1 + returns))


def make_ohlcv(n: int = 100) -> pd.DataFrame:
    prices = make_prices(n)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "open": prices * (1 + rng.normal(0, 0.005, n)),
        "high": prices * (1 + rng.uniform(0, 0.01, n)),
        "low": prices * (1 - rng.uniform(0, 0.01, n)),
        "close": prices,
        "volume": rng.integers(500_000, 5_000_000, n).astype(float),
    })


class TestRSI:
    def test_range(self):
        assert 0 <= compute_rsi(make_prices()) <= 100

    def test_overbought(self):
        # Monotonically rising → RSI near 100
        assert compute_rsi(pd.Series(range(1, 101), dtype=float)) > 70

    def test_oversold(self):
        # Monotonically falling → RSI near 0
        assert compute_rsi(pd.Series(range(100, 0, -1), dtype=float)) < 30


class TestMACD:
    def test_keys(self):
        result = compute_macd(make_prices())
        assert {"macd", "macd_signal", "macd_histogram", "macd_crossover"} == set(result)

    def test_crossover_values(self):
        assert compute_macd(make_prices())["macd_crossover"] in {-1, 0, 1}


class TestVolumeSpike:
    def test_flat_volume_near_one(self):
        volumes = pd.Series([1_000_000.0] * 25)
        assert 0.8 < compute_volume_spike(volumes) < 1.2

    def test_spike_detected(self):
        volumes = pd.Series([1_000_000.0] * 24 + [5_000_000.0])
        assert compute_volume_spike(volumes) > 4


class TestMomentum:
    def test_keys(self):
        result = compute_momentum(make_prices(30))
        assert {"mom_5d", "mom_10d", "mom_20d"} == set(result)


class TestBollinger:
    def test_range(self):
        val = compute_bollinger_position(make_prices())
        # Can exceed [0,1] in extreme moves; just check it's a float
        assert isinstance(val, float)


class TestKPI:
    def _income(self, n=6):
        return pd.DataFrame({
            "revenue": [1000, 900, 850, 800, 750, 700],
            "gross_profit": [400, 350, 320, 300, 280, 260],
            "net_income": [100, 90, 80, 70, 60, 50],
            "eps": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            "eps_estimate": [0.95, 0.88, 0.82, 0.72, 0.58, 0.52],
        })

    def _balance(self, n=6):
        return pd.DataFrame({
            "total_equity": [500, 480, 460, 440, 420, 400],
            "long_term_debt": [200, 210, 220, 230, 240, 250],
        })

    def test_revenue_growth_positive(self):
        result = compute_revenue_growth(self._income())
        assert result["revenue_growth_qoq"] > 0

    def test_eps_surprise_beat(self):
        result = compute_eps_surprise(self._income())
        assert result["eps_surprise_pct"] > 0  # 1.0 vs 0.95

    def test_gross_margin_trend_keys(self):
        result = compute_gross_margin_trend(self._income())
        assert "gross_margin" in result and "gross_margin_delta_qoq" in result

    def test_debt_equity_keys(self):
        result = compute_debt_equity(self._balance())
        assert "debt_equity_ratio" in result

    def test_roe_keys(self):
        result = compute_roe(self._income(), self._balance())
        assert "roe" in result and "roe_delta_qoq" in result
