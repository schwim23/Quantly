"""Tests for technical and KPI feature computation."""

import numpy as np
import pandas as pd
import pytest

from quantly.features.technical import (
    compute_rsi,
    compute_macd_signal,
    compute_volume_spike,
    compute_momentum,
    compute_bollinger_position,
    compute_distance_from_52w,
)
from quantly.features.kpi import (
    compute_revenue_growth,
    compute_eps_surprise,
    compute_gross_margin_trend,
)


def make_prices(n: int = 100, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.02, n)
    prices = pd.Series(100 * np.cumprod(1 + returns))
    return prices


class TestRSI:
    def test_rsi_in_range(self):
        prices = make_prices()
        rsi = compute_rsi(prices)
        assert 0 <= rsi <= 100

    def test_rsi_overbought(self):
        # Consistently rising prices should give high RSI
        prices = pd.Series(range(1, 101), dtype=float)
        rsi = compute_rsi(prices)
        assert rsi > 70

    def test_rsi_oversold(self):
        # Consistently falling prices should give low RSI
        prices = pd.Series(range(100, 0, -1), dtype=float)
        rsi = compute_rsi(prices)
        assert rsi < 30


class TestMACD:
    def test_macd_returns_all_keys(self):
        prices = make_prices()
        result = compute_macd_signal(prices)
        assert set(result.keys()) == {"macd", "signal", "histogram", "crossover"}

    def test_crossover_valid_values(self):
        prices = make_prices()
        result = compute_macd_signal(prices)
        assert result["crossover"] in {-1, 0, 1}


class TestVolumSpike:
    def test_no_spike_returns_near_one(self):
        volumes = pd.Series([1_000_000] * 25)
        spike = compute_volume_spike(volumes)
        assert 0.8 < spike < 1.2

    def test_large_spike(self):
        volumes = pd.Series([1_000_000] * 24 + [5_000_000])
        spike = compute_volume_spike(volumes)
        assert spike > 4


class TestMomentum:
    def test_returns_all_keys(self):
        prices = make_prices()
        result = compute_momentum(prices)
        assert set(result.keys()) == {"mom_5d", "mom_10d", "mom_20d"}


class TestKPIFeatures:
    def make_income(self, n: int = 6) -> pd.DataFrame:
        return pd.DataFrame({
            "fiscal_date_ending": pd.date_range("2024-01-01", periods=n, freq="QE"),
            "revenue": [1000, 900, 850, 800, 750, 700],
            "gross_profit": [400, 350, 320, 300, 280, 260],
            "net_income": [100, 90, 80, 70, 60, 50],
            "eps": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            "eps_estimate": [0.95, 0.88, 0.82, 0.72, 0.58, 0.52],
        })

    def test_revenue_growth_positive(self):
        income = self.make_income()
        result = compute_revenue_growth(income)
        assert result["revenue_growth_qoq"] > 0   # 1000 > 900

    def test_eps_surprise_beat(self):
        income = self.make_income()
        result = compute_eps_surprise(income)
        # EPS 1.0 vs estimate 0.95 = positive surprise
        assert result["eps_surprise_pct"] > 0

    def test_gross_margin_trend(self):
        income = self.make_income()
        result = compute_gross_margin_trend(income)
        assert "gross_margin" in result
        assert "gross_margin_delta_qoq" in result
        assert 0 < result["gross_margin"] < 1
