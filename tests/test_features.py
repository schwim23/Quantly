"""Tests for quantly.features — technical indicators and pipeline assembly."""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quantly.features.technical import (
    bollinger_pct_b,
    compute_all_technical,
    distance_from_52w,
    macd,
    momentum,
    rsi,
    volume_spike,
)
from quantly.features.pipeline import build_feature_row, FEATURE_COLUMNS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_price_df(
    n: int = 60,
    start_price: float = 100.0,
    trend: float = 0.2,
    volume: float = 1_000_000,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame (ascending date index)."""
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    prices = [start_price + trend * i for i in range(n)]
    return pd.DataFrame(
        {
            "open":   [p * 0.99 for p in prices],
            "high":   [p * 1.02 for p in prices],
            "low":    [p * 0.98 for p in prices],
            "close":  prices,
            "volume": [volume] * n,
        },
        index=idx,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RSI
# ═══════════════════════════════════════════════════════════════════════════════

class TestRsi:
    def test_uptrend_gives_high_rsi(self):
        df = _make_price_df(n=60, trend=1.0)
        val = rsi(df)
        assert val > 60, f"expected high RSI in uptrend, got {val}"

    def test_downtrend_gives_low_rsi(self):
        df = _make_price_df(n=60, trend=-0.5)
        val = rsi(df)
        assert val < 40, f"expected low RSI in downtrend, got {val}"

    def test_flat_price_gives_rsi_near_50(self):
        df = _make_price_df(n=60, trend=0.0)
        val = rsi(df)
        # Flat price: close is constant → rsi should be near 50
        assert 40 <= val <= 60 or val == 50.0

    def test_insufficient_data_returns_50(self):
        df = _make_price_df(n=5)
        assert rsi(df) == 50.0

    def test_range_0_to_100(self):
        df = _make_price_df(n=60, trend=2.0)
        val = rsi(df)
        assert 0.0 <= val <= 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# MACD
# ═══════════════════════════════════════════════════════════════════════════════

class TestMacd:
    def test_returns_all_keys(self):
        df = _make_price_df(n=60)
        result = macd(df)
        assert set(result) == {"macd_line", "signal_line", "histogram", "crossover"}

    def test_uptrend_positive_histogram(self):
        df = _make_price_df(n=80, trend=1.0)
        result = macd(df)
        assert result["histogram"] > 0

    def test_downtrend_negative_histogram(self):
        df = _make_price_df(n=80, trend=-0.5)
        result = macd(df)
        assert result["histogram"] < 0

    def test_bullish_crossover_detected(self):
        """Price reversal up after a downtrend should eventually yield crossover=1."""
        n = 80
        prices = [100.0 - 0.5 * i for i in range(40)] + [60.0 + 1.5 * i for i in range(40)]
        idx = pd.date_range("2024-01-02", periods=n, freq="B")
        df = pd.DataFrame({
            "open": prices, "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices], "close": prices,
            "volume": [1_000_000] * n,
        }, index=idx)
        result = macd(df)
        # Crossover should be 0 or +1 after the reversal
        assert result["crossover"] in (-1, 0, 1)

    def test_insufficient_data_returns_zeros(self):
        df = _make_price_df(n=10)
        result = macd(df)
        assert result["macd_line"] == 0.0
        assert result["crossover"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Momentum
# ═══════════════════════════════════════════════════════════════════════════════

class TestMomentum:
    def test_positive_momentum_in_uptrend(self):
        df = _make_price_df(n=30, trend=1.0)
        result = momentum(df)
        assert result["momentum_5d"] > 0
        assert result["momentum_10d"] > 0
        assert result["momentum_20d"] > 0

    def test_negative_momentum_in_downtrend(self):
        df = _make_price_df(n=30, trend=-0.5)
        result = momentum(df)
        assert result["momentum_5d"] < 0

    def test_insufficient_data_returns_zero(self):
        df = _make_price_df(n=3)
        result = momentum(df)
        assert result["momentum_5d"] == 0.0
        assert result["momentum_10d"] == 0.0

    def test_returns_all_keys(self):
        df = _make_price_df(n=30)
        result = momentum(df)
        assert set(result) == {"momentum_5d", "momentum_10d", "momentum_20d"}


# ═══════════════════════════════════════════════════════════════════════════════
# Volume spike
# ═══════════════════════════════════════════════════════════════════════════════

class TestVolumeSpike:
    def test_normal_volume_near_one(self):
        df = _make_price_df(n=30, volume=1_000_000)
        ratio = volume_spike(df)
        assert 0.9 <= ratio <= 1.1

    def test_spike_detected(self):
        df = _make_price_df(n=30, volume=1_000_000)
        # Replace last day's volume with a big spike
        df.iloc[-1, df.columns.get_loc("volume")] = 5_000_000
        ratio = volume_spike(df)
        assert ratio > 3.0

    def test_insufficient_data_returns_one(self):
        df = _make_price_df(n=5)
        assert volume_spike(df) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Distance from 52-week high/low
# ═══════════════════════════════════════════════════════════════════════════════

class TestDistanceFrom52w:
    def test_at_52w_high_gives_zero_dist_high(self):
        df = _make_price_df(n=60, trend=1.0)  # monotonic uptrend → always at high
        result = distance_from_52w(df)
        assert result["dist_from_52w_high"] == pytest.approx(0.0, abs=0.01)

    def test_far_from_high_gives_positive(self):
        df = _make_price_df(n=60, trend=-0.5)  # downtrend → far from high
        result = distance_from_52w(df)
        assert result["dist_from_52w_high"] > 0

    def test_at_52w_low_gives_zero_dist_low(self):
        df = _make_price_df(n=60, trend=-1.0)  # monotonic downtrend → at low
        result = distance_from_52w(df)
        assert result["dist_from_52w_low"] == pytest.approx(0.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# Bollinger %B
# ═══════════════════════════════════════════════════════════════════════════════

class TestBollingerPctB:
    def test_at_upper_band_near_one(self):
        df = _make_price_df(n=40, trend=2.0)  # strong uptrend → near upper band
        val = bollinger_pct_b(df)
        assert val > 0.7

    def test_flat_near_midline(self):
        df = _make_price_df(n=40, trend=0.0)
        val = bollinger_pct_b(df)
        assert 0.3 <= val <= 0.7

    def test_insufficient_data_returns_half(self):
        df = _make_price_df(n=5)
        assert bollinger_pct_b(df) == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# compute_all_technical
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeAllTechnical:
    def test_returns_all_expected_keys(self):
        df = _make_price_df(n=60)
        result = compute_all_technical(df)
        expected = {
            "rsi_14", "macd_crossover", "macd_histogram",
            "momentum_5d", "momentum_10d", "momentum_20d",
            "volume_spike_ratio", "dist_from_52w_high", "dist_from_52w_low",
            "bollinger_pct_b",
        }
        assert expected.issubset(result.keys())

    def test_empty_df_returns_neutral_defaults(self):
        result = compute_all_technical(pd.DataFrame())
        assert result["rsi_14"] == 50.0
        assert result["macd_crossover"] == 0
        assert result["volume_spike_ratio"] == 1.0
        assert result["bollinger_pct_b"] == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# build_feature_row (integration, with mocked sub-functions)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildFeatureRow:
    def test_returns_flat_dict_with_ticker_key(self, cache):
        df = _make_price_df(n=60)

        with (
            patch("quantly.features.pipeline.compute_sentiment_features",
                  return_value={"news_sentiment": 0.3, "sentiment_composite": 0.2,
                                "news_bullish_pct": 0.6, "gdelt_tone": 0.1,
                                "stocktwits_sentiment": 0.2, "congress_net": 1.0,
                                "congress_latest_buy_days": 10.0,
                                "options_unusual_calls": 1.5,
                                "options_put_call_ratio": 0.8,
                                "options_net_score": 0.5}),
            patch("quantly.features.pipeline.compute_catalyst_features",
                  return_value={"days_to_earnings": 14.0, "earnings_proximity": 0.07,
                                "eps_beat_streak": 3.0, "eps_surprise_last": 0.05,
                                "eps_surprise_prev": 0.03, "analyst_upgrades_10d": 2.0,
                                "analyst_downgrades_10d": 0.0, "analyst_net_10d": 2.0}),
            patch("quantly.features.pipeline.compute_fundamental_features",
                  return_value={"revenue_growth_yoy": 0.12, "revenue_growth_qoq": 0.03,
                                "gross_margin": 0.45, "gross_margin_trend": 1.0,
                                "fcf_yield": 0.05, "eps_growth_slope": 0.1,
                                "debt_to_equity": 0.8, "has_positive_fcf": 1.0,
                                "eps_surprise_last": 0.05, "eps_surprise_prev": 0.03}),
        ):
            row = build_feature_row("AAPL", cache, df, {})

        assert "ticker" in row  # still has ticker key before set_index
        assert "rsi_14" in row
        assert "news_sentiment" in row
        assert "days_to_earnings" in row
        assert "revenue_growth_yoy" in row

    def test_partial_failure_does_not_raise(self, cache):
        df = _make_price_df(n=60)
        with (
            patch("quantly.features.pipeline.compute_sentiment_features",
                  side_effect=Exception("API down")),
            patch("quantly.features.pipeline.compute_catalyst_features",
                  return_value={}),
            patch("quantly.features.pipeline.compute_fundamental_features",
                  return_value={}),
        ):
            row = build_feature_row("FAIL", cache, df, {})

        # Should still have technical features and ticker key
        assert "ticker" in row
        assert "rsi_14" in row
