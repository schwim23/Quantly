"""Tests for quantly.data.universe — screener logic (network calls mocked)."""
from __future__ import annotations

import pandas as pd
import pytest

from quantly.data.cache import Cache
from quantly.data.universe import screen_universe


class TestScreenUniverse:
    def _make_prices_df(self, rows: list[dict]) -> pd.DataFrame:
        """Build a prices snapshot DataFrame as expected by screen_universe."""
        df = pd.DataFrame(rows).set_index("ticker")
        return df

    def test_filters_low_price(self, cache: Cache):
        df = self._make_prices_df([
            {"ticker": "CHEAP", "close": 2.0, "dollar_volume": 10_000_000},
            {"ticker": "OK",    "close": 20.0, "dollar_volume": 10_000_000},
        ])
        result = screen_universe(["CHEAP", "OK"], df, min_price=5.0, min_dollar_volume=1_000_000)
        assert "OK" in result
        assert "CHEAP" not in result

    def test_filters_low_volume(self, cache: Cache):
        df = self._make_prices_df([
            {"ticker": "ILLIQUID", "close": 50.0, "dollar_volume": 100_000},
            {"ticker": "LIQUID",   "close": 50.0, "dollar_volume": 10_000_000},
        ])
        result = screen_universe(
            ["ILLIQUID", "LIQUID"], df,
            min_price=5.0, min_dollar_volume=5_000_000
        )
        assert "LIQUID" in result
        assert "ILLIQUID" not in result

    def test_empty_universe_returns_empty(self, cache: Cache):
        df = pd.DataFrame(columns=["close", "dollar_volume"]).rename_axis("ticker")
        result = screen_universe([], df, min_price=5.0, min_dollar_volume=5_000_000)
        assert result == []

    def test_empty_prices_df_returns_empty(self, cache: Cache):
        df = pd.DataFrame(columns=["close", "dollar_volume"]).rename_axis("ticker")
        result = screen_universe(["AAPL"], df, min_price=5.0, min_dollar_volume=5_000_000)
        assert result == []

    def test_ticker_not_in_prices_is_excluded(self, cache: Cache):
        df = self._make_prices_df([
            {"ticker": "AAPL", "close": 180.0, "dollar_volume": 50_000_000},
        ])
        result = screen_universe(
            ["AAPL", "NOTINDF"], df,
            min_price=5.0, min_dollar_volume=5_000_000
        )
        assert "AAPL" in result
        assert "NOTINDF" not in result

    def test_all_pass_filter(self, cache: Cache):
        df = self._make_prices_df([
            {"ticker": "A", "close": 100.0, "dollar_volume": 20_000_000},
            {"ticker": "B", "close": 200.0, "dollar_volume": 30_000_000},
            {"ticker": "C", "close": 50.0,  "dollar_volume": 10_000_000},
        ])
        result = screen_universe(
            ["A", "B", "C"], df,
            min_price=5.0, min_dollar_volume=5_000_000
        )
        assert set(result) == {"A", "B", "C"}

    def test_uses_config_defaults_when_no_overrides(self, cache: Cache):
        """screen_universe should use cfg.min_price / cfg.min_dollar_volume if not overridden."""
        df = self._make_prices_df([
            {"ticker": "PASS", "close": 100.0, "dollar_volume": 10_000_000},
            {"ticker": "FAIL", "close": 1.0,   "dollar_volume": 100_000},
        ])
        result = screen_universe(["PASS", "FAIL"], df)
        assert "PASS" in result
        assert "FAIL" not in result
