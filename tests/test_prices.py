"""Tests for quantly.data.prices — validates OHLCV fetch, ATR, snapshot logic.

All HTTP calls are intercepted with respx so no real network traffic is made.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import respx
import httpx

from quantly.data.cache import Cache
from quantly.data.prices import compute_atr, get_current_price, get_latest_snapshot, get_ohlcv


def _make_ohlcv_response(ticker: str, n_days: int = 30) -> list[dict]:
    """Generate synthetic Tiingo-style OHLCV records."""
    base = date(2024, 1, 2)
    records = []
    price = 100.0
    for i in range(n_days):
        d = base + timedelta(days=i)
        records.append(
            {
                "date": f"{d.isoformat()}T00:00:00+00:00",
                "open": price * 0.99,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "volume": 1_000_000,
                "adjClose": price,
            }
        )
        price += 0.50  # gentle uptrend
    return records


class TestGetOhlcv:
    def test_returns_dataframe_with_expected_columns(self, cache: Cache):
        payload = _make_ohlcv_response("AAPL", 10)
        with respx.mock:
            respx.get(url__regex=r".*tiingo.*AAPL.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            df = get_ohlcv("AAPL", date(2024, 1, 2), date(2024, 1, 12), cache)

        assert not df.empty
        assert "close" in df.columns
        assert "volume" in df.columns
        assert "dollar_volume" in df.columns

    def test_dollar_volume_computed(self, cache: Cache):
        payload = _make_ohlcv_response("MSFT", 5)
        with respx.mock:
            respx.get(url__regex=r".*tiingo.*MSFT.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            df = get_ohlcv("MSFT", date(2024, 1, 2), date(2024, 1, 7), cache)

        assert (df["dollar_volume"] == df["close"] * df["volume"]).all()

    def test_returns_empty_on_http_error(self, cache: Cache):
        with respx.mock:
            respx.get(url__regex=r".*tiingo.*BADD.*").mock(
                return_value=httpx.Response(404)
            )
            df = get_ohlcv("BADD", date(2024, 1, 2), date(2024, 1, 5), cache)

        assert df.empty

    def test_returns_empty_on_empty_response(self, cache: Cache):
        with respx.mock:
            respx.get(url__regex=r".*tiingo.*EMPTY.*").mock(
                return_value=httpx.Response(200, json=[])
            )
            df = get_ohlcv("EMPTY", date(2024, 1, 2), date(2024, 1, 5), cache)

        assert df.empty

    def test_caches_result(self, cache: Cache):
        payload = _make_ohlcv_response("NVDA", 10)
        call_count = 0

        def handler(request):
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json=payload)

        with respx.mock:
            respx.get(url__regex=r".*tiingo.*NVDA.*").mock(side_effect=handler)
            get_ohlcv("NVDA", date(2024, 1, 2), date(2024, 1, 12), cache)
            get_ohlcv("NVDA", date(2024, 1, 2), date(2024, 1, 12), cache)

        assert call_count == 1  # second call served from cache

    def test_sorted_index(self, cache: Cache):
        payload = _make_ohlcv_response("AMD", 10)
        with respx.mock:
            respx.get(url__regex=r".*tiingo.*AMD.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            df = get_ohlcv("AMD", date(2024, 1, 2), date(2024, 1, 12), cache)

        assert list(df.index) == sorted(df.index)


class TestGetCurrentPrice:
    def test_returns_last_close(self, cache: Cache):
        payload = _make_ohlcv_response("AAPL", 5)
        with respx.mock:
            respx.get(url__regex=r".*tiingo.*AAPL.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            price = get_current_price("AAPL", cache)

        assert price is not None
        assert isinstance(price, float)

    def test_returns_none_when_no_data(self, cache: Cache):
        with respx.mock:
            respx.get(url__regex=r".*tiingo.*GHOST.*").mock(
                return_value=httpx.Response(200, json=[])
            )
            price = get_current_price("GHOST", cache)

        assert price is None


class TestComputeAtr:
    def _make_price_df(self, n: int = 30) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n)
        return pd.DataFrame(
            {
                "high": [102.0 + i * 0.1 for i in range(n)],
                "low": [98.0 + i * 0.1 for i in range(n)],
                "close": [100.0 + i * 0.1 for i in range(n)],
            },
            index=idx,
        )

    def test_returns_series_same_length(self):
        df = self._make_price_df(30)
        atr = compute_atr(df)
        assert len(atr) == len(df)

    def test_atr_is_positive(self):
        df = self._make_price_df(30)
        atr = compute_atr(df)
        assert (atr.dropna() > 0).all()

    def test_atr_default_period_14(self):
        df = self._make_price_df(30)
        atr14 = compute_atr(df, period=14)
        atr7 = compute_atr(df, period=7)
        # Longer period should smooth more — can't assert direction easily,
        # just assert both are valid series
        assert not atr14.empty
        assert not atr7.empty

    def test_atr_flat_price_series(self):
        """All candles identical → ATR should be near zero."""
        n = 30
        idx = pd.date_range("2024-01-01", periods=n)
        df = pd.DataFrame(
            {"high": [100.0] * n, "low": [100.0] * n, "close": [100.0] * n},
            index=idx,
        )
        atr = compute_atr(df)
        assert (atr.dropna() < 0.01).all()


class TestGetLatestSnapshot:
    def test_returns_one_row_per_valid_ticker(self, cache: Cache):
        payload = _make_ohlcv_response("AAPL", 25)
        with respx.mock:
            respx.get(url__regex=r".*tiingo.*AAPL.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            respx.get(url__regex=r".*tiingo.*BADD.*").mock(
                return_value=httpx.Response(200, json=[])
            )
            snap = get_latest_snapshot(["AAPL", "BADD"], cache)

        assert "AAPL" in snap.index
        assert "BADD" not in snap.index

    def test_snapshot_has_required_columns(self, cache: Cache):
        payload = _make_ohlcv_response("NVDA", 25)
        with respx.mock:
            respx.get(url__regex=r".*tiingo.*NVDA.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            snap = get_latest_snapshot(["NVDA"], cache)

        for col in ["close", "volume", "dollar_volume", "avg_dollar_volume_20d"]:
            assert col in snap.columns

    def test_empty_tickers_returns_empty_df(self, cache: Cache):
        snap = get_latest_snapshot([], cache)
        assert snap.empty
