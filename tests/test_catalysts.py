"""Tests for quantly.data.catalysts — all HTTP mocked with respx."""
from __future__ import annotations

from datetime import date, timedelta

import httpx
import pytest
import respx

from quantly.data.cache import Cache
from quantly.data.catalysts import (
    get_analyst_actions,
    get_days_to_earnings,
    get_eps_surprise_streak,
    get_upcoming_earnings,
)


def _finnhub_router(calendar_payload):
    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "calendar/earnings" in url:
            return httpx.Response(200, json=calendar_payload)
        return httpx.Response(200, json={})
    return _handler


def _fmp_router(upgrades_payload):
    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "upgrades-downgrades" in url:
            return httpx.Response(200, json=upgrades_payload)
        return httpx.Response(200, json=[])
    return _handler


class TestUpcomingEarnings:
    def test_returns_dict_of_ticker_to_date(self, cache: Cache):
        future = (date.today() + timedelta(days=10)).isoformat()
        payload = {"earningsCalendar": [
            {"symbol": "AAPL", "date": future},
            {"symbol": "MSFT", "date": future},
        ]}
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*finnhub.*").mock(
                side_effect=_finnhub_router(payload)
            )
            result = get_upcoming_earnings(["AAPL", "MSFT"], cache)

        assert "AAPL" in result
        assert isinstance(result["AAPL"], date)

    def test_ticker_not_in_calendar_omitted(self, cache: Cache):
        future = (date.today() + timedelta(days=5)).isoformat()
        payload = {"earningsCalendar": [{"symbol": "AAPL", "date": future}]}
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*finnhub.*").mock(
                side_effect=_finnhub_router(payload)
            )
            result = get_upcoming_earnings(["AAPL", "NVDA"], cache)

        assert "AAPL" in result
        assert "NVDA" not in result

    def test_api_error_returns_empty(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*finnhub.*").mock(
                return_value=httpx.Response(500)
            )
            result = get_upcoming_earnings(["AAPL"], cache)

        assert result == {}

    def test_empty_ticker_list(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*finnhub.*").mock(
                return_value=httpx.Response(200, json={"earningsCalendar": []})
            )
            result = get_upcoming_earnings([], cache)

        assert result == {}


class TestDaysToEarnings:
    def test_returns_correct_days(self):
        upcoming = {"AAPL": date.today() + timedelta(days=7)}
        assert get_days_to_earnings("AAPL", upcoming) == 7

    def test_returns_99_when_not_in_map(self):
        assert get_days_to_earnings("AAPL", {}) == 99

    def test_past_date_returns_zero(self):
        upcoming = {"AAPL": date.today() - timedelta(days=3)}
        assert get_days_to_earnings("AAPL", upcoming) == 0


class TestAnalystActions:
    def _make_actions(self, actions: list[tuple[str, str]]) -> list[dict]:
        return [{"newGrade": grade, "publishedDate": dt} for grade, dt in actions]

    def test_counts_upgrades_and_downgrades(self, cache: Cache):
        recent = (date.today() - timedelta(days=3)).isoformat()
        payload = self._make_actions([
            ("Buy", recent), ("Outperform", recent), ("Sell", recent),
        ])
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*financialmodelingprep.*").mock(
                side_effect=_fmp_router(payload)
            )
            result = get_analyst_actions("AAPL", cache)

        assert result["upgrades"] == 2
        assert result["downgrades"] == 1
        assert result["net"] == 1

    def test_ignores_old_actions(self, cache: Cache):
        old = (date.today() - timedelta(days=30)).isoformat()
        payload = self._make_actions([("Buy", old)])
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*financialmodelingprep.*").mock(
                side_effect=_fmp_router(payload)
            )
            result = get_analyst_actions("AAPL", cache, lookback_days=10)

        assert result["upgrades"] == 0

    def test_api_error_returns_zeros(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*financialmodelingprep.*").mock(
                return_value=httpx.Response(500)
            )
            result = get_analyst_actions("ERR", cache)

        assert result == {"upgrades": 0, "downgrades": 0, "net": 0}


class TestEpsSurpriseStreak:
    def test_three_consecutive_beats(self):
        earnings = [
            {"actualEarningResult": 2.0, "estimatedEarning": 1.8},
            {"actualEarningResult": 1.9, "estimatedEarning": 1.7},
            {"actualEarningResult": 1.5, "estimatedEarning": 1.4},
        ]
        assert get_eps_surprise_streak(earnings) == 3

    def test_first_miss_breaks_streak(self):
        earnings = [
            {"actualEarningResult": 1.5, "estimatedEarning": 1.6},  # miss
            {"actualEarningResult": 1.9, "estimatedEarning": 1.7},
        ]
        assert get_eps_surprise_streak(earnings) == 0

    def test_streak_stops_at_miss(self):
        earnings = [
            {"actualEarningResult": 2.0, "estimatedEarning": 1.8},
            {"actualEarningResult": 1.4, "estimatedEarning": 1.5},  # miss
            {"actualEarningResult": 1.3, "estimatedEarning": 1.2},
        ]
        assert get_eps_surprise_streak(earnings) == 1

    def test_empty_returns_zero(self):
        assert get_eps_surprise_streak([]) == 0
