"""Tests for fundamentals.py and sec_filings.py — all HTTP mocked with respx."""
from __future__ import annotations

import httpx
import pytest
import respx

from quantly.data.cache import Cache
from quantly.data.fundamentals import (
    get_fundamentals_summary,
    passes_fundamentals_gate,
)
from quantly.data.sec_filings import _extract_risk_keywords


# ── Helper factories ──────────────────────────────────────────────────────────

def _income_stmt(revenue: float, gross_profit: float, eps: float) -> dict:
    return {"revenue": revenue, "grossProfit": gross_profit, "eps": eps}


def _cashflow(fcf: float) -> dict:
    return {"freeCashFlow": fcf}


def _fmp_router(income=None, cashflow=None, metrics=None):
    """Build a respx callback that routes FMP requests by URL path fragment."""
    income = income or []
    cashflow = cashflow or []
    metrics = metrics or []

    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "income-statement" in url:
            return httpx.Response(200, json=income)
        if "cash-flow-statement" in url:
            return httpx.Response(200, json=cashflow)
        if "key-metrics" in url:
            return httpx.Response(200, json=metrics)
        return httpx.Response(200, json=[])

    return _handler


class TestFundamentalsSummary:
    def test_revenue_growth_yoy(self, cache: Cache):
        income = [
            _income_stmt(110, 55, 2.0),  # latest
            _income_stmt(100, 50, 1.8),
            _income_stmt(95,  48, 1.7),
            _income_stmt(90,  45, 1.5),
            _income_stmt(100, 50, 1.6),  # year-ago quarter
        ]
        handler = _fmp_router(income=income, cashflow=[_cashflow(5_000_000)],
                               metrics=[{"debtToEquity": 0.5}])
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*financialmodelingprep.*").mock(side_effect=handler)
            summary = get_fundamentals_summary("AAPL", cache)

        assert abs(summary["revenue_growth_yoy"] - 0.10) < 0.01

    def test_positive_fcf_flagged(self, cache: Cache):
        income = [_income_stmt(100, 50, 2.0)] * 8
        handler = _fmp_router(income=income, cashflow=[_cashflow(10_000_000)] * 4)
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*financialmodelingprep.*").mock(side_effect=handler)
            summary = get_fundamentals_summary("MSFT", cache)

        assert summary["has_positive_fcf"] == 1

    def test_negative_fcf_flagged(self, cache: Cache):
        income = [_income_stmt(100, 50, 2.0)] * 8
        handler = _fmp_router(income=income, cashflow=[_cashflow(-5_000_000)] * 4)
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*financialmodelingprep.*").mock(side_effect=handler)
            summary = get_fundamentals_summary("BURN", cache)

        assert summary["has_positive_fcf"] == 0

    def test_revenue_growth_gt10_flag(self, cache: Cache):
        income = [_income_stmt(115, 57, 2.0)] + [_income_stmt(100, 50, 1.8)] * 7
        handler = _fmp_router(income=income)
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*financialmodelingprep.*").mock(side_effect=handler)
            summary = get_fundamentals_summary("GROW", cache)

        assert summary["revenue_growth_gt10"] == 1

    def test_api_error_returns_defaults(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*financialmodelingprep.*").mock(
                return_value=httpx.Response(500)
            )
            summary = get_fundamentals_summary("ERR", cache)

        assert summary["revenue_growth_yoy"] == 0.0
        assert summary["has_positive_fcf"] == 0

    def test_gross_margin_trend_expanding(self, cache: Cache):
        # Margin goes from 0.40 → 0.45 → 0.48 → 0.50 (expanding, latest first)
        income = [
            _income_stmt(100, 50, 2.0),   # latest: 50% margin
            _income_stmt(100, 48, 1.9),
            _income_stmt(100, 45, 1.8),
            _income_stmt(100, 40, 1.7),   # oldest: 40% margin
        ] * 2
        handler = _fmp_router(income=income)
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*financialmodelingprep.*").mock(side_effect=handler)
            summary = get_fundamentals_summary("GROW2", cache)

        assert summary["gross_margin_trend"] == 1  # expanding


class TestFundamentalsGate:
    def test_passes_with_positive_fcf_and_low_debt(self):
        summary = {"has_positive_fcf": 1, "revenue_growth_gt10": 0, "debt_to_equity": 1.5}
        assert passes_fundamentals_gate(summary) is True

    def test_passes_via_growth_exception(self):
        summary = {"has_positive_fcf": 0, "revenue_growth_gt10": 1, "debt_to_equity": 2.0}
        assert passes_fundamentals_gate(summary) is True

    def test_fails_no_fcf_no_growth(self):
        summary = {"has_positive_fcf": 0, "revenue_growth_gt10": 0, "debt_to_equity": 1.0}
        assert passes_fundamentals_gate(summary) is False

    def test_fails_extreme_debt(self):
        summary = {"has_positive_fcf": 1, "revenue_growth_gt10": 1, "debt_to_equity": 15.0}
        assert passes_fundamentals_gate(summary) is False

    def test_empty_summary_fails(self):
        assert passes_fundamentals_gate({}) is False


class TestRiskKeywords:
    def test_finds_risk_keywords(self):
        text = (
            "Item 1A. Risk Factors\n"
            "We face significant litigation and regulatory risk.\n"
            "Our debt levels may adversely affect operations.\n"
            "Competition in our market is intense.\n"
            "Item 2. Management Discussion\n"
        )
        keywords = _extract_risk_keywords(text)
        assert "litigation" in keywords
        assert "regulatory" in keywords
        assert "debt" in keywords

    def test_empty_text_returns_empty_set(self):
        assert _extract_risk_keywords("") == set()

    def test_no_risk_section_returns_empty_set(self):
        assert isinstance(_extract_risk_keywords("No risk section here."), set)


class TestEpsSurpriseStreak:
    def test_consecutive_beats(self):
        from quantly.data.catalysts import get_eps_surprise_streak
        earnings = [
            {"actualEarningResult": 2.0, "estimatedEarning": 1.8},
            {"actualEarningResult": 1.9, "estimatedEarning": 1.7},
            {"actualEarningResult": 1.5, "estimatedEarning": 1.4},
        ]
        assert get_eps_surprise_streak(earnings) == 3

    def test_miss_breaks_streak(self):
        from quantly.data.catalysts import get_eps_surprise_streak
        earnings = [
            {"actualEarningResult": 2.0, "estimatedEarning": 1.8},
            {"actualEarningResult": 1.5, "estimatedEarning": 1.6},  # miss
            {"actualEarningResult": 1.4, "estimatedEarning": 1.2},
        ]
        assert get_eps_surprise_streak(earnings) == 1

    def test_empty_list(self):
        from quantly.data.catalysts import get_eps_surprise_streak
        assert get_eps_surprise_streak([]) == 0
