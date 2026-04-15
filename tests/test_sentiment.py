"""Tests for all sentiment data sources — all HTTP mocked with respx."""
from __future__ import annotations

from datetime import date, timedelta

import httpx
import pytest
import respx

from quantly.data.cache import Cache
from quantly.data.sentiment.news import get_news_sentiment
from quantly.data.sentiment.gdelt import get_gdelt_tone
from quantly.data.sentiment.stocktwits import get_stocktwits_sentiment
from quantly.data.sentiment.congress import get_congress_sentiment
from quantly.data.sentiment.options_flow import get_options_flow


# ═══════════════════════════════════════════════════════════════════════════════
# Alpha Vantage news sentiment
# ═══════════════════════════════════════════════════════════════════════════════

def _av_payload(ticker: str, scores: list[tuple[float, float]]) -> dict:
    """Build an Alpha Vantage NEWS_SENTIMENT response.
    scores: list of (sentiment_score, relevance_score) tuples.
    """
    feed = []
    for score, relevance in scores:
        feed.append({
            "ticker_sentiment": [{
                "ticker": ticker,
                "ticker_sentiment_score": str(score),
                "relevance_score": str(relevance),
            }]
        })
    return {"feed": feed}


class TestNewsSentiment:
    def test_positive_sentiment(self, cache: Cache):
        payload = _av_payload("AAPL", [(0.5, 0.9), (0.3, 0.8), (0.4, 0.7)])
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*alphavantage.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = get_news_sentiment("AAPL", cache)

        assert result["avg_sentiment"] > 0
        assert result["article_count"] == 3
        assert result["bullish_pct"] > 0

    def test_negative_sentiment(self, cache: Cache):
        payload = _av_payload("TSLA", [(-0.4, 0.9), (-0.5, 0.8)])
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*alphavantage.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = get_news_sentiment("TSLA", cache)

        assert result["avg_sentiment"] < 0

    def test_empty_feed_returns_neutral(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*alphavantage.*").mock(
                return_value=httpx.Response(200, json={"feed": []})
            )
            result = get_news_sentiment("NVDA", cache)

        assert result["avg_sentiment"] == 0.0
        assert result["article_count"] == 0

    def test_api_error_returns_neutral(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*alphavantage.*").mock(
                return_value=httpx.Response(500)
            )
            result = get_news_sentiment("ERR", cache)

        assert result["avg_sentiment"] == 0.0

    def test_caches_result(self, cache: Cache):
        payload = _av_payload("MSFT", [(0.3, 0.8)])
        call_count = 0

        def _handler(req):
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json=payload)

        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*alphavantage.*").mock(side_effect=_handler)
            get_news_sentiment("MSFT", cache)
            get_news_sentiment("MSFT", cache)

        assert call_count == 1

    def test_ignores_irrelevant_ticker_articles(self, cache: Cache):
        """Articles mentioning a different ticker should not be counted."""
        payload = {
            "feed": [{
                "ticker_sentiment": [
                    {"ticker": "OTHER", "ticker_sentiment_score": "0.8", "relevance_score": "0.9"}
                ]
            }]
        }
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*alphavantage.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = get_news_sentiment("AAPL", cache)

        assert result["article_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# GDELT tone
# ═══════════════════════════════════════════════════════════════════════════════

class TestGdeltTone:
    def test_positive_tone_normalised(self, cache: Cache):
        payload = {"articles": [{"tone": 40.0}, {"tone": 30.0}, {"tone": 50.0}]}
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*gdeltproject.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = get_gdelt_tone("AAPL", cache)

        assert 0 < result["tone"] <= 1.0
        assert result["article_count"] == 3

    def test_negative_tone(self, cache: Cache):
        payload = {"articles": [{"tone": -40.0}, {"tone": -50.0}]}
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*gdeltproject.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = get_gdelt_tone("TSLA", cache)

        assert result["tone"] < 0

    def test_empty_articles_returns_zero(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*gdeltproject.*").mock(
                return_value=httpx.Response(200, json={"articles": []})
            )
            result = get_gdelt_tone("NVDA", cache)

        assert result["tone"] == 0.0
        assert result["article_count"] == 0

    def test_api_error_returns_zero(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*gdeltproject.*").mock(
                return_value=httpx.Response(500)
            )
            result = get_gdelt_tone("ERR", cache)

        assert result["tone"] == 0.0

    def test_tone_clamped_to_minus_one_plus_one(self, cache: Cache):
        payload = {"articles": [{"tone": 999.0}, {"tone": -999.0}]}
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*gdeltproject.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = get_gdelt_tone("CLAMP", cache)

        assert -1.0 <= result["tone"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# StockTwits
# ═══════════════════════════════════════════════════════════════════════════════

def _st_payload(bullish: int, bearish: int) -> dict:
    messages = (
        [{"entities": {"sentiment": {"basic": "Bullish"}}}] * bullish
        + [{"entities": {"sentiment": {"basic": "Bearish"}}}] * bearish
    )
    return {"messages": messages}


class TestStockTwitsSentiment:
    def test_mostly_bullish(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*stocktwits.*").mock(
                return_value=httpx.Response(200, json=_st_payload(8, 2))
            )
            result = get_stocktwits_sentiment("AAPL", cache)

        assert result["bullish_pct"] == 0.8
        assert result["sentiment"] == pytest.approx(0.6, abs=0.01)
        assert result["message_count"] == 10

    def test_mostly_bearish(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*stocktwits.*").mock(
                return_value=httpx.Response(200, json=_st_payload(1, 9))
            )
            result = get_stocktwits_sentiment("TSLA", cache)

        assert result["bullish_pct"] == pytest.approx(0.1, abs=0.01)
        assert result["sentiment"] < 0

    def test_empty_messages_returns_neutral(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*stocktwits.*").mock(
                return_value=httpx.Response(200, json={"messages": []})
            )
            result = get_stocktwits_sentiment("NVDA", cache)

        assert result["sentiment"] == 0.0
        assert result["message_count"] == 0

    def test_api_error_returns_neutral(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*stocktwits.*").mock(
                return_value=httpx.Response(500)
            )
            result = get_stocktwits_sentiment("ERR", cache)

        assert result["sentiment"] == 0.0

    def test_sentiment_maps_50pct_to_zero(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*stocktwits.*").mock(
                return_value=httpx.Response(200, json=_st_payload(5, 5))
            )
            result = get_stocktwits_sentiment("FLAT", cache)

        assert result["sentiment"] == pytest.approx(0.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# Congressional trading
# ═══════════════════════════════════════════════════════════════════════════════

def _congress_trade(transaction: str, days_ago: int = 5) -> dict:
    d = (date.today() - timedelta(days=days_ago)).isoformat()
    return {"Transaction": transaction, "Date": d, "Amount": "$1,001 - $15,000"}


class TestCongressSentiment:
    def test_recent_buys_counted(self, cache: Cache):
        payload = [
            _congress_trade("Purchase", days_ago=10),
            _congress_trade("Purchase", days_ago=20),
            _congress_trade("Sale", days_ago=5),
        ]
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*quiverquant.*congresstrading.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = get_congress_sentiment("AAPL", cache, lookback_days=90)

        assert result["recent_buys"] == 2
        assert result["recent_sells"] == 1
        assert result["net_congress"] == 1

    def test_old_trades_excluded(self, cache: Cache):
        payload = [_congress_trade("Purchase", days_ago=200)]  # beyond 90-day window
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*quiverquant.*congresstrading.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = get_congress_sentiment("MSFT", cache, lookback_days=90)

        assert result["recent_buys"] == 0

    def test_api_error_returns_zeros(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*quiverquant.*").mock(
                return_value=httpx.Response(500)
            )
            result = get_congress_sentiment("ERR", cache)

        assert result["net_congress"] == 0
        assert result["latest_buy_days"] == 99

    def test_latest_buy_days_computed(self, cache: Cache):
        payload = [_congress_trade("Purchase", days_ago=3)]
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*quiverquant.*congresstrading.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = get_congress_sentiment("NVDA", cache)

        assert result["latest_buy_days"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Options flow
# ═══════════════════════════════════════════════════════════════════════════════

def _options_row(calls: float, puts: float, days_ago: int = 0) -> dict:
    d = (date.today() - timedelta(days=days_ago)).isoformat()
    return {"Date": d, "CallVolume": calls, "PutVolume": puts}


class TestOptionsFlow:
    def test_unusual_call_ratio_above_one(self, cache: Cache):
        # 30 days of baseline (1000 calls), then recent spike (5000 calls)
        payload = (
            [_options_row(1000, 800, days_ago=i) for i in range(5, 35)]
            + [_options_row(5000, 500, days_ago=i) for i in range(0, 5)]
        )
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*quiverquant.*options.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = get_options_flow("AAPL", cache)

        assert result["unusual_call_ratio"] > 1.0

    def test_low_put_call_ratio_is_bullish(self, cache: Cache):
        payload = [_options_row(2000, 400, days_ago=i) for i in range(35)]
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*quiverquant.*options.*").mock(
                return_value=httpx.Response(200, json=payload)
            )
            result = get_options_flow("NVDA", cache)

        assert result["put_call_ratio"] < 1.0

    def test_empty_data_returns_neutral(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*quiverquant.*options.*").mock(
                return_value=httpx.Response(200, json=[])
            )
            result = get_options_flow("EMPTY", cache)

        assert result["unusual_call_ratio"] == 1.0
        assert result["put_call_ratio"] == 1.0

    def test_api_error_returns_neutral(self, cache: Cache):
        with respx.mock(assert_all_called=False) as m:
            m.get(url__regex=r".*quiverquant.*options.*").mock(
                return_value=httpx.Response(500)
            )
            result = get_options_flow("ERR", cache)

        assert result["net_options_score"] == 0.0
