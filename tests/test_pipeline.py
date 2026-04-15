"""Tests for the daily pipeline orchestrator.

Strategy: mock all external sub-systems (screen_universe, build_feature_matrix,
score_candidates, get_fundamentals_summary, get_current_price, ShadowBook) so
the tests exercise the pipeline's orchestration logic without any network or
disk I/O.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantly.pipeline import run_pipeline, _get_atr, _run_exit_checks, _record_nav_snapshot


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scores(tickers: list[str]) -> pd.DataFrame:
    """Return a mock scores DataFrame as score_candidates would return."""
    rows = []
    for i, ticker in enumerate(tickers):
        rows.append({
            "ticker": ticker,
            "conviction": 80.0 - i * 5,
            "top_drivers": ["rsi_14", "momentum_5d", "news_sentiment"],
            "shap_values": {"rsi_14": 0.3, "momentum_5d": 0.2, "news_sentiment": 0.1},
        })
    return pd.DataFrame(rows).set_index("ticker")


def _make_features(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [[0.0] * 5] * len(tickers),
        index=tickers,
        columns=["rsi_14", "momentum_5d", "news_sentiment", "days_to_earnings", "fcf_yield"],
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRunPipelineOrchestration:
    """Tests the end-to-end orchestration flow with all sub-systems mocked."""

    @patch("quantly.pipeline.get_current_price", return_value=150.0)
    @patch("quantly.pipeline.get_fundamentals_summary", return_value={"fcf_ttm": 1.0})
    @patch("quantly.pipeline.passes_fundamentals_gate", return_value=True)
    @patch("quantly.pipeline.score_candidates")
    @patch("quantly.pipeline.build_feature_matrix")
    @patch("quantly.pipeline.screen_universe", return_value=["AAPL", "MSFT", "NVDA"])
    @patch("quantly.pipeline.get_latest_snapshot", return_value=pd.DataFrame())
    @patch("quantly.pipeline.load_model")
    def test_happy_path_returns_picks(
        self, mock_model, mock_snapshot, mock_screen,
        mock_features, mock_scores, mock_gate, mock_funda, mock_price,
        tmp_path,
    ):
        mock_model.return_value = MagicMock()
        mock_features.return_value = _make_features(["AAPL", "MSFT", "NVDA"])
        mock_scores.return_value = _make_scores(["AAPL", "MSFT", "NVDA"])

        book = MagicMock()
        book.get_open_positions.return_value = []
        cache = MagicMock()

        picks = run_pipeline(
            as_of=date(2024, 1, 2),
            book=book,
            cache=cache,
        )
        assert len(picks) == 3
        assert picks[0]["conviction"] >= picks[-1]["conviction"]  # sorted desc
        assert all("entry_price" in p for p in picks)
        assert all("stop_price" in p for p in picks)
        assert all("max_hold_date" in p for p in picks)

    @patch("quantly.pipeline.load_model", return_value=None)
    def test_no_model_returns_empty(self, mock_model, tmp_path):
        book = MagicMock()
        cache = MagicMock()
        picks = run_pipeline(tickers=["AAPL"], as_of=date(2024, 1, 2), book=book, cache=cache)
        assert picks == []

    @patch("quantly.pipeline.get_current_price", return_value=150.0)
    @patch("quantly.pipeline.passes_fundamentals_gate", return_value=False)
    @patch("quantly.pipeline.get_fundamentals_summary", return_value={})
    @patch("quantly.pipeline.score_candidates")
    @patch("quantly.pipeline.build_feature_matrix")
    @patch("quantly.pipeline.load_model")
    def test_all_fail_fundamentals_gate_returns_empty(
        self, mock_model, mock_features, mock_scores, mock_funda, mock_gate, mock_price,
    ):
        mock_model.return_value = MagicMock()
        mock_features.return_value = _make_features(["AAPL"])
        mock_scores.return_value = _make_scores(["AAPL"])

        book = MagicMock()
        book.get_open_positions.return_value = []
        cache = MagicMock()

        picks = run_pipeline(tickers=["AAPL"], as_of=date(2024, 1, 2), book=book, cache=cache)
        assert picks == []

    @patch("quantly.pipeline.get_current_price", return_value=None)
    @patch("quantly.pipeline.passes_fundamentals_gate", return_value=True)
    @patch("quantly.pipeline.get_fundamentals_summary", return_value={"fcf_ttm": 1.0})
    @patch("quantly.pipeline.score_candidates")
    @patch("quantly.pipeline.build_feature_matrix")
    @patch("quantly.pipeline.load_model")
    def test_no_price_skips_ticker(
        self, mock_model, mock_features, mock_scores, mock_funda, mock_gate, mock_price,
    ):
        mock_model.return_value = MagicMock()
        mock_features.return_value = _make_features(["AAPL"])
        mock_scores.return_value = _make_scores(["AAPL"])

        book = MagicMock()
        book.get_open_positions.return_value = []
        cache = MagicMock()

        picks = run_pipeline(tickers=["AAPL"], as_of=date(2024, 1, 2), book=book, cache=cache)
        assert picks == []

    @patch("quantly.pipeline.build_feature_matrix", return_value=pd.DataFrame())
    @patch("quantly.pipeline.load_model")
    def test_empty_feature_matrix_returns_empty(self, mock_model, mock_features):
        mock_model.return_value = MagicMock()
        book = MagicMock()
        book.get_open_positions.return_value = []
        cache = MagicMock()

        picks = run_pipeline(tickers=["AAPL"], as_of=date(2024, 1, 2), book=book, cache=cache)
        assert picks == []

    @patch("quantly.pipeline.get_current_price", return_value=150.0)
    @patch("quantly.pipeline.passes_fundamentals_gate", return_value=True)
    @patch("quantly.pipeline.get_fundamentals_summary", return_value={"fcf_ttm": 1.0})
    @patch("quantly.pipeline.score_candidates")
    @patch("quantly.pipeline.build_feature_matrix")
    @patch("quantly.pipeline.load_model")
    def test_picks_saved_to_book(
        self, mock_model, mock_features, mock_scores, mock_funda, mock_gate, mock_price,
    ):
        mock_model.return_value = MagicMock()
        mock_features.return_value = _make_features(["AAPL", "MSFT"])
        mock_scores.return_value = _make_scores(["AAPL", "MSFT"])

        book = MagicMock()
        book.get_open_positions.return_value = []
        cache = MagicMock()

        run_pipeline(tickers=["AAPL", "MSFT"], as_of=date(2024, 1, 2), book=book, cache=cache)
        book.save_picks.assert_called_once()
        saved_picks = book.save_picks.call_args[0][0]
        assert len(saved_picks) == 2

    @patch("quantly.pipeline.get_current_price", return_value=150.0)
    @patch("quantly.pipeline.passes_fundamentals_gate", return_value=True)
    @patch("quantly.pipeline.get_fundamentals_summary", return_value={"fcf_ttm": 1.0})
    @patch("quantly.pipeline.score_candidates")
    @patch("quantly.pipeline.build_feature_matrix")
    @patch("quantly.pipeline.load_model")
    def test_picks_sorted_by_conviction_desc(
        self, mock_model, mock_features, mock_scores, mock_funda, mock_gate, mock_price,
    ):
        mock_model.return_value = MagicMock()
        mock_features.return_value = _make_features(["AAPL", "MSFT", "NVDA"])
        mock_scores.return_value = _make_scores(["AAPL", "MSFT", "NVDA"])

        book = MagicMock()
        book.get_open_positions.return_value = []
        cache = MagicMock()

        picks = run_pipeline(tickers=["AAPL", "MSFT", "NVDA"], as_of=date(2024, 1, 2),
                              book=book, cache=cache)
        convictions = [p["conviction"] for p in picks]
        assert convictions == sorted(convictions, reverse=True)

    def test_ticker_override_skips_screen(self):
        """When tickers are provided, screen_universe must NOT be called."""
        with (
            patch("quantly.pipeline.screen_universe") as mock_screen,
            patch("quantly.pipeline.load_model", return_value=None),
        ):
            book = MagicMock()
            book.get_open_positions.return_value = []
            cache = MagicMock()
            run_pipeline(tickers=["AAPL"], as_of=date(2024, 1, 2), book=book, cache=cache)
            mock_screen.assert_not_called()


class TestRunExitChecks:
    def test_closes_stopped_out_positions(self, tmp_path):
        book = MagicMock()
        book.get_open_positions.return_value = [
            {
                "id": 1, "ticker": "AAPL",
                "entry_date": "2024-01-02", "entry_price": 185.0,
                "stop_price": 179.0, "shares": 10,
            }
        ]
        book.check_exits.return_value = [
            {
                "id": 1, "ticker": "AAPL",
                "entry_date": "2024-01-02", "entry_price": 185.0,
                "stop_price": 179.0, "shares": 10,
                "exit_reason": "stop",
            }
        ]
        cache = MagicMock()
        with patch("quantly.pipeline.get_current_price", return_value=177.0):
            _run_exit_checks(book, cache, date(2024, 1, 15))
        book.close_position.assert_called_once_with(1, date(2024, 1, 15), 177.0)

    def test_ratchets_stop_for_survivors(self):
        book = MagicMock()
        book.get_open_positions.return_value = [
            {
                "id": 2, "ticker": "MSFT",
                "entry_date": "2024-01-02", "entry_price": 400.0,
                "stop_price": 388.0, "shares": 5,
            }
        ]
        book.check_exits.return_value = []  # no exits
        book.ratchet_stop.return_value = 420.0  # 430 - 2*5 = 420 > 388
        cache = MagicMock()

        with (
            patch("quantly.pipeline.get_current_price", return_value=430.0),
            patch("quantly.pipeline._get_atr", return_value=5.0),
        ):
            _run_exit_checks(book, cache, date(2024, 1, 10))

        book.ratchet_stop.assert_called_once_with(2, 430.0, 5.0)

    def test_no_open_positions_does_nothing(self):
        book = MagicMock()
        book.get_open_positions.return_value = []
        cache = MagicMock()
        _run_exit_checks(book, cache, date(2024, 1, 10))
        book.close_position.assert_not_called()
        book.ratchet_stop.assert_not_called()


class TestRecordNavSnapshot:
    def test_records_nav_with_open_positions(self):
        book = MagicMock()
        book.get_open_positions.return_value = [
            {"ticker": "AAPL", "entry_price": 185.0, "shares": 10},
        ]
        cache = MagicMock()

        with patch("quantly.pipeline.get_current_price", return_value=200.0):
            _record_nav_snapshot(book, cache, date(2024, 1, 10))

        book.record_nav.assert_called_once()
        call_args = book.record_nav.call_args[0]
        nav_date, portfolio_value, cash, positions_value = call_args
        assert nav_date == date(2024, 1, 10)
        assert positions_value == pytest.approx(2000.0)  # 200 * 10

    def test_records_nav_with_no_positions(self):
        book = MagicMock()
        book.get_open_positions.return_value = []
        cache = MagicMock()

        _record_nav_snapshot(book, cache, date(2024, 1, 10))
        book.record_nav.assert_called_once()
        _, portfolio_value, cash, positions_value = book.record_nav.call_args[0]
        assert positions_value == 0.0
