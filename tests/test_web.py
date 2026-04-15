"""Tests for the FastAPI web dashboard.

Uses FastAPI's TestClient with the shadow book fully mocked so tests are
deterministic and require no real DB or network I/O.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantly.web.app import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """TestClient that patches the scheduler so it doesn't fire."""
    with (
        patch("quantly.web.app._start_scheduler"),
        patch("quantly.web.app.shadow_db.init_db"),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def _pending_pick(pick_id: int = 1, ticker: str = "AAPL") -> dict:
    return {
        "id": pick_id,
        "ticker": ticker,
        "conviction": 80.0,
        "signal_date": "2024-01-02",
        "entry_price": 185.0,
        "stop_price": 179.0,
        "max_hold_date": "2024-02-01",
        "status": "pending",
        "shap_drivers": [{"feature": "rsi_14", "value": 0.3}],
    }


def _open_position(pos_id: int = 1, ticker: str = "AAPL") -> dict:
    return {
        "id": pos_id,
        "ticker": ticker,
        "entry_date": "2024-01-02",
        "entry_price": 185.0,
        "current_price": 195.0,
        "shares": 10,
        "stop_price": 179.0,
        "max_hold_date": "2024-02-01",
        "status": "open",
        "unrealized_pnl": 100.0,
        "cost_basis": 1850.0,
        "market_value": 1950.0,
        "return_pct": 0.054,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════════
# Picks page
# ═══════════════════════════════════════════════════════════════════════════════

class TestPicksPage:
    def test_renders_picks(self, client):
        pick = _pending_pick()
        with (
            patch("quantly.web.app._get_book") as mock_book_fn,
        ):
            book = MagicMock()
            book.get_picks_for_date.return_value = [pick]
            mock_book_fn.return_value = book

            resp = client.get("/")
        assert resp.status_code == 200
        assert "AAPL" in resp.text
        assert "80" in resp.text  # conviction

    def test_empty_picks_shows_placeholder(self, client):
        with patch("quantly.web.app._get_book") as mock_book_fn:
            book = MagicMock()
            book.get_picks_for_date.return_value = []
            book.get_pending_picks.return_value = []
            mock_book_fn.return_value = book

            resp = client.get("/")
        assert resp.status_code == 200
        assert "No picks yet" in resp.text

    def test_falls_back_to_pending_when_no_today(self, client):
        pick = _pending_pick()
        with patch("quantly.web.app._get_book") as mock_book_fn:
            book = MagicMock()
            book.get_picks_for_date.return_value = []  # nothing for today
            book.get_pending_picks.return_value = [pick]  # but pending exists
            mock_book_fn.return_value = book

            resp = client.get("/")
        assert resp.status_code == 200
        assert "AAPL" in resp.text

    def test_approved_pick_shows_badge(self, client):
        pick = {**_pending_pick(), "status": "approved"}
        with patch("quantly.web.app._get_book") as mock_book_fn:
            book = MagicMock()
            book.get_picks_for_date.return_value = [pick]
            mock_book_fn.return_value = book

            resp = client.get("/")
        assert "Approved" in resp.text


class TestApproveRejectPick:
    def test_approve_returns_badge(self, client):
        with (
            patch("quantly.web.app._get_book") as mock_book_fn,
            patch("quantly.web.app._get_cache"),
            patch("quantly.web.app.get_current_price", return_value=185.0),
        ):
            book = MagicMock()
            book.get_pending_picks.return_value = [_pending_pick(pick_id=1)]
            mock_book_fn.return_value = book

            resp = client.post("/picks/1/approve")
        assert resp.status_code == 200
        assert "Approved" in resp.text
        book.approve_pick.assert_called_once_with(1)

    def test_approve_unknown_pick_returns_404(self, client):
        with (
            patch("quantly.web.app._get_book") as mock_book_fn,
            patch("quantly.web.app._get_cache"),
        ):
            book = MagicMock()
            book.get_pending_picks.return_value = []
            mock_book_fn.return_value = book

            resp = client.post("/picks/9999/approve")
        assert resp.status_code == 404

    def test_reject_returns_badge(self, client):
        with patch("quantly.web.app._get_book") as mock_book_fn:
            book = MagicMock()
            mock_book_fn.return_value = book

            resp = client.post("/picks/1/reject")
        assert resp.status_code == 200
        assert "Rejected" in resp.text
        book.reject_pick.assert_called_once_with(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Shadow Book page
# ═══════════════════════════════════════════════════════════════════════════════

class TestBookPage:
    def test_renders_open_positions(self, client):
        pos = _open_position()
        with (
            patch("quantly.web.app._get_book") as mock_book_fn,
            patch("quantly.web.app._get_cache"),
            patch("quantly.web.app.get_current_price", return_value=195.0),
        ):
            book = MagicMock()
            book.get_open_positions.return_value = [pos]
            pnl_df = pd.DataFrame([pos])
            book.unrealized_pnl.return_value = pnl_df
            mock_book_fn.return_value = book

            resp = client.get("/book")
        assert resp.status_code == 200
        assert "AAPL" in resp.text
        assert "185" in resp.text  # entry price

    def test_empty_book_shows_placeholder(self, client):
        with (
            patch("quantly.web.app._get_book") as mock_book_fn,
            patch("quantly.web.app._get_cache"),
        ):
            book = MagicMock()
            book.get_open_positions.return_value = []
            book.unrealized_pnl.return_value = pd.DataFrame()
            mock_book_fn.return_value = book

            resp = client.get("/book")
        assert resp.status_code == 200
        assert "No open positions" in resp.text


class TestExitPosition:
    def test_exit_returns_confirmation(self, client):
        pos = {
            "id": 1, "ticker": "AAPL", "entry_price": 185.0,
            "shares": 10, "status": "open",
        }
        with patch("quantly.web.app._get_book") as mock_book_fn:
            book = MagicMock()
            book.get_position.return_value = pos
            mock_book_fn.return_value = book

            resp = client.post("/book/1/exit", data={"exit_price": "200.0"})
        assert resp.status_code == 200
        assert "200.00" in resp.text or "Exited" in resp.text
        book.close_position.assert_called_once()

    def test_exit_unknown_position_returns_404(self, client):
        with patch("quantly.web.app._get_book") as mock_book_fn:
            book = MagicMock()
            book.get_position.return_value = None
            mock_book_fn.return_value = book

            resp = client.post("/book/9999/exit", data={"exit_price": "200.0"})
        assert resp.status_code == 404

    def test_exit_closed_position_returns_404(self, client):
        with patch("quantly.web.app._get_book") as mock_book_fn:
            book = MagicMock()
            book.get_position.return_value = {"id": 1, "status": "closed"}
            mock_book_fn.return_value = book

            resp = client.post("/book/1/exit", data={"exit_price": "200.0"})
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# Performance page
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformancePage:
    def _make_nav(self) -> pd.DataFrame:
        return pd.DataFrame({
            "nav_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "portfolio_value": [10000.0, 10500.0],
            "cash": [5000.0, 5000.0],
            "positions_value": [5000.0, 5500.0],
        })

    def test_renders_metrics(self, client):
        trade_df = pd.DataFrame({
            "ticker": ["AAPL"],
            "return_pct": [0.08],
            "entry_price": [185.0],
            "exit_price": [200.0],
            "shares": [10],
            "entry_date": ["2024-01-02"],
            "exit_date": ["2024-01-20"],
        })
        nav = self._make_nav()

        with (
            patch("quantly.web.app._get_book") as mock_book_fn,
            patch("quantly.web.app._get_cache"),
            patch("quantly.web.app.get_ohlcv", return_value=pd.DataFrame()),
        ):
            book = MagicMock()
            book.get_trade_history.return_value = trade_df
            book.get_nav_history.return_value = nav
            mock_book_fn.return_value = book

            resp = client.get("/performance")
        assert resp.status_code == 200
        assert "Win Rate" in resp.text
        assert "Sharpe" in resp.text
        assert "Max Drawdown" in resp.text

    def test_empty_history_renders_without_error(self, client):
        with (
            patch("quantly.web.app._get_book") as mock_book_fn,
            patch("quantly.web.app._get_cache"),
        ):
            book = MagicMock()
            book.get_trade_history.return_value = pd.DataFrame()
            book.get_nav_history.return_value = pd.DataFrame()
            mock_book_fn.return_value = book

            resp = client.get("/performance")
        assert resp.status_code == 200
        assert "No NAV history" in resp.text or "Performance" in resp.text

    def test_closed_trades_shown(self, client):
        trade_df = pd.DataFrame({
            "ticker": ["TSLA"],
            "return_pct": [0.12],
            "entry_price": [250.0],
            "exit_price": [280.0],
            "shares": [4],
            "entry_date": ["2024-01-02"],
            "exit_date": ["2024-01-20"],
        })
        with (
            patch("quantly.web.app._get_book") as mock_book_fn,
            patch("quantly.web.app._get_cache"),
            patch("quantly.web.app.get_ohlcv", return_value=pd.DataFrame()),
        ):
            book = MagicMock()
            book.get_trade_history.return_value = trade_df
            book.get_nav_history.return_value = pd.DataFrame()
            mock_book_fn.return_value = book

            resp = client.get("/performance")
        assert resp.status_code == 200
        assert "TSLA" in resp.text


# ═══════════════════════════════════════════════════════════════════════════════
# Admin — pipeline trigger
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineTrigger:
    def test_run_pipeline_returns_success(self, client):
        # run_pipeline is imported inside the route function body, so patch the source
        with patch("quantly.pipeline.run_pipeline", return_value=[{"ticker": "AAPL"}]):
            resp = client.post("/pipeline/run")
        assert resp.status_code == 200
        assert "Pipeline" in resp.text or "complete" in resp.text
