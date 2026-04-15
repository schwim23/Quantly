"""Tests for the shadow book — DB layer, ShadowBook class, performance analytics."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from quantly.shadow_book import db
from quantly.shadow_book.book import ShadowBook
from quantly.shadow_book.performance import (
    benchmark_vs_spy,
    compute_daily_returns,
    compute_trade_returns,
    portfolio_summary,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Return a fresh DB path in a temp directory."""
    p = tmp_path / "shadow_book.db"
    db.init_db(p)
    return p


@pytest.fixture
def book(tmp_db: Path) -> ShadowBook:
    return ShadowBook(db_path=tmp_db)


# ═══════════════════════════════════════════════════════════════════════════════
# DB layer
# ═══════════════════════════════════════════════════════════════════════════════

class TestDbInit:
    def test_creates_tables(self, tmp_db: Path):
        with db.get_conn(tmp_db) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert {"positions", "picks", "daily_nav"}.issubset(tables)

    def test_init_idempotent(self, tmp_db: Path):
        # Calling init_db again should not raise
        db.init_db(tmp_db)


class TestDbPositions:
    def test_insert_and_retrieve(self, tmp_db: Path):
        rid = db.insert_position(
            ticker="AAPL",
            entry_date=date(2024, 1, 2),
            entry_price=185.0,
            shares=10,
            stop_price=179.0,
            max_hold_date=date(2024, 2, 1),
            path=tmp_db,
        )
        pos = db.get_position(rid, path=tmp_db)
        assert pos is not None
        assert pos["ticker"] == "AAPL"
        assert pos["status"] == "open"

    def test_close_position(self, tmp_db: Path):
        rid = db.insert_position(
            "MSFT", date(2024, 1, 2), 400.0, 5, 388.0, date(2024, 2, 1),
            path=tmp_db,
        )
        db.close_position(rid, date(2024, 1, 20), 420.0, path=tmp_db)
        pos = db.get_position(rid, path=tmp_db)
        assert pos["status"] == "closed"
        assert pos["exit_price"] == 420.0

    def test_update_stop(self, tmp_db: Path):
        rid = db.insert_position(
            "NVDA", date(2024, 1, 2), 600.0, 3, 580.0, date(2024, 2, 1),
            path=tmp_db,
        )
        db.update_stop(rid, 595.0, path=tmp_db)
        assert db.get_position(rid, path=tmp_db)["stop_price"] == 595.0

    def test_get_open_positions_excludes_closed(self, tmp_db: Path):
        r1 = db.insert_position(
            "A", date(2024, 1, 2), 100.0, 10, 95.0, date(2024, 2, 1),
            path=tmp_db,
        )
        r2 = db.insert_position(
            "B", date(2024, 1, 2), 200.0, 5, 190.0, date(2024, 2, 1),
            path=tmp_db,
        )
        db.close_position(r1, date(2024, 1, 15), 110.0, path=tmp_db)
        open_positions = db.get_open_positions(path=tmp_db)
        tickers = [p["ticker"] for p in open_positions]
        assert "B" in tickers
        assert "A" not in tickers

    def test_unknown_position_returns_none(self, tmp_db: Path):
        assert db.get_position(9999, path=tmp_db) is None


class TestDbPicks:
    def test_upsert_and_retrieve(self, tmp_db: Path):
        db.upsert_pick(
            signal_date=date(2024, 1, 2),
            ticker="AAPL",
            conviction=75.0,
            shap_drivers=[{"feature": "rsi", "value": 0.8}],
            entry_price=185.0,
            stop_price=179.0,
            path=tmp_db,
        )
        picks = db.get_pending_picks(path=tmp_db)
        assert len(picks) == 1
        assert picks[0]["ticker"] == "AAPL"
        assert isinstance(picks[0]["shap_drivers"], list)

    def test_upsert_is_idempotent(self, tmp_db: Path):
        for _ in range(3):
            db.upsert_pick(
                signal_date=date(2024, 1, 2),
                ticker="TSLA",
                conviction=60.0,
                path=tmp_db,
            )
        picks = db.get_picks_for_date(date(2024, 1, 2), path=tmp_db)
        assert len(picks) == 1

    def test_update_status(self, tmp_db: Path):
        db.upsert_pick(
            signal_date=date(2024, 1, 2),
            ticker="GOOG",
            conviction=80.0,
            path=tmp_db,
        )
        picks = db.get_pending_picks(path=tmp_db)
        db.update_pick_status(picks[0]["id"], "approved", path=tmp_db)
        # Should no longer appear as pending
        assert len(db.get_pending_picks(path=tmp_db)) == 0


class TestDbDailyNav:
    def test_record_and_retrieve(self, tmp_db: Path):
        db.record_daily_nav(date(2024, 1, 2), 10_000.0, 5_000.0, 5_000.0, path=tmp_db)
        history = db.get_nav_history(path=tmp_db)
        assert len(history) == 1
        assert history[0]["portfolio_value"] == 10_000.0

    def test_upsert_updates_in_place(self, tmp_db: Path):
        db.record_daily_nav(date(2024, 1, 2), 10_000.0, 5_000.0, 5_000.0, path=tmp_db)
        db.record_daily_nav(date(2024, 1, 2), 10_500.0, 5_000.0, 5_500.0, path=tmp_db)
        history = db.get_nav_history(path=tmp_db)
        assert len(history) == 1
        assert history[0]["portfolio_value"] == 10_500.0


# ═══════════════════════════════════════════════════════════════════════════════
# ShadowBook class
# ═══════════════════════════════════════════════════════════════════════════════

class TestShadowBookPositions:
    def test_open_sets_max_hold_date(self, book: ShadowBook):
        rid = book.open_position(
            ticker="AAPL",
            entry_date=date(2024, 1, 2),
            entry_price=185.0,
            shares=10,
            stop_price=179.0,
        )
        pos = book.get_position(rid)
        assert pos["max_hold_date"] > pos["entry_date"]

    def test_close_records_exit(self, book: ShadowBook):
        rid = book.open_position("TSLA", date(2024, 1, 2), 250.0, 4, 240.0)
        book.close_position(rid, date(2024, 1, 20), 275.0)
        pos = book.get_position(rid)
        assert pos["status"] == "closed"
        assert pos["exit_price"] == 275.0

    def test_ratchet_stop_moves_up(self, book: ShadowBook):
        rid = book.open_position("NVDA", date(2024, 1, 2), 600.0, 3, 588.0)
        new_stop = book.ratchet_stop(rid, current_price=650.0, atr=6.0)
        # New ATR stop = 650 - 2*6 = 638 > 588 → should update
        assert new_stop > 588.0
        assert book.get_position(rid)["stop_price"] == new_stop

    def test_ratchet_stop_does_not_fall(self, book: ShadowBook):
        rid = book.open_position("META", date(2024, 1, 2), 400.0, 5, 390.0)
        # Price barely moves — ATR-based new stop (397 - 2*6 = 385) < current 390
        new_stop = book.ratchet_stop(rid, current_price=397.0, atr=6.0)
        assert new_stop == 390.0  # unchanged

    def test_ratchet_stop_on_missing_position_raises(self, book: ShadowBook):
        with pytest.raises(ValueError):
            book.ratchet_stop(9999, current_price=100.0, atr=2.0)

    def test_trade_history_includes_return_pct(self, book: ShadowBook):
        rid = book.open_position("AMZN", date(2024, 1, 2), 180.0, 5, 174.0)
        book.close_position(rid, date(2024, 1, 20), 198.0)
        hist = book.get_trade_history()
        assert not hist.empty
        assert "return_pct" in hist.columns
        assert hist.iloc[0]["return_pct"] == pytest.approx(0.10, abs=0.001)

    def test_trade_history_empty_when_no_closed_positions(self, book: ShadowBook):
        book.open_position("GOOG", date(2024, 1, 2), 150.0, 7, 144.0)
        hist = book.get_trade_history()
        assert hist.empty


class TestShadowBookExitChecks:
    def test_detects_stop_out(self, book: ShadowBook):
        book.open_position("AAPL", date(2024, 1, 2), 185.0, 10, 179.0)
        exits = book.check_exits({"AAPL": 178.0})
        assert len(exits) == 1
        assert exits[0]["exit_reason"] == "stop"

    def test_detects_max_hold_breach(self, book: ShadowBook):
        entry = date(2024, 1, 2)
        book.open_position("MSFT", entry, 400.0, 5, 388.0)
        # Well past max hold date
        far_future = entry + timedelta(days=60)
        exits = book.check_exits({"MSFT": 410.0}, as_of=far_future)
        assert len(exits) == 1
        assert exits[0]["exit_reason"] == "max_hold"

    def test_no_exit_when_price_above_stop_and_within_hold(self, book: ShadowBook):
        book.open_position("TSLA", date(2024, 1, 2), 250.0, 4, 240.0)
        exits = book.check_exits({"TSLA": 260.0}, as_of=date(2024, 1, 5))
        assert exits == []

    def test_skips_tickers_not_in_prices(self, book: ShadowBook):
        book.open_position("NVDA", date(2024, 1, 2), 600.0, 3, 588.0)
        exits = book.check_exits({})   # no prices provided
        assert exits == []


class TestShadowBookUnrealizedPnl:
    def test_positive_pnl_when_price_rises(self, book: ShadowBook):
        book.open_position("AAPL", date(2024, 1, 2), 185.0, 10, 179.0)
        df = book.unrealized_pnl({"AAPL": 200.0})
        assert df.iloc[0]["unrealized_pnl"] == pytest.approx(150.0)
        assert df.iloc[0]["return_pct"] == pytest.approx(0.0811, abs=0.001)

    def test_negative_pnl_when_price_falls(self, book: ShadowBook):
        book.open_position("META", date(2024, 1, 2), 400.0, 5, 390.0)
        df = book.unrealized_pnl({"META": 380.0})
        assert df.iloc[0]["unrealized_pnl"] == pytest.approx(-100.0)

    def test_empty_when_no_open_positions(self, book: ShadowBook):
        df = book.unrealized_pnl({"AAPL": 200.0})
        assert df.empty


class TestShadowBookPicks:
    def test_save_and_retrieve_picks(self, book: ShadowBook):
        picks = [
            {
                "ticker": "AAPL",
                "conviction": 82.0,
                "shap_drivers": [{"feature": "rsi", "value": 0.9}],
                "entry_price": 185.0,
                "stop_price": 179.0,
            },
            {"ticker": "MSFT", "conviction": 71.0},
        ]
        book.save_picks(picks, signal_date=date(2024, 1, 2))
        pending = book.get_pending_picks()
        assert len(pending) == 2
        tickers = {p["ticker"] for p in pending}
        assert tickers == {"AAPL", "MSFT"}

    def test_approve_removes_from_pending(self, book: ShadowBook):
        book.save_picks([{"ticker": "GOOG", "conviction": 78.0}], date(2024, 1, 2))
        pending = book.get_pending_picks()
        book.approve_pick(pending[0]["id"])
        assert len(book.get_pending_picks()) == 0

    def test_reject_removes_from_pending(self, book: ShadowBook):
        book.save_picks([{"ticker": "TSLA", "conviction": 55.0}], date(2024, 1, 2))
        pending = book.get_pending_picks()
        book.reject_pick(pending[0]["id"])
        assert len(book.get_pending_picks()) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Performance analytics
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeTradeReturns:
    def test_returns_series_from_history(self):
        df = pd.DataFrame({"return_pct": [0.05, -0.02, 0.08]})
        s = compute_trade_returns(df)
        assert len(s) == 3

    def test_empty_dataframe_gives_empty_series(self):
        s = compute_trade_returns(pd.DataFrame())
        assert s.empty


class TestComputeDailyReturns:
    def test_pct_change_computed(self):
        nav = pd.DataFrame({
            "nav_date": [date(2024, 1, i) for i in range(2, 8)],
            "portfolio_value": [10000, 10100, 10050, 10200, 10300, 10150],
        })
        r = compute_daily_returns(nav)
        assert len(r) == 5  # pct_change drops first

    def test_empty_returns_empty(self):
        assert compute_daily_returns(pd.DataFrame()).empty


class TestBenchmarkVsSpy:
    def _nav(self, values: list[float]) -> pd.DataFrame:
        start = date(2024, 1, 2)
        return pd.DataFrame({
            "nav_date": [start + timedelta(days=i) for i in range(len(values))],
            "portfolio_value": values,
        })

    def _spy(self, values: list[float], start: date = date(2024, 1, 2)) -> pd.Series:
        idx = [start + timedelta(days=i) for i in range(len(values))]
        return pd.Series(values, index=idx)

    def test_both_start_at_100(self):
        result = benchmark_vs_spy(self._nav([10000, 10500, 10800]), self._spy([450, 460, 470]))
        assert result.iloc[0]["portfolio"] == pytest.approx(100.0)
        assert result.iloc[0]["spy"] == pytest.approx(100.0)

    def test_portfolio_outperformance_shows(self):
        result = benchmark_vs_spy(
            self._nav([10000, 11000, 12000]),
            self._spy([450, 450, 450]),  # flat SPY
        )
        assert result.iloc[-1]["portfolio"] > result.iloc[-1]["spy"]

    def test_no_overlap_returns_empty(self):
        nav = self._nav([10000, 10500])
        spy = self._spy([450, 460], start=date(2030, 1, 1))
        result = benchmark_vs_spy(nav, spy)
        assert result.empty

    def test_empty_nav_returns_empty(self):
        result = benchmark_vs_spy(pd.DataFrame(), self._spy([450, 460]))
        assert result.empty


class TestPortfolioSummary:
    def test_keys_present(self):
        trades = pd.DataFrame({"return_pct": [0.05, -0.02, 0.08], "ticker": ["A", "B", "C"]})
        nav = pd.DataFrame({
            "nav_date": [date(2024, 1, i) for i in range(2, 8)],
            "portfolio_value": [10000, 10100, 10050, 10200, 10300, 10150],
        })
        summary = portfolio_summary(trades, nav)
        assert "sharpe" in summary
        assert "max_drawdown" in summary
        assert "win_rate" in summary
        assert "portfolio_return" in summary
        assert "spy_return" in summary

    def test_spy_return_computed_when_provided(self):
        nav = pd.DataFrame({
            "nav_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "portfolio_value": [10000.0, 10500.0],
        })
        spy = pd.Series([450.0, 459.0], index=[date(2024, 1, 2), date(2024, 1, 3)])
        summary = portfolio_summary(pd.DataFrame(), nav, spy_prices=spy)
        assert summary["spy_return"] == pytest.approx(0.02, abs=0.001)

    def test_empty_inputs_return_zeros(self):
        summary = portfolio_summary(pd.DataFrame(), pd.DataFrame())
        assert summary["win_rate"] == 0.0
        assert summary["portfolio_return"] == 0.0
        assert summary["spy_return"] is None
