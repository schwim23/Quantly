"""Shadow book: high-level position tracker built on top of db.py.

This is the interface used by the pipeline and the web dashboard.
It abstracts raw SQL behind domain-language methods and handles
derived calculations (unrealized P&L, stop ratchet, exit triggers).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from quantly.risk.stops import (
    compute_max_hold_date,
    is_max_hold_exceeded,
    is_stopped_out,
    update_trailing_stop,
)
from quantly.risk.sizing import compute_position_size
from quantly.shadow_book import db


class ShadowBook:
    """High-level interface for the shadow trading book.

    Args:
        db_path: Override the default DB path (useful in tests).
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._path = db_path
        db.init_db(db_path)

    # ── Opening / closing positions ───────────────────────────────────────────

    def open_position(
        self,
        ticker: str,
        entry_date: date,
        entry_price: float,
        shares: int,
        stop_price: float,
        notes: str = "",
    ) -> int:
        """Add a new open position.  Returns the DB row id."""
        max_hold = compute_max_hold_date(entry_date)
        return db.insert_position(
            ticker=ticker,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
            stop_price=stop_price,
            max_hold_date=max_hold,
            notes=notes,
            path=self._path,
        )

    def close_position(
        self,
        position_id: int,
        exit_date: date,
        exit_price: float,
    ) -> None:
        """Mark a position closed at the given price."""
        db.close_position(position_id, exit_date, exit_price, path=self._path)

    # ── Stop management ───────────────────────────────────────────────────────

    def ratchet_stop(
        self,
        position_id: int,
        current_price: float,
        atr: float,
    ) -> float:
        """Ratchet the trailing stop upward if warranted. Returns new stop."""
        pos = db.get_position(position_id, path=self._path)
        if pos is None or pos["status"] != "open":
            raise ValueError(f"Position {position_id} not found or not open")
        new_stop = update_trailing_stop(pos["stop_price"], current_price, atr)
        if new_stop > pos["stop_price"]:
            db.update_stop(position_id, new_stop, path=self._path)
        return new_stop

    # ── Exit checks ───────────────────────────────────────────────────────────

    def check_exits(
        self,
        current_prices: dict[str, float],
        as_of: Optional[date] = None,
    ) -> list[dict]:
        """Return list of positions that should be exited today.

        Checks:
            1. Stop-out: current price ≤ stop_price
            2. Max-hold: as_of ≥ max_hold_date

        Returns:
            List of position dicts with an extra ``exit_reason`` key.
        """
        as_of = as_of or date.today()
        to_exit = []
        for pos in db.get_open_positions(path=self._path):
            ticker = pos["ticker"]
            price = current_prices.get(ticker)
            if price is None:
                continue
            entry = date.fromisoformat(pos["entry_date"])
            if is_stopped_out(price, pos["stop_price"]):
                pos["exit_reason"] = "stop"
                to_exit.append(pos)
            elif is_max_hold_exceeded(entry, as_of=as_of):
                pos["exit_reason"] = "max_hold"
                to_exit.append(pos)
        return to_exit

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_open_positions(self) -> list[dict]:
        return db.get_open_positions(path=self._path)

    def get_position(self, position_id: int) -> Optional[dict]:
        return db.get_position(position_id, path=self._path)

    def get_trade_history(self) -> pd.DataFrame:
        """Return all closed positions as a DataFrame with return_pct column."""
        rows = db.get_closed_positions(path=self._path)
        if not rows:
            return pd.DataFrame(columns=[
                "id", "ticker", "entry_date", "entry_price", "shares",
                "exit_date", "exit_price", "return_pct",
            ])
        df = pd.DataFrame(rows)
        df["return_pct"] = (df["exit_price"] - df["entry_price"]) / df["entry_price"]
        return df

    # ── NAV snapshots ─────────────────────────────────────────────────────────

    def record_nav(
        self,
        nav_date: date,
        portfolio_value: float,
        cash: float,
        positions_value: float,
    ) -> None:
        db.record_daily_nav(nav_date, portfolio_value, cash, positions_value,
                            path=self._path)

    def get_nav_history(self) -> pd.DataFrame:
        rows = db.get_nav_history(path=self._path)
        if not rows:
            return pd.DataFrame(columns=["nav_date", "portfolio_value",
                                         "cash", "positions_value"])
        df = pd.DataFrame(rows)
        df["nav_date"] = pd.to_datetime(df["nav_date"]).dt.date
        return df

    # ── Unrealized P&L ────────────────────────────────────────────────────────

    def unrealized_pnl(self, current_prices: dict[str, float]) -> pd.DataFrame:
        """Return open positions enriched with current P&L columns."""
        positions = db.get_open_positions(path=self._path)
        if not positions:
            return pd.DataFrame()
        rows = []
        for pos in positions:
            price = current_prices.get(pos["ticker"])
            if price is None:
                price = pos["entry_price"]
            cost_basis = pos["entry_price"] * pos["shares"]
            mkt_value = price * pos["shares"]
            rows.append({
                **pos,
                "current_price": price,
                "cost_basis": round(cost_basis, 2),
                "market_value": round(mkt_value, 2),
                "unrealized_pnl": round(mkt_value - cost_basis, 2),
                "return_pct": round((price - pos["entry_price"]) / pos["entry_price"], 4),
            })
        return pd.DataFrame(rows)

    # ── Picks ─────────────────────────────────────────────────────────────────

    def save_picks(self, picks: list[dict], signal_date: date) -> None:
        """Persist a list of pick dicts from the pipeline to the DB."""
        for pick in picks:
            db.upsert_pick(
                signal_date=signal_date,
                ticker=pick["ticker"],
                conviction=pick.get("conviction", 0.0),
                shap_drivers=pick.get("shap_drivers"),
                entry_price=pick.get("entry_price"),
                stop_price=pick.get("stop_price"),
                max_hold_date=pick.get("max_hold_date"),
                path=self._path,
            )

    def get_pending_picks(self) -> list[dict]:
        return db.get_pending_picks(path=self._path)

    def approve_pick(self, pick_id: int) -> None:
        db.update_pick_status(pick_id, "approved", path=self._path)

    def reject_pick(self, pick_id: int) -> None:
        db.update_pick_status(pick_id, "rejected", path=self._path)

    def get_picks_for_date(self, signal_date: date) -> list[dict]:
        return db.get_picks_for_date(signal_date, path=self._path)
