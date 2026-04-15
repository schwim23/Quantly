"""SQLite schema and low-level queries for the shadow book.

Tables
------
positions
    One row per trade (open or closed). Shadow trades approved from the dashboard.

picks
    Raw signal output from the pipeline: ranked candidates per signal date.
    Status tracks whether the user approved, rejected, or ignored the pick.

daily_nav
    End-of-day portfolio snapshots: total NAV, cash, and mark-to-market
    positions value.  Used for drawdown tracking and vs-SPY chart.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Generator, Optional

from quantly.config import get_config

# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    entry_date      TEXT    NOT NULL,   -- ISO date YYYY-MM-DD
    entry_price     REAL    NOT NULL,
    shares          INTEGER NOT NULL,
    stop_price      REAL    NOT NULL,
    max_hold_date   TEXT    NOT NULL,   -- ISO date
    status          TEXT    NOT NULL DEFAULT 'open',  -- open | closed
    exit_date       TEXT,              -- NULL until closed
    exit_price      REAL,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS picks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_date     TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    conviction      REAL    NOT NULL,  -- 0–100
    shap_drivers    TEXT,              -- JSON array of {feature, value}
    entry_price     REAL,
    stop_price      REAL,
    max_hold_date   TEXT,
    status          TEXT    NOT NULL DEFAULT 'pending',  -- pending | approved | rejected
    UNIQUE(signal_date, ticker)
);

CREATE TABLE IF NOT EXISTS daily_nav (
    nav_date            TEXT PRIMARY KEY,  -- ISO date
    portfolio_value     REAL NOT NULL,
    cash                REAL NOT NULL,
    positions_value     REAL NOT NULL
);
"""


# ── Connection helpers ────────────────────────────────────────────────────────

def _db_path() -> Path:
    return get_config().shadow_book_db_path


def init_db(path: Optional[Path] = None) -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    db = path or _db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(_DDL)


@contextmanager
def get_conn(path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding an auto-committing sqlite3 connection."""
    db = path or _db_path()
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Positions ─────────────────────────────────────────────────────────────────

def insert_position(
    ticker: str,
    entry_date: date,
    entry_price: float,
    shares: int,
    stop_price: float,
    max_hold_date: date,
    notes: str = "",
    path: Optional[Path] = None,
) -> int:
    """Insert an open position. Returns the new row id."""
    with get_conn(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO positions
                (ticker, entry_date, entry_price, shares, stop_price,
                 max_hold_date, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, 'open', ?)
            """,
            (
                ticker,
                entry_date.isoformat(),
                entry_price,
                shares,
                stop_price,
                max_hold_date.isoformat(),
                notes,
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]


def close_position(
    position_id: int,
    exit_date: date,
    exit_price: float,
    path: Optional[Path] = None,
) -> None:
    """Mark a position as closed."""
    with get_conn(path) as conn:
        conn.execute(
            """
            UPDATE positions
               SET status = 'closed',
                   exit_date  = ?,
                   exit_price = ?
             WHERE id = ? AND status = 'open'
            """,
            (exit_date.isoformat(), exit_price, position_id),
        )


def update_stop(
    position_id: int,
    new_stop: float,
    path: Optional[Path] = None,
) -> None:
    """Update the trailing stop price for an open position."""
    with get_conn(path) as conn:
        conn.execute(
            "UPDATE positions SET stop_price = ? WHERE id = ? AND status = 'open'",
            (new_stop, position_id),
        )


def get_open_positions(path: Optional[Path] = None) -> list[dict]:
    with get_conn(path) as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE status = 'open' ORDER BY entry_date"
        ).fetchall()
    return [dict(r) for r in rows]


def get_position(position_id: int, path: Optional[Path] = None) -> Optional[dict]:
    with get_conn(path) as conn:
        row = conn.execute(
            "SELECT * FROM positions WHERE id = ?", (position_id,)
        ).fetchone()
    return dict(row) if row else None


def get_closed_positions(path: Optional[Path] = None) -> list[dict]:
    with get_conn(path) as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE status = 'closed' ORDER BY exit_date"
        ).fetchall()
    return [dict(r) for r in rows]


# ── Picks ─────────────────────────────────────────────────────────────────────

def upsert_pick(
    signal_date: date,
    ticker: str,
    conviction: float,
    shap_drivers: list[dict] | None = None,
    entry_price: float | None = None,
    stop_price: float | None = None,
    max_hold_date: date | None = None,
    path: Optional[Path] = None,
) -> None:
    """Insert or update a pick (idempotent on signal_date+ticker)."""
    shap_json = json.dumps(shap_drivers) if shap_drivers else None
    max_hold_str = max_hold_date.isoformat() if max_hold_date else None
    with get_conn(path) as conn:
        conn.execute(
            """
            INSERT INTO picks
                (signal_date, ticker, conviction, shap_drivers,
                 entry_price, stop_price, max_hold_date, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
            ON CONFLICT(signal_date, ticker) DO UPDATE SET
                conviction   = excluded.conviction,
                shap_drivers = excluded.shap_drivers,
                entry_price  = excluded.entry_price,
                stop_price   = excluded.stop_price,
                max_hold_date= excluded.max_hold_date
            """,
            (
                signal_date.isoformat(),
                ticker,
                conviction,
                shap_json,
                entry_price,
                stop_price,
                max_hold_str,
            ),
        )


def update_pick_status(
    pick_id: int,
    status: str,
    path: Optional[Path] = None,
) -> None:
    """Set pick status to 'approved' or 'rejected'."""
    with get_conn(path) as conn:
        conn.execute(
            "UPDATE picks SET status = ? WHERE id = ?",
            (status, pick_id),
        )


def get_pending_picks(path: Optional[Path] = None) -> list[dict]:
    with get_conn(path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM picks
             WHERE status = 'pending'
             ORDER BY signal_date DESC, conviction DESC
            """
        ).fetchall()
    parsed = []
    for r in rows:
        d = dict(r)
        if d.get("shap_drivers"):
            d["shap_drivers"] = json.loads(d["shap_drivers"])
        parsed.append(d)
    return parsed


def get_picks_for_date(
    signal_date: date,
    path: Optional[Path] = None,
) -> list[dict]:
    with get_conn(path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM picks
             WHERE signal_date = ?
             ORDER BY conviction DESC
            """,
            (signal_date.isoformat(),),
        ).fetchall()
    parsed = []
    for r in rows:
        d = dict(r)
        if d.get("shap_drivers"):
            d["shap_drivers"] = json.loads(d["shap_drivers"])
        parsed.append(d)
    return parsed


# ── Daily NAV ─────────────────────────────────────────────────────────────────

def record_daily_nav(
    nav_date: date,
    portfolio_value: float,
    cash: float,
    positions_value: float,
    path: Optional[Path] = None,
) -> None:
    """Upsert today's portfolio snapshot."""
    with get_conn(path) as conn:
        conn.execute(
            """
            INSERT INTO daily_nav (nav_date, portfolio_value, cash, positions_value)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(nav_date) DO UPDATE SET
                portfolio_value = excluded.portfolio_value,
                cash            = excluded.cash,
                positions_value = excluded.positions_value
            """,
            (nav_date.isoformat(), portfolio_value, cash, positions_value),
        )


def get_nav_history(path: Optional[Path] = None) -> list[dict]:
    with get_conn(path) as conn:
        rows = conn.execute(
            "SELECT * FROM daily_nav ORDER BY nav_date"
        ).fetchall()
    return [dict(r) for r in rows]
