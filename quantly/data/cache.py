"""SQLite-backed disk cache to avoid repeat API calls and respect free-tier rate limits.

Keys are namespaced strings, e.g. "tiingo:AAPL:2024-01-15".
Values are pickled Python objects (DataFrames, dicts, lists, etc.).
TTL is enforced on read — expired entries are lazily deleted.
"""
from __future__ import annotations

import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

# Default TTLs (seconds)
TTL_PRICE = 4 * 3600          # price data: 4 hours
TTL_FUNDAMENTAL = 24 * 3600   # fundamentals: 24 hours
TTL_SENTIMENT = 2 * 3600      # sentiment/news: 2 hours
TTL_UNIVERSE = 24 * 3600      # universe list: 24 hours
TTL_CATALYST = 6 * 3600       # earnings calendar: 6 hours


class Cache:
    """Thread-safe SQLite cache with TTL expiry and prefix-based invalidation."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key        TEXT    PRIMARY KEY,
                    value      BLOB    NOT NULL,
                    expires_at REAL    NOT NULL,
                    created_at REAL    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Return cached value for *key*, or None if missing / expired."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        if time.time() > row["expires_at"]:
            self.delete(key)
            return None
        return pickle.loads(row["value"])

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Store *value* under *key* with a TTL of *ttl_seconds*."""
        now = time.time()
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache (key, value, expires_at, created_at)
                   VALUES (?, ?, ?, ?)""",
                (key, blob, now + ttl_seconds, now),
            )

    def delete(self, key: str) -> None:
        """Remove *key* from the cache."""
        with self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def get_or_fetch(
        self,
        key: str,
        fetch_fn: Callable[[], T],
        ttl_seconds: int = 3600,
    ) -> T:
        """Return cached value if present; otherwise call *fetch_fn*, cache, and return."""
        cached = self.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        result = fetch_fn()
        self.set(key, result, ttl_seconds)
        return result

    def purge_expired(self) -> int:
        """Delete all expired entries. Returns number of rows removed."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM cache WHERE expires_at < ?", (time.time(),)
            )
            return cursor.rowcount

    def clear_prefix(self, prefix: str) -> int:
        """Delete all keys whose name starts with *prefix*. Returns count removed."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM cache WHERE key LIKE ?", (f"{prefix}%",)
            )
            return cursor.rowcount

    def clear_all(self) -> int:
        """Wipe the entire cache. Returns count removed."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM cache")
            return cursor.rowcount
