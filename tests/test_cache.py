"""Tests for quantly.data.cache — validates TTL, persistence, prefix ops."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from quantly.data.cache import Cache


class TestBasicGetSet:
    def test_set_and_get(self, cache: Cache):
        cache.set("k1", {"value": 42}, ttl_seconds=60)
        assert cache.get("k1") == {"value": 42}

    def test_get_missing_returns_none(self, cache: Cache):
        assert cache.get("does_not_exist") is None

    def test_overwrite_existing_key(self, cache: Cache):
        cache.set("k2", "first", ttl_seconds=60)
        cache.set("k2", "second", ttl_seconds=60)
        assert cache.get("k2") == "second"

    def test_delete_removes_key(self, cache: Cache):
        cache.set("k3", "data", ttl_seconds=60)
        cache.delete("k3")
        assert cache.get("k3") is None

    def test_delete_nonexistent_is_safe(self, cache: Cache):
        cache.delete("ghost")  # should not raise


class TestTTLExpiry:
    def test_expired_entry_returns_none(self, cache: Cache):
        cache.set("exp", "data", ttl_seconds=1)
        time.sleep(1.1)
        assert cache.get("exp") is None

    def test_fresh_entry_survives(self, cache: Cache):
        cache.set("fresh", "alive", ttl_seconds=60)
        assert cache.get("fresh") == "alive"


class TestGetOrFetch:
    def test_fetches_on_cache_miss(self, cache: Cache):
        calls: list[int] = []

        def fetch():
            calls.append(1)
            return "fetched"

        result = cache.get_or_fetch("new_key", fetch, ttl_seconds=60)
        assert result == "fetched"
        assert len(calls) == 1

    def test_uses_cache_on_hit(self, cache: Cache):
        cache.set("hit_key", "cached", ttl_seconds=60)
        calls: list[int] = []

        def fetch():
            calls.append(1)
            return "should_not_be_called"

        result = cache.get_or_fetch("hit_key", fetch, ttl_seconds=60)
        assert result == "cached"
        assert len(calls) == 0

    def test_second_call_uses_cache(self, cache: Cache):
        calls: list[int] = []

        def fetch():
            calls.append(1)
            return "value"

        cache.get_or_fetch("once", fetch, ttl_seconds=60)
        cache.get_or_fetch("once", fetch, ttl_seconds=60)
        assert len(calls) == 1


class TestBulkOperations:
    def test_purge_expired_removes_stale(self, cache: Cache):
        cache.set("e1", "x", ttl_seconds=1)
        cache.set("e2", "x", ttl_seconds=1)
        cache.set("keep", "x", ttl_seconds=60)
        time.sleep(1.1)
        removed = cache.purge_expired()
        assert removed == 2
        assert cache.get("keep") == "x"

    def test_clear_prefix_removes_matching(self, cache: Cache):
        cache.set("ticker:AAPL:price", "a", ttl_seconds=60)
        cache.set("ticker:MSFT:price", "b", ttl_seconds=60)
        cache.set("universe:sp500", "c", ttl_seconds=60)
        removed = cache.clear_prefix("ticker:")
        assert removed == 2
        assert cache.get("universe:sp500") == "c"

    def test_clear_all(self, cache: Cache):
        cache.set("a", 1, ttl_seconds=60)
        cache.set("b", 2, ttl_seconds=60)
        removed = cache.clear_all()
        assert removed == 2
        assert cache.get("a") is None


class TestComplexTypes:
    def test_stores_and_retrieves_dataframe(self, cache: Cache):
        df = pd.DataFrame({"open": [100.0, 101.0], "close": [102.0, 103.0]})
        cache.set("df", df, ttl_seconds=60)
        result = cache.get("df")
        pd.testing.assert_frame_equal(df, result)

    def test_stores_list_of_dicts(self, cache: Cache):
        data = [{"ticker": "AAPL", "score": 0.87}, {"ticker": "MSFT", "score": 0.72}]
        cache.set("picks", data, ttl_seconds=60)
        assert cache.get("picks") == data

    def test_stores_nested_dict(self, cache: Cache):
        data = {"meta": {"version": 1}, "values": [1, 2, 3]}
        cache.set("nested", data, ttl_seconds=60)
        assert cache.get("nested") == data


class TestPersistence:
    def test_survives_new_cache_instance(self, tmp_path: Path):
        """Data written by one Cache instance is readable by another on same file."""
        path = tmp_path / "persist.db"
        c1 = Cache(path)
        c1.set("persistent", "hello", ttl_seconds=3600)

        c2 = Cache(path)
        assert c2.get("persistent") == "hello"
