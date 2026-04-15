"""Shared pytest fixtures used across the test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

from quantly.config import Config, reset_config
from quantly.data.cache import Cache


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Ensure each test starts with a fresh Config singleton."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """A Config instance pointing all paths at a temp directory."""
    cfg = Config()
    cfg.data_dir = tmp_path
    return cfg


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    """A Cache instance backed by a temp SQLite file."""
    return Cache(tmp_path / "test_cache.db")
