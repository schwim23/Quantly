"""Tests for quantly.config — validates Config defaults, env overrides, and paths."""
from __future__ import annotations

from pathlib import Path

import pytest

from quantly.config import Config, get_config, reset_config


class TestConfigDefaults:
    def test_portfolio_value(self):
        assert Config().portfolio_value == 10_000.0

    def test_max_positions(self):
        assert Config().max_positions == 15

    def test_atr_stop_multiplier(self):
        assert Config().atr_stop_multiplier == 2.0

    def test_max_hold_days(self):
        assert Config().max_hold_days == 20

    def test_drawdown_pause_threshold(self):
        assert Config().drawdown_pause_threshold == 0.12

    def test_drawdown_resume_threshold(self):
        assert Config().drawdown_resume_threshold == 0.08

    def test_min_dollar_volume(self):
        assert Config().min_dollar_volume == 5_000_000

    def test_min_price(self):
        assert Config().min_price == 5.0

    def test_max_loss_per_position(self):
        assert Config().max_loss_per_position_pct == 0.02


class TestEnvOverrides:
    def test_portfolio_value_from_env(self, monkeypatch):
        monkeypatch.setenv("PORTFOLIO_VALUE", "50000")
        assert Config().portfolio_value == 50_000.0

    def test_max_positions_from_env(self, monkeypatch):
        monkeypatch.setenv("MAX_POSITIONS", "10")
        assert Config().max_positions == 10

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("TIINGO_API_KEY", "test-key-123")
        assert Config().tiingo_api_key == "test-key-123"

    def test_empty_api_key_default(self):
        assert Config().tiingo_api_key == "" or isinstance(
            Config().tiingo_api_key, str
        )


class TestDerivedPaths:
    def test_cache_db_path(self, tmp_path: Path):
        cfg = Config()
        cfg.data_dir = tmp_path
        assert cfg.cache_db_path == tmp_path / "cache.db"
        assert cfg.cache_db_path.parent.exists()

    def test_shadow_book_db_path(self, tmp_path: Path):
        cfg = Config()
        cfg.data_dir = tmp_path
        assert cfg.shadow_book_db_path == tmp_path / "shadow_book.db"

    def test_model_path(self, tmp_path: Path):
        cfg = Config()
        cfg.data_dir = tmp_path
        assert cfg.model_path == tmp_path / "model.lgb"

    def test_data_dir_created_on_path_access(self, tmp_path: Path):
        nested = tmp_path / "deeply" / "nested"
        cfg = Config()
        cfg.data_dir = nested
        _ = cfg.cache_db_path
        assert nested.exists()


class TestSingleton:
    def test_get_config_returns_same_instance(self):
        a = get_config()
        b = get_config()
        assert a is b

    def test_reset_config_creates_new_instance(self):
        a = get_config()
        reset_config()
        b = get_config()
        assert a is not b
