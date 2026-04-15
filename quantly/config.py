"""Central configuration loaded from environment / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── API keys ──────────────────────────────────────────────────────────────
    tiingo_api_key: str = field(default_factory=lambda: os.getenv("TIINGO_API_KEY", ""))
    fmp_api_key: str = field(default_factory=lambda: os.getenv("FMP_API_KEY", ""))
    alpha_vantage_api_key: str = field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY", "")
    )
    finnhub_api_key: str = field(default_factory=lambda: os.getenv("FINNHUB_API_KEY", ""))
    quiver_api_key: str = field(default_factory=lambda: os.getenv("QUIVER_API_KEY", ""))
    unusual_whales_api_key: str = field(
        default_factory=lambda: os.getenv("UNUSUAL_WHALES_API_KEY", "")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "data"))
    )

    # ── Capital & position sizing ─────────────────────────────────────────────
    portfolio_value: float = field(
        default_factory=lambda: float(os.getenv("PORTFOLIO_VALUE", "10000"))
    )
    max_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_POSITIONS", "15"))
    )
    min_position_pct: float = 0.07   # 7% of portfolio minimum
    max_position_pct: float = 0.10   # 10% of portfolio maximum
    max_loss_per_position_pct: float = 0.02  # 2% of portfolio max loss on a single position

    # ── Risk management ───────────────────────────────────────────────────────
    atr_stop_multiplier: float = 2.0   # stop = entry - 2 × ATR(14)
    max_hold_days: int = 20            # force exit after 20 trading days
    drawdown_pause_threshold: float = 0.12   # pause new entries at -12% portfolio drawdown
    drawdown_resume_threshold: float = 0.08  # resume when recovered to -8%
    reentry_cooldown_days: int = 5     # days before same ticker can re-enter after stop-out

    # ── Universe ──────────────────────────────────────────────────────────────
    min_dollar_volume: float = 5_000_000   # $5M avg daily dollar volume (20-day)
    min_price: float = 5.0                 # no penny stocks

    # ── ML pipeline ──────────────────────────────────────────────────────────
    top_candidates: int = 20          # candidates passed to the fundamentals gate
    min_conviction_score: float = 60.0  # minimum score to display in dashboard

    # ── Scheduler ────────────────────────────────────────────────────────────
    scan_hour: int = 7    # 7 am ET pre-market
    scan_minute: int = 0

    # ── Derived paths (computed properties) ──────────────────────────────────
    @property
    def cache_db_path(self) -> Path:
        path = self.data_dir / "cache.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def shadow_book_db_path(self) -> Path:
        path = self.data_dir / "shadow_book.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def model_path(self) -> Path:
        path = self.data_dir / "model.lgb"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


_config: Config | None = None


def get_config() -> Config:
    """Return the global singleton Config, creating it on first call."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset singleton — used in tests to pick up monkeypatched env vars."""
    global _config
    _config = None
