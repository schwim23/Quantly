"""Global configuration — loaded from .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    # Required
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

    # Optional — Reddit sentiment (free tier, needs app registration)
    REDDIT_CLIENT_ID: str = os.environ.get("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET: str = os.environ.get("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT: str = os.environ.get("REDDIT_USER_AGENT", "quantly/0.1")

    @property
    def reddit_enabled(self) -> bool:
        return bool(self.REDDIT_CLIENT_ID and self.REDDIT_CLIENT_SECRET)

    # Pipeline settings
    UNIVERSE: str = os.environ.get("QUANTLY_UNIVERSE", "sp500")
    MIN_VOLUME: int = int(os.environ.get("QUANTLY_MIN_VOLUME", "500000"))
    TOP_N_FOR_CLAUDE: int = int(os.environ.get("QUANTLY_TOP_N_FOR_CLAUDE", "20"))
    CACHE_TTL: int = int(os.environ.get("QUANTLY_CACHE_TTL", "86400"))

    # Storage
    CACHE_DIR: Path = Path(os.environ.get("QUANTLY_CACHE_DIR", "data/cache"))
    DB_PATH: Path = Path(os.environ.get("QUANTLY_DB_PATH", "data/paper_trades.db"))

    # Label construction
    PREDICTION_HORIZON_DAYS: int = 15       # trading days
    OUTPERFORMANCE_THRESHOLD: float = 0.03  # 3% above SPY = positive label
    UNDERPERFORMANCE_THRESHOLD: float = 0.02

    # Claude
    CLAUDE_MODEL: str = "claude-opus-4-6"
    CLAUDE_MAX_TOKENS: int = 4096


config = Config()
