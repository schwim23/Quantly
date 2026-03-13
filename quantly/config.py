"""Global configuration loaded from environment variables and .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    # API keys
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
    POLYGON_API_KEY: str = os.environ.get("POLYGON_API_KEY", "")
    ALPHA_VANTAGE_API_KEY: str = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    BLOOMBERG_API_KEY: str = os.environ.get("BLOOMBERG_API_KEY", "")
    REDDIT_CLIENT_ID: str = os.environ.get("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET: str = os.environ.get("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT: str = os.environ.get("REDDIT_USER_AGENT", "quantly/0.1")

    # Pipeline settings
    UNIVERSE: str = os.environ.get("QUANTLY_UNIVERSE", "all_us")
    MIN_VOLUME: int = int(os.environ.get("QUANTLY_MIN_VOLUME", "500000"))
    TOP_N_FOR_CLAUDE: int = int(os.environ.get("QUANTLY_TOP_N_FOR_CLAUDE", "20"))

    # Storage paths
    CACHE_DIR: Path = Path(os.environ.get("QUANTLY_CACHE_DIR", "data/cache"))
    DB_PATH: Path = Path(os.environ.get("QUANTLY_DB_PATH", "data/paper_trades.db"))

    # Model settings
    PREDICTION_HORIZON_DAYS: int = 15       # trading days for label construction
    OUTPERFORMANCE_THRESHOLD: float = 0.03  # 3% outperformance vs SPY = positive label
    UNDERPERFORMANCE_THRESHOLD: float = 0.02

    # Claude model
    CLAUDE_MODEL: str = "claude-opus-4-6"
    CLAUDE_MAX_TOKENS: int = 4096


config = Config()
