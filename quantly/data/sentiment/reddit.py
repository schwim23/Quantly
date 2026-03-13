"""Fetch and score Reddit sentiment from finance-related subreddits.

Subreddits monitored: wallstreetbets, stocks, investing, options.
Uses PRAW (Python Reddit API Wrapper).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from quantly.config import config
from quantly.data.cache import cached

logger = logging.getLogger(__name__)

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options"]


@cached("reddit")
def get_ticker_mentions(
    ticker: str,
    start: date,
    end: date,
    subreddits: list[str] = SUBREDDITS,
) -> pd.DataFrame:
    """Fetch Reddit posts and comments mentioning a ticker.

    Returns:
        DataFrame with columns: created_at, subreddit, title, text, score, num_comments
    """
    import praw

    reddit = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
    )
    # TODO: implement pushshift or PRAW search across subreddits
    raise NotImplementedError


def compute_mention_velocity(
    mentions: pd.DataFrame,
    window_days: int = 7,
) -> dict[str, float]:
    """Compute how fast mention volume is growing.

    Returns:
        Dict with keys: mentions_last_7d, mentions_prior_7d, velocity_ratio, avg_sentiment
    """
    # TODO: implement rolling mention count and growth rate
    raise NotImplementedError
