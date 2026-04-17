"""Reddit sentiment via PRAW (free tier — needs app registration only).

Register a free app at https://www.reddit.com/prefs/apps (type: script).
Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env.

If Reddit credentials are not configured, this module returns empty results
gracefully — the pipeline continues without Reddit sentiment.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta, timezone

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from quantly.config import config
from quantly.data.cache import cached

logger = logging.getLogger(__name__)
_analyzer = SentimentIntensityAnalyzer()

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options"]


@cached("reddit_mentions", ttl=3_600)
def get_ticker_mentions(
    ticker: str,
    days_back: int = 7,
) -> pd.DataFrame:
    """Fetch recent Reddit posts mentioning a ticker.

    Returns empty DataFrame if Reddit credentials are not configured.

    Returns:
        DataFrame with columns: created_at, subreddit, title, score, num_comments
    """
    if not config.reddit_enabled:
        logger.debug("Reddit not configured — skipping for %s", ticker)
        return pd.DataFrame(columns=["created_at", "subreddit", "title", "score", "num_comments"])

    import praw

    reddit = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
    )

    rows = []
    query = f"{ticker} stock"
    cutoff = (date.today() - timedelta(days=days_back)).isoformat()

    for sub_name in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.search(query, sort="new", time_filter="week", limit=25):
                rows.append({
                    "created_at": pd.Timestamp.fromtimestamp(post.created_utc, tz=timezone.utc),
                    "subreddit": sub_name,
                    "title": post.title,
                    "score": post.score,
                    "num_comments": post.num_comments,
                })
        except Exception as e:
            logger.debug("Reddit search failed for r/%s: %s", sub_name, e)

    if not rows:
        return pd.DataFrame(columns=["created_at", "subreddit", "title", "score", "num_comments"])

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df.sort_values("created_at", ascending=False).reset_index(drop=True)


def score_reddit_mentions(mentions: pd.DataFrame) -> pd.DataFrame:
    """Add VADER sentiment to each Reddit post title."""
    if mentions.empty:
        return mentions.assign(vader_compound=float("nan"))
    df = mentions.copy()
    df["vader_compound"] = df["title"].apply(
        lambda t: _analyzer.polarity_scores(str(t))["compound"]
    )
    return df


def compute_reddit_features(ticker: str) -> dict[str, float]:
    """Compute all Reddit-derived features for a ticker.

    Returns:
        Dict with keys: reddit_mention_count_7d, reddit_avg_sentiment,
        reddit_velocity_ratio (7d vs prior 7d), reddit_available
    """
    if not config.reddit_enabled:
        return {
            "reddit_mention_count_7d": float("nan"),
            "reddit_avg_sentiment": float("nan"),
            "reddit_velocity_ratio": float("nan"),
            "reddit_available": 0.0,
        }

    mentions = get_ticker_mentions(ticker, days_back=14)
    if mentions.empty:
        return {
            "reddit_mention_count_7d": 0.0,
            "reddit_avg_sentiment": 0.0,
            "reddit_velocity_ratio": 1.0,
            "reddit_available": 1.0,
        }

    scored = score_reddit_mentions(mentions)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    recent = scored[scored["created_at"] >= cutoff]
    prior = scored[scored["created_at"] < cutoff]

    recent_count = len(recent)
    prior_count = max(len(prior), 1)
    velocity = recent_count / prior_count

    avg_sentiment = float(recent["vader_compound"].mean()) if not recent.empty else 0.0

    return {
        "reddit_mention_count_7d": float(recent_count),
        "reddit_avg_sentiment": avg_sentiment,
        "reddit_velocity_ratio": velocity,
        "reddit_available": 1.0,
    }
