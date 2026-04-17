"""Local disk cache for API/scrape responses.

Avoids redundant fetches and respects rate limits.
Entries expire after a configurable TTL (default 24 h).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from quantly.config import config

logger = logging.getLogger(__name__)


def _cache_path(namespace: str, key: str) -> Path:
    hashed = hashlib.md5(key.encode()).hexdigest()  # noqa: S324
    path = config.CACHE_DIR / namespace
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{hashed}.pkl"


def cache_get(namespace: str, key: str, ttl: int | None = None) -> Any | None:
    ttl = ttl if ttl is not None else config.CACHE_TTL
    path = _cache_path(namespace, key)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > ttl:
        path.unlink(missing_ok=True)
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301
    except Exception:
        path.unlink(missing_ok=True)
        return None


def cache_set(namespace: str, key: str, value: Any) -> None:
    path = _cache_path(namespace, key)
    with path.open("wb") as f:
        pickle.dump(value, f)


def cached(namespace: str, ttl: int | None = None) -> Callable:
    """Decorator: cache the return value by function arguments."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
            result = cache_get(namespace, key, ttl)
            if result is not None:
                logger.debug("Cache hit: %s", namespace)
                return result
            result = fn(*args, **kwargs)
            cache_set(namespace, key, result)
            return result
        return wrapper
    return decorator
