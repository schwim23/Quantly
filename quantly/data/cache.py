"""Local disk cache for API responses.

Avoids redundant API calls and respects rate limits.
Cache entries expire after a configurable TTL.
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

DEFAULT_TTL = 86_400  # 24 hours in seconds


def _cache_path(namespace: str, key: str) -> Path:
    hashed = hashlib.md5(key.encode()).hexdigest()
    path = config.CACHE_DIR / namespace
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{hashed}.pkl"


def cache_get(namespace: str, key: str, ttl: int = DEFAULT_TTL) -> Any | None:
    path = _cache_path(namespace, key)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > ttl:
        path.unlink()
        return None
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def cache_set(namespace: str, key: str, value: Any) -> None:
    path = _cache_path(namespace, key)
    with path.open("wb") as f:
        pickle.dump(value, f)


def cached(namespace: str, ttl: int = DEFAULT_TTL) -> Callable:
    """Decorator: cache the return value of a function by its arguments."""
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
