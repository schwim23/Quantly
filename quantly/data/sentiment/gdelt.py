"""GDELT Project — global news tone scores, updated every 15 minutes.

GDELT is 100% free, Google-backed. We query the GKG (Global Knowledge Graph)
for tone scores on company-related articles over the past 7 days.

Tone score range: -100 (most negative) to +100 (most positive).
We normalise to [-1, +1] for consistency with other sentiment sources.

Ref: https://www.gdeltproject.org/
BigQuery or DOC endpoint used here (no key required).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import httpx

from quantly.data.cache import Cache, TTL_SENTIMENT

logger = logging.getLogger(__name__)

# GDELT DOC 2.0 API — returns JSON matching articles
_GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _company_name_from_ticker(ticker: str) -> str:
    """Very basic ticker→name mapping for GDELT queries.
    For unknown tickers we just use the ticker itself.
    """
    _MAP = {
        "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA",
        "AMZN": "Amazon", "GOOGL": "Google", "META": "Meta",
        "TSLA": "Tesla", "AMD": "AMD", "AVGO": "Broadcom",
        "QCOM": "Qualcomm", "INTC": "Intel", "MU": "Micron",
    }
    return _MAP.get(ticker.upper(), ticker)


def get_gdelt_tone(ticker: str, cache: Cache) -> dict[str, float]:
    """Return 7-day average GDELT tone score for *ticker*.

    Returns dict with keys:
        tone          - average tone normalised to [-1, +1]
        article_count - number of articles found
    """
    key = f"gdelt:tone:{ticker}:{date.today()}"

    def _fetch() -> dict[str, float]:
        query_term = _company_name_from_ticker(ticker)
        today = date.today()
        week_ago = today - timedelta(days=7)

        params = {
            "query": f'"{query_term}" sourcelang:english',
            "mode": "ArtList",
            "maxrecords": "50",
            "startdatetime": week_ago.strftime("%Y%m%d000000"),
            "enddatetime": today.strftime("%Y%m%d235959"),
            "format": "json",
        }

        try:
            resp = httpx.get(_GDELT_DOC_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("GDELT tone fetch for %s: %s", ticker, exc)
            return {"tone": 0.0, "article_count": 0}

        articles = data.get("articles", [])
        if not articles:
            return {"tone": 0.0, "article_count": 0}

        tones: list[float] = []
        for art in articles:
            try:
                tone_raw = float(art.get("tone", 0))
                # GDELT tone is comma-separated: "tone,positive,negative,polarity,..."
                # When parsed as JSON it's already a number
                tones.append(tone_raw)
            except (ValueError, TypeError):
                continue

        if not tones:
            return {"tone": 0.0, "article_count": 0}

        avg_tone = sum(tones) / len(tones)
        # Normalise from [-100, +100] to [-1, +1]
        normalised = max(-1.0, min(1.0, avg_tone / 100.0))

        return {"tone": round(normalised, 4), "article_count": len(tones)}

    return cache.get_or_fetch(key, _fetch, ttl_seconds=TTL_SENTIMENT)
