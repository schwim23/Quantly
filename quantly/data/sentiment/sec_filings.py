"""Fetch and analyze SEC EDGAR 10-Q and 10-K filings.

Extracts:
- MD&A (Management Discussion & Analysis) text for tone analysis
- Risk factor section delta (new risks vs. prior filing)
- Key financial table extraction
"""

from __future__ import annotations

import logging

import pandas as pd

from quantly.data.cache import cached

logger = logging.getLogger(__name__)


@cached("sec_filings")
def get_latest_filing(ticker: str, form_type: str = "10-Q") -> dict:
    """Fetch the most recent 10-Q or 10-K for a ticker from EDGAR.

    Returns:
        Dict with keys: filing_date, period_of_report, mda_text, risk_factors_text, url
    """
    # TODO: use sec-edgar-downloader to fetch and parse filing
    raise NotImplementedError


def score_mda_sentiment(mda_text: str) -> dict[str, float]:
    """Score the MD&A section for tone: positive, negative, uncertain word ratios.

    Uses a finance-specific word list (Loughran-McDonald dictionary).

    Returns:
        Dict with keys: positive_ratio, negative_ratio, uncertainty_ratio, litigious_ratio
    """
    # TODO: implement Loughran-McDonald word list scoring
    raise NotImplementedError


def compute_risk_factor_delta(
    current_text: str,
    prior_text: str,
) -> dict[str, int]:
    """Compare risk factor sections between two filings.

    Returns:
        Dict with keys: new_risks_added (count), risks_removed (count), net_change
    """
    # TODO: implement sentence-level diff between filings
    raise NotImplementedError
