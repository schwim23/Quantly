"""SEC EDGAR 10-Q / 10-K filing analysis (free, no API key required).

Extracts MD&A text and risk factor sections from annual/quarterly filings
and scores them for tone using the Loughran-McDonald finance word list
approximated via VADER + domain-specific word lists.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from quantly.data.cache import cached

logger = logging.getLogger(__name__)
_analyzer = SentimentIntensityAnalyzer()

# Loughran-McDonald inspired finance-specific negative words
_LM_NEGATIVE = [
    "loss", "decline", "decrease", "impairment", "litigation", "adverse",
    "risk", "uncertainty", "negative", "deterioration", "defaulted", "bankruptcy",
    "investigation", "penalty", "lawsuit", "unfavorable", "volatile",
]
_LM_POSITIVE = [
    "growth", "increase", "improvement", "strong", "record", "expand",
    "profitable", "favorable", "robust", "successful", "exceed", "outperform",
]
_LM_UNCERTAINTY = [
    "may", "might", "could", "uncertain", "unclear", "potential", "approximately",
    "estimate", "subject to", "depends", "contingent", "possible",
]


@cached("sec_10q", ttl=86_400 * 7)
def get_latest_filing(ticker: str, form_type: str = "10-Q") -> dict:
    """Fetch the most recent 10-Q or 10-K for a ticker from SEC EDGAR.

    Returns:
        Dict with keys: filing_date, form_type, full_text, mda_text, risk_text
    """
    try:
        from sec_edgar_downloader import Downloader
    except ImportError:
        logger.warning("sec-edgar-downloader not installed")
        return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        dl = Downloader("Quantly", "research@quantly.local", tmpdir)
        try:
            dl.get(form_type, ticker, limit=1, download_details=True)
        except Exception as e:
            logger.debug("%s download failed for %s: %s", form_type, ticker, e)
            # Try 10-K if 10-Q fails
            if form_type == "10-Q":
                return get_latest_filing(ticker, "10-K")
            return {}

        base = Path(tmpdir) / "sec-edgar-filings" / ticker / form_type
        filing_dirs = sorted(base.glob("*"), reverse=True)
        if not filing_dirs:
            return {}

        full_text = ""
        for txt_file in filing_dirs[0].glob("*.txt"):
            try:
                content = txt_file.read_text(errors="ignore")
                full_text = re.sub(r"<[^>]+>", " ", content)
                full_text = re.sub(r"\s+", " ", full_text).strip()
                break
            except Exception:
                pass

        if not full_text:
            return {}

        return {
            "filing_date": filing_dirs[0].name[:10],
            "form_type": form_type,
            "full_text": full_text,
            "mda_text": _extract_section(full_text, "management", "quantitative"),
            "risk_text": _extract_section(full_text, "risk factor", "unresolved"),
        }


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """Heuristically extract a section from filing text by keyword markers."""
    text_lower = text.lower()
    start = text_lower.find(start_marker)
    if start == -1:
        return text[:3000]  # fallback: return first 3000 chars
    end = text_lower.find(end_marker, start + len(start_marker))
    if end == -1:
        return text[start : start + 5000]
    return text[start:end][:5000]


def score_mda_sentiment(filing: dict) -> dict[str, float]:
    """Score the MD&A section for financial tone.

    Returns:
        Dict with keys: mda_vader_score, mda_positive_ratio,
        mda_negative_ratio, mda_uncertainty_ratio
    """
    text = filing.get("mda_text") or filing.get("full_text", "")
    if not text:
        return {
            "mda_vader_score": 0.0,
            "mda_positive_ratio": 0.0,
            "mda_negative_ratio": 0.0,
            "mda_uncertainty_ratio": 0.0,
        }

    text_lower = text.lower()
    words = text_lower.split()
    total = max(len(words), 1)

    pos_hits = sum(1 for w in _LM_POSITIVE if w in text_lower)
    neg_hits = sum(1 for w in _LM_NEGATIVE if w in text_lower)
    unc_hits = sum(1 for w in _LM_UNCERTAINTY if w in text_lower)

    vader = _analyzer.polarity_scores(text[:1000])["compound"]

    return {
        "mda_vader_score": vader,
        "mda_positive_ratio": pos_hits / (pos_hits + neg_hits + 1),
        "mda_negative_ratio": neg_hits / (pos_hits + neg_hits + 1),
        "mda_uncertainty_ratio": unc_hits / total,
    }
