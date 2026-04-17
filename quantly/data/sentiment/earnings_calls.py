"""Earnings call content via SEC EDGAR 8-K filings (free, no API key required).

Companies file 8-K forms for material events. Exhibit 99.1 typically contains
the press release (earnings results) and sometimes the prepared remarks.
Full transcripts are not in SEC filings — this module extracts the earnings
press release tone as a proxy for prepared remarks sentiment.
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

# Forward-looking / guidance language word lists
_POSITIVE_GUIDANCE = [
    "growth", "increase", "expand", "record", "exceed", "strong", "robust",
    "confident", "optimistic", "outperform", "ahead", "raised", "raised guidance",
]
_NEGATIVE_GUIDANCE = [
    "headwind", "decline", "challenging", "uncertain", "reduce", "lower",
    "disappoint", "miss", "below", "difficult", "pressure", "concern",
]
_UNCERTAINTY_WORDS = [
    "may", "might", "could", "uncertain", "unclear", "subject to", "depends",
    "potential", "approximately", "estimate", "expect",
]


@cached("earnings_8k", ttl=86_400 * 7)  # refresh weekly
def get_latest_earnings_release(ticker: str) -> dict:
    """Fetch the most recent earnings press release text via SEC EDGAR 8-K.

    Uses sec-edgar-downloader to fetch the latest 8-K filing for the ticker,
    then extracts text from the document.

    Returns:
        Dict with keys: filing_date, text, excerpt (first 2000 chars)
    """
    try:
        from sec_edgar_downloader import Downloader
    except ImportError:
        logger.warning("sec-edgar-downloader not installed; skipping earnings release fetch")
        return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        dl = Downloader("Quantly", "research@quantly.local", tmpdir)
        try:
            dl.get("8-K", ticker, limit=1, download_details=True)
        except Exception as e:
            logger.debug("8-K download failed for %s: %s", ticker, e)
            return {}

        # Find the downloaded filing text
        base = Path(tmpdir) / "sec-edgar-filings" / ticker / "8-K"
        filing_dirs = sorted(base.glob("*"), reverse=True)
        if not filing_dirs:
            return {}

        text = ""
        for filing_dir in filing_dirs[:1]:
            for txt_file in filing_dir.glob("*.txt"):
                try:
                    content = txt_file.read_text(errors="ignore")
                    # Strip SGML/HTML tags
                    text = re.sub(r"<[^>]+>", " ", content)
                    text = re.sub(r"\s+", " ", text).strip()
                    break
                except Exception:
                    pass

        if not text:
            return {}

        return {
            "filing_date": filing_dirs[0].name[:10] if filing_dirs else "",
            "text": text,
            "excerpt": text[:2000],
        }


def score_earnings_tone(filing: dict) -> dict[str, float]:
    """Score earnings press release tone.

    Returns:
        Dict with keys: overall_sentiment, guidance_positive_ratio,
        guidance_negative_ratio, uncertainty_ratio
    """
    if not filing or not filing.get("text"):
        return {
            "ec_overall_sentiment": 0.0,
            "ec_guidance_positive_ratio": 0.0,
            "ec_guidance_negative_ratio": 0.0,
            "ec_uncertainty_ratio": 0.0,
        }

    text = filing["text"].lower()
    words = text.split()
    total = max(len(words), 1)

    pos_count = sum(1 for w in _POSITIVE_GUIDANCE if w in text)
    neg_count = sum(1 for w in _NEGATIVE_GUIDANCE if w in text)
    unc_count = sum(1 for w in _UNCERTAINTY_WORDS if w in text)

    vader_score = _analyzer.polarity_scores(filing["excerpt"])["compound"]

    return {
        "ec_overall_sentiment": vader_score,
        "ec_guidance_positive_ratio": pos_count / (pos_count + neg_count + 1),
        "ec_guidance_negative_ratio": neg_count / (pos_count + neg_count + 1),
        "ec_uncertainty_ratio": unc_count / total,
    }
