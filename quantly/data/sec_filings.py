"""SEC EDGAR — fetch 10-Q/10-K filings and extract MD&A risk-factor delta.

EDGAR API is free, no key required. Rate limit: 10 req/sec (global).
We keep requests well below this limit via caching.

Ref: https://www.sec.gov/developer
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

import httpx

from quantly.data.cache import Cache, TTL_FUNDAMENTAL

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Quantly research@quantly.local"}
_EDGAR_BASE = "https://data.sec.gov"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def _edgar_get(url: str) -> Optional[Any]:
    try:
        resp = httpx.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("EDGAR GET %s error: %s", url, exc)
        return None


# ── CIK lookup ────────────────────────────────────────────────────────────────

def get_cik(ticker: str, cache: Cache) -> Optional[str]:
    """Return zero-padded 10-digit CIK for a ticker, or None if not found."""
    key = f"edgar:cik:{ticker}"

    def _fetch() -> Optional[str]:
        url = "https://efts.sec.gov/LATEST/search-index?q=%22{}%22&dateRange=custom&startdt=2000-01-01&forms=10-K".format(
            ticker
        )
        # Use the company tickers JSON which is simpler
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        data = _edgar_get(tickers_url)
        if not data:
            return None
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                return cik
        return None

    return cache.get_or_fetch(key, _fetch, ttl_seconds=86400 * 7)  # 7-day cache


# ── Recent filings ────────────────────────────────────────────────────────────

def get_recent_filings(cik: str, cache: Cache, form_type: str = "10-Q") -> list[dict]:
    """Return metadata for the most recent filings of *form_type* for CIK."""
    key = f"edgar:filings:{cik}:{form_type}"

    def _fetch() -> list[dict]:
        url = _SUBMISSIONS_URL.format(cik=cik)
        data = _edgar_get(url)
        if not data:
            return []
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        result = []
        for form, dt, acc in zip(forms, dates, accessions):
            if form == form_type:
                result.append({"form": form, "date": dt, "accession": acc})
        return result[:5]  # newest 5

    return cache.get_or_fetch(key, _fetch, ttl_seconds=TTL_FUNDAMENTAL)


# ── Risk factor delta ─────────────────────────────────────────────────────────

def _extract_risk_keywords(text: str) -> set[str]:
    """Pull out distinctive risk phrases from filing text (simple heuristic)."""
    # Look for sentences in the Risk Factors section containing key risk words
    risk_section = re.search(
        r"(?i)risk factors?(.*?)(?:item\s+[23]|management.{0,30}discussion)",
        text,
        re.DOTALL,
    )
    if not risk_section:
        return set()

    section_text = risk_section.group(1)
    risk_phrases = re.findall(
        r"\b(litigation|regulatory|competition|debt|loss|decline|adverse|"
        r"breach|cyber|lawsuit|penalty|investigation|recall|shortage|"
        r"going concern|material weakness|restatement|fraud|sanction)\b",
        section_text,
        re.IGNORECASE,
    )
    return {p.lower() for p in risk_phrases}


def get_risk_factor_delta(ticker: str, cache: Cache) -> dict[str, Any]:
    """Compare risk keywords between the two most recent 10-Q filings.

    Returns:
        new_risks    - set of risk keywords present in latest but not prior filing
        removed_risks - set of risk keywords in prior but not latest filing
        delta_score  - len(new_risks) - len(removed_risks)  (-ve = improving)
    """
    key = f"edgar:risk_delta:{ticker}"

    def _compute() -> dict[str, Any]:
        cik = get_cik(ticker, cache)
        if not cik:
            return {"new_risks": [], "removed_risks": [], "delta_score": 0}

        filings = get_recent_filings(cik, cache, form_type="10-Q")
        if len(filings) < 2:
            return {"new_risks": [], "removed_risks": [], "delta_score": 0}

        def _fetch_text(accession: str) -> str:
            acc_clean = accession.replace("-", "")
            idx_url = (
                f"https://www.sec.gov/Archives/edgar/full-index/"
                f"../{acc_clean[:4]}/{acc_clean[4:6]}/{accession}-index.json"
            )
            idx_data = _edgar_get(idx_url)
            if not idx_data:
                return ""
            # Try to fetch the primary document
            files = idx_data.get("directory", {}).get("item", [])
            doc = next(
                (f for f in files if f.get("name", "").endswith(".htm")), None
            )
            if not doc:
                return ""
            doc_url = f"https://www.sec.gov/Archives/edgar/{acc_clean[:4]}/{acc_clean[4:6]}/{doc['name']}"
            try:
                resp = httpx.get(doc_url, headers=_HEADERS, timeout=20)
                resp.raise_for_status()
                # Strip HTML tags for plain text
                text = re.sub(r"<[^>]+>", " ", resp.text)
                return text[:50_000]  # cap at 50K chars
            except Exception:
                return ""

        latest_text = _fetch_text(filings[0]["accession"])
        prior_text = _fetch_text(filings[1]["accession"])

        latest_risks = _extract_risk_keywords(latest_text)
        prior_risks = _extract_risk_keywords(prior_text)

        new_risks = latest_risks - prior_risks
        removed_risks = prior_risks - latest_risks

        return {
            "new_risks": sorted(new_risks),
            "removed_risks": sorted(removed_risks),
            "delta_score": len(new_risks) - len(removed_risks),
        }

    return cache.get_or_fetch(key, _compute, ttl_seconds=TTL_FUNDAMENTAL)
