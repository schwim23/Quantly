"""Claude API integration for deep-dive stock analysis."""

from __future__ import annotations

import json
import logging

import anthropic

from quantly.claude_analyst.prompts import SYSTEM_PROMPT, build_candidate_prompt
from quantly.config import config

logger = logging.getLogger(__name__)


def analyze_candidates(candidates: list[dict]) -> dict:
    """Send top-N ML candidates to Claude for filtering, ranking, and thesis writing.

    Args:
        candidates: List of candidate dicts (built by formatter.format_all_candidates)

    Returns:
        Dict with keys: filtered_out, filtered_out_reasons, picks
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    prompt = build_candidate_prompt(candidates)

    logger.info("Sending %d candidates to Claude (%s)...", len(candidates), config.CLAUDE_MODEL)

    message = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=config.CLAUDE_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text

    # Extract JSON from response (Claude may add surrounding prose)
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        result = json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Failed to parse Claude response as JSON: %s\nRaw: %s", e, text[:500])
        raise

    logger.info(
        "Claude: %d picks kept, %d filtered out",
        len(result.get("picks", [])),
        len(result.get("filtered_out", [])),
    )
    return result
