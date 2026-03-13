"""Claude-powered deep-dive analysis of ML-ranked stock candidates.

Sends top-N candidates to Claude with all available context and receives:
- Filtered pick list (removes low-quality signals)
- Conviction scores (1-10)
- 3-sentence investment thesis per stock
- Key risk per stock
"""

from __future__ import annotations

import json
import logging

import anthropic

from quantly.claude_analyst.prompts import SYSTEM_PROMPT, build_candidate_prompt
from quantly.config import config

logger = logging.getLogger(__name__)


def analyze_candidates(candidates: list[dict]) -> dict:
    """Send candidates to Claude for deep-dive analysis.

    Args:
        candidates: List of candidate dicts (see prompts.build_candidate_prompt for schema)

    Returns:
        Dict with keys: filtered_out, filtered_out_reasons, picks
        Each pick has: ticker, conviction, thesis, biggest_risk, key_signals
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    user_prompt = build_candidate_prompt(candidates)

    logger.info("Sending %d candidates to Claude for analysis...", len(candidates))

    message = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=config.CLAUDE_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    response_text = message.content[0].text

    # Parse JSON from response
    try:
        # Find JSON block in response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        json_str = response_text[start:end]
        result = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Failed to parse Claude response as JSON: %s", e)
        logger.debug("Raw response: %s", response_text)
        raise

    logger.info(
        "Claude analysis complete: %d picks, %d filtered out",
        len(result.get("picks", [])),
        len(result.get("filtered_out", [])),
    )
    return result
