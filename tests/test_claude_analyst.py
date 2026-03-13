"""Tests for Claude analyst prompt building and response parsing."""

import json
import pytest

from quantly.claude_analyst.prompts import build_candidate_prompt, SYSTEM_PROMPT


class TestPromptBuilder:
    def make_candidates(self, n: int = 3) -> list[dict]:
        return [
            {
                "ticker": f"TICK{i}",
                "ml_rank": i,
                "ensemble_prob": 0.8 - i * 0.1,
                "top_shap_features": [
                    {"feature": "rsi_14", "shap_value": 0.12, "direction": "bullish"},
                    {"feature": "news_sentiment_7d_avg", "shap_value": 0.08, "direction": "bullish"},
                ],
                "recent_headlines": [
                    f"Headline 1 for TICK{i}",
                    f"Headline 2 for TICK{i}",
                ],
                "mda_excerpt": f"Management discussion excerpt for TICK{i}.",
                "earnings_call_excerpt": f"Prepared remarks excerpt for TICK{i}.",
                "reddit_summary": f"Reddit: mentions 150, velocity 2.1x, sentiment 0.3",
            }
            for i in range(1, n + 1)
        ]

    def test_prompt_contains_all_tickers(self):
        candidates = self.make_candidates(3)
        prompt = build_candidate_prompt(candidates)
        for c in candidates:
            assert c["ticker"] in prompt

    def test_prompt_contains_shap_features(self):
        candidates = self.make_candidates(2)
        prompt = build_candidate_prompt(candidates)
        assert "rsi_14" in prompt
        assert "news_sentiment_7d_avg" in prompt

    def test_prompt_is_nonempty(self):
        candidates = self.make_candidates(1)
        prompt = build_candidate_prompt(candidates)
        assert len(prompt) > 100

    def test_system_prompt_mentions_json(self):
        assert "JSON" in SYSTEM_PROMPT

    def test_system_prompt_mentions_no_hallucination(self):
        # Should warn against using training knowledge for current prices
        assert "training" in SYSTEM_PROMPT.lower() or "provided" in SYSTEM_PROMPT.lower()

    def test_empty_candidates(self):
        prompt = build_candidate_prompt([])
        assert isinstance(prompt, str)
