"""Unit tests for Claude analyst prompt building."""

from quantly.claude_analyst.prompts import SYSTEM_PROMPT, build_candidate_prompt


def make_candidates(n=3):
    return [
        {
            "ticker": f"T{i}",
            "ml_rank": i,
            "ensemble_prob": 0.8 - i * 0.1,
            "top_shap_features": [
                {"feature": "rsi_14", "shap_value": 0.12, "direction": "bullish"},
                {"feature": "news_sentiment_7d_avg", "shap_value": 0.08, "direction": "bullish"},
            ],
            "recent_headlines": [f"Headline {j} for T{i}" for j in range(3)],
            "earnings_excerpt": f"Strong Q{i} results.",
            "mda_excerpt": f"Management expects continued growth in T{i}.",
            "reddit_summary": "mentions: 120, velocity: 1.8x, sentiment: 0.25",
        }
        for i in range(1, n + 1)
    ]


class TestPromptBuilder:
    def test_all_tickers_in_prompt(self):
        candidates = make_candidates(3)
        prompt = build_candidate_prompt(candidates)
        for c in candidates:
            assert c["ticker"] in prompt

    def test_shap_features_in_prompt(self):
        prompt = build_candidate_prompt(make_candidates(1))
        assert "rsi_14" in prompt

    def test_nonempty_output(self):
        assert len(build_candidate_prompt(make_candidates(1))) > 200

    def test_empty_candidates(self):
        result = build_candidate_prompt([])
        assert isinstance(result, str)

    def test_system_prompt_has_json_schema(self):
        assert "JSON" in SYSTEM_PROMPT

    def test_system_prompt_no_hallucination_warning(self):
        # Must warn against using training knowledge for current data
        assert "training" in SYSTEM_PROMPT.lower()
