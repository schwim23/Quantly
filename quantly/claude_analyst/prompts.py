"""Prompt templates for Claude's stock analysis deep-dive.

Claude is given ML-ranked candidates and asked to:
1. Filter out stocks with obvious flaws
2. Re-rank with a conviction score (1-10)
3. Write a 3-sentence investment thesis per stock
4. Identify the single biggest risk to the thesis
"""

from __future__ import annotations


SYSTEM_PROMPT = """You are a professional equity analyst with expertise in identifying
short-term (1–4 week) trading opportunities. You are reviewing a list of stocks that
a machine learning model has flagged as potentially outperforming the S&P 500 over
the next 15 trading days.

Your job:
1. FILTER: Remove stocks where the data reveals an obvious reason the thesis is flawed.
2. RANK: Assign a conviction score from 1 (low) to 10 (high) to each remaining stock.
3. THESIS: Write exactly 3 sentences explaining why this stock may outperform.
4. RISK: Identify the single biggest risk that could invalidate the thesis.

Rules:
- Only use the data provided to you. Do NOT rely on your training knowledge for
  current prices, recent events, or financial figures — they may be outdated.
- Be honest about uncertainty. If the data is mixed, say so.
- Flag any accounting red flags, regulatory risks, or broken narratives you notice.
- Output valid JSON following the schema below exactly.

Output format:
{
  "filtered_out": ["TICKER1", "TICKER2"],
  "filtered_out_reasons": {"TICKER1": "reason", "TICKER2": "reason"},
  "picks": [
    {
      "ticker": "AAPL",
      "conviction": 8,
      "thesis": "Sentence 1. Sentence 2. Sentence 3.",
      "biggest_risk": "One sentence describing the key risk.",
      "key_signals": ["signal_1", "signal_2", "signal_3"]
    }
  ]
}
"""


def build_candidate_prompt(candidates: list[dict]) -> str:
    """Build the user-turn prompt from a list of candidate dicts.

    Each candidate dict should contain:
    - ticker, ml_rank, ensemble_prob
    - top_shap_features (list of feature dicts)
    - recent_headlines (list of strings)
    - earnings_call_excerpt (string, last 500 chars of prepared remarks)
    - mda_excerpt (string, last 500 chars of MD&A)
    - reddit_summary (string)
    """
    import json
    lines = [
        "Here are the top ML-ranked stock candidates for your analysis.",
        "For each stock, I provide: ML rank, probability score, top predictive signals,",
        "recent news, earnings call tone, SEC filing MD&A tone, and Reddit sentiment.",
        "",
        "Candidates:",
        "",
    ]
    for i, c in enumerate(candidates, 1):
        lines.append(f"--- Candidate {i}: {c['ticker']} ---")
        lines.append(f"ML Rank: #{c['ml_rank']}  |  Model Confidence: {c['ensemble_prob']:.1%}")
        lines.append("")
        lines.append("Top predictive signals (SHAP):")
        for feat in c.get("top_shap_features", []):
            direction = "+" if feat["shap_value"] > 0 else "-"
            lines.append(f"  {direction} {feat['feature']}: {feat['shap_value']:+.3f}")
        lines.append("")
        if c.get("recent_headlines"):
            lines.append("Recent headlines (last 7 days):")
            for h in c["recent_headlines"][:5]:
                lines.append(f"  • {h}")
            lines.append("")
        if c.get("earnings_call_excerpt"):
            lines.append(f"Earnings call excerpt: {c['earnings_call_excerpt']}")
            lines.append("")
        if c.get("mda_excerpt"):
            lines.append(f"MD&A excerpt: {c['mda_excerpt']}")
            lines.append("")
        if c.get("reddit_summary"):
            lines.append(f"Reddit sentiment: {c['reddit_summary']}")
            lines.append("")

    lines.append("Please analyze these candidates and return your assessment as JSON.")
    return "\n".join(lines)
