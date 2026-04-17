"""Prompt templates for Claude's deep-dive stock analysis."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a professional equity analyst reviewing a shortlist of stocks
flagged by a machine learning model as likely to outperform the S&P 500 over the next
1–4 weeks (15 trading days).

Your tasks:
1. FILTER: Remove stocks where the provided data reveals an obvious flaw in the thesis.
2. RANK: Assign a conviction score 1–10 to each remaining stock.
3. THESIS: Write exactly 3 sentences explaining why this stock may outperform.
4. RISK: Identify the single biggest risk that could invalidate the thesis.

Rules:
- Only use data provided to you. Do NOT use your training knowledge for current prices,
  recent earnings, or current events — that data may be stale or wrong.
- Flag accounting red flags, regulatory risk, or broken narratives if you see them.
- Be honest about uncertainty. Mixed signals deserve a lower conviction score.
- Output valid JSON matching the schema below exactly — no extra text outside the JSON.

Output schema:
{
  "filtered_out": ["TICKER1"],
  "filtered_out_reasons": {"TICKER1": "one-sentence reason"},
  "picks": [
    {
      "ticker": "AAPL",
      "conviction": 8,
      "thesis": "Sentence 1. Sentence 2. Sentence 3.",
      "biggest_risk": "One sentence.",
      "key_signals": ["signal_a", "signal_b", "signal_c"]
    }
  ]
}
"""


def build_candidate_prompt(candidates: list[dict]) -> str:
    """Build the user-turn prompt from a list of candidate dicts.

    Each candidate should contain:
      ticker, ml_rank, ensemble_prob, top_shap_features,
      recent_headlines, earnings_excerpt, mda_excerpt, reddit_summary
    """
    lines = [
        "Analyze the following ML-ranked stock candidates.",
        "For each I provide: ML rank, model confidence, top predictive signals (SHAP),",
        "recent news, earnings press release excerpt, SEC MD&A excerpt, and Reddit sentiment.",
        "",
    ]
    for i, c in enumerate(candidates, 1):
        lines.append(f"--- #{i}: {c['ticker']}  (ML rank #{c['ml_rank']}, confidence {c['ensemble_prob']:.1%}) ---")
        lines.append("Top SHAP signals:")
        for feat in c.get("top_shap_features", []):
            sign = "+" if feat["shap_value"] > 0 else "-"
            lines.append(f"  {sign} {feat['feature']} ({feat['direction']}): {feat['shap_value']:+.3f}")
        if c.get("recent_headlines"):
            lines.append("Recent news:")
            for h in c["recent_headlines"][:5]:
                lines.append(f"  • {h}")
        if c.get("earnings_excerpt"):
            lines.append(f"Earnings press release excerpt: {c['earnings_excerpt']}")
        if c.get("mda_excerpt"):
            lines.append(f"SEC MD&A excerpt: {c['mda_excerpt']}")
        if c.get("reddit_summary"):
            lines.append(f"Reddit: {c['reddit_summary']}")
        lines.append("")

    lines.append("Return your analysis as JSON following the schema in your instructions.")
    return "\n".join(lines)
