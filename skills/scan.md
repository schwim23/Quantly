# /scan — Run a Full Market Scan

Runs the complete Quantly pipeline end-to-end and outputs today's top stock picks.

## What This Does

1. Fetches the current US equity universe and applies liquidity/volume screener
2. Computes all KPI, technical, and sentiment features for surviving candidates
3. Scores candidates with the trained LightGBM/XGBoost model
4. Sends the top-20 ML-ranked stocks to Claude for deep-dive analysis
5. Outputs the final ranked pick list with thesis and risk flags

## Steps

1. **Check prerequisites**
   - Confirm `.env` file has all required API keys
   - Confirm trained model exists in `models/` (run `/train` first if not)
   - Check data cache freshness (stale if > 24 hours old)

2. **Run the pipeline**
   ```bash
   python -m quantly scan
   ```
   Optional flags:
   - `--universe sp500` — limit to S&P 500 for a faster run
   - `--top-n 10` — how many stocks to send to Claude (default: 20)
   - `--min-volume 500000` — minimum daily volume filter
   - `--no-cache` — force fresh data fetch

3. **Interpret the output**
   - Each pick shows: ticker, conviction score (1–10), 3-sentence thesis, top SHAP features, biggest risk
   - Picks are sorted by conviction score descending
   - Any stock with a red risk flag should be treated with extra caution

4. **Save results**
   ```bash
   python -m quantly scan --output results/scan_$(date +%Y%m%d).json
   ```

## Caveats
- Full scan over all US equities takes ~10–20 minutes due to data fetching
- Claude deep-dive costs ~$0.10–0.30 per full run depending on transcript length
- Do not run more than once per day — use cached results for intraday re-runs
