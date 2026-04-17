# /scan — Run a Full Market Scan

Runs the complete pipeline and outputs today's top stock picks.

## Steps

1. **Check prerequisites**
   - `.env` has `ANTHROPIC_API_KEY` set
   - Trained model exists in `models/` (run `/train` first if not)

2. **Run**
   ```bash
   python -m quantly scan                        # S&P 500 default
   python -m quantly scan --universe nasdaq100
   python -m quantly scan --universe all_us       # slow — 8,000+ tickers
   python -m quantly scan --tickers AAPL MSFT NVDA  # specific tickers
   python -m quantly scan --paper-trade           # record picks for tracking
   python -m quantly scan --output results/scan_$(date +%Y%m%d).json
   ```

3. **Timing**
   - S&P 500: ~5–10 minutes (feature compute + Claude call)
   - All US: ~45–90 minutes (use with caution, cache helps on re-runs)

4. **Interpret output**
   - Picks sorted by Claude conviction (1–10)
   - Each pick shows: thesis (3 sentences), key signals (SHAP), biggest risk
   - Red = conviction < 4, Yellow = 4–6, Green = 7+

## Data sources used (all free)
- Prices: yfinance (Yahoo Finance)
- Fundamentals: yfinance quarterly financials
- News: yfinance news feed + VADER sentiment
- Filings: SEC EDGAR 10-Q MD&A
- Earnings: SEC EDGAR 8-K press releases
- Reddit: PRAW (if credentials configured in .env)
