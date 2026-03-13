# Quantly — AI Stock Picker

An AI-powered swing trading signal engine that combines financial KPI analysis,
multi-source market sentiment, and Claude-powered reasoning to identify US equities
with a high probability of positive price movement over a **1–4 week horizon**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                     CLI Entry Point                  │
│                  python -m quantly                   │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   Universe Screener     │  Stage 1: fast pass over all US equities
          │  (liquidity + volume)   │  → filters to ~500–1000 liquid candidates
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Feature Pipeline      │  Stage 2: compute KPI + sentiment features
          │  KPIs | Sentiment | TA  │  for screened candidates
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  ML Scoring Engine      │  Stage 3: LightGBM/XGBoost ranks candidates
          │  (LightGBM / XGBoost)   │  by predicted 1-4 week return probability
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  Claude Deep-Dive       │  Stage 4: Claude reasons over top-N stocks,
          │  (Anthropic API)        │  filters noise, explains the thesis
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  CLI Output / Reporter  │  Stage 5: ranked picks with confidence,
          │                         │  rationale, and risk flags
          └─────────────────────────┘
```

---

## Data Sources

| Source | Purpose | Access |
|---|---|---|
| **Polygon.io** | Real-time & historical OHLCV, fundamentals, options flow | REST API |
| **Alpha Vantage** | Earnings, income statement, balance sheet, cash flow | REST API |
| **Bloomberg** | Professional-grade price data, news, company fundamentals | Terminal/API |
| **SEC EDGAR** | 10-Q/10-K filings for KPI and MD&A extraction | Free REST API |
| **Reddit (Pushshift / Reddit API)** | WallStreetBets and finance subreddit sentiment | Reddit API |
| **Financial news** | Headlines from Bloomberg/Reuters via Polygon news endpoint | REST API |
| **Earnings call transcripts** | Seeking Alpha / Motley Fool scrape or licensed source | Scraper/API |

---

## Feature Groups

### 1. Financial KPIs (from SEC filings + Alpha Vantage / Bloomberg)
- Revenue growth QoQ and YoY
- EPS surprise vs. consensus estimate
- Gross margin trend (expanding vs. contracting)
- Free cash flow yield
- Debt/equity ratio change
- Return on equity (ROE) trend
- Inventory turnover (for product companies)
- Operating leverage (revenue growth vs. opex growth)

### 2. Technical / Price Features
- RSI(14), MACD signal line crossover
- Distance from 52-week high/low
- Volume spike ratio (current vs. 20-day avg)
- Price momentum: 5d, 10d, 20d returns
- ATR (volatility) relative to sector peers
- Bollinger Band position

### 3. Sentiment Features (NLP-extracted)
- News headline sentiment score (rolling 7-day avg)
- Earnings call tone: positive/negative/uncertain word ratios
- SEC filing risk-factor delta (new risks added vs. removed)
- Reddit mention velocity and sentiment polarity
- Short interest ratio and short squeeze potential

### 4. Association / Cross-Feature
- KPI beat + positive sentiment convergence score
- Volume spike + news catalyst alignment
- Sector rotation momentum (sector ETF flow vs. stock)
- Earnings surprise + upward guidance revision combo

---

## ML Model Design

### Model Type
**LightGBM** (primary) with **XGBoost** as ensemble member.

### Label Construction
- **Positive label (1)**: stock outperforms SPY by >3% within 15 trading days
- **Negative label (0)**: stock underperforms SPY by >2% or is flat within 15 days

### Training Strategy
- **Walk-forward validation**: train on rolling 2-year window, validate on next quarter
- Minimum 3 years of historical data for initial training
- Retrain weekly with fresh data

### Feature Importance
- Log SHAP values for every prediction — feeds Claude's explanation step
- Top features surface as bullet points in CLI output

### Anti-Overfitting Rules
- No lookahead bias: all features computed strictly from data available at signal date
- Purge 5-day gap between train and validation sets
- Embargo period: exclude stocks with events (splits, mergers) from training labels

---

## Claude Integration (Stage 4 Deep-Dive)

Claude receives the **top 20 ML-ranked candidates** and for each stock is given:

1. SHAP feature importance breakdown (which signals fired)
2. Recent news headlines (last 7 days)
3. Earnings call transcript excerpt (most recent)
4. SEC filing MD&A excerpt (most recent 10-Q)
5. Reddit sentiment summary

Claude's tasks:
- **Filter**: remove stocks where the thesis has an obvious flaw (regulatory risk, accounting red flags, broken narrative)
- **Rank**: re-rank the filtered set with a conviction score (1–10)
- **Explain**: write a 3-sentence investment thesis per stock
- **Flag risks**: identify the single biggest risk to the thesis

Output is the **final pick list** (typically 5–10 stocks).

---

## Backtesting & Validation

### Backtesting Engine
- Walk-forward backtest over 3+ years of history
- Metrics reported: Sharpe ratio, max drawdown, win rate, avg return per trade
- Compare against SPY buy-and-hold benchmark
- Separate results by market regime (bull/bear/sideways)

### Paper Trading
- After backtest validation, run in paper trading mode
- Track picks vs. actual market outcomes weekly
- Store results in local SQLite DB for ongoing accuracy tracking
- Weekly report: how last week's picks performed

---

## Project Structure

```
quantly/
├── CLAUDE.md                    # This file
├── pyproject.toml               # Dependencies (uv/pip)
├── .env.example                 # API key template
│
├── quantly/
│   ├── __init__.py
│   ├── __main__.py              # CLI entry point
│   │
│   ├── data/
│   │   ├── universe.py          # US equity universe fetcher + screener
│   │   ├── prices.py            # OHLCV from Polygon.io / Bloomberg
│   │   ├── fundamentals.py      # KPIs from Alpha Vantage / Bloomberg / SEC
│   │   ├── sentiment/
│   │   │   ├── news.py          # News headline fetcher + scorer
│   │   │   ├── sec_filings.py   # EDGAR 10-Q/10-K fetcher + NLP
│   │   │   ├── earnings_calls.py# Transcript fetcher + tone analysis
│   │   │   └── reddit.py        # Reddit sentiment scraper
│   │   └── cache.py             # Local disk cache (avoid repeat API calls)
│   │
│   ├── features/
│   │   ├── technical.py         # TA features (RSI, MACD, volume, etc.)
│   │   ├── kpi.py               # Financial KPI features
│   │   ├── sentiment_features.py# Aggregate sentiment into model features
│   │   └── pipeline.py          # Combines all features into model input df
│   │
│   ├── models/
│   │   ├── train.py             # Walk-forward training loop
│   │   ├── predict.py           # Inference + SHAP value generation
│   │   ├── labels.py            # Label construction (outperformance vs SPY)
│   │   └── evaluate.py          # Backtest metrics: Sharpe, drawdown, win rate
│   │
│   ├── claude_analyst/
│   │   ├── analyst.py           # Claude API calls for deep-dive analysis
│   │   ├── prompts.py           # Prompt templates for filtering/ranking/thesis
│   │   └── formatter.py         # Format ML output for Claude consumption
│   │
│   ├── backtest/
│   │   ├── engine.py            # Walk-forward backtest orchestrator
│   │   └── paper_trading.py     # Paper trade tracker (SQLite)
│   │
│   ├── cli/
│   │   ├── commands.py          # CLI commands (scan, backtest, paper, report)
│   │   └── display.py           # Rich-formatted terminal output
│   │
│   └── config.py                # Config loading from env + settings file
│
├── skills/
│   ├── scan.md                  # Run a full market scan
│   ├── backtest.md              # Run/analyze backtests
│   ├── train.md                 # Retrain ML models
│   ├── paper-report.md          # Weekly paper trading performance report
│   └── add-feature.md           # Guide for adding a new feature to the pipeline
│
├── tests/
│   ├── test_features.py
│   ├── test_labels.py
│   ├── test_backtest.py
│   └── test_claude_analyst.py
│
└── data/                        # Local cache and SQLite DB (gitignored)
    ├── cache/
    └── paper_trades.db
```

---

## Key Design Principles

1. **No lookahead bias** — every feature and label must use only data available at signal date
2. **Caching first** — all API responses cached locally to avoid rate limits and cost
3. **Two-stage pipeline** — fast ML pass over all equities, then expensive Claude calls only on top-N
4. **Explainability** — every pick comes with SHAP features + Claude rationale, never a black box
5. **Honest backtesting** — walk-forward only, purge gaps, benchmark against SPY
6. **Modular data sources** — swap Bloomberg for Polygon.io without touching model code
7. **Paper trading feedback loop** — weekly outcome tracking closes the signal quality loop

---

## Development Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env  # Fill in API keys

# Run a market scan (full pipeline)
python -m quantly scan

# Run with specific universe
python -m quantly scan --universe sp500
python -m quantly scan --tickers AAPL MSFT NVDA

# Backtest the current model
python -m quantly backtest --years 3

# Train / retrain the ML model
python -m quantly train

# Paper trading report
python -m quantly paper report

# Run tests
pytest tests/ -v
```

---

## API Keys Required

Set in `.env` (see `.env.example`):

```
ANTHROPIC_API_KEY=         # Claude API (required)
POLYGON_API_KEY=           # Polygon.io
ALPHA_VANTAGE_API_KEY=     # Alpha Vantage
BLOOMBERG_API_KEY=         # Bloomberg (optional, premium)
REDDIT_CLIENT_ID=          # Reddit API
REDDIT_CLIENT_SECRET=      # Reddit API
```

---

## Common Pitfalls to Avoid

- **Survivorship bias**: always use point-in-time universe (delisted stocks must be included in backtest history)
- **Sentiment timing**: news/Reddit sentiment must be tagged to the time it was published, not today
- **Earnings date leakage**: never include post-earnings data in pre-earnings features
- **Overfitting to bull markets**: validate model performance across different market regimes separately
- **API rate limits**: always respect rate limits; use the cache layer aggressively
- **Claude hallucination on financials**: Claude should only reason from provided data, never generate financial numbers from memory

---

## When Adding a New Feature

1. Add raw data fetcher in `quantly/data/`
2. Transform into model-ready numeric feature in `quantly/features/`
3. Register feature in `quantly/features/pipeline.py`
4. Run `python -m quantly backtest --years 1` to validate improvement
5. Update this CLAUDE.md if the architecture changes significantly

---

## When Modifying the ML Model

1. Change model config in `quantly/models/train.py`
2. Always run walk-forward backtest before declaring improvement
3. Compare SHAP feature importances before and after — confirm the new model uses sensible signals
4. Log Sharpe ratio, win rate, and max drawdown in a comment in the commit message
