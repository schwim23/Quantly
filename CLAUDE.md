# Quantly — AI Stock Picker

AI-powered swing trading signal engine. Analyzes financial KPIs, technical indicators,
and multi-source market sentiment to identify US equities likely to outperform the S&P 500
over a **1–4 week horizon** (15 trading days).

**All data sources are free — no paid API keys required.**

---

## Architecture

```
┌─────────────────────────────────────────┐
│          CLI  (python -m quantly)        │
└──────────────────┬──────────────────────┘
                   │
      ┌────────────▼────────────┐
      │   Universe + Screener   │  SEC EDGAR tickers / Wikipedia S&P 500
      │                         │  yfinance liquidity filter
      └────────────┬────────────┘
                   │
      ┌────────────▼────────────┐
      │    Feature Pipeline     │  Technical + KPI + Sentiment
      │                         │  ~25 features per ticker
      └────────────┬────────────┘
                   │
      ┌────────────▼────────────┐
      │   ML Scoring Engine     │  LightGBM + XGBoost ensemble
      │                         │  SHAP values for explainability
      └────────────┬────────────┘
                   │
      ┌────────────▼────────────┐
      │   Claude Deep-Dive      │  Top-20 ML candidates →
      │   (claude-opus-4-6)     │  filter + rank + thesis + risk
      └────────────┬────────────┘
                   │
      ┌────────────▼────────────┐
      │   CLI Output            │  Rich-formatted pick list
      │   + Paper Trading DB    │  SQLite outcome tracking
      └─────────────────────────┘
```

---

## Data Sources (all free, no API key required)

| Source | What it provides | How accessed |
|---|---|---|
| **yfinance** | OHLCV prices, income stmt, balance sheet, cash flow, company info, news | `pip install yfinance` |
| **SEC EDGAR** | 10-Q/10-K filings (MD&A, risk factors), 8-K earnings press releases, full US equity universe | `sec-edgar-downloader` + direct JSON |
| **Wikipedia** | S&P 500 and NASDAQ 100 constituent lists | `pd.read_html()` |
| **Reddit (PRAW)** | WSB/stocks/investing sentiment — **optional**, free tier only | Register free app at reddit.com/prefs/apps |

**Only `ANTHROPIC_API_KEY` is required to run the full pipeline.**

---

## Feature Groups

### Technical (from yfinance OHLCV)
- RSI(14)
- MACD line, signal, histogram, crossover flag
- Volume spike ratio (vs 20-day avg)
- Price momentum: 5d, 10d, 20d
- ATR(14) — volatility
- Bollinger Band position (0 = lower band, 1 = upper)
- % distance from 52-week high / low

### Financial KPIs (from yfinance quarterly fundamentals)
- Revenue growth QoQ and YoY
- EPS vs estimate (surprise %) — NaN when estimate unavailable
- Gross margin level and QoQ delta
- Free cash flow yield (FCF / market cap)
- Debt/equity ratio and QoQ change
- Return on equity (ROE) and QoQ change
- Beta, short ratio, forward P/E

### Sentiment (from SEC EDGAR + yfinance news + Reddit)
- News VADER sentiment — 7-day rolling average (yfinance)
- News headline count — 7-day window
- Earnings tone: overall sentiment, guidance positive/negative ratio, uncertainty ratio (SEC 8-K)
- MD&A tone: VADER score, Loughran-McDonald positive/negative/uncertainty ratios (SEC 10-Q/10-K)
- Reddit: mention count (7d), velocity ratio (vs prior 7d), avg sentiment — optional

---

## ML Model

### Label construction
- **Positive (1)**: stock outperforms SPY by > 3% over 15 trading days
- **Negative (0)**: stock underperforms SPY by > 2% over 15 trading days
- **Excluded (NaN)**: ambiguous result — excluded from training

### Model
- **LightGBM** + **XGBoost** ensemble (average of both probabilities)
- Walk-forward training: 2-year rolling window, monthly refit
- SHAP values computed for every prediction — feed into Claude prompt

### Anti-overfitting rules
- No lookahead: all features use only data available at `signal_date`
- 5-day purge gap between train and validation sets
- Validate across at least one bear market period (2022)

---

## Claude Integration

Claude receives the **top 20 ML-ranked candidates** and for each gets:
- SHAP feature breakdown (which signals fired, direction, magnitude)
- Recent news headlines (last 5 from yfinance)
- Earnings press release excerpt (SEC 8-K)
- MD&A excerpt (SEC 10-Q)
- Reddit sentiment summary (if available)

Claude's output:
- **Filter** — removes stocks with obvious thesis flaws
- **Conviction** — 1–10 score per stock
- **Thesis** — exactly 3 sentences per stock
- **Risk** — single biggest risk to the thesis

**Claude must only use data provided — never rely on training knowledge for current prices or events.**

---

## Project Structure

```
quantly/
├── CLAUDE.md
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── quantly/
│   ├── config.py                     # env vars, settings
│   ├── data/
│   │   ├── cache.py                  # disk cache (pickle, TTL-based)
│   │   ├── universe.py               # SEC EDGAR tickers + Wikipedia lists
│   │   ├── prices.py                 # yfinance OHLCV
│   │   ├── fundamentals.py           # yfinance quarterly financials
│   │   └── sentiment/
│   │       ├── news.py               # yfinance news + VADER scoring
│   │       ├── earnings_calls.py     # SEC EDGAR 8-K press releases
│   │       ├── sec_filings.py        # SEC EDGAR 10-Q/10-K MD&A
│   │       └── reddit.py             # PRAW (optional)
│   ├── features/
│   │   ├── technical.py              # RSI, MACD, momentum, Bollinger, ATR
│   │   ├── kpi.py                    # revenue growth, EPS surprise, FCF yield, ROE
│   │   ├── sentiment_features.py     # aggregate sentiment into model features
│   │   └── pipeline.py              # master orchestrator → feature DataFrame
│   ├── models/
│   │   ├── labels.py                 # forward return labels (train/backtest only)
│   │   ├── train.py                  # LightGBM + XGBoost walk-forward training
│   │   ├── predict.py                # inference + SHAP values
│   │   └── evaluate.py               # Sharpe, drawdown, win rate
│   ├── claude_analyst/
│   │   ├── prompts.py               # system prompt + candidate prompt builder
│   │   ├── analyst.py               # Anthropic API call + JSON parsing
│   │   └── formatter.py             # assemble data package for Claude
│   ├── backtest/
│   │   ├── engine.py                # walk-forward backtest orchestrator
│   │   └── paper_trading.py         # SQLite pick tracker + outcome updater
│   └── cli/
│       ├── commands.py              # Click CLI: scan, backtest, train, paper
│       └── display.py               # Rich terminal output
│
├── skills/
│   ├── scan.md, backtest.md, train.md, paper-report.md, add-feature.md
├── tests/
│   ├── test_features.py, test_evaluate.py, test_prompts.py
└── data/
    └── cache/                        # gitignored
```

---

## Setup

```bash
# 1. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtualenv and install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. Configure API keys
cp .env.example .env
nano .env   # set ANTHROPIC_API_KEY (only required key)

# 4. Run tests
pytest tests/ -v

# 5. Train model (first time — takes 20–60 min)
python -m quantly train

# 6. Run a scan
python -m quantly scan
```

## Development Commands

```bash
python -m quantly scan                          # full scan, S&P 500
python -m quantly scan --tickers AAPL NVDA MSFT # quick test with specific tickers
python -m quantly scan --paper-trade            # also record picks
python -m quantly train                         # retrain ML models
python -m quantly backtest --years 3            # validate model
python -m quantly paper report                  # paper trading performance
python -m quantly paper positions               # open positions
pytest tests/ -v                                # run unit tests
```

---

## Key Design Principles

1. **No lookahead bias** — every feature uses only data available at `signal_date`
2. **Free by default** — all data fetched without API keys (except Anthropic)
3. **Cache aggressively** — every API/scrape result cached 24h+ to avoid rate limits
4. **Two-stage pipeline** — cheap ML pass first, expensive Claude calls only on top-20
5. **Explainability always** — every pick backed by SHAP features + Claude thesis
6. **Honest backtesting** — walk-forward only, include bear markets, compare vs SPY
7. **Graceful degradation** — Reddit/SEC parsing failures don't crash the pipeline

---

## Common Pitfalls

- **yfinance rate limits**: don't call `yf.Ticker().info` in a tight loop — use `yf.download()` for bulk price data
- **Survivorship bias**: when backtesting, use tickers that existed at the signal date, not today's constituents
- **Earnings date leakage**: never include post-earnings data in pre-earnings features
- **SEC EDGAR parsing**: 8-K/10-Q filing formats vary widely — always check text extraction quality
- **Reddit API changes**: Reddit periodically changes free tier limits — check PRAW docs if errors occur
- **NaN propagation**: yfinance returns NaN for missing quarters — `fillna(0)` before model inference only, not before feature inspection

---

## When Adding a New Feature

See `/add-feature` skill for the full checklist. Short version:
1. Add data fetch in `quantly/data/` with `@cached` decorator
2. Add transform in `quantly/features/`
3. Wire into `pipeline.py`
4. Verify no lookahead, check NaN rate
5. Run `python -m quantly backtest --years 1` to validate

## When Modifying the ML Model

1. Change config in `quantly/models/train.py`
2. Retrain: `python -m quantly train`
3. Validate: `python -m quantly backtest --years 2`
4. Only deploy if Sharpe ratio is equal or better than before
