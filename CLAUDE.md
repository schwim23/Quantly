# Quantly

A swing + trend-following stock signal engine with a web dashboard for approving picks
and tracking a shadow trading book against SPY.

---

## Strategy Design

### Trading Style
Hybrid of two complementary styles:
- **Swing trading** — 2–10 day holds, momentum and short-term catalyst driven
- **Position / trend following** — multi-week holds, fundamental quality + sustained momentum

The same signal pipeline feeds both; holding period and exit rules determine which mode a
position falls into.

### Universe
- US equities only
- S&P 500 + Russell 1000 (large/mid cap for liquidity)
- Minimum $5M average daily dollar volume filter
- No sector exclusions; soft affinity for AI chip / semiconductor exposure is captured
  as a feature weight, not a hard filter

### Capital & Positions
- Starting capital: $10,000 (shadow book only until validated)
- 10–15 open positions at a time
- Position sizing: volatility-scaled, targeting 7–10% of portfolio per position
  - Scale down for high-beta names so dollar risk is equal across positions
- No leverage

---

## Signal Architecture

### Core Philosophy
Three-layer stack:

```
[ Sentiment + Catalyst signals ]  ← secret sauce, ML-weighted
          ↓
[ ML Scoring Engine (LightGBM) ]  ← ranks candidates by predicted return probability
          ↓
[ Fundamentals Quality Gate ]     ← filters out weak balance sheets before final picks
```

The edge is in the **weighting of sentiment and catalyst signals** — harder to arbitrage
than pure technicals because it requires NLP, source breadth, and timing precision.
Fundamentals act as a filter, not a signal generator.

### Stage 1 — Universe Screening (fast pass)
Run daily pre-market. Filter full US equity universe down to ~300–500 liquid candidates:
- Price > $5
- Avg daily dollar volume > $5M (20-day)
- Not within 5 days of a known data gap (splits, mergers in progress)

### Stage 2 — Feature Pipeline
Compute all features for screened candidates. See Feature Groups below.

### Stage 3 — ML Scoring (LightGBM)
- Ranks candidates by predicted probability of outperforming SPY over next 15 trading days
- Outputs top 20 candidates with SHAP feature importance per prediction
- Retrain weekly on rolling 2-year walk-forward window

### Stage 4 — Fundamentals Gate
Filter top 20 through minimum quality bar:
- Positive free cash flow (TTM) OR revenue growth > 10% YoY (growth exception)
- Debt/equity ratio not in top decile of universe
- No SEC filing red flags (recent restatements, going concern language)

### Stage 5 — Dashboard Output
Final ranked picks (typically 8–12) displayed in web dashboard with:
- ML conviction score
- Top 3 SHAP drivers
- Suggested entry, stop level (2× ATR), and max hold date
- User approves → position added to shadow book

---

## Feature Groups

### Sentiment Features (highest weight — the edge)
- **News sentiment score** — Alpha Vantage AI-scored headlines, 7-day rolling avg
- **GDELT tone score** — 15-min updated global news tone (-100 to +100), stock-level
- **StockTwits sentiment** — bull/bear ratio per ticker, scraped daily
- **Congressional trading** — Quiver Quant: recent buys by Congress members in this stock
- **Options flow** — Quiver Quant / Unusual Whales: unusual call activity vs. 30-day avg

### Catalyst Features (high weight)
- Days until / since earnings announcement
- EPS surprise magnitude (last 2 quarters)
- Analyst upgrade/downgrade in last 10 days
- Revenue guidance revision direction
- Earnings call tone delta (current vs. prior quarter, from FMP transcripts)
- SEC MD&A risk-factor delta (new risks added vs. removed, from EDGAR)

### Technical / Price Features (medium weight)
- RSI(14)
- MACD signal line crossover (binary + magnitude)
- Price momentum: 5d, 10d, 20d returns
- Volume spike ratio: current vs. 20-day avg
- Distance from 52-week high/low (normalized)
- ATR(14) relative to sector median
- Bollinger Band %B position

### Fundamental Features (used as gate + weak signal)
- Revenue growth QoQ and YoY
- Gross margin trend (expanding = +1, contracting = -1)
- Free cash flow yield
- EPS growth trend (3-quarter slope)
- Debt/equity ratio (percentile rank within universe)

---

## Data Sources

| Category | Source | Cost | Notes |
|---|---|---|---|
| Price / OHLCV | **Tiingo** | Free / $7/mo | 30yr history, clean API |
| Fundamentals + Earnings | **Financial Modeling Prep (FMP)** | Free (250 req/day) | Earnings transcripts, calendar, KPIs |
| AI-scored news sentiment | **Alpha Vantage** | Free | Built-in sentiment on news feed |
| Broad news tone | **GDELT Project** | Free | Google-backed, 15-min updates |
| Catalyst calendar | **Finnhub** | Free | Earnings + economic events |
| SEC filings | **SEC EDGAR** | Free | No key needed, 10 req/sec |
| Congressional trades + options flow | **Quiver Quant** | Free tier | Underused alpha signal |
| Options flow (secondary) | **Unusual Whales** | Free tier | Real-time unusual activity |
| Social sentiment | **StockTwits** | Free (scrape) | Bull/bear ratio per ticker |

All API responses cached locally to disk (SQLite) to avoid repeat calls and rate limits.

---

## ML Model

### Model Type
**LightGBM** — gradient boosted trees, handles mixed feature types well, fast inference,
native SHAP support.

### Label
- **Positive (1)**: stock outperforms SPY by >3% within 15 trading days of signal date
- **Negative (0)**: underperforms SPY by >2% or is flat within 15 days

### Training
- Walk-forward validation: train on rolling 2-year window, validate on next quarter
- Minimum 3 years of history for initial training
- Retrain weekly with fresh data
- Purge 5-day gap between train and test sets (no leakage)
- Survivorship bias prevention: include delisted stocks in training history

### Explainability
- SHAP values computed for every prediction
- Top 3 features per pick surfaced in dashboard

---

## Risk Management

### Position-Level Rules
- **Stop loss**: 2× ATR(14) trailing stop from entry price, updated daily
- **Take profit**: none — let winners run, trail the stop upward
- **Max hold**: 20 trading days hard cap; exit at close on day 20 if stop not triggered
- **Max loss per position**: size down pre-entry so that hitting the ATR stop costs no
  more than 2% of total portfolio

### Portfolio-Level Rules
- **Drawdown kill switch**: if portfolio drops 12% from peak NAV, pause all new entries
  until portfolio recovers to within 8% of peak
- **Max concurrent positions**: 15 (prevents over-concentration on hot signal days)
- **Re-entry rule**: after a stop-out, same ticker cannot re-enter for 5 trading days

---

## Web Dashboard

### Pages
1. **Today's Picks** — ranked list of new signals, each with:
   - Ticker, company name, sector
   - ML conviction score (0–100)
   - Top 3 signal drivers (SHAP-based)
   - Suggested entry price, stop price, max hold date
   - "Approve" button → logs to shadow book
2. **Shadow Book** — open positions table:
   - Entry date/price, current price, current stop, days held, unrealized P&L
   - Mark as "Exited" when you execute the trade (logs exit price + date)
3. **Performance** — shadow book analytics:
   - Portfolio P&L vs SPY (cumulative chart)
   - Win rate, avg return per trade, Sharpe ratio, max drawdown
   - Rolling 30-day performance

### Tech Stack
- **Backend**: FastAPI (Python)
- **Frontend**: Jinja2 templates + HTMX (no JS framework, keeps it simple)
- **Database**: SQLite (positions, trades, daily snapshots)
- **Scheduler**: APScheduler (runs daily pipeline pre-market ~7am ET)
- **Styling**: Tailwind CSS via CDN

---

## Project Structure

```
quantly/
├── CLAUDE.md
├── pyproject.toml
├── .env.example
│
├── quantly/
│   ├── __init__.py
│   ├── __main__.py              # Entry point: launch web server or run pipeline
│   │
│   ├── data/
│   │   ├── universe.py          # Universe screener (liquidity filter)
│   │   ├── prices.py            # OHLCV from Tiingo
│   │   ├── fundamentals.py      # KPIs from FMP + SEC EDGAR
│   │   ├── sentiment/
│   │   │   ├── news.py          # Alpha Vantage AI-scored news
│   │   │   ├── gdelt.py         # GDELT tone scores
│   │   │   ├── stocktwits.py    # StockTwits bull/bear scraper
│   │   │   ├── congress.py      # Quiver Quant congressional trades
│   │   │   └── options_flow.py  # Quiver Quant + Unusual Whales flow
│   │   ├── catalysts.py         # Finnhub earnings calendar + FMP transcripts
│   │   ├── sec_filings.py       # EDGAR 10-Q/10-K MD&A + risk factor delta
│   │   └── cache.py             # SQLite-backed disk cache
│   │
│   ├── features/
│   │   ├── technical.py         # RSI, MACD, momentum, volume, ATR, BBands
│   │   ├── sentiment.py         # Aggregate all sentiment sources into model features
│   │   ├── catalyst.py          # Earnings surprise, guidance, analyst actions
│   │   ├── fundamental.py       # FCF yield, margins, revenue growth, D/E
│   │   └── pipeline.py          # Assembles full feature matrix for model input
│   │
│   ├── models/
│   │   ├── train.py             # Walk-forward training loop
│   │   ├── predict.py           # Inference + SHAP values
│   │   ├── labels.py            # Label construction (outperformance vs SPY)
│   │   └── evaluate.py          # Sharpe, drawdown, win rate metrics
│   │
│   ├── risk/
│   │   ├── sizing.py            # Volatility-scaled position sizing
│   │   └── stops.py             # ATR trailing stop calculator, kill switch logic
│   │
│   ├── shadow_book/
│   │   ├── book.py              # Position tracker (open, close, P&L)
│   │   ├── performance.py       # Portfolio metrics vs SPY benchmark
│   │   └── db.py                # SQLite schema and queries
│   │
│   ├── pipeline.py              # Daily orchestrator: screen → features → score → picks
│   │
│   └── web/
│       ├── app.py               # FastAPI app + routes
│       ├── templates/
│       │   ├── base.html
│       │   ├── picks.html       # Today's Picks page
│       │   ├── book.html        # Shadow Book page
│       │   └── performance.html # Performance page
│       └── static/
│           └── style.css
│
└── data/                        # Gitignored: cache + DB
    ├── cache.db
    └── shadow_book.db
```

---

## Key Design Principles

1. **Sentiment + catalyst is the edge** — weight these features highest; they are harder
   to arbitrage than technicals because they require NLP, timing, and multi-source breadth
2. **Fundamentals as a filter, not a signal** — prevent chasing garbage on hype;
   don't use fundamentals to rank, only to gate
3. **Asymmetric payoff profile** — small defined losses (ATR stop), unlimited upside
   (trail and hold), time-boxed if flat (20-day cap)
4. **Shadow book first** — no real money until the system has 4–6 weeks of validated
   paper performance with real-time signal generation (not backtested)
5. **No lookahead bias** — every feature strictly uses data available at signal date;
   5-day purge gap in training; point-in-time universe
6. **Cache aggressively** — all API responses cached in SQLite; free tiers have tight
   rate limits
7. **Human in the loop** — system signals, human approves, system tracks; no auto-execution

---

## Development Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env  # Fill in API keys

# Run web dashboard (starts scheduler + server)
python -m quantly serve

# Run pipeline manually (generate today's picks)
python -m quantly scan

# Train / retrain ML model
python -m quantly train

# Run tests
pytest tests/ -v
```

---

## API Keys Required

```
ANTHROPIC_API_KEY=        # Reserved for future Claude analyst integration
TIINGO_API_KEY=
FMP_API_KEY=              # Financial Modeling Prep
ALPHA_VANTAGE_API_KEY=
FINNHUB_API_KEY=
QUIVER_API_KEY=           # Quiver Quant
UNUSUAL_WHALES_API_KEY=   # Optional, free tier
```

---

## Pitfalls to Avoid

- **Survivorship bias**: point-in-time universe only; delisted stocks included in training
- **Sentiment timing**: tag all sentiment to publication time, not today's date
- **Earnings leakage**: never include post-earnings data in pre-earnings features
- **Regime blindness**: validate model separately in bull, bear, and sideways periods
- **API rate limits**: always go through the cache layer; never call APIs directly in model code
- **Overfitting on recency**: walk-forward only; no peeking at future data during training
