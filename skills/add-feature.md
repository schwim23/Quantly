# /add-feature — Add a New Feature to the Pipeline

Step-by-step guide for safely adding a new KPI, sentiment, or technical feature.

## Overview

Features flow through three layers:
1. **Raw data** — fetched in `quantly/data/`
2. **Feature transform** — computed in `quantly/features/`
3. **Pipeline registration** — added to `quantly/features/pipeline.py`

## Steps

### 1. Define the feature clearly

Before writing code, answer:
- What does this feature measure?
- Is it available in real-time (no lookahead)?
- What data source provides it?
- Is it already computable from existing raw data, or does it need a new fetch?

### 2. Add the raw data fetch (if needed)

Add to the appropriate file in `quantly/data/`:
- `prices.py` — OHLCV, volume, market cap
- `fundamentals.py` — KPIs from filings or API
- `sentiment/news.py` — news headline data
- `sentiment/sec_filings.py` — EDGAR filing data
- `sentiment/earnings_calls.py` — transcript data
- `sentiment/reddit.py` — Reddit post/comment data

Always cache API responses in `quantly/data/cache.py`.

### 3. Implement the feature transform

Add your feature computation in `quantly/features/`:
- **Technical features** → `technical.py`
- **KPI / fundamental features** → `kpi.py`
- **Sentiment-derived features** → `sentiment_features.py`

Feature functions must:
- Accept a DataFrame of raw data for a single ticker
- Return a scalar (or small dict of scalars) for a single observation date
- Handle missing data gracefully (return `np.nan`, never crash)
- Be vectorizable across a DataFrame of tickers

### 4. Register in the pipeline

In `quantly/features/pipeline.py`, add your feature to the `FEATURE_REGISTRY`:
```python
FEATURE_REGISTRY = [
    # ... existing features ...
    ("my_new_feature", compute_my_new_feature, {"param": value}),
]
```

### 5. Validate: no lookahead bias

Run the lookahead check:
```bash
python -m quantly features check-lookahead --feature my_new_feature
```

This verifies that for every row, the feature only uses data from before the signal date.

### 6. Check NaN rate and variance

```bash
python -m quantly features stats --feature my_new_feature
```

Red flags:
- NaN rate > 20%: data source too sparse, reconsider
- Variance ≈ 0: feature is constant, useless

### 7. Backtest with the new feature

```bash
python -m quantly backtest --years 1
```

Compare Sharpe ratio and win rate vs. the baseline without the feature.
Only merge the feature if it improves or is neutral to performance.

### 8. Update CLAUDE.md

Add the new feature to the **Feature Groups** section of `CLAUDE.md` with a one-line description.

## Common Mistakes

- **Lookahead via joining on future data**: always join on `signal_date`, never on `outcome_date`
- **Point-in-time KPIs**: use the filing date, not the report date — earnings are often restated
- **Survivor bias in sentiment**: historical Reddit posts may be deleted; note gaps in the data
- **Double-counting**: check if the new feature is highly correlated (>0.9) with an existing feature — if so, it adds little value
