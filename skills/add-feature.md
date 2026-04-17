# /add-feature — Add a New Feature to the Pipeline

## Checklist

1. **Define it clearly** — what does it measure? Is it available before the signal date?
2. **Add raw data fetch** in `quantly/data/` (use `@cached` decorator)
3. **Add transform** in `quantly/features/` — must return a float or dict of floats
4. **Wire into pipeline** in `quantly/features/sentiment_features.py` or `kpi.py` / `technical.py`
5. **Add call** in `quantly/features/pipeline.py` → `build_features_for_ticker()`
6. **Verify no lookahead** — all data must be from before `signal_date`
7. **Check NaN rate** — if > 30% NaN, the data source is too sparse
8. **Run 1-year backtest** — confirm Sharpe doesn't drop
9. **Update CLAUDE.md** Feature Groups section

## Common mistakes
- Joining on outcome date instead of signal date → lookahead bias
- Using restated financials instead of originally-filed values → point-in-time violation
- High correlation with existing feature → adds no value, creates noise
