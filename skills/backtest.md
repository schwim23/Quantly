# /backtest — Run and Analyze Backtests

Runs a walk-forward backtest of the ML model and reports performance metrics.

## What This Does

- Re-runs the full prediction pipeline on historical data
- Uses walk-forward cross-validation (no lookahead bias)
- Reports Sharpe ratio, win rate, max drawdown, avg return vs. SPY
- Breaks results down by year, sector, and market regime (bull/bear/sideways)

## Steps

1. **Run a full 3-year backtest**
   ```bash
   python -m quantly backtest --years 3
   ```

2. **Run a quick 1-year backtest** (faster, for iteration)
   ```bash
   python -m quantly backtest --years 1
   ```

3. **Backtest a specific date range**
   ```bash
   python -m quantly backtest --start 2022-01-01 --end 2023-12-31
   ```

4. **Run with detailed per-trade output**
   ```bash
   python -m quantly backtest --years 3 --verbose
   ```

## Key Metrics to Evaluate

| Metric | Target | Red Flag |
|---|---|---|
| Sharpe Ratio | > 1.0 | < 0.5 |
| Win Rate | > 55% | < 48% |
| Max Drawdown | < 20% | > 35% |
| Avg Return vs SPY | > +2% per trade | Negative |
| Annualized Return | > 15% | < SPY benchmark |

## Interpreting Results

- **Check for regime sensitivity**: if model only works in bull markets, it's not robust
- **Check SHAP stability**: run `python -m quantly backtest --shap-report` to see if feature importances are consistent across time periods
- **Survivorship bias check**: confirm delisted stocks are present in the historical universe

## When to Re-run
- After adding or removing any feature
- After changing the ML model hyperparameters
- Monthly, to check for model drift

## Anti-Patterns
- Do NOT cherry-pick favorable date ranges to present results
- Do NOT adjust model parameters after seeing backtest results on the full dataset (data snooping)
- Always include 2022 (bear market) in validation — any model only tested on 2023–2024 bull runs is suspect
