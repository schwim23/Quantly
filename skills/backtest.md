# /backtest — Run a Walk-Forward Backtest

## Usage

```bash
python -m quantly backtest                      # 3-year default, S&P 500
python -m quantly backtest --years 1            # quick validation
python -m quantly backtest --universe nasdaq100
```

## Key metrics to evaluate

| Metric | Target | Red flag |
|---|---|---|
| Sharpe Ratio | > 1.0 | < 0.5 |
| Win Rate | > 55% | < 48% |
| Max Drawdown | < −20% | > −35% |
| Avg Outperformance | > +2% | Negative |
| Annualized Return | > 15% | < SPY |

## Anti-patterns to avoid
- Do NOT cherry-pick favorable date ranges
- Always include 2022 (bear market) in validation
- If Sharpe drops after adding a new feature, revert it
