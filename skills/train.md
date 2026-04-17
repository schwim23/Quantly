# /train — Train the ML Model

## Usage

```bash
python -m quantly train                # 3-year training window, S&P 500
python -m quantly train --years 2 --universe nasdaq100
```

## What it does
1. Fetches universe and screens for liquidity
2. Builds feature matrix at monthly signal dates going back N years
3. Computes forward return labels (15-day outperformance vs SPY)
4. Trains LightGBM + XGBoost on the most recent N-year rolling window
5. Saves models to `models/lgbm_latest.pkl`, `models/xgb_latest.pkl`

## After training, always validate
```bash
python -m quantly backtest --years 1
```
If Sharpe < 0.5, investigate feature quality (NaN rates, leakage).

## Feature checklist before training
- [ ] No features with > 30% NaN rate
- [ ] No lookahead bias — all features use only pre-signal-date data
- [ ] `models/feature_names.txt` matches pipeline output columns
