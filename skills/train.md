# /train — Train or Retrain the ML Model

Trains the LightGBM/XGBoost ensemble on historical feature data with walk-forward validation.

## When to Run

- Initial setup (no model exists yet)
- Weekly refresh to incorporate new market data
- After adding new features to the pipeline
- After backtest reveals model drift

## Steps

1. **Ensure feature data is available**
   ```bash
   python -m quantly features build --years 3
   ```
   This fetches historical data and computes all features. Takes 30–60 minutes on first run.
   Subsequent runs are fast due to caching.

2. **Train the model**
   ```bash
   python -m quantly train
   ```
   This runs walk-forward training with the default config.

3. **Train with custom hyperparameters**
   ```bash
   python -m quantly train --config configs/lgbm_aggressive.yaml
   ```

4. **Run hyperparameter search** (slow, do infrequently)
   ```bash
   python -m quantly train --tune --trials 100
   ```

5. **Validate after training**
   Always run a backtest immediately after training:
   ```bash
   python -m quantly backtest --years 1
   ```
   If Sharpe drops significantly vs. prior model, investigate before deploying.

## Model Files

Trained models are saved to `models/`:
- `models/lgbm_latest.pkl` — current production model
- `models/xgb_latest.pkl` — XGBoost ensemble member
- `models/metadata.json` — training date, feature list, validation metrics

## Feature Engineering Checklist

Before training, verify:
- [ ] No NaN-heavy features (> 20% NaN rates indicate a data problem)
- [ ] No features with near-zero variance (useless to the model)
- [ ] No leaky features (check `quantly/models/labels.py` for label construction dates)
- [ ] Feature list in `pipeline.py` matches what was used in prior training run

## Label Construction

Labels are defined in `quantly/models/labels.py`:
- **Positive (1)**: stock total return beats SPY by > 3% within 15 trading days of signal date
- **Negative (0)**: stock underperforms SPY by > 2% or flat within 15 days
- **Excluded**: stocks within 3 days of earnings release, M&A announcements, or trading halts

Adjust the outperformance threshold in `config.py` if you want to target different return magnitudes.
