# /paper-report — Weekly Paper Trading Performance Report

Generates a performance report of all paper trades tracked since paper trading began.

## What This Does

- Reads all picks from `data/paper_trades.db`
- Fetches actual price outcomes for picks that have matured (>15 trading days old)
- Computes win rate, average return, and performance vs. SPY for each week's batch
- Highlights biggest winners and losers
- Flags any systematic patterns in wrong picks (e.g., always wrong on high-short-interest stocks)

## Steps

1. **Generate the weekly report**
   ```bash
   python -m quantly paper report
   ```

2. **View all open (not yet matured) positions**
   ```bash
   python -m quantly paper positions
   ```

3. **Record a new scan's picks into paper trading**
   ```bash
   python -m quantly scan | python -m quantly paper record
   ```
   Or combined:
   ```bash
   python -m quantly scan --paper-trade
   ```

4. **Export to CSV for analysis**
   ```bash
   python -m quantly paper report --export paper_results.csv
   ```

## Report Sections

### Summary Stats
- Total picks tracked
- Win rate (% that outperformed SPY by >3%)
- Average outperformance per pick
- Best and worst weeks

### By Feature Cohort
- Picks that fired on KPI beats vs. sentiment vs. technical — which cohort performs best?
- This tells you which signal types are actually adding value

### Drift Detection
- Compare last 4-week rolling win rate vs. backtest baseline
- Alert if rolling win rate drops >10% below backtest win rate (model drift signal)

## Database Schema

`data/paper_trades.db` — SQLite:
```
paper_trades(
  id, ticker, scan_date, signal_score, claude_conviction,
  entry_price, thesis, top_features,
  outcome_date, outcome_price, outperformance_vs_spy,
  label  -- 1=win, 0=loss, NULL=pending
)
```

## Interpreting Results

- **Win rate > backtest baseline**: model generalizing well to live conditions
- **Win rate < backtest baseline by > 10%**: investigate for data leakage in backtest or market regime change
- **Consistent losers in one sector**: add sector-specific features or exclude that sector
- **Claude re-rankings consistently better than ML rank**: consider increasing weight of sentiment features
