# /paper-report — Weekly Paper Trading Report

## Usage

```bash
python -m quantly paper report             # update outcomes + show report
python -m quantly paper positions          # show open positions
python -m quantly paper report --export results/paper_$(date +%Y%m%d).csv
```

## What it does
- Fetches actual price outcomes for picks older than 15 trading days
- Computes outperformance vs SPY for each pick
- Reports win rate, avg outperformance, best/worst pick, rolling 4-week win rate

## Interpreting the report
- **Win rate > backtest baseline**: model generalising well
- **Win rate < backtest by > 10%**: possible model drift — consider retraining
- **Consistent losses in one sector**: add sector exclusion or sector-specific features
- **Claude re-ranks consistently beat ML rank**: increase weight on sentiment features
