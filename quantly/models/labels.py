"""Label construction for ML training.

Label definition (from strategy spec):
  Positive (1): stock outperforms SPY by >3% within 15 trading days of signal date
  Negative (0): underperforms SPY by >2% OR is flat within 15 days

We compute forward returns for each ticker relative to SPY and apply
the outperformance threshold. A 5-day purge gap is enforced between
train and test sets to prevent leakage.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FORWARD_DAYS = 15          # prediction horizon
OUTPERFORM_THRESHOLD = 0.03   # must beat SPY by 3%
UNDERPERFORM_THRESHOLD = -0.02  # or lag SPY by 2%
PURGE_GAP_DAYS = 5            # days between train end and test start


def compute_forward_return(
    ticker_prices: pd.Series,
    spy_prices: pd.Series,
    signal_date: date,
    forward_days: int = FORWARD_DAYS,
) -> Optional[float]:
    """Compute ticker return minus SPY return over *forward_days* from *signal_date*.

    Args:
        ticker_prices: Close prices indexed by date (datetime.date).
        spy_prices: SPY close prices, same index type.
        signal_date: The date the signal is generated.
        forward_days: Look-ahead period in trading days.

    Returns:
        Excess return (float) or None if insufficient data.
    """
    # Find signal date position
    dates = sorted(ticker_prices.index)
    try:
        start_idx = dates.index(signal_date)
    except ValueError:
        # Try to find nearest date
        later = [d for d in dates if d >= signal_date]
        if not later:
            return None
        start_idx = dates.index(later[0])

    end_idx = start_idx + forward_days
    if end_idx >= len(dates):
        return None

    start_date = dates[start_idx]
    end_date = dates[end_idx]

    ticker_start = ticker_prices.get(start_date)
    ticker_end = ticker_prices.get(end_date)
    spy_start = spy_prices.get(start_date)
    spy_end = spy_prices.get(end_date)

    if any(v is None or v == 0 for v in [ticker_start, ticker_end, spy_start, spy_end]):
        return None

    ticker_ret = (ticker_end - ticker_start) / ticker_start
    spy_ret = (spy_end - spy_start) / spy_start
    return float(ticker_ret - spy_ret)


def make_label(excess_return: Optional[float]) -> Optional[int]:
    """Convert excess return to binary label.

    Returns:
        1 if outperforms SPY by >3%
        0 if underperforms by >2% or flat
        None if ambiguous (between thresholds) — excluded from training
    """
    if excess_return is None:
        return None
    if excess_return >= OUTPERFORM_THRESHOLD:
        return 1
    if excess_return <= UNDERPERFORM_THRESHOLD:
        return 0
    return None  # ambiguous zone — drop from training set


def build_label_series(
    tickers: list[str],
    all_prices: dict[str, pd.Series],  # ticker → close price series
    spy_prices: pd.Series,
    signal_dates: list[date],
) -> pd.DataFrame:
    """Build a label DataFrame for all (ticker, signal_date) pairs.

    Returns DataFrame with columns: ticker, signal_date, excess_return, label
    Rows where label is None (ambiguous zone) are excluded.
    """
    rows = []
    for ticker in tickers:
        prices = all_prices.get(ticker)
        if prices is None or prices.empty:
            continue
        for sig_date in signal_dates:
            excess = compute_forward_return(prices, spy_prices, sig_date)
            label = make_label(excess)
            if label is None:
                continue
            rows.append({
                "ticker": ticker,
                "signal_date": sig_date,
                "excess_return": excess,
                "label": label,
            })

    if not rows:
        return pd.DataFrame(columns=["ticker", "signal_date", "excess_return", "label"])

    return pd.DataFrame(rows)


def purge_gap(
    train_df: pd.DataFrame,
    test_start: date,
    gap_days: int = PURGE_GAP_DAYS,
) -> pd.DataFrame:
    """Remove rows from *train_df* within *gap_days* before *test_start*.

    Prevents leakage: a signal on day T could see forward prices that
    overlap with the test period.
    """
    cutoff = test_start - timedelta(days=gap_days)
    return train_df[train_df["signal_date"] <= cutoff]
