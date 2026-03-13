"""Label construction for supervised ML training.

Labels are constructed STRICTLY using forward returns measured AFTER the signal date.
This module must NEVER be used at inference time — only during training/backtesting.

Label definition:
  Positive (1): stock outperforms SPY by > OUTPERFORMANCE_THRESHOLD within HORIZON trading days
  Negative (0): stock underperforms SPY by > UNDERPERFORMANCE_THRESHOLD within HORIZON trading days
  Excluded (NaN): ambiguous outcome, stocks near corporate events, or data unavailable
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from quantly.config import config

logger = logging.getLogger(__name__)

HORIZON = config.PREDICTION_HORIZON_DAYS
OUTPERFORM_THRESH = config.OUTPERFORMANCE_THRESHOLD
UNDERPERFORM_THRESH = config.UNDERPERFORMANCE_THRESHOLD


def compute_forward_return(
    ticker: str,
    signal_date: date,
    horizon_days: int = HORIZON,
) -> float | None:
    """Compute the total return from signal_date to signal_date + horizon_days.

    This function is ONLY for training/backtesting. Never call at inference time.

    Returns:
        Float return, or None if data unavailable
    """
    from quantly.data.prices import get_ohlcv

    end = signal_date + timedelta(days=horizon_days + 10)  # buffer for non-trading days
    try:
        ohlcv = get_ohlcv(ticker, signal_date, end)
        if len(ohlcv) < horizon_days:
            return None
        entry = ohlcv["close"].iloc[0]
        exit_ = ohlcv["close"].iloc[horizon_days - 1]
        return (exit_ - entry) / entry
    except Exception:
        return None


def compute_labels(
    tickers: list[str],
    signal_dates: list[date],
) -> pd.DataFrame:
    """Compute labels for a list of (ticker, signal_date) pairs.

    Returns:
        DataFrame with columns: ticker, signal_date, stock_return, spy_return,
        outperformance, label (1/0/NaN)
    """
    from quantly.data.prices import get_spy_returns

    rows = []
    for ticker, signal_date in zip(tickers, signal_dates, strict=True):
        stock_ret = compute_forward_return(ticker, signal_date)
        spy_ret = compute_forward_return("SPY", signal_date)

        if stock_ret is None or spy_ret is None:
            label = np.nan
            outperformance = np.nan
        else:
            outperformance = stock_ret - spy_ret
            if outperformance > OUTPERFORM_THRESH:
                label = 1
            elif outperformance < -UNDERPERFORM_THRESH:
                label = 0
            else:
                label = np.nan  # ambiguous — exclude from training

        rows.append({
            "ticker": ticker,
            "signal_date": signal_date,
            "stock_return": stock_ret,
            "spy_return": spy_ret,
            "outperformance": outperformance,
            "label": label,
        })

    return pd.DataFrame(rows)
