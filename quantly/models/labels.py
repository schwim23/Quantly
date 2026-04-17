"""Label construction — ONLY used for training and backtesting, never at inference.

Positive (1):  stock outperforms SPY by > OUTPERFORMANCE_THRESHOLD over HORIZON trading days
Negative (0):  stock underperforms SPY by > UNDERPERFORMANCE_THRESHOLD
Excluded (NaN): ambiguous outcome or data unavailable
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from quantly.config import config

logger = logging.getLogger(__name__)


def compute_forward_return(ticker: str, signal_date: date, horizon: int = config.PREDICTION_HORIZON_DAYS) -> float | None:
    """Total return from signal_date to signal_date + horizon trading days."""
    from quantly.data.prices import get_ohlcv

    end = signal_date + timedelta(days=horizon + 14)
    try:
        ohlcv = get_ohlcv(ticker, signal_date, end)
        trading_days = ohlcv[ohlcv.index > pd.Timestamp(signal_date)]
        if len(trading_days) < horizon:
            return None
        entry = float(ohlcv["close"].iloc[0])
        exit_ = float(trading_days["close"].iloc[horizon - 1])
        return (exit_ - entry) / entry
    except Exception:
        return None


def compute_labels(tickers: list[str], signal_dates: list[date]) -> pd.DataFrame:
    """Compute labels for (ticker, signal_date) pairs.

    Returns:
        DataFrame with columns: ticker, signal_date, stock_return, spy_return,
        outperformance, label (1 / 0 / NaN)
    """
    rows = []
    for ticker, signal_date in zip(tickers, signal_dates, strict=True):
        stock_ret = compute_forward_return(ticker, signal_date)
        spy_ret = compute_forward_return("SPY", signal_date)

        if stock_ret is None or spy_ret is None:
            outperformance, label = float("nan"), float("nan")
        else:
            outperformance = stock_ret - spy_ret
            if outperformance > config.OUTPERFORMANCE_THRESHOLD:
                label = 1.0
            elif outperformance < -config.UNDERPERFORMANCE_THRESHOLD:
                label = 0.0
            else:
                label = float("nan")  # ambiguous — exclude from training

        rows.append({
            "ticker": ticker,
            "signal_date": signal_date,
            "stock_return": stock_ret,
            "spy_return": spy_ret,
            "outperformance": outperformance,
            "label": label,
        })

    return pd.DataFrame(rows)
