"""Backtest evaluation metrics.

Computes performance statistics from a set of picks and their outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.05, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio.

    Args:
        returns: Daily or per-trade returns series
        risk_free_rate: Annual risk-free rate (default 5% for current environment)
        periods_per_year: 252 for daily, ~17 if per 15-day trade

    Returns:
        Annualized Sharpe ratio
    """
    excess = returns - risk_free_rate / periods_per_year
    if returns.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def compute_max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from peak equity.

    Returns:
        Max drawdown as a negative float (e.g., -0.25 = 25% drawdown)
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()


def compute_win_rate(outcomes: pd.Series) -> float:
    """Fraction of picks that were positive (label=1).

    Args:
        outcomes: Series of 1/0 labels

    Returns:
        Win rate between 0 and 1
    """
    return outcomes.mean()


def compute_summary_metrics(
    picks_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute all performance metrics from a backtest results DataFrame.

    Args:
        picks_df: DataFrame with columns: ticker, signal_date, outperformance, label

    Returns:
        Dict with keys: sharpe, max_drawdown, win_rate, avg_outperformance,
        total_picks, annualized_return
    """
    returns = picks_df["outperformance"].dropna()
    labeled = picks_df["label"].dropna()

    return {
        "sharpe": compute_sharpe(returns, periods_per_year=17),  # ~17 15-day periods/year
        "max_drawdown": compute_max_drawdown(returns),
        "win_rate": compute_win_rate(labeled),
        "avg_outperformance": returns.mean(),
        "total_picks": len(returns),
        "annualized_return": (1 + returns.mean()) ** 17 - 1,
    }
