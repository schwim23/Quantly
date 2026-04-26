"""Backtest performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.05, periods_per_year: int = 17) -> float:
    """Annualized Sharpe ratio. periods_per_year=17 for ~15-day trade cycles."""
    if returns.std() == 0:
        m = float(returns.mean())
        if m > 0:
            return 100.0
        if m < 0:
            return -100.0
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def compute_max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def compute_summary_metrics(picks_df: pd.DataFrame) -> dict[str, float]:
    """All performance metrics from a backtest results DataFrame.

    Args:
        picks_df: DataFrame with columns: outperformance, label
    """
    returns = picks_df["outperformance"].dropna()
    labeled = picks_df["label"].dropna()

    if returns.empty:
        return {}

    return {
        "sharpe": compute_sharpe(returns),
        "max_drawdown": compute_max_drawdown(returns),
        "win_rate": float(labeled.mean()),
        "avg_outperformance": float(returns.mean()),
        "total_picks": int(len(returns)),
        "annualized_return": float((1 + returns.mean()) ** 17 - 1),
    }
