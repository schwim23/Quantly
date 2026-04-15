"""Portfolio-level and model-level evaluation metrics.

Used to assess model quality from closed shadow book positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Annualised Sharpe ratio from a series of daily portfolio returns.

    Args:
        returns: Daily return series (fractions, e.g. 0.01 = 1%).
        risk_free_rate: Annual risk-free rate (default 5%).

    Returns:
        Annualised Sharpe ratio. Returns 0.0 if insufficient data or zero variance.
    """
    if len(returns) < 5:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((excess.mean() / std) * np.sqrt(252))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown as a positive fraction.

    Args:
        equity_curve: Portfolio value over time (absolute NAV, not returns).

    Returns:
        Maximum drawdown as a positive fraction (e.g. 0.15 = 15% drawdown).
    """
    if equity_curve.empty:
        return 0.0
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(abs(drawdown.min()))


def win_rate(trade_returns: pd.Series) -> float:
    """Fraction of trades with positive return.

    Args:
        trade_returns: Per-trade return series (fractions).

    Returns:
        Win rate in [0, 1]. Returns 0.0 if no trades.
    """
    if trade_returns.empty:
        return 0.0
    return float((trade_returns > 0).mean())


def compute_portfolio_metrics(
    trade_history: pd.DataFrame,
    portfolio_values: pd.Series,
) -> dict[str, float]:
    """Compute full suite of portfolio metrics.

    Args:
        trade_history: DataFrame with columns [ticker, entry_date, exit_date,
                       entry_price, exit_price, return_pct].
        portfolio_values: Daily portfolio NAV series indexed by date.

    Returns:
        Dict with: sharpe, max_drawdown, win_rate, avg_return, trade_count.
    """
    if trade_history.empty:
        return {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "trade_count": 0,
        }

    trade_returns = trade_history.get("return_pct", pd.Series(dtype=float))

    daily_returns = portfolio_values.pct_change().dropna()

    return {
        "sharpe": sharpe_ratio(daily_returns),
        "max_drawdown": max_drawdown(portfolio_values),
        "win_rate": win_rate(trade_returns),
        "avg_return": float(trade_returns.mean()) if not trade_returns.empty else 0.0,
        "trade_count": len(trade_history),
    }


def model_precision(predictions: pd.Series, labels: pd.Series) -> float:
    """Precision of positive predictions (how often conviction ≥ 60 → actually positive).

    Args:
        predictions: Conviction scores (0–100).
        labels: Actual labels (1 = outperformed, 0 = underperformed).

    Returns:
        Precision in [0, 1].
    """
    if predictions.empty or labels.empty:
        return 0.0
    positive_mask = predictions >= 60
    if positive_mask.sum() == 0:
        return 0.0
    return float(labels[positive_mask].mean())
