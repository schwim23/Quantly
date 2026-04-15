"""Portfolio performance analytics — P&L, vs-SPY benchmark, metrics.

Functions here work on DataFrames produced by ShadowBook and are
consumed by the web dashboard's performance page.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from quantly.models.evaluate import (
    compute_portfolio_metrics,
    max_drawdown,
    sharpe_ratio,
    win_rate,
)


# ── P&L helpers ───────────────────────────────────────────────────────────────

def compute_trade_returns(trade_history: pd.DataFrame) -> pd.Series:
    """Return a Series of per-trade return_pct values (closed trades only).

    Args:
        trade_history: DataFrame with at minimum a ``return_pct`` column,
            as produced by ``ShadowBook.get_trade_history()``.

    Returns:
        pd.Series of floats (empty series if no trades).
    """
    if trade_history.empty or "return_pct" not in trade_history.columns:
        return pd.Series(dtype=float)
    return trade_history["return_pct"].dropna()


def compute_daily_returns(nav_history: pd.DataFrame) -> pd.Series:
    """Convert NAV snapshots to daily percentage returns.

    Args:
        nav_history: DataFrame with ``nav_date`` and ``portfolio_value``
            columns, as produced by ``ShadowBook.get_nav_history()``.

    Returns:
        pd.Series indexed by date.
    """
    if nav_history.empty or "portfolio_value" not in nav_history.columns:
        return pd.Series(dtype=float)
    s = nav_history.set_index("nav_date")["portfolio_value"].sort_index()
    return s.pct_change().dropna()


# ── SPY benchmark ─────────────────────────────────────────────────────────────

def benchmark_vs_spy(
    nav_history: pd.DataFrame,
    spy_prices: pd.Series,
) -> pd.DataFrame:
    """Align portfolio NAV vs SPY on the same date axis.

    Both series are rebased to 100 at the first common date so they
    can be overlaid on a single chart.

    Args:
        nav_history: From ``ShadowBook.get_nav_history()``.
        spy_prices: Daily closing prices for SPY, indexed by date.

    Returns:
        DataFrame with columns: ``date``, ``portfolio``, ``spy``.
        Returns an empty DataFrame if there is no overlap.
    """
    if nav_history.empty:
        return pd.DataFrame(columns=["date", "portfolio", "spy"])

    nav = nav_history.set_index("nav_date")["portfolio_value"].sort_index()

    # Align indices
    common = nav.index.intersection(spy_prices.index)
    if len(common) == 0:
        return pd.DataFrame(columns=["date", "portfolio", "spy"])

    nav_aligned = nav.loc[common]
    spy_aligned = spy_prices.loc[common]

    # Rebase to 100 at start
    base_nav = nav_aligned.iloc[0]
    base_spy = spy_aligned.iloc[0]
    rebased_portfolio = (nav_aligned / base_nav * 100).round(2)
    rebased_spy = (spy_aligned / base_spy * 100).round(2)

    return pd.DataFrame({
        "date": common,
        "portfolio": rebased_portfolio.values,
        "spy": rebased_spy.values,
    })


# ── Summary metrics ───────────────────────────────────────────────────────────

def portfolio_summary(
    trade_history: pd.DataFrame,
    nav_history: pd.DataFrame,
    spy_prices: Optional[pd.Series] = None,
) -> dict:
    """Return a summary dict for the performance dashboard page.

    Keys
    ----
    sharpe, max_drawdown, win_rate, avg_return, trade_count
        Core metrics from evaluate.py.
    spy_return
        Total SPY return over the same period (fractional, e.g. 0.12).
        None if spy_prices not provided or no overlap.
    portfolio_return
        Total portfolio return from first to last NAV snapshot.
    """
    trade_returns = compute_trade_returns(trade_history)
    equity = nav_history["portfolio_value"] if not nav_history.empty else pd.Series(dtype=float)
    metrics = compute_portfolio_metrics(
        trade_history if not trade_history.empty else pd.DataFrame(),
        equity,
    )

    # Total portfolio return
    if len(equity) >= 2:
        portfolio_return = round((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0], 4)
    else:
        portfolio_return = 0.0

    # SPY return over same window
    spy_return = None
    if spy_prices is not None and not nav_history.empty:
        nav_dates = nav_history.set_index("nav_date").index
        common = nav_dates.intersection(spy_prices.index)
        if len(common) >= 2:
            spy_return = round(
                (spy_prices.loc[common[-1]] - spy_prices.loc[common[0]])
                / spy_prices.loc[common[0]],
                4,
            )

    return {
        **metrics,
        "portfolio_return": portfolio_return,
        "spy_return": spy_return,
    }
