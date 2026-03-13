"""Financial KPI features computed from fundamental data.

All features use point-in-time data (no lookahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_revenue_growth(income: pd.DataFrame) -> dict[str, float]:
    """Revenue growth QoQ and YoY.

    Args:
        income: Quarterly income statement DataFrame, sorted by fiscal_date_ending desc

    Returns:
        Dict with keys: revenue_growth_qoq, revenue_growth_yoy
    """
    if len(income) < 5:
        return {"revenue_growth_qoq": np.nan, "revenue_growth_yoy": np.nan}
    rev = income["revenue"]
    return {
        "revenue_growth_qoq": (rev.iloc[0] - rev.iloc[1]) / abs(rev.iloc[1]),
        "revenue_growth_yoy": (rev.iloc[0] - rev.iloc[4]) / abs(rev.iloc[4]),
    }


def compute_eps_surprise(income: pd.DataFrame) -> dict[str, float]:
    """EPS surprise percentage vs. analyst consensus.

    Returns:
        Dict with keys: eps_surprise_pct (positive = beat, negative = miss)
    """
    if "eps_estimate" not in income.columns or len(income) == 0:
        return {"eps_surprise_pct": np.nan}
    eps_actual = income["eps"].iloc[0]
    eps_est = income["eps_estimate"].iloc[0]
    if eps_est == 0:
        return {"eps_surprise_pct": np.nan}
    return {"eps_surprise_pct": (eps_actual - eps_est) / abs(eps_est)}


def compute_gross_margin_trend(income: pd.DataFrame) -> dict[str, float]:
    """Gross margin level and QoQ trend.

    Returns:
        Dict with keys: gross_margin, gross_margin_delta_qoq
    """
    if len(income) < 2:
        return {"gross_margin": np.nan, "gross_margin_delta_qoq": np.nan}
    margin = income["gross_profit"] / income["revenue"]
    return {
        "gross_margin": margin.iloc[0],
        "gross_margin_delta_qoq": margin.iloc[0] - margin.iloc[1],
    }


def compute_fcf_yield(
    cashflow: pd.DataFrame,
    market_cap: float,
) -> dict[str, float]:
    """Free cash flow yield = FCF / market cap.

    Returns:
        Dict with key: fcf_yield
    """
    if len(cashflow) == 0 or market_cap <= 0:
        return {"fcf_yield": np.nan}
    fcf = cashflow["free_cash_flow"].iloc[0]
    return {"fcf_yield": fcf / market_cap}


def compute_debt_equity_change(balance: pd.DataFrame) -> dict[str, float]:
    """Change in debt/equity ratio QoQ.

    Returns:
        Dict with keys: debt_equity_ratio, debt_equity_delta_qoq
    """
    if len(balance) < 2:
        return {"debt_equity_ratio": np.nan, "debt_equity_delta_qoq": np.nan}
    de = balance["long_term_debt"] / balance["total_equity"].replace(0, np.nan)
    return {
        "debt_equity_ratio": de.iloc[0],
        "debt_equity_delta_qoq": de.iloc[0] - de.iloc[1],
    }


def compute_roe_trend(income: pd.DataFrame, balance: pd.DataFrame) -> dict[str, float]:
    """Return on equity and QoQ trend.

    Returns:
        Dict with keys: roe, roe_delta_qoq
    """
    if len(income) < 2 or len(balance) < 2:
        return {"roe": np.nan, "roe_delta_qoq": np.nan}
    roe = income["net_income"] / balance["total_equity"].replace(0, np.nan)
    return {
        "roe": roe.iloc[0],
        "roe_delta_qoq": roe.iloc[0] - roe.iloc[1],
    }


def compute_operating_leverage(income: pd.DataFrame) -> dict[str, float]:
    """Operating leverage = revenue growth / opex growth (QoQ).

    High positive = scaling efficiently. Negative = costs growing faster than revenue.

    Returns:
        Dict with key: operating_leverage_qoq
    """
    # TODO: requires operating expenses column in income statement
    return {"operating_leverage_qoq": np.nan}


def compute_all_kpi_features(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    market_cap: float,
) -> dict[str, float]:
    """Compute all KPI features from fundamental data.

    Returns:
        Dict of feature_name -> float value
    """
    features: dict[str, float] = {}
    features.update(compute_revenue_growth(income))
    features.update(compute_eps_surprise(income))
    features.update(compute_gross_margin_trend(income))
    features.update(compute_fcf_yield(cashflow, market_cap))
    features.update(compute_debt_equity_change(balance))
    features.update(compute_roe_trend(income, balance))
    features.update(compute_operating_leverage(income))
    return features
