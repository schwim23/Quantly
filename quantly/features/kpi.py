"""Financial KPI features from yfinance fundamental data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(denominator):
        return float("nan")
    return numerator / denominator


def compute_revenue_growth(income: pd.DataFrame) -> dict[str, float]:
    if len(income) < 5 or "revenue" not in income.columns:
        return {"revenue_growth_qoq": float("nan"), "revenue_growth_yoy": float("nan")}
    rev = income["revenue"]
    return {
        "revenue_growth_qoq": _safe_ratio(rev.iloc[0] - rev.iloc[1], abs(rev.iloc[1])),
        "revenue_growth_yoy": _safe_ratio(rev.iloc[0] - rev.iloc[4], abs(rev.iloc[4])),
    }


def compute_eps_surprise(income: pd.DataFrame) -> dict[str, float]:
    """EPS vs estimate — NaN if estimate not available (common with free data)."""
    if "eps" not in income.columns or len(income) == 0:
        return {"eps_surprise_pct": float("nan")}
    eps = income["eps"].iloc[0]
    est = income.get("eps_estimate", pd.Series([float("nan")])).iloc[0]
    return {"eps_surprise_pct": _safe_ratio(eps - est, abs(est))}


def compute_gross_margin_trend(income: pd.DataFrame) -> dict[str, float]:
    if len(income) < 2 or "gross_profit" not in income.columns or "revenue" not in income.columns:
        return {"gross_margin": float("nan"), "gross_margin_delta_qoq": float("nan")}
    margin = income["gross_profit"] / income["revenue"].replace(0, float("nan"))
    return {
        "gross_margin": float(margin.iloc[0]),
        "gross_margin_delta_qoq": float(margin.iloc[0] - margin.iloc[1]),
    }


def compute_fcf_yield(cashflow: pd.DataFrame, market_cap: float) -> dict[str, float]:
    if len(cashflow) == 0 or not market_cap or market_cap <= 0:
        return {"fcf_yield": float("nan")}
    if "free_cash_flow" not in cashflow.columns:
        return {"fcf_yield": float("nan")}
    fcf = cashflow["free_cash_flow"].iloc[0]
    return {"fcf_yield": _safe_ratio(fcf, market_cap)}


def compute_debt_equity(balance: pd.DataFrame) -> dict[str, float]:
    if len(balance) < 2:
        return {"debt_equity_ratio": float("nan"), "debt_equity_delta_qoq": float("nan")}
    debt = balance.get("long_term_debt", pd.Series([float("nan")] * len(balance)))
    equity = balance.get("total_equity", balance.get("common_equity", pd.Series([float("nan")] * len(balance))))
    de = debt / equity.replace(0, float("nan"))
    return {
        "debt_equity_ratio": float(de.iloc[0]),
        "debt_equity_delta_qoq": float(de.iloc[0] - de.iloc[1]),
    }


def compute_roe(income: pd.DataFrame, balance: pd.DataFrame) -> dict[str, float]:
    if len(income) < 2 or len(balance) < 2:
        return {"roe": float("nan"), "roe_delta_qoq": float("nan")}
    equity = balance.get("total_equity", balance.get("common_equity", pd.Series([float("nan")] * len(balance))))
    if "net_income" not in income.columns:
        return {"roe": float("nan"), "roe_delta_qoq": float("nan")}
    roe = income["net_income"] / equity.replace(0, float("nan"))
    return {
        "roe": float(roe.iloc[0]),
        "roe_delta_qoq": float(roe.iloc[0] - roe.iloc[1]) if len(roe) >= 2 else float("nan"),
    }


def compute_all_kpi_features(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    market_cap: float,
) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(compute_revenue_growth(income))
    features.update(compute_eps_surprise(income))
    features.update(compute_gross_margin_trend(income))
    features.update(compute_fcf_yield(cashflow, market_cap))
    features.update(compute_debt_equity(balance))
    features.update(compute_roe(income, balance))
    return features
