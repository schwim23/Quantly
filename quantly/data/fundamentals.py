"""Fetch financial KPIs from Alpha Vantage, Bloomberg, or SEC EDGAR."""

from __future__ import annotations

import logging

import pandas as pd

from quantly.data.cache import cached

logger = logging.getLogger(__name__)


@cached("fundamentals_income")
def get_income_statement(ticker: str, source: str = "alpha_vantage") -> pd.DataFrame:
    """Quarterly income statement data.

    Returns:
        DataFrame with columns: fiscal_date_ending, revenue, gross_profit,
        operating_income, net_income, eps, eps_estimate, eps_surprise_pct
    """
    if source == "alpha_vantage":
        return _fetch_av_income(ticker)
    raise ValueError(f"Unknown source: {source}")


@cached("fundamentals_balance")
def get_balance_sheet(ticker: str, source: str = "alpha_vantage") -> pd.DataFrame:
    """Quarterly balance sheet data.

    Returns:
        DataFrame with columns: fiscal_date_ending, total_assets, total_liabilities,
        total_equity, long_term_debt, cash_and_equivalents
    """
    if source == "alpha_vantage":
        return _fetch_av_balance(ticker)
    raise ValueError(f"Unknown source: {source}")


@cached("fundamentals_cashflow")
def get_cash_flow(ticker: str, source: str = "alpha_vantage") -> pd.DataFrame:
    """Quarterly cash flow statement.

    Returns:
        DataFrame with columns: fiscal_date_ending, operating_cashflow,
        capital_expenditures, free_cash_flow
    """
    if source == "alpha_vantage":
        return _fetch_av_cashflow(ticker)
    raise ValueError(f"Unknown source: {source}")


def _fetch_av_income(ticker: str) -> pd.DataFrame:
    # TODO: implement Alpha Vantage INCOME_STATEMENT endpoint
    raise NotImplementedError


def _fetch_av_balance(ticker: str) -> pd.DataFrame:
    # TODO: implement Alpha Vantage BALANCE_SHEET endpoint
    raise NotImplementedError


def _fetch_av_cashflow(ticker: str) -> pd.DataFrame:
    # TODO: implement Alpha Vantage CASH_FLOW endpoint
    raise NotImplementedError
