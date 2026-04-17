"""Financial KPI data via yfinance (free, no API key required).

yfinance exposes quarterly income statements, balance sheets, and
cash flow statements directly from Yahoo Finance.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from quantly.data.cache import cached

logger = logging.getLogger(__name__)


def _ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol)


@cached("fundamentals_income")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_income_statement(ticker: str) -> pd.DataFrame:
    """Quarterly income statement from yfinance.

    Returns:
        DataFrame with columns: revenue, gross_profit, operating_income,
        net_income, eps, eps_estimate, eps_surprise_pct
        Index: fiscal period end dates (most recent first)
    """
    t = _ticker(ticker)
    df = t.quarterly_income_stmt

    if df is None or df.empty:
        logger.warning("No income statement for %s", ticker)
        return pd.DataFrame()

    # yfinance returns rows=metrics, cols=dates — transpose to rows=dates
    df = df.T.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=False)

    # Normalise column names to snake_case
    rename = {
        "Total Revenue": "revenue",
        "Gross Profit": "gross_profit",
        "Operating Income": "operating_income",
        "Net Income": "net_income",
        "Basic EPS": "eps",
        "Diluted EPS": "eps_diluted",
        "Operating Expense": "operating_expense",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # EPS estimate / surprise not available from yfinance — fill with NaN
    if "eps" in df.columns:
        df["eps_estimate"] = float("nan")
        df["eps_surprise_pct"] = float("nan")

    return df


@cached("fundamentals_balance")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_balance_sheet(ticker: str) -> pd.DataFrame:
    """Quarterly balance sheet from yfinance.

    Returns:
        DataFrame with columns: total_assets, total_liabilities, total_equity,
        long_term_debt, cash_and_equivalents
        Index: fiscal period end dates (most recent first)
    """
    t = _ticker(ticker)
    df = t.quarterly_balance_sheet

    if df is None or df.empty:
        logger.warning("No balance sheet for %s", ticker)
        return pd.DataFrame()

    df = df.T.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=False)

    rename = {
        "Total Assets": "total_assets",
        "Total Liabilities Net Minority Interest": "total_liabilities",
        "Stockholders Equity": "total_equity",
        "Long Term Debt": "long_term_debt",
        "Cash And Cash Equivalents": "cash_and_equivalents",
        "Common Stock Equity": "common_equity",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


@cached("fundamentals_cashflow")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_cash_flow(ticker: str) -> pd.DataFrame:
    """Quarterly cash flow statement from yfinance.

    Returns:
        DataFrame with columns: operating_cashflow, capital_expenditures, free_cash_flow
        Index: fiscal period end dates (most recent first)
    """
    t = _ticker(ticker)
    df = t.quarterly_cash_flow

    if df is None or df.empty:
        logger.warning("No cash flow for %s", ticker)
        return pd.DataFrame()

    df = df.T.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=False)

    rename = {
        "Operating Cash Flow": "operating_cashflow",
        "Capital Expenditure": "capital_expenditures",
        "Free Cash Flow": "free_cash_flow",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Compute FCF if not provided directly
    if "free_cash_flow" not in df.columns:
        if "operating_cashflow" in df.columns and "capital_expenditures" in df.columns:
            df["free_cash_flow"] = df["operating_cashflow"] + df["capital_expenditures"]

    return df


@cached("fundamentals_info")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_company_info(ticker: str) -> dict:
    """Company metadata from yfinance: market cap, sector, shares outstanding, etc.

    Returns:
        Dict with keys: market_cap, shares_outstanding, sector, industry,
        short_ratio, beta, forward_pe, trailing_pe
    """
    info = _ticker(ticker).info
    return {
        "market_cap": info.get("marketCap"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "short_ratio": info.get("shortRatio"),
        "beta": info.get("beta"),
        "forward_pe": info.get("forwardPE"),
        "trailing_pe": info.get("trailingPE"),
        "earnings_date": info.get("earningsTimestamp"),
    }
