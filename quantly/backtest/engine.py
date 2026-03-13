"""Walk-forward backtest orchestrator.

Simulates the full pipeline (features → ML scoring → label comparison)
over historical date ranges without lookahead bias.

Walk-forward protocol:
  - Train on 2-year rolling window
  - Validate on next quarter
  - 5-day purge gap between train and validation sets
  - Report metrics per quarter, per year, and overall
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Iterator

import pandas as pd

from quantly.models.evaluate import compute_summary_metrics

logger = logging.getLogger(__name__)


def date_range_quarterly(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield (quarter_start, quarter_end) tuples over a date range."""
    current = start
    while current < end:
        quarter_end = current + timedelta(days=90)
        yield current, min(quarter_end, end)
        current = quarter_end


def run_backtest(
    years: int = 3,
    universe: str = "sp500",
    top_n: int = 20,
) -> dict:
    """Run a full walk-forward backtest.

    Args:
        years: Number of years of history to backtest over
        universe: Which stock universe to use ("all_us" | "sp500" | "nasdaq100")
        top_n: How many picks to simulate per scan date

    Returns:
        Dict with keys: overall_metrics, quarterly_metrics, annual_metrics,
        all_picks (DataFrame)
    """
    end_date = date.today()
    start_date = date(end_date.year - years, end_date.month, end_date.day)

    quarterly_results = []
    all_picks = []

    for q_start, q_end in date_range_quarterly(start_date, end_date):
        logger.info("Backtesting quarter %s to %s", q_start, q_end)
        # TODO: implement quarterly backtest loop
        # 1. Get universe for q_start
        # 2. Build features as of q_start (using data available at q_start)
        # 3. Load / train model on data prior to q_start
        # 4. Score candidates
        # 5. Compute labels (forward returns from q_start)
        # 6. Collect results

    if not all_picks:
        return {"error": "No backtest results generated"}

    all_picks_df = pd.concat(all_picks, ignore_index=True)
    overall = compute_summary_metrics(all_picks_df)

    return {
        "overall_metrics": overall,
        "quarterly_metrics": quarterly_results,
        "all_picks": all_picks_df,
    }
