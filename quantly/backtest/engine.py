"""Walk-forward backtest orchestrator."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Iterator

import pandas as pd

from quantly.models.evaluate import compute_summary_metrics

logger = logging.getLogger(__name__)


def quarterly_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    current = start
    while current < end:
        q_end = current + timedelta(days=90)
        yield current, min(q_end, end)
        current = q_end


def run_backtest(years: int = 3, universe: str = "sp500", top_n: int = 20) -> dict:
    """Walk-forward backtest over historical data.

    For each quarter in the date range:
      1. Build features as of quarter start (point-in-time, no lookahead)
      2. Score with model trained on prior data
      3. Compute actual forward returns as labels

    Returns:
        Dict with keys: overall_metrics, all_picks (DataFrame)
    """
    from quantly.data.universe import get_universe_tickers, screen_by_liquidity
    from quantly.features.pipeline import build_feature_matrix
    from quantly.models.labels import compute_labels
    from quantly.models.predict import score_candidates
    from quantly.models.train import load_models

    end_date = date.today()
    start_date = date(end_date.year - years, end_date.month, end_date.day)

    all_picks: list[pd.DataFrame] = []

    for q_start, q_end in quarterly_windows(start_date, end_date):
        logger.info("Backtesting %s → %s", q_start, q_end)
        try:
            tickers = get_universe_tickers(universe)
            screened = screen_by_liquidity(tickers)

            feature_matrix = build_feature_matrix(screened, q_start)
            scored = score_candidates(feature_matrix, top_n=top_n)

            pick_tickers = scored["ticker"].tolist()
            labels_df = compute_labels(pick_tickers, [q_start] * len(pick_tickers))

            result = scored.merge(labels_df[["ticker", "outperformance", "label"]], on="ticker")
            result["signal_date"] = q_start
            all_picks.append(result)
        except Exception as e:
            logger.warning("Backtest quarter %s failed: %s", q_start, e)

    if not all_picks:
        return {"error": "No backtest results — check data availability and model training"}

    all_picks_df = pd.concat(all_picks, ignore_index=True)
    return {
        "overall_metrics": compute_summary_metrics(all_picks_df),
        "all_picks": all_picks_df,
    }
