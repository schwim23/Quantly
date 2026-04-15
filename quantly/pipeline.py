"""Daily pipeline orchestrator.

Sequence
--------
1. Screen universe → ~300–500 liquid US equities
2. Build feature matrix (technical, sentiment, catalyst, fundamental)
3. Score with LightGBM → top 20 candidates ranked by conviction
4. Filter through fundamentals quality gate
5. Attach entry/stop/max-hold metadata
6. Persist ranked picks to shadow book DB
7. Run exit checks on open positions (stop-out, max-hold breach)
8. Record end-of-day NAV snapshot

Designed to run pre-market (~7 am ET) via the APScheduler job in the web app.
Can also be triggered manually: ``python -m quantly scan``.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

from quantly.config import get_config
from quantly.data.cache import Cache
from quantly.data.fundamentals import get_fundamentals_summary, passes_fundamentals_gate
from quantly.data.prices import compute_atr, get_current_price, get_latest_snapshot, get_ohlcv
from quantly.data.universe import screen_universe
from quantly.features.pipeline import build_feature_matrix
from quantly.models.predict import score_candidates
from quantly.models.train import load_model
from quantly.risk.sizing import compute_position_size, max_concurrent_positions_reached
from quantly.risk.stops import compute_initial_stop, compute_max_hold_date
from quantly.shadow_book.book import ShadowBook

logger = logging.getLogger(__name__)

# Number of ML-ranked candidates to pass through to the fundamentals gate
_CANDIDATE_POOL = 20


# ── Main entry point ──────────────────────────────────────────────────────────

def run_pipeline(
    tickers: Optional[list[str]] = None,
    as_of: Optional[date] = None,
    book: Optional[ShadowBook] = None,
    cache: Optional[Cache] = None,
) -> list[dict]:
    """Run the full signal pipeline and return today's ranked picks.

    Args:
        tickers: Override universe with a specific list (for debugging / tests).
        as_of: Signal date (default: today).
        book: ShadowBook instance (default: production DB).
        cache: Cache instance (default: production cache).

    Returns:
        List of pick dicts, sorted by conviction descending.  Each dict has:
            ticker, conviction, top_drivers, shap_drivers,
            entry_price, stop_price, max_hold_date,
            position_size (dict from compute_position_size)
    """
    cfg = get_config()
    today = as_of or date.today()
    cache = cache or Cache()
    book = book or ShadowBook()

    # ── 1. Universe screen ────────────────────────────────────────────────────
    if tickers:
        candidates = tickers
        logger.info("Using %d override tickers", len(candidates))
    else:
        logger.info("Screening universe…")
        snapshot = get_latest_snapshot([], cache)  # Tiingo batch endpoint
        candidates = screen_universe(snapshot, cache)
        logger.info("Universe screened: %d candidates", len(candidates))

    if not candidates:
        logger.warning("No candidates after screen — aborting pipeline")
        return []

    # ── 2. Feature matrix ─────────────────────────────────────────────────────
    logger.info("Building feature matrix for %d tickers…", len(candidates))
    features_df = build_feature_matrix(candidates, cache, as_of_date=today)

    if features_df.empty:
        logger.warning("Feature matrix empty — aborting pipeline")
        return []

    # ── 3. ML scoring ─────────────────────────────────────────────────────────
    model = load_model()
    if model is None:
        logger.warning("No trained model found — run 'python -m quantly train' first")
        return []

    logger.info("Scoring %d candidates…", len(features_df))
    scores_df = score_candidates(features_df, model=model)

    if scores_df.empty:
        return []

    top_candidates = scores_df.head(_CANDIDATE_POOL)

    # ── 4. Fundamentals gate ──────────────────────────────────────────────────
    logger.info("Applying fundamentals gate to top %d…", len(top_candidates))
    passed_tickers = []
    for ticker in top_candidates.index:
        try:
            summary = get_fundamentals_summary(ticker, cache)
            if passes_fundamentals_gate(summary):
                passed_tickers.append(ticker)
            else:
                logger.debug("%s: failed fundamentals gate", ticker)
        except Exception as exc:
            logger.warning("%s: fundamentals gate error: %s — passing through", ticker, exc)
            passed_tickers.append(ticker)  # fail open so we don't lose all picks on API outage

    if not passed_tickers:
        logger.warning("All candidates failed fundamentals gate")
        return []

    logger.info("%d tickers passed fundamentals gate", len(passed_tickers))

    # ── 5. Build pick metadata ────────────────────────────────────────────────
    open_count = len(book.get_open_positions())
    picks: list[dict] = []

    for ticker in passed_tickers:
        score_row = scores_df.loc[ticker]
        conviction = float(score_row["conviction"])
        top_drivers = list(score_row["top_drivers"])
        shap_drivers = [
            {"feature": f, "value": round(v, 4)}
            for f, v in (score_row.get("shap_values") or {}).items()
            if f in top_drivers
        ]

        # Current price and ATR for stop calculation
        entry_price = get_current_price(ticker, cache)
        if not entry_price:
            logger.debug("%s: no current price — skipping", ticker)
            continue

        atr = _get_atr(ticker, today, cache)
        stop_price = compute_initial_stop(entry_price, atr) if atr else entry_price * 0.95

        max_hold = compute_max_hold_date(today)

        position_size = compute_position_size(
            entry_price=entry_price,
            stop_price=stop_price,
            portfolio_value=cfg.portfolio_value,
        )

        picks.append({
            "ticker": ticker,
            "conviction": conviction,
            "top_drivers": top_drivers,
            "shap_drivers": shap_drivers,
            "entry_price": round(entry_price, 2),
            "stop_price": round(stop_price, 4),
            "max_hold_date": max_hold,
            "position_size": position_size,
        })

    # ── 6. Save picks to DB ───────────────────────────────────────────────────
    if picks:
        book.save_picks(picks, signal_date=today)
        logger.info("Saved %d picks for %s", len(picks), today)

    # ── 7. Daily exit checks on open positions ────────────────────────────────
    _run_exit_checks(book, cache, today)

    # ── 8. NAV snapshot ───────────────────────────────────────────────────────
    _record_nav_snapshot(book, cache, today)

    return sorted(picks, key=lambda p: p["conviction"], reverse=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_atr(ticker: str, as_of: date, cache: Cache) -> Optional[float]:
    """Fetch ATR(14) for a ticker. Returns None on failure."""
    from datetime import timedelta

    start = as_of - timedelta(days=60)
    try:
        df = get_ohlcv(ticker, start, as_of, cache)
        if df.empty:
            return None
        atr_series = compute_atr(df)
        return float(atr_series.iloc[-1]) if not atr_series.empty else None
    except Exception as exc:
        logger.debug("%s: ATR computation failed: %s", ticker, exc)
        return None


def _run_exit_checks(book: ShadowBook, cache: Cache, today: date) -> None:
    """Check all open positions for stop-outs or max-hold breaches."""
    open_positions = book.get_open_positions()
    if not open_positions:
        return

    tickers = [p["ticker"] for p in open_positions]
    prices: dict[str, float] = {}
    for ticker in tickers:
        price = get_current_price(ticker, cache)
        if price:
            prices[ticker] = price

    to_exit = book.check_exits(prices, as_of=today)
    for pos in to_exit:
        price = prices.get(pos["ticker"], pos["entry_price"])
        book.close_position(pos["id"], today, price)
        logger.info(
            "Auto-closed %s (id=%d) at %.2f — reason: %s",
            pos["ticker"], pos["id"], price, pos["exit_reason"],
        )

    # Ratchet trailing stops for positions that weren't exited
    exited_ids = {p["id"] for p in to_exit}
    for pos in open_positions:
        if pos["id"] in exited_ids:
            continue
        ticker = pos["ticker"]
        price = prices.get(ticker)
        if price is None:
            continue
        atr = _get_atr(ticker, today, cache)
        if atr:
            new_stop = book.ratchet_stop(pos["id"], price, atr)
            if new_stop > pos["stop_price"]:
                logger.info("%s: trailing stop ratcheted %.4f → %.4f", ticker,
                            pos["stop_price"], new_stop)


def _record_nav_snapshot(book: ShadowBook, cache: Cache, today: date) -> None:
    """Compute current portfolio value and record a NAV snapshot."""
    cfg = get_config()
    open_positions = book.get_open_positions()

    positions_value = 0.0
    for pos in open_positions:
        price = get_current_price(pos["ticker"], cache) or pos["entry_price"]
        positions_value += price * pos["shares"]

    # Cash is approximated: starting capital minus cost basis of open positions
    cost_basis = sum(p["entry_price"] * p["shares"] for p in open_positions)
    cash = max(0.0, cfg.portfolio_value - cost_basis)
    portfolio_value = cash + positions_value

    book.record_nav(today, portfolio_value, cash, positions_value)
    logger.info(
        "NAV snapshot: portfolio=%.2f  cash=%.2f  positions=%.2f",
        portfolio_value, cash, positions_value,
    )
