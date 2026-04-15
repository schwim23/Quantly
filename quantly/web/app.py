"""FastAPI web application — picks approval, shadow book, performance dashboard.

Routes
------
GET  /                         Today's Picks page
POST /picks/{id}/approve       HTMX: approve a pick → open position
POST /picks/{id}/reject        HTMX: reject a pick → remove from list
GET  /book                     Shadow Book (open positions)
POST /book/{id}/exit           HTMX: mark a position as manually exited
GET  /performance              Performance vs SPY page
POST /pipeline/run             Admin: trigger pipeline manually
GET  /health                   Liveness probe
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from quantly.config import get_config
from quantly.data.cache import Cache
from quantly.data.prices import get_current_price, get_ohlcv, compute_atr
from quantly.shadow_book import db as shadow_db
from quantly.shadow_book.book import ShadowBook
from quantly.shadow_book.performance import benchmark_vs_spy, portfolio_summary

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"

@asynccontextmanager
async def lifespan(application: FastAPI):
    shadow_db.init_db()
    _start_scheduler()
    yield


app = FastAPI(title="Quantly", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
templates = Jinja2Templates(directory=_TEMPLATES_DIR)


def _get_book() -> ShadowBook:
    return ShadowBook()


def _get_cache() -> Cache:
    return Cache()


# ── Scheduler (APScheduler) ───────────────────────────────────────────────────

def _start_scheduler() -> None:
    """Start the background scheduler for daily pipeline execution."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        from quantly.pipeline import run_pipeline

        scheduler = BackgroundScheduler(timezone="America/New_York")
        scheduler.add_job(
            run_pipeline,
            trigger=CronTrigger(hour=7, minute=0),
            id="daily_pipeline",
            replace_existing=True,
        )
        scheduler.start()
        logger.info("Scheduler started — daily pipeline at 07:00 ET")
    except Exception as exc:
        logger.warning("Scheduler failed to start: %s", exc)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ── Picks ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def picks_page(request: Request) -> Response:
    book = _get_book()
    today = date.today()

    # Show today's picks first, fall back to most recent signal date
    picks = book.get_picks_for_date(today)
    signal_date = today

    if not picks:
        # Try the most recent date in the DB
        all_pending = book.get_pending_picks()
        if all_pending:
            picks = all_pending
            signal_date_str = picks[0].get("signal_date", str(today))
            signal_date = date.fromisoformat(signal_date_str)

    return templates.TemplateResponse(
        request,
        "picks.html",
        {
            "picks": picks,
            "signal_date": signal_date,
            "page": "picks",
        },
    )


@app.post("/picks/{pick_id}/approve", response_class=HTMLResponse)
async def approve_pick(request: Request, pick_id: int) -> Response:
    """Approve a pick: mark approved and open a shadow position."""
    book = _get_book()
    cache = _get_cache()

    # Load the pick
    all_pending = book.get_pending_picks()
    pick = next((p for p in all_pending if p["id"] == pick_id), None)
    if pick is None:
        raise HTTPException(status_code=404, detail="Pick not found or already actioned")

    book.approve_pick(pick_id)

    # Open position if we have price data
    try:
        entry_price = pick.get("entry_price") or get_current_price(pick["ticker"], cache)
        stop_price = pick.get("stop_price", 0.0)
        signal_date = date.fromisoformat(pick["signal_date"])

        if entry_price and stop_price:
            # Determine share count from position sizing
            cfg = get_config()
            from quantly.risk.sizing import compute_position_size
            sizing = compute_position_size(
                entry_price=entry_price,
                stop_price=stop_price,
                portfolio_value=cfg.portfolio_value,
            )
            book.open_position(
                ticker=pick["ticker"],
                entry_date=signal_date,
                entry_price=entry_price,
                shares=sizing["shares"],
                stop_price=stop_price,
                notes=f"Approved pick — conviction {pick.get('conviction', 0):.0f}",
            )
            logger.info("Opened position: %s @ %.2f (%d shares)", pick["ticker"],
                        entry_price, sizing["shares"])
    except Exception as exc:
        logger.warning("Could not open position for %s: %s", pick["ticker"], exc)

    # Return HTMX replacement: a "approved" badge replacing the action buttons
    return HTMLResponse(
        f'<span class="px-3 py-1 rounded-full text-sm font-medium bg-green-100 '
        f'text-green-800">Approved</span>'
    )


@app.post("/picks/{pick_id}/reject", response_class=HTMLResponse)
async def reject_pick(request: Request, pick_id: int) -> Response:
    """Reject a pick: mark rejected."""
    book = _get_book()
    book.reject_pick(pick_id)
    return HTMLResponse(
        f'<span class="px-3 py-1 rounded-full text-sm font-medium bg-red-100 '
        f'text-red-800">Rejected</span>'
    )


# ── Shadow Book ───────────────────────────────────────────────────────────────

@app.get("/book", response_class=HTMLResponse)
async def book_page(request: Request) -> Response:
    book = _get_book()
    cache = _get_cache()

    open_positions = book.get_open_positions()
    current_prices: dict[str, float] = {}
    for pos in open_positions:
        price = get_current_price(pos["ticker"], cache)
        if price:
            current_prices[pos["ticker"]] = price

    pnl_df = book.unrealized_pnl(current_prices)
    positions_with_pnl = pnl_df.to_dict("records") if not pnl_df.empty else []

    return templates.TemplateResponse(
        request,
        "book.html",
        {
            "positions": positions_with_pnl,
            "page": "book",
        },
    )


@app.post("/book/{position_id}/exit", response_class=HTMLResponse)
async def exit_position(
    request: Request,
    position_id: int,
    exit_price: float = Form(...),
) -> Response:
    """Mark a position as manually exited at the given price."""
    book = _get_book()
    pos = book.get_position(position_id)
    if pos is None or pos["status"] != "open":
        raise HTTPException(status_code=404, detail="Open position not found")

    book.close_position(position_id, date.today(), exit_price)

    pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
    colour = "green" if pnl_pct >= 0 else "red"
    sign = "+" if pnl_pct >= 0 else ""
    return HTMLResponse(
        f'<td colspan="8" class="px-4 py-2 text-center text-sm text-{colour}-600 font-medium">'
        f'Exited at ${exit_price:.2f} ({sign}{pnl_pct:.1f}%)'
        f'</td>'
    )


# ── Performance ───────────────────────────────────────────────────────────────

@app.get("/performance", response_class=HTMLResponse)
async def performance_page(request: Request) -> Response:
    book = _get_book()
    cache = _get_cache()

    trade_history = book.get_trade_history()
    nav_history = book.get_nav_history()

    # Fetch SPY price history for benchmark
    spy_prices = None
    try:
        if not nav_history.empty:
            from datetime import timedelta
            start = nav_history["nav_date"].min() - timedelta(days=5)
            spy_df = get_ohlcv("SPY", start, date.today(), cache)
            if not spy_df.empty:
                spy_prices = spy_df["close"]
                spy_prices.index = spy_prices.index.map(
                    lambda x: x.date() if hasattr(x, "date") else x
                )
    except Exception as exc:
        logger.warning("SPY fetch failed: %s", exc)

    summary = portfolio_summary(trade_history, nav_history, spy_prices=spy_prices)

    # Build chart data for benchmark_vs_spy
    chart_data: list[dict] = []
    if spy_prices is not None and not nav_history.empty:
        comparison = benchmark_vs_spy(nav_history, spy_prices)
        if not comparison.empty:
            chart_data = comparison.to_dict("records")

    # Closed trades table
    closed_trades = trade_history.to_dict("records") if not trade_history.empty else []

    return templates.TemplateResponse(
        request,
        "performance.html",
        {
            "summary": summary,
            "chart_data": chart_data,
            "closed_trades": closed_trades,
            "page": "performance",
        },
    )


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.post("/pipeline/run", response_class=HTMLResponse)
async def run_pipeline_now(request: Request) -> Response:
    """Manually trigger the signal pipeline (admin action)."""
    try:
        from quantly.pipeline import run_pipeline
        picks = run_pipeline()
        return HTMLResponse(
            f'<p class="text-green-700 text-sm font-medium">'
            f'Pipeline complete — {len(picks)} picks generated.</p>'
        )
    except Exception as exc:
        logger.exception("Pipeline run failed")
        return HTMLResponse(
            f'<p class="text-red-700 text-sm font-medium">Pipeline failed: {exc}</p>',
            status_code=500,
        )
