"""Paper trading tracker.

Records scan picks to SQLite and tracks actual outcomes after the prediction horizon.
Provides weekly performance reports to close the feedback loop.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session

from quantly.config import config

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    scan_date = Column(String, nullable=False)          # ISO date string
    signal_score = Column(Float)                        # ML ensemble probability
    claude_conviction = Column(Integer)                 # Claude's 1-10 conviction score
    entry_price = Column(Float)
    thesis = Column(String)
    top_features = Column(String)                       # JSON string
    biggest_risk = Column(String)
    outcome_date = Column(String)                       # ISO date string, NULL if pending
    outcome_price = Column(Float)
    outperformance_vs_spy = Column(Float)
    label = Column(Integer)                             # 1=win, 0=loss, NULL=pending
    created_at = Column(DateTime, default=datetime.utcnow)


def get_engine():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{config.DB_PATH}")


def init_db() -> None:
    """Create tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def record_picks(picks: list[dict], scan_date: date) -> None:
    """Save a batch of picks from a scan to the paper trading DB.

    Args:
        picks: List of pick dicts from Claude analyst output
        scan_date: Date the scan was run
    """
    import json

    from quantly.data.prices import get_ohlcv

    engine = get_engine()
    init_db()

    with Session(engine) as session:
        for pick in picks:
            ticker = pick["ticker"]
            try:
                ohlcv = get_ohlcv(ticker, scan_date, scan_date)
                entry_price = ohlcv["close"].iloc[-1] if len(ohlcv) > 0 else None
            except Exception:
                entry_price = None

            trade = PaperTrade(
                ticker=ticker,
                scan_date=scan_date.isoformat(),
                signal_score=pick.get("ensemble_prob"),
                claude_conviction=pick.get("conviction"),
                entry_price=entry_price,
                thesis=pick.get("thesis"),
                top_features=json.dumps(pick.get("key_signals", [])),
                biggest_risk=pick.get("biggest_risk"),
            )
            session.add(trade)
        session.commit()
    logger.info("Recorded %d picks for %s", len(picks), scan_date)


def update_outcomes(horizon_days: int = 15) -> None:
    """Fetch actual price outcomes for matured trades and compute performance.

    Runs daily; updates any picks that are now past the prediction horizon.
    """
    from quantly.data.prices import get_ohlcv, get_spy_returns

    engine = get_engine()
    cutoff = date.today() - timedelta(days=horizon_days)

    with Session(engine) as session:
        pending = session.execute(
            text("SELECT * FROM paper_trades WHERE label IS NULL AND scan_date <= :cutoff"),
            {"cutoff": cutoff.isoformat()},
        ).fetchall()

        for trade in pending:
            ticker = trade.ticker
            scan_date = date.fromisoformat(trade.scan_date)
            try:
                ohlcv = get_ohlcv(ticker, scan_date, date.today())
                if len(ohlcv) < horizon_days:
                    continue
                exit_price = ohlcv["close"].iloc[horizon_days - 1]
                stock_ret = (exit_price - trade.entry_price) / trade.entry_price

                spy = get_spy_returns(scan_date, date.today())
                spy_ret = (1 + spy).cumprod().iloc[horizon_days - 2] - 1

                outperformance = stock_ret - spy_ret
                label = 1 if outperformance > 0.03 else 0 if outperformance < -0.02 else None

                session.execute(
                    text("""UPDATE paper_trades SET
                        outcome_date=:od, outcome_price=:op,
                        outperformance_vs_spy=:out, label=:lbl
                        WHERE id=:id"""),
                    {
                        "od": date.today().isoformat(),
                        "op": exit_price,
                        "out": outperformance,
                        "lbl": label,
                        "id": trade.id,
                    },
                )
            except Exception as e:
                logger.warning("Could not update outcome for %s: %s", ticker, e)

        session.commit()


def get_performance_report() -> dict:
    """Generate a performance report from all completed paper trades.

    Returns:
        Dict with keys: total_trades, win_rate, avg_outperformance,
        best_pick, worst_pick, rolling_4w_win_rate
    """
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM paper_trades WHERE label IS NOT NULL", engine)

    if df.empty:
        return {"message": "No completed trades yet."}

    return {
        "total_trades": len(df),
        "win_rate": df["label"].mean(),
        "avg_outperformance": df["outperformance_vs_spy"].mean(),
        "best_pick": df.loc[df["outperformance_vs_spy"].idxmax(), "ticker"],
        "worst_pick": df.loc[df["outperformance_vs_spy"].idxmin(), "ticker"],
        "rolling_4w_win_rate": df.tail(40)["label"].mean(),
    }
