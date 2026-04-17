"""Paper trading tracker — records picks to SQLite and measures outcomes."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta

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
    scan_date = Column(String, nullable=False)
    signal_score = Column(Float)
    claude_conviction = Column(Integer)
    entry_price = Column(Float)
    thesis = Column(String)
    top_features = Column(String)
    biggest_risk = Column(String)
    outcome_date = Column(String)
    outcome_price = Column(Float)
    outperformance_vs_spy = Column(Float)
    label = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


def _engine():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{config.DB_PATH}")


def init_db() -> None:
    Base.metadata.create_all(_engine())


def record_picks(picks: list[dict], scan_date: date) -> None:
    """Save a batch of Claude-approved picks to the paper trading DB."""
    from quantly.data.prices import get_ohlcv

    engine = _engine()
    init_db()

    with Session(engine) as session:
        for pick in picks:
            ticker = pick["ticker"]
            try:
                ohlcv = get_ohlcv(ticker, scan_date, scan_date)
                entry_price = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else None
            except Exception:
                entry_price = None

            session.add(PaperTrade(
                ticker=ticker,
                scan_date=scan_date.isoformat(),
                signal_score=pick.get("ensemble_prob"),
                claude_conviction=pick.get("conviction"),
                entry_price=entry_price,
                thesis=pick.get("thesis"),
                top_features=json.dumps(pick.get("key_signals", [])),
                biggest_risk=pick.get("biggest_risk"),
            ))
        session.commit()
    logger.info("Recorded %d picks for %s", len(picks), scan_date)


def update_outcomes(horizon_days: int = config.PREDICTION_HORIZON_DAYS) -> None:
    """Fetch actual price outcomes for matured trades and write results."""
    from quantly.data.prices import get_ohlcv, get_spy_returns

    engine = _engine()
    cutoff = (date.today() - timedelta(days=horizon_days)).isoformat()

    with Session(engine) as session:
        pending = session.execute(
            text("SELECT * FROM paper_trades WHERE label IS NULL AND scan_date <= :c"),
            {"c": cutoff},
        ).fetchall()

        for trade in pending:
            scan = date.fromisoformat(trade.scan_date)
            try:
                ohlcv = get_ohlcv(trade.ticker, scan, date.today())
                trading = ohlcv[ohlcv.index > pd.Timestamp(scan)]
                if len(trading) < horizon_days:
                    continue
                exit_price = float(trading["close"].iloc[horizon_days - 1])
                if not trade.entry_price:
                    continue
                stock_ret = (exit_price - trade.entry_price) / trade.entry_price

                spy_rets = get_spy_returns(scan, date.today())
                spy_cumret = float((1 + spy_rets).cumprod().iloc[min(horizon_days - 1, len(spy_rets) - 1)] - 1)

                outperformance = stock_ret - spy_cumret
                label = 1 if outperformance > config.OUTPERFORMANCE_THRESHOLD else 0 if outperformance < -config.UNDERPERFORMANCE_THRESHOLD else None

                session.execute(
                    text("""UPDATE paper_trades SET outcome_date=:od, outcome_price=:op,
                        outperformance_vs_spy=:out, label=:lbl WHERE id=:id"""),
                    {"od": date.today().isoformat(), "op": exit_price,
                     "out": outperformance, "lbl": label, "id": trade.id},
                )
            except Exception as e:
                logger.debug("Outcome update failed for %s: %s", trade.ticker, e)

        session.commit()


def get_performance_report() -> dict:
    engine = _engine()
    try:
        df = pd.read_sql("SELECT * FROM paper_trades WHERE label IS NOT NULL", engine)
    except Exception:
        return {"message": "No completed trades yet."}

    if df.empty:
        return {"message": "No completed trades yet."}

    return {
        "total_trades": len(df),
        "win_rate": float(df["label"].mean()),
        "avg_outperformance": float(df["outperformance_vs_spy"].mean()),
        "best_pick": df.loc[df["outperformance_vs_spy"].idxmax(), "ticker"],
        "worst_pick": df.loc[df["outperformance_vs_spy"].idxmin(), "ticker"],
        "rolling_4w_win_rate": float(df.tail(40)["label"].mean()),
    }
