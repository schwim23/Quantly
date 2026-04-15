"""Technical / price-based feature computation.

All indicators implemented directly with pandas/numpy — no external TA library.
Every function takes an OHLCV DataFrame (indexed by date, sorted ascending)
and returns a scalar value representing the most recent reading.

Implements:
  - RSI(14)
  - MACD signal line crossover (binary + magnitude)
  - Price momentum: 5d, 10d, 20d returns
  - Volume spike ratio vs 20-day average
  - Distance from 52-week high/low (normalised)
  - ATR(14) relative ratio (already in prices.py; used here for features)
  - Bollinger Band %B position
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(df: pd.DataFrame, period: int = 14) -> float:
    """Return the latest RSI value (0–100). NaN-safe; returns 50.0 if insufficient data."""
    close = df["close"]
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    last_gain = gain.iloc[-1]
    last_loss = loss.iloc[-1]
    if last_loss == 0 and last_gain == 0:
        return 50.0  # flat price — no movement
    if last_loss == 0:
        return 100.0  # only gains, no losses → RSI at ceiling
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    val = rsi_series.iloc[-1]
    return float(val) if not np.isnan(val) else 50.0


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, float]:
    """Return MACD line, signal line, histogram, and crossover flag.

    Returns dict with:
        macd_line       - MACD line value
        signal_line     - signal line value
        histogram       - macd_line - signal_line
        crossover       - +1 if bullish crossover today, -1 bearish, 0 neutral
    """
    close = df["close"]
    if len(close) < slow + signal:
        return {"macd_line": 0.0, "signal_line": 0.0, "histogram": 0.0, "crossover": 0}

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    crossover = 0
    if len(histogram) >= 2:
        if histogram.iloc[-2] < 0 and histogram.iloc[-1] >= 0:
            crossover = 1   # bullish crossover
        elif histogram.iloc[-2] > 0 and histogram.iloc[-1] <= 0:
            crossover = -1  # bearish crossover

    return {
        "macd_line": round(float(macd_line.iloc[-1]), 4),
        "signal_line": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
        "crossover": crossover,
    }


def momentum(df: pd.DataFrame) -> dict[str, float]:
    """Return price momentum over 5, 10, and 20 trading days.

    Returns fractions (e.g. 0.05 = +5%). Returns 0.0 if insufficient data.
    """
    close = df["close"]

    def _ret(n: int) -> float:
        if len(close) < n + 1:
            return 0.0
        prev = close.iloc[-(n + 1)]
        if prev == 0:
            return 0.0
        return float((close.iloc[-1] - prev) / prev)

    return {
        "momentum_5d": round(_ret(5), 4),
        "momentum_10d": round(_ret(10), 4),
        "momentum_20d": round(_ret(20), 4),
    }


def volume_spike(df: pd.DataFrame, window: int = 20) -> float:
    """Return today's volume relative to *window*-day average.

    >1.5 = elevated; >3.0 = spike. Returns 1.0 if insufficient data.
    """
    vol = df["volume"]
    if len(vol) < window + 1:
        return 1.0
    avg = vol.iloc[-(window + 1):-1].mean()
    if avg == 0:
        return 1.0
    return round(float(vol.iloc[-1] / avg), 3)


def distance_from_52w(df: pd.DataFrame) -> dict[str, float]:
    """Return distance from 52-week high and low as fractions.

    Returns dict with:
        dist_from_52w_high  - (high - close) / high  (0 = at high)
        dist_from_52w_low   - (close - low) / low    (>0 = above low)
    """
    close = df["close"]
    if len(close) < 5:
        return {"dist_from_52w_high": 0.0, "dist_from_52w_low": 0.0}

    window = min(252, len(close))
    period = close.iloc[-window:]
    high_52w = period.max()
    low_52w = period.min()
    latest = close.iloc[-1]

    dist_high = float((high_52w - latest) / high_52w) if high_52w else 0.0
    dist_low = float((latest - low_52w) / low_52w) if low_52w else 0.0

    return {
        "dist_from_52w_high": round(dist_high, 4),
        "dist_from_52w_low": round(dist_low, 4),
    }


def bollinger_pct_b(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> float:
    """Return Bollinger Band %B: position of close within the bands.

    0.0 = at lower band, 0.5 = at midline, 1.0 = at upper band.
    Values outside [0, 1] are possible (outside the bands).
    Returns 0.5 if insufficient data.
    """
    close = df["close"]
    if len(close) < period:
        return 0.5
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std

    latest_close = close.iloc[-1]
    latest_upper = upper.iloc[-1]
    latest_lower = lower.iloc[-1]
    band_width = latest_upper - latest_lower

    if band_width == 0 or np.isnan(band_width):
        return 0.5
    pct_b = (latest_close - latest_lower) / band_width
    return round(float(pct_b), 4)


def compute_all_technical(df: pd.DataFrame) -> dict[str, float]:
    """Compute all technical features for a price DataFrame and return as flat dict."""
    if df.empty:
        return {
            "rsi_14": 50.0,
            "macd_crossover": 0,
            "macd_histogram": 0.0,
            "momentum_5d": 0.0,
            "momentum_10d": 0.0,
            "momentum_20d": 0.0,
            "volume_spike_ratio": 1.0,
            "dist_from_52w_high": 0.0,
            "dist_from_52w_low": 0.0,
            "bollinger_pct_b": 0.5,
        }

    macd_vals = macd(df)
    mom = momentum(df)
    dist = distance_from_52w(df)

    return {
        "rsi_14": rsi(df),
        "macd_crossover": macd_vals["crossover"],
        "macd_histogram": macd_vals["histogram"],
        "momentum_5d": mom["momentum_5d"],
        "momentum_10d": mom["momentum_10d"],
        "momentum_20d": mom["momentum_20d"],
        "volume_spike_ratio": volume_spike(df),
        "dist_from_52w_high": dist["dist_from_52w_high"],
        "dist_from_52w_low": dist["dist_from_52w_low"],
        "bollinger_pct_b": bollinger_pct_b(df),
    }
