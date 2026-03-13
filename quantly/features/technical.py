"""Technical analysis features computed from OHLCV price data.

All features are backward-looking — no lookahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(prices: pd.Series, window: int = 14) -> float:
    """Relative Strength Index at the most recent date."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean().iloc[-1]
    avg_loss = loss.rolling(window).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd_signal(prices: pd.Series) -> dict[str, float]:
    """MACD line, signal line, and histogram at most recent date.

    Returns:
        Dict with keys: macd, signal, histogram, crossover (1=bullish, -1=bearish, 0=none)
    """
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    crossover = 0
    if len(macd_line) >= 2:
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            crossover = 1   # bullish crossover
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            crossover = -1  # bearish crossover
    return {
        "macd": macd_line.iloc[-1],
        "signal": signal_line.iloc[-1],
        "histogram": histogram.iloc[-1],
        "crossover": crossover,
    }


def compute_volume_spike(volumes: pd.Series, window: int = 20) -> float:
    """Ratio of today's volume to the 20-day average. > 1.5 = significant spike."""
    avg = volumes.rolling(window).mean().iloc[-2]  # -2 to avoid including today
    return volumes.iloc[-1] / avg if avg > 0 else 1.0


def compute_momentum(prices: pd.Series) -> dict[str, float]:
    """Return momentum over 5, 10, and 20 trading days."""
    return {
        "mom_5d": prices.pct_change(5).iloc[-1],
        "mom_10d": prices.pct_change(10).iloc[-1],
        "mom_20d": prices.pct_change(20).iloc[-1],
    }


def compute_atr(ohlcv: pd.DataFrame, window: int = 14) -> float:
    """Average True Range — measure of volatility."""
    high = ohlcv["high"]
    low = ohlcv["low"]
    prev_close = ohlcv["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean().iloc[-1]


def compute_bollinger_position(prices: pd.Series, window: int = 20) -> float:
    """Position within Bollinger Bands: 0=lower band, 0.5=middle, 1=upper band."""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    band_width = upper.iloc[-1] - lower.iloc[-1]
    if band_width == 0:
        return 0.5
    return (prices.iloc[-1] - lower.iloc[-1]) / band_width


def compute_distance_from_52w(prices: pd.Series) -> dict[str, float]:
    """Distance from 52-week high and low (as % of current price)."""
    one_year = prices.tail(252)
    high_52w = one_year.max()
    low_52w = one_year.min()
    current = prices.iloc[-1]
    return {
        "pct_from_52w_high": (current - high_52w) / high_52w,
        "pct_from_52w_low": (current - low_52w) / low_52w,
    }


def compute_all_technical_features(ohlcv: pd.DataFrame) -> dict[str, float]:
    """Compute all technical features from an OHLCV DataFrame.

    Args:
        ohlcv: DataFrame with columns: open, high, low, close, volume

    Returns:
        Dict of feature_name -> float value
    """
    prices = ohlcv["close"]
    features: dict[str, float] = {}
    features["rsi_14"] = compute_rsi(prices)
    features.update(compute_macd_signal(prices))
    features["volume_spike_20d"] = compute_volume_spike(ohlcv["volume"])
    features.update(compute_momentum(prices))
    features["atr_14"] = compute_atr(ohlcv)
    features["bollinger_position"] = compute_bollinger_position(prices)
    features.update(compute_distance_from_52w(prices))
    return features
