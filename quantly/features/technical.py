"""Technical analysis features from OHLCV data. No external API required."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(prices: pd.Series, window: int = 14) -> float:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    last_gain = float(gain.iloc[-1])
    last_loss = float(loss.iloc[-1])
    if np.isnan(last_gain) or np.isnan(last_loss):
        return float("nan")
    if last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0
    return float(100 - (100 / (1 + last_gain / last_loss)))


def compute_macd(prices: pd.Series) -> dict[str, float]:
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    crossover = 0
    if len(macd_line) >= 2:
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            crossover = 1
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            crossover = -1

    return {
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(signal_line.iloc[-1]),
        "macd_histogram": float(histogram.iloc[-1]),
        "macd_crossover": float(crossover),
    }


def compute_volume_spike(volumes: pd.Series, window: int = 20) -> float:
    """Today's volume divided by the prior N-day average."""
    if len(volumes) < window + 1:
        return float("nan")
    avg = volumes.iloc[-(window + 1):-1].mean()
    return float(volumes.iloc[-1] / avg) if avg > 0 else float("nan")


def compute_momentum(prices: pd.Series) -> dict[str, float]:
    return {
        "mom_5d": float(prices.pct_change(5).iloc[-1]),
        "mom_10d": float(prices.pct_change(10).iloc[-1]),
        "mom_20d": float(prices.pct_change(20).iloc[-1]),
    }


def compute_atr(ohlcv: pd.DataFrame, window: int = 14) -> float:
    high, low, prev_close = ohlcv["high"], ohlcv["low"], ohlcv["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return float(tr.rolling(window).mean().iloc[-1])


def compute_bollinger_position(prices: pd.Series, window: int = 20) -> float:
    """0 = at lower band, 0.5 = middle, 1 = at upper band."""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    band_width = float((upper - lower).iloc[-1])
    if band_width == 0:
        return 0.5
    return float((prices.iloc[-1] - lower.iloc[-1]) / band_width)


def compute_distance_from_52w(prices: pd.Series) -> dict[str, float]:
    year_prices = prices.tail(252)
    high_52w = float(year_prices.max())
    low_52w = float(year_prices.min())
    current = float(prices.iloc[-1])
    return {
        "pct_from_52w_high": (current - high_52w) / high_52w if high_52w else float("nan"),
        "pct_from_52w_low": (current - low_52w) / low_52w if low_52w else float("nan"),
    }


def compute_all_technical_features(ohlcv: pd.DataFrame) -> dict[str, float]:
    """Compute all technical features from an OHLCV DataFrame."""
    prices = ohlcv["close"]
    features: dict[str, float] = {}
    features["rsi_14"] = compute_rsi(prices)
    features.update(compute_macd(prices))
    features["volume_spike_20d"] = compute_volume_spike(ohlcv["volume"])
    features.update(compute_momentum(prices))
    features["atr_14"] = compute_atr(ohlcv)
    features["bollinger_position"] = compute_bollinger_position(prices)
    features.update(compute_distance_from_52w(prices))
    return features
