"""
Shared signal computation and voting logic for the Crypto Trading Bot.

This module contains all indicator computation, signal generation (A, B, C),
the voting system, and ATR-based risk management. It is imported by both
backtest.py and live_signal.py to avoid code duplication.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np

# =============================================================================
# Strategy Constants
# =============================================================================

EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ZSCORE_PERIOD = 20
ZSCORE_THRESHOLD = 2.0
BB_PERIOD = 20
BB_STD = 2.0
VOLUME_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5
ATR_TP_MULT = 2.5

# =============================================================================
# Indicator Computation
# =============================================================================


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on a DataFrame with OHLCV data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume.

    Returns:
        DataFrame with indicator columns appended. NaN values appear during
        the warm-up period for each indicator.
    """
    df = df.copy()

    # --- EMA (Trend) ---
    df["EMA_9"] = ta.ema(df["close"], length=EMA_FAST)
    df["EMA_21"] = ta.ema(df["close"], length=EMA_SLOW)

    # --- RSI ---
    df["RSI_14"] = ta.rsi(df["close"], length=RSI_PERIOD)

    # --- Z-Score (Mean Reversion) ---
    df["SMA_20"] = ta.sma(df["close"], length=ZSCORE_PERIOD)
    df["STDEV_20"] = ta.stdev(df["close"], length=ZSCORE_PERIOD)
    df["Z_SCORE"] = (df["close"] - df["SMA_20"]) / df["STDEV_20"].replace(0, np.nan)

    # --- Bollinger Bands ---
    bb = ta.bbands(df["close"], length=BB_PERIOD, std=BB_STD)
    df["BB_UPPER"] = bb[f"BBU_{BB_PERIOD}_{BB_STD}"]
    df["BB_LOWER"] = bb[f"BBL_{BB_PERIOD}_{BB_STD}"]
    df["BB_MIDDLE"] = bb[f"BBM_{BB_PERIOD}_{BB_STD}"]

    # --- Volume SMA ---
    df["VOLUME_SMA_20"] = ta.sma(df["volume"], length=VOLUME_PERIOD)

    # --- ATR ---
    df["ATR_14"] = ta.atr(
        df["high"], df["low"], df["close"], length=ATR_PERIOD
    )

    return df


# =============================================================================
# Individual Signal Functions
# =============================================================================


def signal_a_trend(
    ema9: float,
    ema21: float,
    ema9_prev: float,
    ema21_prev: float,
    rsi: float,
) -> int:
    """Signal A: Trend Following (EMA crossover + RSI filter).

    LONG:  EMA(9) crosses ABOVE EMA(21) and RSI(14) > 50.
    SHORT: EMA(9) crosses BELOW EMA(21) and RSI(14) < 50.
    Neutral otherwise.

    Returns:
        +1 for LONG, -1 for SHORT, 0 for neutral.
    """
    if any(pd.isna(x) for x in [ema9, ema21, ema9_prev, ema21_prev, rsi]):
        return 0

    ema_cross_up = (ema9_prev <= ema21_prev) and (ema9 > ema21)
    ema_cross_down = (ema9_prev >= ema21_prev) and (ema9 < ema21)

    if ema_cross_up and rsi > 50:
        return 1
    elif ema_cross_down and rsi < 50:
        return -1
    return 0


def signal_b_mean_reversion(z_score: float) -> int:
    """Signal B: Mean Reversion (Z-Score).

    LONG:  Z-Score < -2.0 (price too low, expect bounce up).
    SHORT: Z-Score > +2.0 (price too high, expect drop down).
    Neutral otherwise.

    Returns:
        +1 for LONG, -1 for SHORT, 0 for neutral.
    """
    if pd.isna(z_score):
        return 0

    if z_score < -ZSCORE_THRESHOLD:
        return 1
    elif z_score > ZSCORE_THRESHOLD:
        return -1
    return 0


def signal_c_volume_breakout(
    close: float,
    volume: float,
    volume_sma: float,
    bb_upper: float,
    bb_lower: float,
) -> int:
    """Signal C: Volume Breakout (Volume + Bollinger Bands).

    LONG:  Volume > 1.5x average AND price > upper Bollinger band.
    SHORT: Volume > 1.5x average AND price < lower Bollinger band.
    Neutral otherwise.

    Returns:
        +1 for LONG, -1 for SHORT, 0 for neutral.
    """
    if any(
        pd.isna(x) for x in [close, volume, volume_sma, bb_upper, bb_lower]
    ):
        return 0

    volume_spike = volume > (VOLUME_MULTIPLIER * volume_sma)

    if volume_spike and close > bb_upper:
        return 1
    elif volume_spike and close < bb_lower:
        return -1
    return 0


# =============================================================================
# Voting System
# =============================================================================


def voting_system(
    sig_a: int, sig_b: int, sig_c: int
) -> tuple[str, int]:
    """Combine three signals via majority voting.

    Each signal contributes +1 (LONG), -1 (SHORT), or 0 (neutral).
    Total >= +2  →  LONG
    Total <= -2  →  SHORT
    Otherwise    →  HOLD

    Returns:
        Tuple of (action: 'LONG' | 'SHORT' | 'HOLD', total_score: int).
    """
    total = sig_a + sig_b + sig_c

    if total >= 2:
        return ("LONG", total)
    elif total <= -2:
        return ("SHORT", total)
    return ("HOLD", total)


# =============================================================================
# Risk Management (ATR-based)
# =============================================================================


def calculate_risk(
    entry_price: float, atr: float, action: str
) -> tuple[float, float]:
    """Calculate stop loss and take profit levels based on ATR.

    LONG:
        SL = entry_price - (1.5 × ATR)
        TP = entry_price + (2.5 × ATR)

    SHORT:
        SL = entry_price + (1.5 × ATR)
        TP = entry_price - (2.5 × ATR)

    Args:
        entry_price: Current close price (entry).
        atr: ATR(14) value at entry.
        action: 'LONG' or 'SHORT'.

    Returns:
        Tuple of (stop_loss, take_profit).
    """
    if action == "LONG":
        stop_loss = entry_price - (ATR_STOP_MULT * atr)
        take_profit = entry_price + (ATR_TP_MULT * atr)
    else:  # SHORT
        stop_loss = entry_price + (ATR_STOP_MULT * atr)
        take_profit = entry_price - (ATR_TP_MULT * atr)

    return (stop_loss, take_profit)
