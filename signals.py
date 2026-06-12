"""
Shared signal computation and voting logic for the Crypto Trading Bot.

This module contains all indicator computation, signal generation (A, B, C),
the voting system, and ATR-based risk management.  It is imported by both
backtest.py and live_signal.py to avoid code duplication.

All tunable constants are imported from config.py — the single source of truth.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np

from config import (
    EMA_FAST,
    EMA_SLOW,
    RSI_PERIOD,
    ZSCORE_PERIOD,
    ZSCORE_THRESHOLD,
    BB_PERIOD,
    BB_STD,
    VOLUME_PERIOD,
    VOLUME_MULTIPLIER,
    ATR_PERIOD,
    ATR_STOP_MULT,
    ATR_TP_MULT,
    VOTING_THRESHOLD,
    ema_fast_col,
    ema_slow_col,
    rsi_col,
    sma_col,
    stdev_col,
    atr_col,
    vol_sma_col,
)

# =============================================================================
# Indicator Computation
# =============================================================================


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on a DataFrame with OHLCV data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume.

    Returns:
        DataFrame with indicator columns appended.  NaN values appear during
        the warm-up period for each indicator.  Column names are derived from
        config.py constants (e.g. EMA_9, EMA_21) so they stay in sync.
    """
    df = df.copy()

    # --- EMA (Trend) ---
    df[ema_fast_col()] = ta.ema(df["close"], length=EMA_FAST)
    df[ema_slow_col()] = ta.ema(df["close"], length=EMA_SLOW)

    # --- RSI ---
    df[rsi_col()] = ta.rsi(df["close"], length=RSI_PERIOD)

    # --- Z-Score (Mean Reversion) ---
    df[sma_col()] = ta.sma(df["close"], length=ZSCORE_PERIOD)
    df[stdev_col()] = ta.stdev(df["close"], length=ZSCORE_PERIOD)
    df["Z_SCORE"] = (
        (df["close"] - df[sma_col()]) / df[stdev_col()].replace(0, np.nan)
    )

    # --- Bollinger Bands ---
    bb = ta.bbands(df["close"], length=BB_PERIOD, std=BB_STD)
    # pandas_ta column names vary by version; look up by prefix.
    bbu_col = next(c for c in bb.columns if c.startswith("BBU_"))
    bbl_col = next(c for c in bb.columns if c.startswith("BBL_"))
    bbm_col = next(c for c in bb.columns if c.startswith("BBM_"))
    df["BB_UPPER"] = bb[bbu_col]
    df["BB_LOWER"] = bb[bbl_col]
    df["BB_MIDDLE"] = bb[bbm_col]

    # --- Volume SMA ---
    df[vol_sma_col()] = ta.sma(df["volume"], length=VOLUME_PERIOD)

    # --- ATR ---
    df[atr_col()] = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)

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

    LONG:  EMA(fast) crosses ABOVE EMA(slow) and RSI > 50.
    SHORT: EMA(fast) crosses BELOW EMA(slow) and RSI < 50.
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

    LONG:  Z-Score < -threshold (price too low, expect bounce up).
    SHORT: Z-Score > +threshold (price too high, expect drop down).
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

    LONG:  Volume > multiplier×average AND price > upper Bollinger band.
    SHORT: Volume > multiplier×average AND price < lower Bollinger band.
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
    Total >=  VOTING_THRESHOLD  ->  LONG
    Total <= -VOTING_THRESHOLD  ->  SHORT
    Otherwise                   ->  HOLD

    Returns:
        Tuple of (action: 'LONG' | 'SHORT' | 'HOLD', total_score: int).
    """
    total = sig_a + sig_b + sig_c

    if total >= VOTING_THRESHOLD:
        return ("LONG", total)
    elif total <= -VOTING_THRESHOLD:
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
        SL = entry_price - (ATR_STOP_MULT × ATR)
        TP = entry_price + (ATR_TP_MULT × ATR)

    SHORT:
        SL = entry_price + (ATR_STOP_MULT × ATR)
        TP = entry_price - (ATR_TP_MULT × ATR)

    Args:
        entry_price: Current close price (entry).
        atr: ATR value at entry.
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
