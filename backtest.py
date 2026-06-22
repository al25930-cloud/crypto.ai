"""
Backtest engine for trading strategies.

Runs a strategy on historical OHLCV data and computes performance metrics.

Key rules (from spec):
- Entry at close price when threshold conditions are met
- Exit at SL or TP (conservative: SL-first if both hit in same candle)
- One trade at a time
- Minimum trade duration: 45 minutes (loss applied to equity but trade marked invalid)
- Maximum trade duration: 48 hours (close at current price)
- Cooldown: 4 candles after any exit
- Trading fees: 0.1% per side
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

import config
from indicators import compute_all_conditions

logger = logging.getLogger(__name__)

# Timeframe to timedelta for duration checks
_TIMEFRAME_DELTA = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
}


def backtest_strategy(
    df: pd.DataFrame,
    strategy: dict,
    conditions_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Run a backtest for a single strategy on a DataFrame.

    Args:
        df: DataFrame with OHLCV data and all indicator columns already computed.
            Must have been processed by compute_all_indicators() and dropna().
        strategy: Dict with keys: conditions, threshold, sl, rr, direction.
        conditions_df: Pre-computed boolean condition DataFrame. If None, computed
            from df using strategy['conditions']. Pre-computing and passing this
            in saves time when testing many strategies on the same data.

    Returns:
        Dict with keys:
            total_trades, valid_trades, win_rate, total_rr, total_period_days,
            trading_days, rr_per_day, max_drawdown, avg_trades_per_day, trades,
            invalid_trades, timeout_trades, exit_sl_count, exit_tp_count,
            exit_timeout_count, total_fees
    """
    conditions = strategy["conditions"]
    threshold = strategy["threshold"]
    sl_atr_mult = strategy.get("sl_atr_mult", strategy.get("sl", 1.5))  # Fallback for backward compat
    rr_ratio = strategy["rr"]
    direction = strategy["direction"]

    # Compute conditions if not pre-computed
    if conditions_df is None:
        conditions_df = compute_all_conditions(df, conditions)

    # Pre-compute the condition satisfaction ratio as a vectorized Series
    # This is the fraction of conditions that are True for each candle
    if len(conditions) == 0:
        satisfaction = pd.Series(0.0, index=df.index)
    else:
        satisfaction = conditions_df[conditions].sum(axis=1) / len(conditions)

    # Pre-extract numpy arrays for speed in the inner loop
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df["timestamp"].values
    satisfies = satisfaction.values
    atr_values = df["atr_14"].values

    min_duration = timedelta(minutes=config.MIN_TRADE_DURATION_MINUTES)
    max_duration = timedelta(hours=config.MAX_TRADE_DURATION_HOURS)
    fee_pct = config.TRADING_FEE_PCT / 100.0  # Convert 0.1% to 0.001

    # --- Simulation state ---
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    entry_time: Optional[datetime] = None
    sl_price = 0.0
    tp_price = 0.0
    cooldown_remaining = 0

    trades = []
    equity = 1.0  # Normalized starting equity
    peak_equity = 1.0
    max_drawdown = 0.0
    equity_curve = [1.0]

    n = len(df)

    for i in range(n):
        ts = pd.Timestamp(timestamps[i]).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        if in_position:
            # --- Mark-to-market: track unrealized P&L for drawdown ---
            current_price = closes[i]
            if direction == "LONG":
                unrealized_pnl = (current_price - entry_price) / entry_price
            else:
                unrealized_pnl = (entry_price - current_price) / entry_price
            mtm_equity = equity * (1.0 + unrealized_pnl - (2 * fee_pct))
            peak_equity = max(peak_equity, mtm_equity)
            drawdown = (peak_equity - mtm_equity) / peak_equity if peak_equity > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

            # --- Check exit conditions ---
            duration = ts - entry_time
            high_i = highs[i]
            low_i = lows[i]

            exit_type = None
            exit_price = 0.0

            # Check if duration exceeds max (48h timeout)
            if duration >= max_duration:
                exit_type = "timeout"
                exit_price = closes[i]

            # Check SL/TP (conservative: SL first)
            elif direction == "LONG":
                sl_hit = low_i <= sl_price
                tp_hit = high_i >= tp_price
                if sl_hit:
                    exit_type = "sl"
                    exit_price = sl_price
                elif tp_hit:
                    exit_type = "tp"
                    exit_price = tp_price
            else:  # SHORT
                sl_hit = high_i >= sl_price
                tp_hit = low_i <= tp_price
                if sl_hit:
                    exit_type = "sl"
                    exit_price = sl_price
                elif tp_hit:
                    exit_type = "tp"
                    exit_price = tp_price

            if exit_type is not None:
                # --- Close the trade ---
                # Calculate P&L with fees
                if direction == "LONG":
                    gross_pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    gross_pnl_pct = (entry_price - exit_price) / entry_price

                # Deduct fees (entry + exit)
                net_pnl_pct = gross_pnl_pct - (2 * fee_pct)

                # Calculate RR for this trade using ATR-based risk
                risk_amount = atr_values[entry_idx] * sl_atr_mult
                if risk_amount > 0:
                    trade_rr = ((exit_price - entry_price) / risk_amount) if direction == "LONG" \
                        else ((entry_price - exit_price) / risk_amount)
                else:
                    trade_rr = 0.0

                # Apply realized P&L to equity
                equity *= (1.0 + net_pnl_pct)

                # Check validity (minimum duration)
                is_valid = duration >= min_duration
                invalid_reason = None if is_valid else "too_short"

                trade_record = {
                    "entry_time": entry_time.isoformat(),
                    "exit_time": ts.isoformat(),
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "direction": direction,
                    "result": exit_type,
                    "rr": float(trade_rr),
                    "gross_pnl_pct": float(gross_pnl_pct),
                    "net_pnl_pct": float(net_pnl_pct),
                    "duration_minutes": float(duration.total_seconds() / 60),
                    "valid": is_valid,
                    "invalid_reason": invalid_reason,
                }
                trades.append(trade_record)

                # Reset position
                in_position = False
                cooldown_remaining = config.COOLDOWN_CANDLES

            # Track mark-to-market equity on the curve
            equity_curve.append(mtm_equity if in_position else equity)

        else:
            # --- Check entry conditions ---
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            else:
                # Entry signal: satisfaction ratio >= threshold
                if satisfies[i] >= threshold:
                    in_position = True
                    entry_price = closes[i]
                    entry_time = ts
                    entry_idx = i  # Store index for ATR lookup on exit
                    atr_at_entry = atr_values[i]
                    sl_distance = atr_at_entry * sl_atr_mult

                    if direction == "LONG":
                        sl_price = entry_price - sl_distance
                        tp_price = entry_price + sl_distance * rr_ratio
                    else:  # SHORT
                        sl_price = entry_price + sl_distance
                        tp_price = entry_price - sl_distance * rr_ratio

            equity_curve.append(equity)

    # If still in position at end of data, close at last price
    if in_position:
        last_ts = pd.Timestamp(timestamps[-1]).to_pydatetime()
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        duration = last_ts - entry_time
        exit_price = closes[-1]

        if direction == "LONG":
            gross_pnl_pct = (exit_price - entry_price) / entry_price
        else:
            gross_pnl_pct = (entry_price - exit_price) / entry_price

        net_pnl_pct = gross_pnl_pct - (2 * fee_pct)

        risk_amount = atr_values[entry_idx] * sl_atr_mult
        if risk_amount > 0:
            trade_rr = ((exit_price - entry_price) / risk_amount) if direction == "LONG" \
                else ((entry_price - exit_price) / risk_amount)
        else:
            trade_rr = 0.0

        equity *= (1.0 + net_pnl_pct)
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)

        is_valid = duration >= min_duration
        trades.append({
            "entry_time": entry_time.isoformat(),
            "exit_time": last_ts.isoformat(),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "direction": direction,
            "result": "timeout",
            "rr": float(trade_rr),
            "gross_pnl_pct": float(gross_pnl_pct),
            "net_pnl_pct": float(net_pnl_pct),
            "duration_minutes": float(duration.total_seconds() / 60),
            "valid": is_valid,
            "invalid_reason": None if is_valid else "too_short",
        })

    # --- Compute metrics ---
    valid_trades = [t for t in trades if t["valid"]]
    invalid_trades = [t for t in trades if not t["valid"]]

    total_trades = len(trades)
    num_valid = len(valid_trades)
    num_invalid = len(invalid_trades)
    num_timeout = sum(1 for t in trades if t["result"] == "timeout")
    exit_sl_count = sum(1 for t in trades if t["result"] == "sl")
    exit_tp_count = sum(1 for t in trades if t["result"] == "tp")
    exit_timeout_count = sum(1 for t in trades if t["result"] == "timeout")
    total_fees = total_trades * 2 * fee_pct  # Approximate

    # Win rate: based on valid trades only
    if num_valid > 0:
        wins = sum(1 for t in valid_trades if t["rr"] > 0)
        win_rate = wins / num_valid
    else:
        win_rate = 0.0

    # RR/day: sum of valid trade RRs / total calendar days in the backtest period
    total_rr = sum(t["rr"] for t in valid_trades)

    # Total calendar days in the backtest period (first candle to last candle)
    if n > 1:
        first_ts = pd.Timestamp(timestamps[0]).to_pydatetime()
        last_ts_end = pd.Timestamp(timestamps[-1]).to_pydatetime()
        total_period_days = max((last_ts_end - first_ts).days, 1)
    else:
        total_period_days = 1  # Avoid division by zero

    # Count unique trading days (days with at least one trade entry)
    if trades:
        trade_dates = set()
        for t in trades:
            trade_date = datetime.fromisoformat(t["entry_time"]).date()
            trade_dates.add(trade_date)
        trading_days = max(len(trade_dates), 1)
    else:
        trading_days = 1  # Avoid division by zero

    rr_per_day = total_rr / total_period_days
    avg_trades_per_day = num_valid / total_period_days

    return {
        "total_trades": total_trades,
        "valid_trades": num_valid,
        "invalid_trades": num_invalid,
        "timeout_trades": num_timeout,
        "win_rate": float(win_rate),
        "total_rr": float(total_rr),
        "total_period_days": total_period_days,
        "trading_days": trading_days,
        "rr_per_day": float(rr_per_day),
        "max_drawdown": float(max_drawdown),
        "avg_trades_per_day": float(avg_trades_per_day),
        "exit_sl_count": exit_sl_count,
        "exit_tp_count": exit_tp_count,
        "exit_timeout_count": exit_timeout_count,
        "total_fees": float(total_fees),
        "equity_curve": equity_curve,
        "trades": trades,
    }


def prepare_data(df: pd.DataFrame, condition_keys: Optional[list[str]] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for backtesting: compute indicators, drop NaN, compute conditions.

    This is a convenience function that chains the typical data preparation steps.

    Args:
        df: Raw OHLCV DataFrame.
        condition_keys: If provided, compute only these conditions. Otherwise None
            (conditions computed later per-strategy).

    Returns:
        Tuple of (clean_df, conditions_df). conditions_df is None if condition_keys not provided.
    """
    from indicators import compute_all_indicators

    df = compute_all_indicators(df.copy())
    df = df.dropna().reset_index(drop=True)

    conditions_df = None
    if condition_keys:
        conditions_df = compute_all_conditions(df, condition_keys)

    logger.info(f"Data prepared: {len(df)} candles after warmup drop.")
    return df, conditions_df
