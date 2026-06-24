"""
Live signal generator.

Monitors the latest candle at each close, evaluates the best strategy's
conditions, and sends Discord alerts when signals are generated.

Handles:
- Entry signal detection
- Exit tracking (SL/TP/timeout)
- Cooldown after exits (COOLDOWN_CANDLES candles, recalculated on startup)
- Cooldown expiry notifications
- Missed signal recovery on startup
- State persistence via state.json

Usage:
    python live_signal.py --symbol BTC/USDT
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

import config
from backtest import backtest_strategy
from conditions import get_direction_for_condition, compute_shared_bonus
from data_fetcher import get_latest_candles
from discord_bot import (
    send_cooldown_alert,
    send_entry_signal,
    send_exit_alert,
    send_recovery_alert,
)
from indicators import compute_all_conditions, compute_all_indicators
from strategy import load_strategy

logger = logging.getLogger(__name__)

# Timeframe to seconds for sleep
_TIMEFRAME_SECONDS = {
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
}


class LiveState:
    """Manages the live bot state (position tracking, cooldown, last check)."""

    def __init__(self, state_file: Path = config.STATE_FILE):
        self.state_file = state_file
        self.in_position: bool = False
        self.entry_price: float = 0.0
        self.entry_time: Optional[str] = None
        self.sl: float = 0.0
        self.tp: float = 0.0
        self.direction: str = ""
        self.strategy_id: str = ""
        self.cooldown_remaining: int = 0
        self.cooldown_exit_time: Optional[str] = None  # ISO timestamp of the trade exit that started cooldown
        self.last_check_time: Optional[str] = None
        self._load()

    def _load(self) -> None:
        """Load state from file."""
        if not self.state_file.exists():
            logger.info("No existing state file. Starting fresh.")
            self.last_check_time = datetime.now(timezone.utc).isoformat()
            return
        try:
            with open(self.state_file) as f:
                data = json.load(f)
            self.in_position = data.get("in_position", False)
            self.entry_price = data.get("entry_price", 0.0)
            self.entry_time = data.get("entry_time")
            self.sl = data.get("sl", 0.0)
            self.tp = data.get("tp", 0.0)
            self.direction = data.get("direction", "")
            self.strategy_id = data.get("strategy_id", "")
            self.cooldown_remaining = data.get("cooldown_remaining", 0)
            self.cooldown_exit_time = data.get("cooldown_exit_time")
            self.last_check_time = data.get("last_check_time")
            logger.info(f"State loaded: in_position={self.in_position}, cooldown={self.cooldown_remaining}")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}. Starting fresh.")
            self.last_check_time = datetime.now(timezone.utc).isoformat()

    def save(self) -> None:
        """Save state to file."""
        data = {
            "symbol": config.SYMBOL,
            "in_position": self.in_position,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "sl": self.sl,
            "tp": self.tp,
            "direction": self.direction,
            "strategy_id": self.strategy_id,
            "cooldown_remaining": self.cooldown_remaining,
            "cooldown_exit_time": self.cooldown_exit_time,
            "last_check_time": self.last_check_time,
        }
        try:
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def enter_position(
        self, entry_price: float, sl: float, tp: float,
        direction: str, strategy_id: str,
        entry_time: Optional[datetime] = None,
    ) -> None:
        self.in_position = True
        self.entry_price = entry_price
        self.entry_time = (entry_time or datetime.now(timezone.utc)).isoformat()
        self.sl = sl
        self.tp = tp
        self.direction = direction
        self.strategy_id = strategy_id
        self.save()  # Persist immediately — never lose an open position

    def exit_position(self, exit_time: Optional[datetime] = None) -> None:
        self.in_position = False
        self.cooldown_remaining = config.COOLDOWN_CANDLES
        self.cooldown_exit_time = (exit_time or datetime.now(timezone.utc)).isoformat()
        self.entry_price = 0.0
        self.entry_time = None
        self.sl = 0.0
        self.tp = 0.0
        self.direction = ""
        self.strategy_id = ""
        self.save()  # Persist immediately — never lose exit/cooldown state


class LiveSignalGenerator:
    """Main live signal generator."""

    def __init__(self, symbol: str = config.SYMBOL, timeframe: str = config.TIMEFRAME):
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy: Optional[dict] = None
        self.state = LiveState()
        self.sleep_seconds = _TIMEFRAME_SECONDS.get(timeframe, 900)

    def run(self) -> None:
        """Start the live signal generator."""
        config.setup_logging()
        self._setup_file_logging()

        logger.info("=" * 60)
        logger.info("LIVE SIGNAL GENERATOR")
        logger.info("=" * 60)

        # Load strategy
        self.strategy = load_strategy()
        if self.strategy is None:
            logger.error("No best strategy found. Run training first.")
            return

        logger.info(f"Strategy loaded: {self.strategy['id']}")
        # Show direction mix instead of fixed direction
        conds = self.strategy['conditions']
        shared_conds_list = self.strategy.get('shared_conditions', [])
        shared_bonus_weight = self.strategy.get('shared_bonus_weight', 0.0)
        long_conds = sum(1 for c in conds if get_direction_for_condition(c) == 'LONG')
        short_conds = sum(1 for c in conds if get_direction_for_condition(c) == 'SHORT')
        logger.info(f"  Direction mix: LONG:{long_conds} SHORT:{short_conds} SHARED(bonus):{len(shared_conds_list)}")
        logger.info(f"  Core conditions: {len(conds)} | Shared bonus weight: {shared_bonus_weight:.4f}")
        logger.info(f"  Threshold: {self.strategy['threshold']}")
        logger.info(f"  SL_ATR_MULT: {self.strategy.get('sl_atr_mult', self.strategy.get('sl', 'N/A'))}, RR: {self.strategy['rr']}")
        logger.info(f"Symbol: {self.symbol} | Timeframe: {self.timeframe}")

        # Check for missed signals
        self._check_missed_signals()

        # Recalculate cooldown based on elapsed candles since exit
        self._recalculate_cooldown()

        # Main loop
        logger.info("Entering main loop. Waiting for candle closes...")
        while True:
            try:
                self._check_cycle()
                self.state.save()
                self._sleep_until_next_candle()
            except KeyboardInterrupt:
                logger.info("Shutdown requested. Saving state...")
                self.state.save()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)  # Wait a minute before retrying

    def _check_cycle(self) -> None:
        """Run one check cycle."""
        now = datetime.now(timezone.utc)
        self.state.last_check_time = now.isoformat()
        self.state.save()  # Save at start — ensures last_check_time and cooldown are persisted

        # Fetch latest candles
        df = get_latest_candles(self.symbol, self.timeframe, count=500)
        if df.empty:
            logger.warning("No data fetched. Skipping cycle.")
            return

        # Compute indicators
        df = compute_all_indicators(df.copy())
        df = df.dropna().reset_index(drop=True)

        if df.empty:
            logger.warning("[WARNING] Indicator NaN detected. Skipping signal check.")
            return

        # Get the latest candle
        latest = df.iloc[-1]
        current_price = float(latest["close"])
        current_high = float(latest["high"])
        current_low = float(latest["low"])
        current_time = pd.Timestamp(latest["timestamp"]).to_pydatetime()

        if self.state.in_position:
            self._check_exit(current_price, current_high, current_low, current_time)
        else:
            self._check_entry(df, latest, current_price, current_time)

    def _recalculate_cooldown(self) -> None:
        """On startup, recalculate cooldown based on elapsed candles since the exit that triggered it."""
        if not self.state.cooldown_exit_time:
            return

        try:
            exit_dt = datetime.fromisoformat(self.state.cooldown_exit_time)
            if exit_dt.tzinfo is None:
                exit_dt = exit_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            elapsed_seconds = (now - exit_dt).total_seconds()
            elapsed_candles = int(elapsed_seconds // self.sleep_seconds)

            if elapsed_candles >= config.COOLDOWN_CANDLES:
                logger.info(f"Cooldown already expired ({elapsed_candles} candles elapsed since exit).")
                self.state.cooldown_remaining = 0
                self.state.cooldown_exit_time = None
                send_cooldown_alert(self.symbol)
            else:
                old_remaining = self.state.cooldown_remaining
                self.state.cooldown_remaining = max(0, config.COOLDOWN_CANDLES - elapsed_candles)
                logger.info(
                    f"Cooldown adjusted: {elapsed_candles} candles elapsed since exit, "
                    f"{self.state.cooldown_remaining} remaining (was {old_remaining})."
                )
        except Exception as e:
            logger.warning(f"Failed to recalculate cooldown: {e}. Using saved value.")

    def _check_entry(self, df: pd.DataFrame, latest: pd.Series, price: float, ts: datetime) -> None:
        """Check for entry signal."""
        # Check cooldown
        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            logger.info(f"Cooldown: {self.state.cooldown_remaining} candles remaining.")
            if self.state.cooldown_remaining == 0:
                logger.info("Cooldown expired.")
                self.state.cooldown_exit_time = None
                send_cooldown_alert(self.symbol)
            return

        # Evaluate conditions
        strategy = self.strategy
        conditions = strategy["conditions"]  # Core conditions (LONG/SHORT only)
        shared_conds = strategy.get("shared_conditions", [])
        shared_bonus_weight = strategy.get("shared_bonus_weight", 0.0)

        # Compute conditions for both core and shared
        all_condition_keys = list(set(conditions + shared_conds))
        latest_df = df.iloc[[-1]]  # Single-row DataFrame
        cond_df = compute_all_conditions(latest_df, all_condition_keys)
        last_row = cond_df.iloc[0]

        # Core conditions only (LONG/SHORT) — used for base strength
        long_conds_list = [c for c in conditions if get_direction_for_condition(c) == 'LONG']
        short_conds_list = [c for c in conditions if get_direction_for_condition(c) == 'SHORT']
        long_true = int(last_row[long_conds_list].sum()) if long_conds_list else 0
        short_true = int(last_row[short_conds_list].sum()) if short_conds_list else 0
        long_strength = long_true / len(long_conds_list) if long_conds_list else 0
        short_strength = short_true / len(short_conds_list) if short_conds_list else 0

        # Compute shared bonus (directional filtering + dedup applied)
        if shared_conds and shared_bonus_weight > 0:
            long_bonus = compute_shared_bonus(last_row, shared_conds, shared_bonus_weight, "LONG")
            short_bonus = compute_shared_bonus(last_row, shared_conds, shared_bonus_weight, "SHORT")
        else:
            long_bonus = 0.0
            short_bonus = 0.0

        long_total = min(long_strength + long_bonus, 0.95)  # Clamp to 95%
        short_total = min(short_strength + short_bonus, 0.95)

        if long_total >= strategy["threshold"] and long_strength > short_strength * config.DIRECTION_RATIO:
            direction = "LONG"
        elif short_total >= strategy["threshold"] and short_strength > long_strength * config.DIRECTION_RATIO:
            direction = "SHORT"
        else:
            direction = None  # HOLD — ambiguous or insufficient strength

        if direction is None:
            logger.info(
                f"[HOLD] LONG:{long_strength:.0%}(+{long_bonus:.0%}) SHORT:{short_strength:.0%}(+{short_bonus:.0%}) "
                f"(threshold {strategy['threshold']:.0%}, ratio {config.DIRECTION_RATIO})"
            )
            return

        # Confidence: base strength + bonus for the chosen direction
        base_strength = long_strength if direction == "LONG" else short_strength
        bonus = long_bonus if direction == "LONG" else short_bonus
        core_met = long_true if direction == "LONG" else short_true
        core_total = len(long_conds_list) if direction == "LONG" else len(short_conds_list)
        confidence = min(base_strength + bonus, 0.95) * 100

        sl_atr_mult = strategy.get("sl_atr_mult", strategy.get("sl", 1.5))
        rr = strategy["rr"]
        atr_value = float(latest.get("atr_14", 0))
        sl_distance = atr_value * sl_atr_mult

        if direction == "LONG":
            sl_price = price - sl_distance
            tp_price = price + sl_distance * rr
        else:
            sl_price = price + sl_distance
            tp_price = price - sl_distance * rr

        # Enter position (use candle timestamp for consistency with backtest)
        self.state.enter_position(price, sl_price, tp_price, direction, strategy["id"], entry_time=ts)

        logger.info(
            f"Signal: {direction} at ${price:,.2f} | "
            f"SL ${sl_price:,.2f} | TP ${tp_price:,.2f} | "
            f"Strength LONG:{long_strength:.0%} SHORT:{short_strength:.0%} | "
            f"Confidence {confidence:.0f}% ({base_strength:.0%} base + {bonus:.0%} bonus)"
        )

        # Get strategy historical metrics
        rr_day = strategy.get("results", {}).get("rr_per_day", 0)
        wr = strategy.get("results", {}).get("win_rate", 0)

        send_entry_signal(
            symbol=self.symbol,
            direction=direction,
            entry_price=price,
            sl_price=sl_price,
            tp_price=tp_price,
            rr_ratio=rr,
            confidence=confidence,
            conditions_met=core_met,
            conditions_total=core_total,
            strategy_id=strategy["id"],
            strategy_rr_day=rr_day,
            strategy_win_rate=wr,
            base_confidence=base_strength * 100,
            bonus=bonus * 100,
        )

    def _check_exit(self, current_price: float, candle_high: float, candle_low: float, ts: datetime) -> None:
        """Check if SL/TP/timeout should trigger an exit.

        Uses candle high/low with conservative SL-first detection (matching backtest logic).
        """
        direction = self.state.direction
        entry_price = self.state.entry_price
        sl_price = self.state.sl
        tp_price = self.state.tp

        # Parse entry time
        entry_dt = datetime.fromisoformat(self.state.entry_time)
        if entry_dt.tzinfo is None:
            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
        duration = ts - entry_dt

        # Check timeout
        max_duration = timedelta(hours=config.MAX_TRADE_DURATION_HOURS)
        if duration >= max_duration:
            pnl_pct = self._calc_pnl(entry_price, current_price, direction)
            rr = self._calc_rr(entry_price, current_price, direction, self.state.sl)
            logger.info(f"Timeout exit at ${current_price:,.2f} after {duration}")
            send_exit_alert(self.symbol, direction, entry_price, current_price, "timeout", pnl_pct, rr, duration.total_seconds() / 60)
            self.state.exit_position(exit_time=ts)
            return

        # Check SL/TP using candle high/low (conservative: SL first)
        if direction == "LONG":
            sl_hit = candle_low <= sl_price
            tp_hit = candle_high >= tp_price
            if sl_hit:
                pnl_pct = self._calc_pnl(entry_price, sl_price, direction)
                logger.info(f"SL hit at ${sl_price:,.2f} (candle low: ${candle_low:,.2f})")
                send_exit_alert(self.symbol, direction, entry_price, sl_price, "sl", pnl_pct, -1.0, duration.total_seconds() / 60)
                self.state.exit_position(exit_time=ts)
                return
            if tp_hit:
                pnl_pct = self._calc_pnl(entry_price, tp_price, direction)
                rr = self.strategy["rr"]
                logger.info(f"TP hit at ${tp_price:,.2f} (candle high: ${candle_high:,.2f})")
                send_exit_alert(self.symbol, direction, entry_price, tp_price, "tp", pnl_pct, rr, duration.total_seconds() / 60)
                self.state.exit_position(exit_time=ts)
                return
        else:  # SHORT
            sl_hit = candle_high >= sl_price
            tp_hit = candle_low <= tp_price
            if sl_hit:
                pnl_pct = self._calc_pnl(entry_price, sl_price, direction)
                logger.info(f"SL hit at ${sl_price:,.2f} (candle high: ${candle_high:,.2f})")
                send_exit_alert(self.symbol, direction, entry_price, sl_price, "sl", pnl_pct, -1.0, duration.total_seconds() / 60)
                self.state.exit_position(exit_time=ts)
                return
            if tp_hit:
                pnl_pct = self._calc_pnl(entry_price, tp_price, direction)
                rr = self.strategy["rr"]
                logger.info(f"TP hit at ${tp_price:,.2f} (candle low: ${candle_low:,.2f})")
                send_exit_alert(self.symbol, direction, entry_price, tp_price, "tp", pnl_pct, rr, duration.total_seconds() / 60)
                self.state.exit_position(exit_time=ts)
                return

        # Still open
        hours = int(duration.total_seconds() // 3600)
        mins = int((duration.total_seconds() % 3600) // 60)
        logger.info(f"Position still open. Duration: {hours}h {mins}m. Price: ${current_price:,.2f}")

    def _check_missed_signals(self) -> None:
        """Check for signals that occurred while the bot was offline."""
        if self.state.last_check_time is None:
            return

        # If already in a position, scan for missed SL/TP/timeout instead
        if self.state.in_position:
            try:
                self._check_missed_exit()
            except Exception as e:
                logger.error(f"Error during missed exit scan: {e}. Continuing with normal exit monitoring.")
            # Don't update last_check_time yet — the entry scan below needs to see
            # candles from the original last_check through now. We'll update it after the scan.

        logger.info("Checking for missed signals...")
        last_check = datetime.fromisoformat(self.state.last_check_time)
        if last_check.tzinfo is None:
            last_check = last_check.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        offline_seconds = (now - last_check).total_seconds()
        candles_per_hour = 3600 / self.sleep_seconds
        # Skip recovery if offline for less than one candle
        if offline_seconds < self.sleep_seconds:
            logger.info(f"Offline for less than one candle ({self.timeframe}). No missed signal check needed.")
            return

        offline_hours = offline_seconds / 3600
        logger.info(f"Offline for {offline_hours:.1f} hours. Scanning missed candles...")

        # Fetch candles covering the offline period
        df = get_latest_candles(self.symbol, self.timeframe, count=max(500, int(offline_hours * candles_per_hour) + 100))
        if df.empty:
            return

        df = compute_all_indicators(df.copy())
        df = df.dropna().reset_index(drop=True)

        if df.empty:
            return

        # Filter to only the offline period
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        offline_df = df[df["timestamp"] > last_check].copy()

        if offline_df.empty:
            logger.info("No candles in the offline period.")
            return

        # Evaluate conditions
        strategy = self.strategy
        conditions = strategy["conditions"]  # Core conditions (LONG/SHORT only)
        shared_conds = strategy.get("shared_conditions", [])
        shared_bonus_weight = strategy.get("shared_bonus_weight", 0.0)
        all_condition_keys = list(set(conditions + shared_conds))
        cond_df = compute_all_conditions(offline_df, all_condition_keys)

        for i in range(len(offline_df)):
            row = cond_df.iloc[i]

            # Core conditions only (LONG/SHORT) — used for base strength
            miss_long = [c for c in conditions if get_direction_for_condition(c) == 'LONG']
            miss_short = [c for c in conditions if get_direction_for_condition(c) == 'SHORT']
            long_true = int(row[miss_long].sum()) if miss_long else 0
            short_true = int(row[miss_short].sum()) if miss_short else 0
            long_str = long_true / len(miss_long) if miss_long else 0
            short_str = short_true / len(miss_short) if miss_short else 0

            # Compute shared bonus
            if shared_conds and shared_bonus_weight > 0:
                long_bonus = compute_shared_bonus(row, shared_conds, shared_bonus_weight, "LONG")
                short_bonus = compute_shared_bonus(row, shared_conds, shared_bonus_weight, "SHORT")
            else:
                long_bonus = 0.0
                short_bonus = 0.0

            long_total = min(long_str + long_bonus, 0.95)
            short_total = min(short_str + short_bonus, 0.95)

            if long_total >= strategy["threshold"] and long_str > short_str * config.DIRECTION_RATIO:
                direction = "LONG"
            elif short_total >= strategy["threshold"] and short_str > long_str * config.DIRECTION_RATIO:
                direction = "SHORT"
            else:
                continue  # No clear direction, skip this missed signal

            ts = offline_df.iloc[i]["timestamp"]
            price = float(offline_df.iloc[i]["close"])

            sl_atr_mult = strategy.get("sl_atr_mult", strategy.get("sl", 1.5))
            rr = strategy["rr"]
            atr_value = float(offline_df.iloc[i].get("atr_14", 0))
            sl_distance = atr_value * sl_atr_mult

            if direction == "LONG":
                sl_price = price - sl_distance
                tp_price = price + sl_distance * rr
            else:
                sl_price = price + sl_distance
                tp_price = price - sl_distance * rr

            signal_time = pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
            logger.info(f"[RECOVERY] Missed signal: {direction} at ${price:,.2f} on {signal_time}")

            send_recovery_alert(
                symbol=self.symbol,
                direction=direction,
                entry_price=price,
                sl_price=sl_price,
                tp_price=tp_price,
                rr_ratio=rr,
                signal_time=signal_time,
                expired=True,
            )
            break  # Only report the first missed signal

        # Update last_check_time after scanning — prevents duplicate recovery on crash-restart
        # while still allowing the entry scan above to see the full offline window
        self.state.last_check_time = datetime.now(timezone.utc).isoformat()
        self.state.save()

    def _check_missed_exit(self) -> None:
        """Scan candles since last check for SL/TP/timeout hits that occurred while offline."""
        if not self.state.entry_time:
            return

        last_check = datetime.fromisoformat(self.state.last_check_time)
        if last_check.tzinfo is None:
            last_check = last_check.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        offline_hours = (now - last_check).total_seconds() / 3600

        entry_dt = datetime.fromisoformat(self.state.entry_time)
        if entry_dt.tzinfo is None:
            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
        max_duration = timedelta(hours=config.MAX_TRADE_DURATION_HOURS)

        logger.info(f"Resuming position ({self.state.direction} from {self.state.entry_time}). Scanning offline candles for missed exits...")

        # Fetch candles covering the offline period
        candles_per_hour = 3600 / self.sleep_seconds
        df = get_latest_candles(self.symbol, self.timeframe, count=max(500, int(offline_hours * candles_per_hour) + 100))
        if df.empty:
            logger.warning("No data for missed exit scan.")
            return

        df = compute_all_indicators(df.copy())
        df = df.dropna().reset_index(drop=True)
        if df.empty:
            return

        # Only scan candles after last check
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        offline_df = df[df["timestamp"] > last_check].copy()
        if offline_df.empty:
            logger.info("No new candles since last check.")
            return

        direction = self.state.direction
        entry_price = self.state.entry_price
        sl_price = self.state.sl
        tp_price = self.state.tp

        for i in range(len(offline_df)):
            candle = offline_df.iloc[i]
            candle_high = float(candle["high"])
            candle_low = float(candle["low"])
            candle_close = float(candle["close"])
            candle_ts = pd.Timestamp(candle["timestamp"]).to_pydatetime()

            # Skip candles that predate the entry (use < not <= to include the candle
            # that was open when we entered — entry_dt is clock time, candle_ts is candle open)
            if candle_ts < entry_dt:
                continue

            duration = candle_ts - entry_dt

            # Check timeout
            if duration >= max_duration:
                pnl_pct = self._calc_pnl(entry_price, candle_close, direction)
                rr = self._calc_rr(entry_price, candle_close, direction, sl_price)
                logger.info(f"[RECOVERY] Missed timeout exit at ${candle_close:,.2f} on {pd.Timestamp(candle_ts).strftime('%Y-%m-%d %H:%M')} after {duration}")
                send_exit_alert(self.symbol, direction, entry_price, candle_close, "timeout", pnl_pct, rr, duration.total_seconds() / 60)
                self.state.exit_position(exit_time=candle_ts)
                return

            # Check SL/TP (conservative: SL first)
            if direction == "LONG":
                if candle_low <= sl_price:
                    pnl_pct = self._calc_pnl(entry_price, sl_price, direction)
                    logger.info(f"[RECOVERY] Missed SL hit at ${sl_price:,.2f} on {pd.Timestamp(candle_ts).strftime('%Y-%m-%d %H:%M')}")
                    send_exit_alert(self.symbol, direction, entry_price, sl_price, "sl", pnl_pct, -1.0, duration.total_seconds() / 60)
                    self.state.exit_position(exit_time=candle_ts)
                    return
                if candle_high >= tp_price:
                    pnl_pct = self._calc_pnl(entry_price, tp_price, direction)
                    logger.info(f"[RECOVERY] Missed TP hit at ${tp_price:,.2f} on {pd.Timestamp(candle_ts).strftime('%Y-%m-%d %H:%M')}")
                    send_exit_alert(self.symbol, direction, entry_price, tp_price, "tp", pnl_pct, self.strategy["rr"], duration.total_seconds() / 60)
                    self.state.exit_position(exit_time=candle_ts)
                    return
            else:  # SHORT
                if candle_high >= sl_price:
                    pnl_pct = self._calc_pnl(entry_price, sl_price, direction)
                    logger.info(f"[RECOVERY] Missed SL hit at ${sl_price:,.2f} on {pd.Timestamp(candle_ts).strftime('%Y-%m-%d %H:%M')}")
                    send_exit_alert(self.symbol, direction, entry_price, sl_price, "sl", pnl_pct, -1.0, duration.total_seconds() / 60)
                    self.state.exit_position(exit_time=candle_ts)
                    return
                if candle_low <= tp_price:
                    pnl_pct = self._calc_pnl(entry_price, tp_price, direction)
                    logger.info(f"[RECOVERY] Missed TP hit at ${tp_price:,.2f} on {pd.Timestamp(candle_ts).strftime('%Y-%m-%d %H:%M')}")
                    send_exit_alert(self.symbol, direction, entry_price, tp_price, "tp", pnl_pct, self.strategy["rr"], duration.total_seconds() / 60)
                    self.state.exit_position(exit_time=candle_ts)
                    return

        # No SL/TP/timeout hit found during offline period — position still valid
        logger.info(f"No missed exits found. Position still open ({self.state.direction}).")

    def _calc_pnl(self, entry: float, exit_price: float, direction: str) -> float:
        if direction == "LONG":
            return (exit_price - entry) / entry - 2 * config.TRADING_FEE_PCT / 100
        return (entry - exit_price) / entry - 2 * config.TRADING_FEE_PCT / 100

    def _calc_rr(self, entry: float, exit_price: float, direction: str, sl_price: float) -> float:
        """Calculate RR for a trade using absolute SL price."""
        if sl_price and entry:
            risk = abs(entry - sl_price)
        else:
            return 0.0
        if risk == 0:
            return 0.0
        if direction == "LONG":
            return (exit_price - entry) / risk
        return (entry - exit_price) / risk

    def _sleep_until_next_candle(self) -> None:
        """Sleep until the next candle close."""
        now = datetime.now(timezone.utc)
        seconds_in_current = (now.minute * 60 + now.second) % self.sleep_seconds
        wait = self.sleep_seconds - seconds_in_current + 5  # +5s buffer
        if wait > 0:
            logger.debug(f"Sleeping {wait}s until next candle close.")
            time.sleep(wait)

    def _setup_file_logging(self) -> None:
        """Set up file logging for live mode."""
        log_file = config.LOG_DIR_LIVE / f"live_{datetime.now().strftime('%Y-%m-%d')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.DATE_FORMAT))
        logging.getLogger().addHandler(fh)
        logger.info(f"Live log: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Run the live signal generator.")
    parser.add_argument("--symbol", default=config.SYMBOL, help="Trading pair")
    parser.add_argument("--timeframe", default=config.TIMEFRAME, help="Candle timeframe")
    args = parser.parse_args()

    generator = LiveSignalGenerator(symbol=args.symbol, timeframe=args.timeframe)
    generator.run()


if __name__ == "__main__":
    main()
