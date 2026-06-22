"""
Live signal generator.

Monitors the latest candle at each close, evaluates the best strategy's
conditions, and sends Discord alerts when signals are generated.

Handles:
- Entry signal detection
- Exit tracking (SL/TP/timeout)
- Cooldown after exits (4 candles)
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
from conditions import get_direction_for_condition
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
    ) -> None:
        self.in_position = True
        self.entry_price = entry_price
        self.entry_time = datetime.now(timezone.utc).isoformat()
        self.sl = sl
        self.tp = tp
        self.direction = direction
        self.strategy_id = strategy_id

    def exit_position(self) -> None:
        self.in_position = False
        self.cooldown_remaining = config.COOLDOWN_CANDLES
        self.entry_price = 0.0
        self.entry_time = None
        self.sl = 0.0
        self.tp = 0.0
        self.direction = ""
        self.strategy_id = ""


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
        logger.info(f"  Direction: {self.strategy['direction']}")
        logger.info(f"  Conditions: {len(self.strategy['conditions'])}")
        logger.info(f"  Threshold: {self.strategy['threshold']}")
        logger.info(f"  SL_ATR_MULT: {self.strategy.get('sl_atr_mult', self.strategy.get('sl', 'N/A'))}, RR: {self.strategy['rr']}")
        logger.info(f"Symbol: {self.symbol} | Timeframe: {self.timeframe}")

        # Check for missed signals
        self._check_missed_signals()

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

    def _check_entry(self, df: pd.DataFrame, latest: pd.Series, price: float, ts: datetime) -> None:
        """Check for entry signal."""
        # Check cooldown
        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            logger.info(f"Cooldown: {self.state.cooldown_remaining} candles remaining.")
            if self.state.cooldown_remaining == 0:
                logger.info("Cooldown expired.")
                send_cooldown_alert(self.symbol)
            return

        # Evaluate conditions
        strategy = self.strategy
        conditions = strategy["conditions"]

        # Compute conditions for the latest candle
        cond_df = compute_all_conditions(df, conditions)
        last_row = cond_df.iloc[-1]
        conditions_met = int(last_row.sum())
        conditions_total = len(conditions)
        satisfaction = conditions_met / conditions_total if conditions_total > 0 else 0

        if satisfaction >= strategy["threshold"]:
            direction = strategy["direction"]
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

            # Enter position
            self.state.enter_position(price, sl_price, tp_price, direction, strategy["id"])

            confidence = satisfaction * 100
            logger.info(
                f"Signal: {direction} at ${price:,.2f} | "
                f"SL ${sl_price:,.2f} | TP ${tp_price:,.2f} | "
                f"Confidence {confidence:.0f}% ({conditions_met}/{conditions_total})"
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
                conditions_met=conditions_met,
                conditions_total=conditions_total,
                strategy_id=strategy["id"],
                strategy_rr_day=rr_day,
                strategy_win_rate=wr,
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
            self.state.exit_position()
            return

        # Check SL/TP using candle high/low (conservative: SL first)
        if direction == "LONG":
            sl_hit = candle_low <= sl_price
            tp_hit = candle_high >= tp_price
            if sl_hit:
                pnl_pct = self._calc_pnl(entry_price, sl_price, direction)
                logger.info(f"SL hit at ${sl_price:,.2f} (candle low: ${candle_low:,.2f})")
                send_exit_alert(self.symbol, direction, entry_price, sl_price, "sl", pnl_pct, -1.0, duration.total_seconds() / 60)
                self.state.exit_position()
                return
            if tp_hit:
                pnl_pct = self._calc_pnl(entry_price, tp_price, direction)
                rr = self.strategy["rr"]
                logger.info(f"TP hit at ${tp_price:,.2f} (candle high: ${candle_high:,.2f})")
                send_exit_alert(self.symbol, direction, entry_price, tp_price, "tp", pnl_pct, rr, duration.total_seconds() / 60)
                self.state.exit_position()
                return
        else:  # SHORT
            sl_hit = candle_high >= sl_price
            tp_hit = candle_low <= tp_price
            if sl_hit:
                pnl_pct = self._calc_pnl(entry_price, sl_price, direction)
                logger.info(f"SL hit at ${sl_price:,.2f} (candle high: ${candle_high:,.2f})")
                send_exit_alert(self.symbol, direction, entry_price, sl_price, "sl", pnl_pct, -1.0, duration.total_seconds() / 60)
                self.state.exit_position()
                return
            if tp_hit:
                pnl_pct = self._calc_pnl(entry_price, tp_price, direction)
                rr = self.strategy["rr"]
                logger.info(f"TP hit at ${tp_price:,.2f} (candle low: ${candle_low:,.2f})")
                send_exit_alert(self.symbol, direction, entry_price, tp_price, "tp", pnl_pct, rr, duration.total_seconds() / 60)
                self.state.exit_position()
                return

        # Still open
        hours = int(duration.total_seconds() // 3600)
        mins = int((duration.total_seconds() % 3600) // 60)
        logger.info(f"Position still open. Duration: {hours}h {mins}m. Price: ${current_price:,.2f}")

    def _check_missed_signals(self) -> None:
        """Check for signals that occurred while the bot was offline."""
        if self.state.last_check_time is None:
            return

        logger.info("Checking for missed signals...")
        last_check = datetime.fromisoformat(self.state.last_check_time)
        if last_check.tzinfo is None:
            last_check = last_check.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        offline_hours = (now - last_check).total_seconds() / 3600
        if offline_hours < 0.25:
            logger.info("Offline for less than 15 minutes. No missed signal check needed.")
            return

        logger.info(f"Offline for {offline_hours:.1f} hours. Scanning missed candles...")

        # Fetch candles covering the offline period
        df = get_latest_candles(self.symbol, self.timeframe, count=max(500, int(offline_hours * 4) + 100))
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
        conditions = strategy["conditions"]
        cond_df = compute_all_conditions(offline_df, conditions)

        for i in range(len(offline_df)):
            row = cond_df.iloc[i]
            satisfaction = row.sum() / len(conditions) if conditions else 0

            if satisfaction >= strategy["threshold"]:
                ts = offline_df.iloc[i]["timestamp"]
                price = float(offline_df.iloc[i]["close"])
                direction = strategy["direction"]
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
        log_file = config.LOG_DIR / f"live_{datetime.now().strftime('%Y-%m-%d')}.log"
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
