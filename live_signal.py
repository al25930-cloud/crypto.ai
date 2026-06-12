"""
Crypto Trading Bot — Live Signal Module.

Checks for trading signals every hour for BTC/USDT and ETH/USDT and sends
Discord webhook notifications when signals change (HOLD→LONG, LONG→SHORT, etc.).

Usage:
    python live_signal.py

Requires a .env file in the project root with:
    DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
import requests
from dotenv import load_dotenv

from signals import (
    compute_indicators,
    signal_a_trend,
    signal_b_mean_reversion,
    signal_c_volume_breakout,
    voting_system,
    calculate_risk,
)

# =============================================================================
# Configuration
# =============================================================================

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
CANDLES_NEEDED = 300
SLEEP_SECONDS = 3600  # 1 hour
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # seconds

# =============================================================================
# Logging Setup
# =============================================================================

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "bot.log", encoding="utf-8"),
    ],
)
logging.Formatter.converter = time.gmtime  # UTC timestamps
logger = logging.getLogger(__name__)

# =============================================================================
# State Management (last_signal.json)
# =============================================================================

DEFAULT_STATE: dict = {
    "BTC/USDT": "HOLD",
    "ETH/USDT": "HOLD",
    "last_update": None,
}


def load_state(path: Path = Path("last_signal.json")) -> dict:
    """Load the last signal state from JSON, creating it if missing."""
    if not path.exists():
        logger.info("No last_signal.json found. Creating with default HOLD state.")
        save_state(DEFAULT_STATE, path)
        return DEFAULT_STATE.copy()

    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info("Loaded last_signal.json: %s", state)
        return state
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Corrupted last_signal.json — resetting. Error: %s", e)
        save_state(DEFAULT_STATE, path)
        return DEFAULT_STATE.copy()


def save_state(state: dict, path: Path = Path("last_signal.json")) -> None:
    """Persist the signal state to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)
    logger.debug("Saved state: %s", state)


# =============================================================================
# Data Fetching (with retry)
# =============================================================================


def fetch_latest_candles(
    symbol: str, limit: int = CANDLES_NEEDED
) -> Optional[pd.DataFrame]:
    """Fetch the latest OHLCV candles from Binance with retry logic.

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT'.
        limit: Number of candles to fetch.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume,
        or None if all retries exhausted.
    """
    exchange = ccxt.binance({"enableRateLimit": True})

    for attempt in range(MAX_RETRIES):
        try:
            candles = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
            if not candles:
                raise RuntimeError("Empty response from exchange.")

            df = pd.DataFrame(
                candles,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="ms", utc=True
            )

            if len(df) < limit:
                logger.warning(
                    "%s: got %d candles, expected %d.",
                    symbol,
                    len(df),
                    limit,
                )

            return df

        except (ccxt.NetworkError, ccxt.ExchangeError, RuntimeError) as e:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            logger.warning(
                "%s: fetch attempt %d/%d failed — %s. Retrying in %ds ...",
                symbol,
                attempt + 1,
                MAX_RETRIES,
                e,
                wait,
            )
            time.sleep(wait)

    logger.error(
        "%s: all %d fetch attempts failed. Skipping this cycle.",
        symbol,
        MAX_RETRIES,
    )
    return None


# =============================================================================
# Signal Evaluation
# =============================================================================


def evaluate_signal(df: pd.DataFrame) -> dict:
    """Compute all indicators and evaluate the trading signal for a DataFrame.

    Args:
        df: OHLCV DataFrame (must have columns: open, high, low, close, volume).

    Returns:
        Dictionary with keys: action, total_score, sig_a, sig_b, sig_c,
        entry_price, stop_loss, take_profit, atr, timestamp.
    """
    df = compute_indicators(df)

    # Use the last row (most recent candle)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    # Signal A (needs previous values for crossover detection)
    sig_a = signal_a_trend(
        last["EMA_9"],
        last["EMA_21"],
        prev["EMA_9"],
        prev["EMA_21"],
        last["RSI_14"],
    )

    # Signal B
    sig_b = signal_b_mean_reversion(last["Z_SCORE"])

    # Signal C
    sig_c = signal_c_volume_breakout(
        last["close"],
        last["volume"],
        last["VOLUME_SMA_20"],
        last["BB_UPPER"],
        last["BB_LOWER"],
    )

    action, total_score = voting_system(sig_a, sig_b, sig_c)

    result: dict = {
        "action": action,
        "total_score": total_score,
        "sig_a": sig_a,
        "sig_b": sig_b,
        "sig_c": sig_c,
        "entry_price": round(float(last["close"]), 2),
        "atr": round(float(last["ATR_14"]), 4),
        "stop_loss": None,
        "take_profit": None,
        "timestamp": str(last["timestamp"]) if "timestamp" in df.columns else datetime.now(timezone.utc).isoformat(),
    }

    # Calculate risk levels for actionable signals
    if action in ("LONG", "SHORT"):
        sl, tp = calculate_risk(result["entry_price"], last["ATR_14"], action)
        result["stop_loss"] = round(sl, 2)
        result["take_profit"] = round(tp, 2)

    return result


# =============================================================================
# Discord Notification
# =============================================================================


def send_discord_alert(
    webhook_url: str, symbol: str, signal: dict
) -> bool:
    """Send a trading signal alert to Discord.

    Args:
        webhook_url: Discord webhook URL.
        symbol: Trading pair, e.g. 'BTC/USDT'.
        signal: Dictionary from evaluate_signal().

    Returns:
        True if the message was sent successfully, False otherwise.
    """
    action = signal["action"]
    emoji = "🟢" if action == "LONG" else "🔴"
    sl_label = "Stop Loss" if action == "LONG" else "Stop Loss"
    tp_label = "Take Profit" if action == "LONG" else "Take Profit"

    message = (
        f"**NEW SIGNAL DETECTED: {symbol}**\n\n"
        f"**Action:** {emoji} {action}\n"
        f"**Entry price:** ${signal['entry_price']:,.2f}\n"
        f"**{sl_label}:** ${signal['stop_loss']:,.2f} (1.5 ATR)\n"
        f"**{tp_label}:** ${signal['take_profit']:,.2f} (2.5 ATR)\n\n"
        f"Signals: Trend={signal['sig_a']}, "
        f"MeanRev={signal['sig_b']}, "
        f"Volume={signal['sig_c']} "
        f"→ Total={signal['total_score']} → **{action}**\n\n"
        f"*This is a signal. Execute manually.*"
    )

    payload = {"content": message}

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Discord alert sent for %s: %s", symbol, action)
        return True
    except requests.RequestException as e:
        logger.warning(
            "Discord webhook failed for %s: %s. Signal will be retried next cycle.",
            symbol,
            e,
        )
        return False


# =============================================================================
# Main Loop
# =============================================================================


def main() -> None:
    """Run the live signal loop: check signals hourly, send Discord alerts."""

    # Load environment
    load_dotenv()
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        logger.error(
            "DISCORD_WEBHOOK_URL not set. "
            "Create a .env file with: DISCORD_WEBHOOK_URL=https://..."
        )
        sys.exit(1)

    logger.info("=== Crypto Trading Bot — Live Signals ===")
    logger.info("Symbols: %s | Timeframe: %s | Interval: %ds", SYMBOLS, TIMEFRAME, SLEEP_SECONDS)
    logger.info("Discord webhook configured: %s", webhook_url[:50] + "...")

    # Load previous signal state
    state = load_state()

    try:
        while True:
            cycle_start = datetime.now(timezone.utc)
            logger.info("--- Signal check cycle: %s ---", cycle_start.isoformat())

            for symbol in SYMBOLS:
                df = fetch_latest_candles(symbol)
                if df is None:
                    logger.warning("%s: skipping this cycle due to fetch failure.", symbol)
                    continue

                try:
                    signal = evaluate_signal(df)
                except Exception as e:
                    logger.error("%s: signal evaluation failed — %s", symbol, e, exc_info=True)
                    continue

                prev_action = state.get(symbol, "HOLD")
                new_action = signal["action"]

                logger.info(
                    "%s: action=%s (prev=%s) | A=%d B=%d C=%d | price=%.2f",
                    symbol,
                    new_action,
                    prev_action,
                    signal["sig_a"],
                    signal["sig_b"],
                    signal["sig_c"],
                    signal["entry_price"],
                )

                # Only send alert if signal changed
                if new_action != prev_action:
                    logger.info(
                        "%s: signal changed %s → %s — sending alert.",
                        symbol,
                        prev_action,
                        new_action,
                    )

                    alert_sent = False
                    if new_action in ("LONG", "SHORT"):
                        alert_sent = send_discord_alert(webhook_url, symbol, signal)
                    else:
                        # HOLD signal — send a simpler message
                        try:
                            resp = requests.post(
                                webhook_url,
                                json={
                                    "content": (
                                        f"**SIGNAL UPDATE: {symbol}**\n\n"
                                        f"**Action:** ⚪ HOLD (was {prev_action})\n\n"
                                        f"*No trade. Wait for next signal.*"
                                    )
                                },
                                timeout=10,
                            )
                            resp.raise_for_status()
                            alert_sent = True
                        except requests.RequestException as e:
                            logger.warning("Discord hold notification failed: %s", e)

                    # Only update state if the alert was sent successfully.
                    # On failure, the next cycle will detect the same change and retry.
                    if alert_sent:
                        state[symbol] = new_action
                        state["last_update"] = cycle_start.isoformat()
                        save_state(state)
                    else:
                        logger.info(
                            "%s: alert failed — state NOT updated. Will retry next cycle.",
                            symbol,
                        )
                else:
                    logger.info("%s: no change (%s) — skipping alert.", symbol, new_action)

            # Sleep until next cycle
            elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            sleep_time = max(0, SLEEP_SECONDS - elapsed)
            next_check = datetime.now(timezone.utc).isoformat()
            logger.info("Sleeping %.0fs. Next check ≈ %s", sleep_time, next_check)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down.")
        print("\nBot stopped. Goodbye!")


if __name__ == "__main__":
    main()
