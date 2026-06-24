"""
Discord webhook sender for trading signal alerts.

Sends rich embeds via Discord webhooks for entry signals, exit signals,
cooldown expiry, and missed signal recovery notifications.

Usage (standalone):
    from discord_bot import send_entry_signal, send_exit_alert
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)

# Discord embed colors (decimal)
COLOR_GREEN = 0x00FF00   # Entry signals
COLOR_RED = 0xFF0000     # Exit / SL signals
COLOR_BLUE = 0x3498DB    # Cooldown / recovery
COLOR_YELLOW = 0xFFCC00  # Warnings


def _send_webhook(payload: dict) -> bool:
    """Send a payload to the Discord webhook.

    Args:
        payload: JSON payload dict.

    Returns:
        True if sent successfully, False otherwise.
    """
    webhook_url = config.DISCORD_WEBHOOK_URL
    if not webhook_url:
        logger.warning("Discord webhook URL not configured. Set DISCORD_WEBHOOK_URL in .env")
        return False

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 204:
            logger.info("Discord alert sent.")
            return True
        elif response.status_code == 429:
            retry_after = response.json().get("retry_after", 5)
            logger.warning(f"Discord rate limited. Retry after {retry_after}s. Skipping this cycle.")
            return False
        else:
            logger.error(f"Discord send failed: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Discord send error: {e}. Continuing to next cycle.")
        return False


def send_entry_signal(
    symbol: str,
    direction: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    rr_ratio: float,
    confidence: float,
    conditions_met: int,
    conditions_total: int,
    strategy_id: str,
    strategy_rr_day: float,
    strategy_win_rate: float,
    base_confidence: Optional[float] = None,
    bonus: Optional[float] = None,
) -> bool:
    """Send a new entry signal alert.

    Args:
        symbol: Trading pair.
        direction: "LONG" or "SHORT".
        entry_price: Entry price.
        sl_price: Stop loss price.
        tp_price: Take profit price.
        rr_ratio: Risk-reward ratio.
        confidence: Confidence percentage (0-100).
        conditions_met: Number of core conditions met.
        conditions_total: Total number of core conditions.
        strategy_id: Strategy identifier.
        strategy_rr_day: Strategy's historical RR/day.
        strategy_win_rate: Strategy's historical win rate.
        base_confidence: Optional base confidence (core conditions only, 0-100).
        bonus: Optional SHARED bonus (0-100).

    Returns:
        True if sent successfully.
    """
    sl_pct = abs(entry_price - sl_price) / entry_price * 100
    action_emoji = "🟢" if direction == "LONG" else "🔴"

    embed = {
        "title": f"{action_emoji} NEW SIGNAL DETECTED: {symbol}",
        "color": COLOR_GREEN if direction == "LONG" else COLOR_RED,
        "fields": [
            {"name": "Action", "value": direction, "inline": True},
            {"name": "Entry", "value": f"${entry_price:,.2f}", "inline": True},
            {"name": "\u200b", "value": "\u200b", "inline": True},
            {"name": "Stop Loss", "value": f"${sl_price:,.2f} ({sl_pct:.1f}%)", "inline": True},
            {"name": "Take Profit", "value": f"${tp_price:,.2f} ({rr_ratio:.1f} RR)", "inline": True},
            {"name": "Risk-Reward", "value": str(rr_ratio), "inline": True},
            {"name": "Confidence", "value": f"{confidence:.0f}% ({conditions_met}/{conditions_total} conditions)" if base_confidence is None else f"{confidence:.0f}% ({base_confidence:.0f}% base + {bonus:.0f}% bonus)", "inline": True},
            {"name": "Strategy", "value": strategy_id, "inline": True},
            {"name": "\u200b", "value": "\u200b", "inline": True},
            {"name": "Historical RR/day", "value": f"{strategy_rr_day:.2f}", "inline": True},
            {"name": "Historical Win Rate", "value": f"{strategy_win_rate:.0%}", "inline": True},
            {"name": "\u200b", "value": "\u200b", "inline": True},
        ],
        "footer": {"text": "⚠️ This is a signal. Execute manually with your own position size."},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    payload = {"username": "Crypto Signal Bot", "embeds": [embed]}
    return _send_webhook(payload)


def send_exit_alert(
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    result: str,
    pnl_pct: float,
    rr: float,
    duration_minutes: float,
) -> bool:
    """Send a position closed alert.

    Args:
        symbol: Trading pair.
        direction: "LONG" or "SHORT".
        entry_price: Entry price.
        exit_price: Exit price.
        result: "sl", "tp", or "timeout".
        pnl_pct: Net PnL percentage.
        rr: Trade RR.
        duration_minutes: Duration in minutes.

    Returns:
        True if sent successfully.
    """
    result_labels = {"sl": "STOP LOSS ❌", "tp": "TAKE PROFIT ✅", "timeout": "TIMEOUT ⏰"}
    result_label = result_labels.get(result, result.upper())

    if pnl_pct >= 0:
        color = COLOR_GREEN
        title = f"💰 POSITION CLOSED: {symbol}"
    else:
        color = COLOR_RED
        title = f"🔴 POSITION CLOSED: {symbol}"

    hours = int(duration_minutes // 60)
    mins = int(duration_minutes % 60)
    duration_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"

    embed = {
        "title": title,
        "color": color,
        "fields": [
            {"name": "Result", "value": result_label, "inline": True},
            {"name": "Direction", "value": direction, "inline": True},
            {"name": "\u200b", "value": "\u200b", "inline": True},
            {"name": "Entry", "value": f"${entry_price:,.2f}", "inline": True},
            {"name": "Exit", "value": f"${exit_price:,.2f}", "inline": True},
            {"name": "\u200b", "value": "\u200b", "inline": True},
            {"name": "Profit", "value": f"{pnl_pct:+.2%} ({rr:+.1f} RR)", "inline": True},
            {"name": "Duration", "value": duration_str, "inline": True},
            {"name": "\u200b", "value": "\u200b", "inline": True},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    payload = {"username": "Crypto Signal Bot", "embeds": [embed]}
    return _send_webhook(payload)


def send_cooldown_alert(symbol: str) -> bool:
    """Send a cooldown expired notification.

    Args:
        symbol: Trading pair.

    Returns:
        True if sent successfully.
    """
    embed = {
        "title": f"⏳ COOLDOWN EXPIRED: {symbol}",
        "description": "Cooldown period has ended.\nThe bot is now actively monitoring for new signals.",
        "color": COLOR_BLUE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    payload = {"username": "Crypto Signal Bot", "embeds": [embed]}
    return _send_webhook(payload)


def send_recovery_alert(
    symbol: str,
    direction: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    rr_ratio: float,
    signal_time: str,
    expired: bool,
) -> bool:
    """Send a missed signal recovery notification.

    Args:
        symbol: Trading pair.
        direction: "LONG" or "SHORT".
        entry_price: Original entry price.
        sl_price: Original SL price.
        tp_price: Original TP price.
        rr_ratio: Risk-reward ratio.
        signal_time: ISO timestamp of the original signal.
        expired: Whether the signal has expired.

    Returns:
        True if sent successfully.
    """
    status = "EXPIRED" if expired else "VALID"

    embed = {
        "title": f"📋 [RECOVERY] Missed signal: {symbol}",
        "color": COLOR_YELLOW,
        "fields": [
            {"name": "Direction", "value": direction, "inline": True},
            {"name": "Status", "value": status, "inline": True},
            {"name": "\u200b", "value": "\u200b", "inline": True},
            {"name": "Entry", "value": f"${entry_price:,.2f}", "inline": True},
            {"name": "Stop Loss", "value": f"${sl_price:,.2f}", "inline": True},
            {"name": "Take Profit", "value": f"${tp_price:,.2f} ({rr_ratio:.1f} RR)", "inline": True},
            {"name": "Signal Time", "value": signal_time, "inline": False},
        ],
        "footer": {"text": "ℹ️ Reported for your information only. Do NOT trade this signal."},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    payload = {"username": "Crypto Signal Bot", "embeds": [embed]}
    return _send_webhook(payload)
