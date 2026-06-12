# Crypto Trading Bot — Technical Specification

**Version:** 1.0  
**Date:** 2026-06-12  
**Status:** Ready for Implementation  

---

## 1. Project Overview

Build a Python trading bot for crypto futures (BTC/USDT, ETH/USDT) that generates LONG and SHORT signals and sends them via Discord webhook. The user executes trades manually. The strategy combines trend-following and mean-reversion using classic, well-documented indicators. Risk management is dynamic using ATR.

**Phase 1 (MVP) — Simple, proven, backtested components. No ML, no complex infrastructure.**

---

## 2. Technology Stack

| Component | Library/Tool | Purpose |
|---|---|---|
| Data fetching | `ccxt` | Get OHLCV from Binance |
| Data manipulation | `pandas`, `numpy` | Handle time series |
| Technical indicators | `pandas_ta` | EMA, RSI, Bollinger, ATR, etc. |
| Backtesting | `backtesting.py` | Simulate trades with built-in metrics, slippage, commission |
| Discord alerts | `requests` | Send webhook messages |
| Environment config | `python-dotenv` | Load Discord webhook URL from `.env` |
| Scheduling | `time.sleep()` | Run hourly checks |
| Logging | `logging` (stdlib) | Timestamped logs to console + file with daily rotation |

**Python version:** 3.10+  
**All libraries are free and open-source.**

---

## 3. Strategy Logic (The Core)

### 3.1 Individual Signal Definitions

#### Signal A — Trend Following (EMA + RSI)

- **LONG:** EMA(9) crosses above EMA(21) AND RSI(14) > 50
- **SHORT:** EMA(9) crosses below EMA(21) AND RSI(14) < 50
- **Otherwise:** 0 (neutral)

#### Signal B — Mean Reversion (Z-Score)

- **Formula:** `Z-Score = (current_price - SMA(20)) / stddev(20)`
- **LONG:** Z-Score < -2.0 (price too low, expect bounce up)
- **SHORT:** Z-Score > +2.0 (price too high, expect drop down)
- **Otherwise:** 0 (neutral)

#### Signal C — Volume Breakout (Volume + Bollinger)

- **Volume average:** SMA(20) of volume (not EMA — volume spikes are meaningful as raw values)
- **LONG:** Current volume > 1.5 × SMA_volume(20) AND price > upper Bollinger band(20, 2)
- **SHORT:** Current volume > 1.5 × SMA_volume(20) AND price < lower Bollinger band(20, 2)
- **Otherwise:** 0 (neutral)

### 3.2 Voting System (Final Decision)

Each signal contributes: **+1 for LONG, -1 for SHORT, 0 for neutral.**

Total score = sum(signal A, signal B, signal C)

| Total Score | Final Action | Explanation |
|---|---|---|
| ≥ +2 | LONG | At least two signals agree on LONG |
| ≤ -2 | SHORT | At least two signals agree on SHORT |
| Otherwise | HOLD / NO TRADE | Conflicting signals — do nothing |

---

## 4. Dynamic Risk Management (ATR-based)

Use ATR(14) to set stop loss and take profit.

### For LONG trades
- **Stop Loss** = entry_price - (1.5 × ATR(14))
- **Take Profit** = entry_price + (2.5 × ATR(14))

### For SHORT trades
- **Stop Loss** = entry_price + (1.5 × ATR(14))
- **Take Profit** = entry_price - (2.5 × ATR(14))

**Why these multipliers?**
- 1.5× ATR gives price room to breathe without being too wide.
- 2.5× ATR creates a risk-to-reward ratio of 1:1.66. With a 40% win rate, this yields positive expectancy.

---

## 5. Backtest Module (`backtest.py`)

### 5.1 Parameters

| Parameter | Value | Reason |
|---|---|---|
| Timeframe | 1 hour | Balances signal frequency and noise |
| Date range | 2024-01-01 to 2025-12-31 (24 months) | Covers bull, bear, and sideways markets |
| Symbols | BTC/USDT, ETH/USDT (backtested separately) | Strategy may perform differently per symbol |
| Starting capital | $1000 (simulated) | User scales to own portfolio |
| Risk per trade | 1% of current capital | Conservative, matches user's style |
| Slippage | 0.05% | Realistic for crypto futures |
| Trading fee | 0.04% per trade (Binance futures taker fee) | Standard |
| Backtest engine | `backtesting.py` library | Handles entry/exit, intra-bar checks, slippage, commission, metrics automatically |

### 5.2 Data Management

- **Primary:** Download via `ccxt` from Binance.
- **Caching:** Save to CSV in `data/` directory after first download. Subsequent runs load from CSV.
- **Force redownload:** Support `--redownload` CLI flag to re-fetch data.
- **Directory:** `data/btc_usdt_1h.csv` and `data/eth_usdt_1h.csv`.

### 5.3 Required Output Metrics

After running backtest, print:

1. Total number of trades
2. Win rate (%)
3. Profit factor (gross profit / gross loss)
4. Maximum drawdown (%)
5. Sharpe ratio (annualized, risk-free rate = 0)
6. Final equity curve (optional plot)
7. CSV of all individual trades saved to `data/`

### 5.4 Walk-Forward Validation

**Skipped for MVP.** Can be added later as an enhancement.

---

## 6. Live Signal Module (`live_signal.py`)

### 6.1 Runtime Behavior

1. **On start:** Fetch data and evaluate signals **immediately** — no waiting for top of hour.
2. **After first check:** Sleep 3600 seconds (1 hour), then repeat.
3. **Runs until interrupted:** `Ctrl+C` stops the script.
4. **Command:** `python live_signal.py`

### 6.2 Per-Cycle Logic

1. Fetch latest 300 candles of 1-hour data for BTC/USDT and ETH/USDT from Binance (via `ccxt`).
2. Compute all indicators (EMA, RSI, Z-Score, Bollinger, Volume).
3. Apply the voting system (§3.2) to get final action (LONG / SHORT / HOLD).
4. If action is LONG or SHORT, calculate entry price (current close), stop loss, and take profit using ATR.
5. Compare with last signal stored in `last_signal.json`.
6. Send Discord notification **only if the signal has changed** (e.g., HOLD→LONG, LONG→SHORT).
7. Update `last_signal.json` with the new signal.

### 6.3 Discord Message Format

```
**NEW SIGNAL DETECTED: BTC/USDT**

**Action:** 🔴 SHORT
**Entry price:** $67,200
**Stop Loss:** $68,208 (1.5 ATR)
**Take Profit:** $64,512 (2.5 ATR)

Signals: Trend=-1, MeanRev=0, Volume=0 → Total=-2 → SHORT

*This is a signal. Execute manually.*
```

- **One message per symbol** (separate Discord notifications for BTC and ETH).
- All price values rounded to 2 decimal places.
- UTC timestamps on all messages.

### 6.4 `last_signal.json` Format

**Initial state (auto-created if missing):**
```json
{
    "BTC/USDT": "HOLD",
    "ETH/USDT": "HOLD",
    "last_update": null
}
```

**After first signal:**
```json
{
    "BTC/USDT": "SHORT",
    "ETH/USDT": "HOLD",
    "last_update": "2026-06-12T15:00:00Z"
}
```

---

## 7. Error Handling

### 7.1 CCXT / Binance API Failures

- Wrap API calls in `try/except`.
- Retry up to 3 times with exponential backoff (1s, 2s, 4s).
- If all retries fail: log the error and skip the current cycle.
- **No fallback exchange** (Binance only). If Binance is unreachable, log a clear error.
- User is responsible for ensuring Binance access (VPN if geo-restricted).

### 7.2 Discord Webhook Failures

- Catch the exception, log a warning to console and log file.
- **Do NOT retry.** Continue to the next scheduled check.
- The next hourly check will resend if the signal persists.

### 7.3 Missing Data / Data Quality

- Verify downloaded data has expected columns (timestamp, open, high, low, close, volume).
- If fewer than 300 candles returned, log warning and skip cycle.
- NaN values in indicators: treat as neutral (0) for that signal.

---

## 8. Logging

- **Method:** Python's built-in `logging` module.
- **Output:** Both console (stdout) and file (`logs/bot.log`).
- **Rotation:** Daily file rotation (keep last 7 days).
- **Levels:** INFO (normal operation), WARNING (recoverable errors), ERROR (fatal/unexpected).
- **Timestamps:** All in UTC.
- **Format:** `2026-06-12T15:00:00Z [INFO] Signal check completed: BTC/USDT = HOLD, ETH/USDT = HOLD`

---

## 9. Configuration

### `.env` file (in project root)

```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

Loaded via `python-dotenv`. Add `.env` to `.gitignore`.

### Constants in code

- Date range: `2024-01-01` to `2025-12-31`
- EMA periods: 9, 21
- RSI period: 14
- Z-Score threshold: ±2.0
- Bollinger: 20 period, 2 std
- ATR period: 14
- ATR stop multiplier: 1.5
- ATR take-profit multiplier: 2.5
- Volume multiplier: 1.5
- Slippage: 0.0005 (0.05%)
- Fee: 0.0004 (0.04%)

---

## 10. Project Structure

```
crypto_bot/
├── .env                    # Discord webhook URL (gitignored)
├── .gitignore
├── data/                   # CSV files for cached historical data
│   ├── btc_usdt_1h.csv
│   └── eth_usdt_1h.csv
├── logs/                   # Execution logs (daily rotation, 7-day retention)
│   └── bot.log
├── last_signal.json        # Stores last signal per symbol
├── backtest.py             # Run backtest and report metrics
├── live_signal.py          # Main signal generator (hourly checks)
├── signals.py              # Shared signal/voting logic (imported by both scripts)
├── requirements.txt        # Python dependencies
└── README.md               # Instructions for user
```

### Code Organization

- `signals.py`: Contains all shared logic — indicator computation, signal functions (A, B, C), voting system, ATR risk management. Imported by both `backtest.py` and `live_signal.py` to avoid duplication.
- `backtest.py`: Imports from `signals.py`, handles data fetching/caching, runs backtest via `backtesting.py`, prints metrics.
- `live_signal.py`: Imports from `signals.py`, handles hourly loop, Discord messaging, `last_signal.json` state management.

---

## 11. Requirements

### `requirements.txt`

```
ccxt
pandas
numpy
pandas_ta
backtesting
requests
python-dotenv
```

---

## 12. Testing

**No unit tests for MVP.** The user can add pytest tests later if needed. Focus on core functionality with basic error handling and logging.

---

## 13. Success Criteria

1. Backtest shows **win rate ≥ 40%** and **profit factor ≥ 1.3** on the 24-month period.
2. Live script sends correctly formatted Discord messages **only when signal changes**.
3. No external dependencies (databases, Redis, etc.) — pure Python.
4. User can run the script with a single command and leave it running for hours without crashing.

---

## 14. Out of Scope (MVP)

- Walk-forward validation
- Automatic parameter optimization
- Funding rate tracking
- WebSocket / real-time data
- Fallback exchanges (Binance only)
- Database storage
- Unit tests
- Multi-timeframe analysis
- Portfolio-level backtesting (separate per symbol only)
- GUI / dashboard

---

## 15. Key Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Python version | 3.10+ | Stable, widely supported, modern type hints |
| Webhook storage | `.env` + `python-dotenv` | Standard, secure, easy `.gitignore` |
| Backtest engine | `backtesting.py` library | Battle-tested, built-in metrics, reduces code complexity |
| Volume average | SMA(20) | Standard for volume breakouts; volume spikes are meaningful raw |
| Discord messages | One per symbol | Cleaner, easier to act on each symbol independently |
| Webhook failures | Log only, no retry | Avoids rate limits; next cycle resends if signal persists |
| Backtest scope | Separate per symbol | Strategy may perform differently on BTC vs ETH |
| Walk-forward | Skip for MVP | Reduces complexity; can add later |
| Initial signal state | HOLD for both symbols | Ensures first valid signal triggers Discord notification |
| Data caching | Fetch via ccxt + save CSV | Faster subsequent runs; `--redownload` flag to force refresh |
| Timestamps | UTC | Avoids DST confusion; standard in crypto |
| Check timing | Immediately on start | Gives user quick first result; no alignment to top of hour |
| Backtest date range | 2024-01-01 to 2025-12-31 | 24 months covering bull/bear/sideways |
| Logging | `logging` module | Timestamps, levels, file + console, daily rotation |
| Exchange fallback | None (Binance only) | Keeps code simple; user ensures Binance access |
| Unit tests | None for MVP | Focus on core functionality |
| Shared code | `signals.py` module | Avoids duplication between backtest and live scripts |
