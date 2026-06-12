# Crypto Trading Bot

Automated trading signal generator for crypto futures (BTC/USDT, ETH/USDT) on Binance.
Generates LONG / SHORT / HOLD signals using a combination of trend-following,
mean-reversion, and volume breakout strategies. Sends alerts via Discord webhook.

**⚠️ This is a signal generator only. You execute trades manually.**

---

## Strategy Overview

| Signal | Type | Logic |
|---|---|---|
| **A — Trend** | EMA + RSI | EMA(9) crosses EMA(21) with RSI(14) confirmation |
| **B — Mean Rev** | Z-Score | Z-Score > ±2.0 from SMA(20) |
| **C — Volume** | Volume + BB | Volume > 1.5× avg + Bollinger band break |

**Voting:** ≥ +2 → LONG  |  ≤ -2 → SHORT  |  otherwise → HOLD

**Risk Management:** ATR(14) based stop loss (1.5×) and take profit (2.5×).

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Binance access (use VPN if geo-restricted)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Discord Webhook

Create a `.env` file in the project root:

```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url_here
```

### 4. Run Backtest

```bash
python backtest.py                           # Both BTC and ETH
python backtest.py --symbol BTC/USDT         # Single symbol
python backtest.py --redownload              # Force fresh data download
```

The backtest runs on **2024-01-01 → 2025-12-31** (24 months, 1h candles).
Results include: total trades, win rate, profit factor, max drawdown, Sharpe ratio.

Trade logs are saved to `data/<symbol>_trades.csv`.

### 5. Run Live Signals

```bash
python live_signal.py
```

The script:
- Fetches the latest 300 hourly candles on startup (immediately).
- Evaluates signals for BTC/USDT and ETH/USDT.
- Sends a Discord alert **only when the signal changes** (no spam).
- Sleeps 1 hour, then repeats.
- Press `Ctrl+C` to stop.

---

## Project Structure

```
crypto_bot/
├── .env                  # Discord webhook URL (gitignored)
├── .gitignore
├── data/                 # Cached OHLCV data & trade logs
├── logs/                 # Execution logs
├── last_signal.json      # Tracks last signal per symbol (auto-created)
├── signals.py            # Shared signal logic & indicators
├── backtest.py           # Backtest runner
├── live_signal.py        # Live signal checker
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Error Handling

- **Binance API failures:** Retries 3× with exponential backoff, then skips cycle.
- **Discord failures:** Logged only — no retry. Next cycle resends if signal persists.
- **Missing/corrupt state:** Auto-creates `last_signal.json` with default HOLD state.
- **NaN indicators:** Treated as neutral (0) during warm-up period.

---

## Output Example

### Discord Alert

```
**NEW SIGNAL DETECTED: BTC/USDT**

**Action:** 🔴 SHORT
**Entry price:** $67,200.00
**Stop Loss:** $68,208.00 (1.5 ATR)
**Take Profit:** $64,512.00 (2.5 ATR)

Signals: Trend=-1, MeanRev=0, Volume=0 → Total=-2 → **SHORT**

*This is a signal. Execute manually.*
```

### Backtest Output

```
============================================================
  Backtest Results: BTC/USDT
============================================================
  Period:            2024-01-01 → 2025-12-31
  Timeframe:         1h
  Starting Capital:  $1,000.00
  Commission:        0.04%
  Slippage:          0.05%
────────────────────────────────────────────────────────────
  Total Trades:      156
  Win Rate:          42.31%
  Profit Factor:     1.38
  Max Drawdown:      18.42%
  Sharpe Ratio:      0.87
  Final Equity:      $1,234.56
  Return:            23.46%
============================================================
```

---

## License

MIT
