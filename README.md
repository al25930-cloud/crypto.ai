# Crypto Trading Strategy Optimizer & Signal Generator

A machine learning-based system that tests thousands of trading strategies using Genetic Algorithm + Bayesian Optimization, selects the best one, and generates live trading signals via Discord.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For faster backtesting (optional):
```bash
pip install TA-Lib  # Requires C compiler or pre-built wheel
```

### 2. Configure

Edit `.env` with your Discord webhook URL:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
```

Adjust parameters in `config.py` as needed.

### 3. Train

```bash
# Full training with GA + Bayesian Optimization (default, ~30 min)
python training.py --symbol BTC/USDT

# Quick smoke test with random search (5 min)
python training.py --symbol BTC/USDT --method random --minutes 5
```

Training will:
- Fetch 6 months of historical OHLCV data from Binance
- Run Genetic Algorithm (global exploration) → Bayesian Optimization (local refinement)
- Save the best strategy to `models/best_strategy.json`
- Save top 500 strategies to `models/top_strategies.json`
- Generate an efficiency report showing underperforming conditions

### 4. Validate

```bash
python validation.py --symbol BTC/USDT --period 12
```

Runs the best strategy on a separate 12-month validation period and reports:
- Win rate, profit factor, Sharpe ratio, max drawdown
- Acceptance criteria check (win rate ≥ 35%, drawdown ≤ 20%, profit factor ≥ 1.3)

### 5. Go Live

```bash
python live_signal.py --symbol BTC/USDT
```

The bot will:
- Check every candle close (15 minutes by default)
- Detect missed signals from offline periods
- Send Discord alerts for new signals, exits, and cooldown expiry
- Track position state in `state.json`

## Architecture

```
crypto_trading_bot/
├── config.py                  # All configuration parameters
├── conditions.py              # 53 technical conditions (22 LONG + 22 SHORT + 9 shared)
├── indicators.py              # Indicator computation (TA-Lib primary, pandas_ta fallback)
├── data_fetcher.py            # OHLCV download from Binance via ccxt
├── strategy.py                # Strategy generation, scoring, serialization
├── backtest.py                # Backtest engine with mark-to-market drawdown
├── training.py                # Training loop (GA+Bayesian default, random optional)
├── genetic_optimizer.py       # GA using Individual class
├── bayesian_optimizer.py      # Bayesian optimization using Optuna
├── efficiency.py              # Condition efficiency analysis and auto-removal
├── validation.py              # Full backtest validation with acceptance criteria
├── live_signal.py             # Live signal generator with state management
├── discord_bot.py             # Discord webhook sender with rich embeds
├── data/                      # Cached OHLCV CSV files
├── models/                    # Best strategy, top strategies, efficiency reports
├── logs/                      # Training, validation, and live logs
├── state.json                 # Live bot state (position, cooldown)
├── .env                       # Discord webhook URL
└── requirements.txt
```

## Key Features

- **53 technical conditions** organized into LONG, SHORT, and shared pools
- **GA + Bayesian optimization** as default training (random search for quick testing)
- **Mark-to-market drawdown** tracking during open positions
- **Automatic condition removal** — conditions with efficiency < 0.3 are removed from future runs
- **Missed signal recovery** — detects signals that fired while the bot was offline
- **Cooldown notifications** — alerts when the bot resumes monitoring after a trade
- **Non-overlapping train/validation** periods to prevent data leakage
- **Trading fees** (0.1% per side) included in all backtests
