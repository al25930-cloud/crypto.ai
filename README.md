# Crypto Trading Strategy Optimizer & Signal Generator

A machine learning-based system that tests thousands of trading strategies using Genetic Algorithm + Bayesian Optimization, selects the best one based on performance, and generates live trading signals via Discord.

**You never execute trades automatically** — the system only sends signals. You decide whether and how to act on them.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Configuration](#2-configuration)
3. [Step 1: Train a Strategy](#step-1-train-a-strategy)
4. [Step 2: Review the Efficiency Report](#step-2-review-the-efficiency-report)
5. [Step 3: Validate the Strategy](#step-3-validate-the-strategy)
6. [Step 4: Go Live](#step-4-go-live)
7. [Understanding the Output](#understanding-the-output)
8. [Configuration Reference](#configuration-reference)
9. [File Reference](#file-reference)
10. [Troubleshooting](#troubleshooting)

---

## 1. Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Install dependencies

```bash
cd crypto.ai
pip install -r requirements.txt
```

This installs: ccxt, pandas, numpy, requests, python-dotenv, deap, optuna, pandas_ta.

### Install TA-Lib (optional, recommended)

TA-Lib is a C-based indicator library that makes backtesting **significantly faster**. If it's not installed, the system falls back to pandas_ta (slower but works everywhere).

> **You can skip this entirely.** The system works perfectly fine without TA-Lib — it will just use pandas_ta automatically. TA-Lib is only recommended if you plan to run long training sessions (30+ minutes) and want faster backtesting.

#### Option 1: Prebuilt wheel (Windows — easiest)

1. Download the `.whl` file that matches your Python version from [https://github.com/cgohlke/talib-build/releases](https://github.com/cgohlke/talib-build/releases)
   - Check your Python version: `python --version`
   - Example: `TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl` is for Python 3.11 on 64-bit Windows
2. Install it:
   ```bash
   pip install TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl
   ```

#### Option 2: Conda (Windows/Mac/Linux)

If you use Anaconda or Miniconda, this is the most reliable method:
```bash
conda install -c conda-forge ta-lib
```

#### Option 3: Build from source (Mac/Linux)

```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Ubuntu / Debian
sudo apt install libta-lib0 libta-lib-dev
pip install TA-Lib

# Fedora / RHEL
sudo dnf install ta-lib-devel
pip install TA-Lib
```

#### Common pitfalls

- **`ta-lib-everywhere` on PyPI is deprecated** — it's a dummy package that just redirects to the official `TA-Lib` and does NOT bundle the C library. Don't use it.
- **Missing C headers on Linux** — make sure you install the `-dev` package (e.g., `libta-lib-dev`), not just `libta-lib0`.
- **Python version mismatch** — the `.whl` filename must match your Python version (cp39 = 3.9, cp310 = 3.10, cp311 = 3.11, cp312 = 3.12).

#### Verifying TA-Lib is installed

```bash
python -c "import talib; print(f'TA-Lib version: {talib.__version__}')"
```

If this prints the version number, TA-Lib is working. If you get an import error, the system will automatically fall back to pandas_ta.

If TA-Lib fails to install, don't worry — the system will use pandas_ta automatically. You'll see this message at startup:
```
[WARNING] TA-Lib not found. Using pandas_ta fallback.
```

---

## 2. Configuration

### Discord Webhook (required for live signals)

The system sends trading signals to Discord via a webhook URL. This is already configured in your `.env` file.

To change it later, edit `.env`:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN
```

### Tuning Parameters (optional)

All parameters are in `config.py`. Here are the most important ones:

| Parameter | Default | What it does |
|---|---|---|
| `SYMBOL` | `BTC/USDT` | Which trading pair to trade |
| `TIMEFRAME` | `15m` | Candle interval (`5m`, `15m`, `1h`) |
| `TRAINING_MINUTES` | `30` | How long to train (longer = more strategies tested) |
| `TRAINING_METHOD` | `ga_bayesian` | `ga_bayesian` (serious) or `random` (quick test) |
| `MIN_WIN_RATE` | `0.35` | Minimum 35% win rate to qualify |
| `MAX_DRAWDOWN` | `0.50` | Maximum 50% drawdown to qualify (linear penalty from 15%) |
| `TRADING_FEE_PCT` | `0.1` | Trading fee per side (0.1% = Binance standard) |

**You don't need to change anything to get started.** The defaults are sensible.

---

## Step 1: Train a Strategy

Training is the process of finding the best trading strategy by testing thousands of combinations.

### Quick test (random search, 2 minutes)

```bash
python training.py --symbol BTC/USDT --method random --minutes 2
```

This is useful to verify everything works before committing to a longer run.

### Full training (GA + Bayesian, 30 minutes)

```bash
python training.py --symbol BTC/USDT
```

### What happens during training

1. **Data fetching**: Downloads 6 months of 15-minute candles from Binance (cached to `data/` so subsequent runs are fast)
2. **Indicator computation**: Calculates all technical indicators (EMA, RSI, MACD, Bollinger Bands, etc.)
3. **Phase 1 — Genetic Algorithm**: Evolves a population of 200 strategies over 30 generations. Each generation selects the best performers, combines them (crossover), and introduces random changes (mutation).
4. **Phase 2 — Bayesian Optimization**: Takes the top 10 strategies from GA and refines them using Optuna's Tree-structured Parzen Estimator, testing up to 2,000 additional variations.
5. **Efficiency analysis**: Analyzes which of the 53 conditions are helping vs. hurting. Conditions with efficiency < 0.3 are automatically removed from future training runs.
6. **Save results**: Best strategy → `models/best_strategy.json`, top 500 → `models/top_strategies.json`, efficiency report → `models/condition_efficiency.json`

### What you'll see

```
[2026-06-19 14:00:00] Training started. Method: ga_bayesian | Symbol: BTC/USDT
[2026-06-19 14:00:01] Phase 1: Genetic Algorithm (pop=200, gen=30)
[2026-06-19 14:00:05] GA Gen 0/30 | Best: 1.85 | Avg: 1.12
[2026-06-19 14:00:35] GA Gen 5/30 | Best: 2.10 | Avg: 1.34 | ★ NEW BEST!
...
[2026-06-19 14:05:00] GA: Finished | Best score: 2.45 | Total strategies tested: 4,500
[2026-06-19 14:05:01] Phase 2: Bayesian Optimization (trials=2000)
...
[2026-06-19 14:25:01] Training finished.
   Best score: 2.68
   Best strategy saved to models/best_strategy.json
```

### CLI options

```bash
python training.py --help

# Options:
#   --symbol SYMBOL       Trading pair (default: BTC/USDT)
#   --timeframe TF        Candle timeframe (default: 15m)
#   --minutes N           Training duration in minutes (default: 30)
#   --method METHOD       ga_bayesian (default) or random
```

---

## Step 2: Review the Efficiency Report

After training, check the efficiency report to understand which conditions are performing well or poorly.

The report is saved to `models/condition_efficiency.json` and also printed to the console during training.

**Key alert levels:**
- ⭐ **STRONG** (efficiency > 1.3): This condition appears frequently in top strategies. Keep it.
- ✅ **OK** (0.7–1.3): Normal performance. Keep it.
- ⚠️ **WARNING** (0.5–0.7): Below average but not critical. Monitor it.
- 🔴 **ALERT** (0.3–0.5): Poor performer. Consider removing.
- 🔴🔴 **CRITICAL** (< 0.3): **Automatically removed** from future training runs.

You can restore removed conditions by deleting `models/removed_conditions.json`.

---

## Step 3: Validate the Strategy

**Always validate before going live.** Validation runs the best strategy on a completely separate time period (the last 12 months, non-overlapping with training data) to check if it generalizes.

```bash
python validation.py --symbol BTC/USDT --period 12
```

### What you'll see

```
============================================================
VALIDATION RESULTS
============================================================
  Period:          2024-06-19 to 2025-12-19
  Strategy:        strat_GA_BO_001

  Total trades:    156
  Valid trades:    142
  Invalid trades:  14 (too short)

  Win rate:        48.5%  ✅ PASS (threshold: ≥35%)
  Profit factor:   1.82   ✅ PASS (threshold: ≥1.3)
  Sharpe ratio:    2.1500
  Max drawdown:    18.3%  ✅ PASS (threshold: ≤50%)
  RR/day:          2.4500

  Exit breakdown:
    SL hits:       73
    TP hits:       69
    Timeouts:      14

  ✅ ALL ACCEPTANCE CRITERIA PASSED
============================================================
```

### Acceptance criteria

| Metric | Threshold | What it means |
|---|---|---|
| Win rate | ≥ 35% | At least 1 in 3 trades is a winner |
| Max drawdown | ≤ 50% | Worst peak-to-trough loss is at most 50% (score penalized from 15%) |
| Profit factor | ≥ 1.3 | Gross profit is at least 1.3× gross loss |

**If validation fails**, you can:
1. Retrain with more time: `python training.py --minutes 60`
2. Adjust parameters in `config.py` (e.g., relax `MIN_WIN_RATE` to 0.30)
3. Try a different symbol or timeframe

**If validation passes**, proceed to live mode.

---

## Step 4: Go Live

Start the live signal generator:

```bash
python live_signal.py --symbol BTC/USDT
```

### What it does

1. **Startup**: Loads the best strategy from `models/best_strategy.json`
2. **Missed signal check**: Scans for signals that fired while you were offline
3. **Main loop**: Every 15 minutes (at candle close), evaluates the strategy conditions
4. **Signal detection**: If conditions are met, sends a Discord alert with entry price, SL, TP
5. **Exit tracking**: Monitors the open position for SL/TP hits or 48-hour timeout
6. **Cooldown**: After any exit, waits 4 candles (1 hour) before looking for new signals

### Discord alerts you'll receive

**Entry signal:**
```
🟢 NEW SIGNAL DETECTED: BTC/USDT

Action: LONG
Entry: $67,200.00
Stop Loss: $66,384.00 (1.2%)
Take Profit: $68,880.00 (2.5 RR)
Confidence: 72% (6/10 conditions met)

Strategy ID: strat_GA_BO_001
RR/day: 2.68
Win rate (historical): 52%

⚠️ This is a signal. Execute manually with your own position size.
```

**Exit signal:**
```
💰 POSITION CLOSED: BTC/USDT

Result: TAKE PROFIT ✅
Entry: $67,200.00
Exit: $68,880.00
Profit: +2.5% (+1.0 RR)
Duration: 2h 15m
```

**Cooldown expired:**
```
⏳ COOLDOWN EXPIRED: BTC/USDT

Cooldown period has ended.
The bot is now actively monitoring for new signals.
```

**Missed signal (on startup):**
```
📋 [RECOVERY] Missed signal: BTC/USDT

Direction: LONG
Status: EXPIRED
Entry: $67,200.00
Stop Loss: $66,384.00
Take Profit: $68,880.00

ℹ️ Reported for your information only. Do NOT trade this signal.
```

### State persistence

The bot saves its state to `state.json` after every cycle. If you stop the bot and restart it, it will:
- Remember if you're in a position
- Remember the cooldown timer
- Check for missed signals during the offline period

### Stopping the bot

Press `Ctrl+C`. The bot will save its state and shut down gracefully.

---

## Understanding the Output

### What is RR/day?

**RR/day** (Risk-Reward per Day) is the primary metric. It measures how much risk-reward the strategy earns per trading day. A RR/day of 2.0 means the strategy earns 2× its risk per day on average.

### What is the score?

```
score = rr_per_day × low_trades_penalty × drawdown_penalty
```

**Step 1 — Low trade frequency penalty:**
- If avg trades/day ≤ 2: rr_per_day is multiplied by 0.5 (50% penalty)
- This filters out strategies that rarely trade, since low sample sizes are unreliable

**Step 2 — Drawdown penalty:**
- If max drawdown < 15%: no penalty (penalty = 1.0)
- If max drawdown 15%–50%: linear penalty scaling down to 0
- If max drawdown > 50%: strategy is disqualified

### What are "invalid trades"?

Trades that hit SL/TP within the first 45 minutes are marked as "invalid" (likely noise). The loss is still applied to your equity curve, but the trade is excluded from win rate and RR/day calculations.

### What is the drawdown penalty?

The drawdown penalty prevents the system from selecting strategies that have high returns but also massive drawdowns. A strategy with 50% drawdown gets its score multiplied by 0 (disqualified), while one with 15% drawdown gets no penalty.

---

## Configuration Reference

All parameters are in `config.py`. Here's the complete list:

### Data & Symbol
| Parameter | Default | Description |
|---|---|---|
| `TIMEFRAME` | `"15m"` | Candle interval: `"5m"`, `"15m"`, `"1h"` |
| `SYMBOL` | `"BTC/USDT"` | Trading pair |

### Training
| Parameter | Default | Description |
|---|---|---|
| `TRAINING_MINUTES` | `30` | Training duration in minutes |
| `TRAINING_PERIOD_MONTHS` | `6` | Months of historical data for training |
| `TRAINING_METHOD` | `"ga_bayesian"` | `"ga_bayesian"` or `"random"` |

### Strategy Generation
| Parameter | Default | Description |
|---|---|---|
| `MIN_CONDITIONS` | `8` | Minimum conditions per strategy |
| `MAX_CONDITIONS` | `16` | Maximum conditions per strategy |
| `MIN_THRESHOLD` | `0.5` | Minimum entry threshold (50%) |
| `MAX_THRESHOLD` | `0.7` | Maximum entry threshold (70%) |
| `MIN_SL` | `0.3` | Minimum stop-loss (%) |
| `MAX_SL` | `3.0` | Maximum stop-loss (%) |
| `MIN_RR` | `1.0` | Minimum risk-reward ratio |
| `MAX_RR` | `5.0` | Maximum risk-reward ratio |

### Disqualification
| Parameter | Default | Description |
|---|---|---|
| `MIN_WIN_RATE` | `0.35` | Minimum win rate (35%) |
| `MAX_DRAWDOWN` | `0.50` | Maximum drawdown (50%) |
| `MIN_TRADES_PER_DAY` | `0.5` | Minimum trades per day |
| `MAX_TRADES_PER_DAY` | `10` | Maximum trades per day |
| `LOW_TRADES_THRESHOLD` | `2.0` | If avg trades/day ≤ this, apply 50% penalty |
| `LOW_TRADES_PENALTY` | `0.5` | Score multiplier for low-frequency strategies |

### Trade Rules
| Parameter | Default | Description |
|---|---|---|
| `MIN_TRADE_DURATION_MINUTES` | `45` | Trades shorter than this are "invalid" |
| `MAX_TRADE_DURATION_HOURS` | `48` | Trades open longer than this are force-closed |
| `COOLDOWN_CANDLES` | `4` | Candles to wait after an exit (1 hour on 15m) |
| `TRADING_FEE_PCT` | `0.1` | Trading fee per side (0.1% = Binance standard) |

### GA Parameters
| Parameter | Default | Description |
|---|---|---|
| `GA_POPULATION_SIZE` | `200` | Strategies per generation |
| `GA_GENERATIONS` | `30` | Number of generations |
| `GA_ELITE_COUNT` | `5` | Top strategies preserved each generation |
| `GA_CROSSOVER_PROB` | `0.8` | Probability of crossover |
| `GA_MUTATION_PROB` | `0.2` | Probability of mutation |

### Bayesian Parameters
| Parameter | Default | Description |
|---|---|---|
| `BAYESIAN_N_TRIALS` | `2000` | Total optimization trials |
| `BAYESIAN_STARTUP_TRIALS` | `100` | Random trials before Bayesian model kicks in |

---

## File Reference

### Generated Files

| File | Description |
|---|---|
| `models/best_strategy.json` | The best strategy found during training |
| `models/top_strategies.json` | Top 500 strategies by score |
| `models/condition_efficiency.json` | Efficiency report for all 53 conditions |
| `models/removed_conditions.json` | Conditions auto-removed for poor performance |
| `models/validation_report.json` | Validation backtest results |
| `state.json` | Live bot state (position, cooldown, last check) |
| `data/*.csv` | Cached OHLCV candle data |
| `logs/*.log` | Training, validation, and live signal logs |

### Source Files

| File | Description |
|---|---|
| `config.py` | All configuration parameters |
| `conditions.py` | 53 technical conditions (LONG, SHORT, shared) |
| `indicators.py` | Indicator computation (TA-Lib / pandas_ta) |
| `data_fetcher.py` | OHLCV data download from Binance via ccxt |
| `backtest.py` | Backtest engine with mark-to-market drawdown |
| `strategy.py` | Strategy generation, scoring, save/load |
| `training.py` | Training loop (GA+Bayesian or random) |
| `genetic_optimizer.py` | Genetic Algorithm implementation |
| `bayesian_optimizer.py` | Bayesian Optimization via Optuna |
| `efficiency.py` | Condition efficiency analysis |
| `validation.py` | Full backtest validation with acceptance criteria |
| `live_signal.py` | Live signal generator with state management |
| `discord_bot.py` | Discord webhook sender |

---

## Troubleshooting

### "No best strategy found. Run training first."

You need to run training before validation or live mode:
```bash
python training.py --symbol BTC/USDT --method random --minutes 2
```

### "TA-Lib not found. Using pandas_ta fallback."

This is a warning, not an error. The system works fine with pandas_ta — it's just slower. See [Install TA-Lib](#install-talib-optional-recommended) above.

### "Discord webhook URL not configured"

Your `.env` file is missing or doesn't have the webhook URL. Make sure it exists in the project root with:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### Training finds no valid strategies

On synthetic or very volatile data, the GA may not find strategies that pass the disqualification criteria. Try:
1. Increase training time: `--minutes 60`
2. Relax criteria in `config.py`: lower `MIN_WIN_RATE` to 0.30
3. Use a different symbol with more liquidity

### "Rate limit hit" during data fetching

The Binance API has rate limits. The system handles this automatically with exponential backoff. If it persists, wait a few minutes and try again.

### How often should I retrain?

Market conditions change. Consider retraining weekly or when validation metrics degrade significantly.

### Can I use multiple symbols?

Not simultaneously. Train and run live mode for one symbol at a time. You can run multiple instances with different symbols in separate terminals.

---

## Typical Workflow Summary

```
1.  pip install -r requirements.txt           # Install dependencies
2.  Edit .env with Discord webhook URL         # Configure alerts
3.  python training.py --minutes 2 --method random   # Quick smoke test
4.  python training.py                         # Full GA+Bayesian training (30 min)
5.  Review efficiency report in console         # Check which conditions work
6.  python validation.py --period 12           # Validate on separate data7. Check acceptance criteria (WR≥35%, DD≤50%, PF≥1.3)
8.  python live_signal.py                      # Start receiving signals
9.  Check Discord for entry/exit alerts         # Execute trades manually
```
