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

Training is a two-phase pipeline that tests thousands of strategy combinations to find the one with the highest score. Here's exactly what each stage does:

---

#### Stage 1: Data Preparation

1. **Fetch historical data**: Downloads 6 months of 15-minute candles from Binance (cached to `data/` so subsequent runs are fast)
2. **Compute indicators**: Calculates all technical indicators (EMA, RSI, MACD, Bollinger Bands, ATR, etc.) for every candle
3. **Pre-compute conditions**: All 53 conditions (e.g., `rsi_14_lt_30`, `price_gt_sma_200`) are evaluated for every candle and cached in memory. This avoids recomputing them for each strategy, which makes backtesting ~50x faster.

---

#### Stage 2: Phase 1 — Genetic Algorithm (Global Search)

The GA mimics natural selection to explore a huge search space. It takes ~50% of the total training time.

**How it works, step by step:**

1. **Create initial population**: Generate 200 completely random strategies. Each strategy has:
   - A direction (LONG or SHORT)
   - 7-27 conditions (randomly chosen from the pool of 31 available conditions per direction)
   - A threshold (0.5-0.7) — how many conditions must be true to enter a trade
   - A stop-loss (0.3%-3.0%) — maximum loss per trade
   - A risk-reward ratio (1.0-5.0) — target profit relative to stop-loss

2. **Evaluate every strategy**: Each strategy is backtested against 6 months of historical data. The backtest simulates: "If I had traded this strategy every time its conditions were met, what would my results be?" Each strategy gets a **score**:
   ```
   score = rr_per_day x drawdown_penalty x low_trades_penalty
   ```
   - **rr_per_day**: Risk-reward earned per trading day (higher = better)
   - **drawdown_penalty**: 1.0 if drawdown < 15%, scales linearly to 0 at 50%
   - **low_trades_penalty**: 0.5 if avg trades/day <= 2, else 1.0
   - Strategies that fail basic quality checks are **disqualified** (score = -inf):
     - Win rate < 35%
     - Max drawdown > 50%
     - Trade frequency outside 0.5-10 trades/day

3. **Sort by fitness**: Population sorted by score (highest first). The top strategy is the "best" this generation.

4. **Evolve for 30 generations**. Each generation does the following:

   **a) Elitism** — The top 5 strategies survive unchanged into the next generation. This guarantees we never lose our best strategy.

   **b) Tournament Selection** — For the remaining 195 slots, pick 3 random strategies and the one with the highest score wins a spot. Repeat until all 195 slots are filled. This favors high-scoring strategies but doesn't guarantee them — randomness keeps diversity.

   **c) Crossover (80% chance per pair)** — Pairs of parent strategies are combined to create children:
   ```
   Parent A: [cond1, cond2, cond3, cond4]     Parent B: [cond5, cond6, cond7, cond8]
   Child 1:  [cond1, cond2, cond7, cond8]     Child 2:  [cond5, cond6, cond3, cond4]
   ```
   The first half of conditions comes from one parent, the second half from the other. Numeric parameters (threshold, SL, RR) are averaged. If a child has duplicate conditions, they are removed and replaced with random ones from the pool.

   **d) Mutation (20% chance per individual)** — Randomly change ONE thing:
   - Swap a condition for a different one from the pool
   - Nudge threshold by +/-0.05
   - Nudge stop-loss by +/-0.2%
   - Nudge risk-reward by +/-0.5

   **e) Re-evaluate** — Backtest all new children, sort by score, and check for a new all-time best.

5. **After 30 generations**, the GA has tested ~6,000 strategies and converged toward high-scoring regions. The **top 10 strategies** are passed to Phase 2.

---

#### Stage 3: Phase 2 — Bayesian Optimization (Local Refinement)

The GA found promising *regions* of the search space. Now Bayesian optimization does a focused, intelligent search around those regions. It takes the remaining ~50% of training time.

**How it works:**

1. **Seed with GA results**: The top 10 GA strategies are loaded as the first 10 trials. This gives the optimizer a head start — it already knows what good strategies look like.

2. **Random exploration (100 trials)**: The first 100 trials explore randomly to build an initial model of the search space.

3. **Bayesian-guided search (1,900 trials)**: After the startup phase, Optuna's TPE (Tree-structured Parzen Estimator) model kicks in:
   - It looks at all past trials and their scores
   - It learns: "Strategies with these kinds of conditions, this threshold range, this SL/RR tend to score higher"
   - It suggests new trials that are likely to score well
   - Each trial it learns more, so suggestions get smarter over time

4. **Best strategy found**: After 2,000 total trials, the optimizer returns the highest-scoring strategy it found.

**Why two phases?** The GA is good at exploring a huge space broadly (global search), but it's slow and imprecise. Bayesian optimization is good at refining a narrow region precisely (local search), but it needs good starting points. Combining them gives you both breadth and depth.

---

#### Stage 4: Efficiency Analysis

After all strategies are tested, the system analyzes which of the 53 conditions are helping vs. hurting:

1. For each condition, calculate how often it appears in top-scoring strategies vs. bottom-scoring strategies
2. Compute an **efficiency score**: how much a condition contributes to winning strategies relative to the average
3. Conditions with efficiency < 0.3 are flagged for removal
4. **Pool size safeguard**: The system never removes conditions below 20 per direction (LONG/SHORT), so there are always enough conditions to build strategies
5. **Temporary removals**: Removed conditions are cleared at the start of each training run. All 53 conditions are re-evaluated with fresh market data

---

#### Stage 5: Save Results

1. **Best strategy** → `models/best_strategy.json` — the single best strategy found
2. **Top 500 strategies** → `models/top_strategies.json` — the 500 highest-scoring strategies (sorted by score)
3. **Efficiency report** → `models/condition_efficiency.json` — which conditions helped/hurt
4. **Removed conditions** → `models/removed_conditions.json` — conditions temporarily excluded
5. **Training log** → `logs/training_YYYY-MM-DD_HHMMSS.log` — full log of everything that happened

---

#### How the Score Determines Everything

The **score** is the single metric the entire system optimizes for. Every decision — which strategies survive in the GA, which ones the Bayesian optimizer focuses on, which strategy gets saved as "best" — is based on this score.

```
score = rr_per_day x drawdown_penalty x low_trades_penalty
```

| Component | What it measures | Value range |
|---|---|---|
| **rr_per_day** | Risk-reward earned per trading day | 0 to ~5 (higher = better) |
| **drawdown_penalty** | Penalizes strategies with high drawdown | 0.0 (50% drawdown) to 1.0 (<15% drawdown) |
| **low_trades_penalty** | Penalizes strategies that rarely trade | 0.5 (<=2 trades/day) or 1.0 |

**Disqualified strategies** get score = -inf (instantly lose to everything). A strategy is disqualified if:
- Win rate < 35%
- Max drawdown > 50%
- Average trades/day outside 0.5-10

**What [NEW BEST!] means in the logs**: It appears when a single strategy in the current generation scores higher than any strategy seen in all previous generations. It does NOT mean the average population improved — it means a new champion was found.

---

#### What Determines Which Strategies Survive in the GA?

Three mechanisms control what survives from generation to generation:

| Mechanism | What it does | How many survive |
|---|---|---|
| **Elitism** | Top N individuals copied unchanged to next generation | 5 (`GA_ELITE_COUNT`) |
| **Tournament selection** | Random groups of 3 compete, winner becomes a parent | Probabilistic — favors high fitness but doesn't guarantee it |
| **Score (fitness)** | Higher score = more likely to be selected for reproduction | Higher = better chance |

The key insight: **elitism guarantees the top 5 always survive**, while **tournament selection gives higher-scoring individuals a better chance** (but not certainty) of reproducing. This balance prevents the population from losing diversity while still converging toward better strategies.

#### Pipeline Summary

```
Training Start
    |
    |-- Load 6 months of BTC/USDT data
    |-- Compute all 53 conditions (pre-cached for speed)
    |
    |-- PHASE 1: Genetic Algorithm (~50% of time)
    |   |-- Gen 0: 200 random strategies
    |   |-- Gen 1-30: Evolve via selection + crossover + mutation
    |   |-- ~6,000 strategies tested
    |   +-- Top 10 passed to Phase 2
    |
    |-- PHASE 2: Bayesian Optimization (~50% of time)
    |   |-- Seed with GA's top 10
    |   |-- 100 random trials (build initial model)
    |   |-- 1,900 smart trials (TPE-guided search)
    |   +-- ~2,000 strategies tested
    |
    +-- Save best strategy + top 500 + efficiency report
```

---

### What you'll see

```
Training started | Method: ga_bayesian | Symbol: BTC/USDT | Duration: 30 min
  Objective: Find the strategy with the highest score.
  Score = rr_per_day x drawdown_penalty x low_trades_penalty.
  Disqualified if: win_rate < 35%, max_drawdown > 50%, or avg_trades/day outside 0.5-10.

Phase 1: Genetic Algorithm (global search -- evolve strategies over generations)
GA: Starting | pop=200, gen=30, cx=0.8, mut=0.2, elite=5
  Score = rr_per_day * drawdown_penalty * low_trades_penalty. Higher = better.
GA Gen 0/30 | Best score: 1.85 | Avg score: 1.12 | Pop: 200
GA Gen 1/30 | Best score: 1.92 | Avg score: 1.18 | Tested: 400
GA Gen 5/30 | Best score: 2.10 | Avg score: 1.34 | Tested: 1200 [NEW BEST!]
...
GA: Finished | Best score: 2.45 | Total strategies tested: 4500 | Passing top 10 to Bayesian.

Phase 2: Bayesian Optimization (local refinement -- focus on promising regions)
Bayesian: Starting | trials=2000, startup=100
  Each trial generates a strategy with random conditions, threshold, stop-loss, ...
Bayesian: Seeded 10 strategies from GA.
...
Bayesian: Finished | Best score: 2.68 | Total strategies tested: 2000

Training finished.
  Best score:          2.6800
  Best strategy:       strat_abc123
    Win rate:          52.0%
    RR/day:            2.4500
    Max drawdown:      12.3%
    Valid trades:      142
  Total strategies tested: 6500
  Time elapsed:        1800s (30.0m)
  Average:             3.6 strats/sec
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

**Pool size safeguard:** The system never removes conditions below 20 per direction (LONG/SHORT), so there are always enough conditions to build strategies.

**Temporary removals:** Removed conditions are cleared at the start of each training run. All 53 conditions are re-evaluated with fresh market data, so conditions that underperformed in a previous market regime can recover.

You can also manually restore removed conditions by deleting `models/removed_conditions.json`.

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

### Reading the Logs

Here's what each log line means:

| Log Field | Meaning |
|---|---|
| **Best score** | The highest score found so far in this phase. This is the score of the best strategy. |
| **Avg score** | The average score of all living strategies in the current generation (GA only). Shows if the population is improving overall. |
| **Tested** | Cumulative number of strategies evaluated so far. |
| **Pop** | Population size (number of strategies in each generation). |
| **[NEW BEST!]** | A new all-time best score was found this generation. |
| **Passing top 10** | The top 10 GA strategies are passed as starting points for the Bayesian optimizer. |
| **startup** | Number of random trials before the Bayesian TPE model kicks in (exploration phase). |

The **score** is the primary metric the system optimizes for. It's calculated as:

```
score = rr_per_day x drawdown_penalty x low_trades_penalty
```

- **rr_per_day**: Risk-reward earned per trading day (higher = better)
- **drawdown_penalty**: 1.0 if drawdown < 15%, scales linearly to 0 at 50%
- **low_trades_penalty**: 0.5 if avg trades/day <= 2, else 1.0
- **Disqualified** (score = -inf): win_rate < 35%, drawdown > 50%, or trades/day outside 0.5-10

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

### Strategy Generation (percentage-based)
| Parameter | Default | Description |
|---|---|---|
| `MIN_CONDITION_PERCENTAGE` | `0.25` | Minimum conditions as % of pool (25% of 31 = ~8) |
| `MAX_CONDITION_PERCENTAGE` | `0.90` | Maximum conditions as % of pool (90% of 31 = ~28) |
| `MIN_CONDITIONS_ABSOLUTE` | `3` | Safety floor -- never go below 3 conditions |
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
