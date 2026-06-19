# Crypto Trading Strategy Optimizer & Signal Generator — Specification

**Date:** 2026-06-19  
**Status:** Draft  
**Language:** Python  
**Data Source:** Binance via ccxt  
**Mode:** Signal-only (no automatic trade execution)

---

## 1. Project Overview

The system will:

1. Define a universe of **53 technical conditions** (22 LONG-specific, 22 SHORT-specific, 9 shared).
2. Generate random strategies, each consisting of 8–16 conditions, a threshold (50–70%), a stop-loss percentage (0.3–3%), and a risk-reward ratio (1–5).
3. Optimize strategies using **Genetic Algorithm (DEAP) + Bayesian Optimization (Optuna)** as the default training method, with **pure random search** available as a quick-testing mode.
4. Backtest each strategy quickly on historical data with **0.1% trading fees per side**.
5. Score each strategy by **RR/day** (sum of all trade RR divided by number of trading days).
6. Apply a **drawdown penalty** to the score.
7. Keep track of the best strategy found so far.
8. At the end of the training session, save the best strategy (and top 500 strategies) and generate an **efficiency report** showing which conditions are underperforming.
9. Allow the user to run a **full backtest validation** on the selected strategy.
10. Run in **live mode**: check the latest candle at close, generate signals, and send Discord alerts.
11. Handle **missed signals**: when the user starts the bot, check if any signals were missed while offline and send informational recovery notifications.

---

## 2. Data and Timeframe

| Parameter | Value |
|---|---|
| Timeframe | 15 minutes (configurable to 5m or 1h) |
| Data source | Binance via ccxt |
| Symbol | BTC/USDT (single symbol) |
| Historical data period for training | Last 6 months (configurable) |
| Historical data period for validation | Last 12 months (strictly non-overlapping, after training period) |
| Caching | Save OHLCV data to CSV in `data/` folder to avoid re-downloading |
| Indicator library | **TA-Lib** (primary, C-based, fast) with **pandas_ta** fallback (pure Python) |

### 2.1. Data Fetching

- Use `ccxt.binance().fetch_ohlcv(symbol, timeframe, since, limit)` with pagination (500 candles per request).
- Enable `exchange.enableRateLimit = True` for automatic rate-limit handling.
- Implement incremental caching: check local CSV for latest timestamp, only fetch data newer than that.
- Handle `ccxt.RateLimitExceeded` and `ccxt.NetworkError` with exponential backoff retries.

### 2.2. Data Periods (Non-Overlapping)

```
Training period:   [6 months ago] to [today]    (e.g., 2025-12-19 to 2026-06-19)
Validation period: [18 months ago] to [6 months ago]  (e.g., 2024-12-19 to 2025-12-19)
```

No overlap between training and validation data to prevent data leakage.

### 2.3. Indicator Library Selection

```python
import logging

logger = logging.getLogger(__name__)

try:
    import talib
    USE_TA_LIB = True
    logger.info("TA-Lib loaded successfully. Using TA-Lib for indicators.")
except ImportError:
    import pandas_ta as ta
    USE_TA_LIB = False
    logger.warning("[WARNING] TA-Lib not found. Using pandas_ta fallback.")
```

TA-Lib is preferred for speed (critical when testing thousands of strategies) and accuracy. pandas_ta serves as a fallback for systems where TA-Lib compilation is problematic (e.g., some Windows environments).

### 2.4. Indicator Warmup Handling

Indicators like EMA(200) require 200+ preceding candles before they produce valid values. This warmup period must be handled consistently in both backtest and live modes.

**Backtest mode:**
- After computing all indicators and conditions on the full dataset, **drop all rows where any indicator/condition value is NaN** using `df.dropna()`.
- This means the backtest starts later than the beginning of the data period (losing the first ~200 candles). This is acceptable and realistic.
- Each strategy may use different conditions with different warmup lengths. The `dropna()` approach handles this automatically — only rows where ALL used indicators are valid are kept.

**Live mode:**
- **Fetch at least 500 candles** to ensure all indicators (including EMA 200) have valid values with buffer.
- If any indicator returns NaN for the latest candle, **skip signal generation** for that cycle and log a warning:
  ```
  [WARNING] Indicator NaN detected at 2026-06-20 14:00:00. Skipping signal check.
  ```
- This prevents false signals from incomplete indicator calculations.

---

## 3. List of Conditions (53 Total)

Conditions are organized into three pools: **LONG-only**, **SHORT-only**, and **Shared** (direction-neutral).

### 3.1. LONG-Only Conditions (22)

```python
CONDITIONS_LONG = {
    # Trend (bullish)
    "ema_9_gt_21":          "EMA(9) > EMA(21)",
    "ema_12_gt_26":         "EMA(12) > EMA(26)",
    "ema_20_gt_50":         "EMA(20) > SMA(50)",
    "sma_20_gt_50":         "SMA(20) > SMA(50)",
    "price_gt_sma_50":      "Close > SMA(50)",
    "price_gt_sma_200":     "Close > SMA(200)",
    "macd_gt_signal":       "MACD line > Signal line",
    "macd_hist_gt_0":       "MACD histogram > 0",

    # Momentum (bullish / oversold reversal)
    "rsi_14_gt_50":         "RSI(14) > 50",
    "rsi_14_lt_30":         "RSI(14) < 30 (oversold reversal)",
    "rsi_21_gt_50":         "RSI(21) > 50",
    "rsi_21_lt_30":         "RSI(21) < 30 (oversold reversal)",
    "stoch_k_gt_20":        "Stochastic %K > 20",
    "stoch_k_lt_80":        "Stochastic %K < 80",
    "cci_14_gt_-100":       "CCI(14) > -100",
    "cci_14_lt_100":        "CCI(14) < 100",
    "williams_gt_-80":      "Williams %R > -80",
    "williams_lt_-20":      "Williams %R < -20",

    # Volatility (bullish)
    "price_lt_bb_lower_20_2":   "Close < BB lower (20,2) — oversold bounce",
    "price_gt_bb_upper_20_2":   "Close > BB upper (20,2) — breakout",
    "price_lt_bb_lower_20_1_5": "Close < BB lower (20,1.5) — oversold bounce",
    "price_gt_bb_upper_20_1_5": "Close > BB upper (20,1.5) — breakout",
}
```

### 3.2. SHORT-Only Conditions (22)

```python
CONDITIONS_SHORT = {
    # Trend (bearish)
    "ema_9_lt_21":          "EMA(9) < EMA(21)",
    "ema_12_lt_26":         "EMA(12) < EMA(26)",
    "ema_20_lt_50":         "EMA(20) < SMA(50)",
    "sma_20_lt_50":         "SMA(20) < SMA(50)",
    "price_lt_sma_50":      "Close < SMA(50)",
    "price_lt_sma_200":     "Close < SMA(200)",
    "macd_lt_signal":       "MACD line < Signal line",
    "macd_hist_lt_0":       "MACD histogram < 0",

    # Momentum (bearish / overbought reversal)
    "rsi_14_lt_50":         "RSI(14) < 50",
    "rsi_14_gt_70":         "RSI(14) > 70 (overbought reversal)",
    "rsi_21_lt_50":         "RSI(21) < 50",
    "rsi_21_gt_70":         "RSI(21) > 70 (overbought reversal)",
    "stoch_k_gt_80":        "Stochastic %K > 80 (overbought)",
    "stoch_k_lt_20":        "Stochastic %K < 20 (oversold breakdown)",
    "cci_14_gt_100":        "CCI(14) > 100 (overbought)",
    "cci_14_lt_-100":       "CCI(14) < -100 (breakdown)",
    "williams_gt_-20":      "Williams %R > -20 (overbought)",
    "williams_lt_-80":      "Williams %R < -80 (breakdown)",

    # Volatility (bearish)
    "price_gt_bb_upper_20_2_s":   "Close > BB upper (20,2) — overbought reversal",
    "price_lt_bb_lower_20_2_s":   "Close < BB lower (20,2) — breakdown",
    "price_gt_bb_upper_20_1_5_s": "Close > BB upper (20,1.5) — overbought reversal",
    "price_lt_bb_lower_20_1_5_s": "Close < BB lower (20,1.5) — breakdown",
}
```

### 3.3. Shared Conditions (9) — Used by Both LONG and SHORT

```python
CONDITIONS_SHARED = {
    "atr_gt_sma_atr_20":        "ATR(14) > SMA(ATR, 20) — high volatility",
    "volume_gt_sma_20_1_5":     "Volume > SMA(Volume, 20) × 1.5",
    "volume_gt_sma_20_2_0":     "Volume > SMA(Volume, 20) × 2.0",
    "obv_gt_sma_obv_20":        "OBV > SMA(OBV, 20)",
    "adx_14_gt_25":             "ADX(14) > 25 — trending",
    "adx_14_gt_30":             "ADX(14) > 30 — strong trend",
    "adx_14_lt_20":             "ADX(14) < 20 — ranging/no trend",
    "price_gt_high_20_1_02":    "Close > Highest(High, 20) × 1.02 — breakout",
    "price_lt_low_20_0_98":     "Close < Lowest(Low, 20) × 0.98 — breakdown",
}
```

**Total unique conditions: 53** (22 LONG + 22 SHORT + 9 shared)

### 3.4. Condition Pool Selection During Strategy Generation

When generating a random strategy:
1. Choose direction (LONG or SHORT) with 50% probability each.
2. If LONG: pick 8–16 conditions from `CONDITIONS_LONG + CONDITIONS_SHARED` (31 pool).
3. If SHORT: pick 8–16 conditions from `CONDITIONS_SHORT + CONDITIONS_SHARED` (31 pool).
4. Each condition is computed once per candle using the appropriate indicator function.

---

## 4. Strategy Definition

A strategy is a JSON object:

```json
{
    "id": "strat_001",
    "method": "ga_bayesian",
    "conditions": ["ema_9_gt_21", "rsi_14_gt_50", "price_gt_sma_50", "adx_14_gt_25", ...],
    "threshold": 0.6,
    "sl": 1.2,
    "rr": 2.5,
    "direction": "LONG"
}
```

### 4.1. Entry Logic

- **LONG:** Enter if `(number of True conditions) / (total conditions) >= threshold`
- **SHORT:** Enter if `(number of True bearish conditions) / (total conditions) >= threshold`
  - Uses the **same threshold logic as LONG**. The conditions themselves are already bearish, so no inversion is needed.

### 4.2. Exit Logic

- **Stop Loss (LONG):** `entry_price × (1 - sl/100)`
- **Stop Loss (SHORT):** `entry_price × (1 + sl/100)`
- **Take Profit (LONG):** `entry_price × (1 + sl × rr / 100)`
- **Take Profit (SHORT):** `entry_price × (1 - sl × rr / 100)`

Trade closes when SL or TP is hit. No other exit conditions.

### 4.3. SL/TP Hit Detection (Intra-Candle)

Since we only have OHLCV data (not tick data), when both SL and TP are within a single candle's range:

- **Conservative approach:** Assume SL is hit before TP.
- For LONG: If `candle_low <= sl_price` → SL hit (loss). If `candle_high >= tp_price` → TP hit (win). If both → SL hit first.
- For SHORT: If `candle_high >= sl_price` → SL hit (loss). If `candle_low <= tp_price` → TP hit (win). If both → SL hit first.

### 4.4. Trade Rules

| Rule | Value |
|---|---|
| Only one trade at a time | Yes |
| Minimum trade duration | 45 minutes (3 candles on 15m) |
| Maximum trade duration | 48 hours (close at current price if still open) |
| Cooldown after exit | 4 candles (1 hour on 15m). **Applies to ALL exits:** SL hit, TP hit, or 48-hour timeout. Same cooldown period regardless of exit reason. |
| Entry price | Close price of the signal candle |
| Signal timing | Fire immediately on candle close |

### 4.5. Minimum Trade Duration Handling

When SL is hit within the first 45 minutes:
1. The trade is **marked as "invalid"** (`"valid": false`).
2. The **loss IS applied to the equity curve** (real capital impact).
3. The trade is **excluded from** win rate, total_trades, and RR/day calculations.
4. The trade **IS included in** max_drawdown calculation.
5. The trade is logged with reason `"too_short"`.

```python
# Example trade record
{
    "entry_time": "2026-06-20T14:00:00Z",
    "exit_time": "2026-06-20T14:30:00Z",
    "entry_price": 67200.0,
    "exit_price": 66384.0,
    "direction": "LONG",
    "result": "SL",
    "rr": -1.0,
    "duration_minutes": 30,
    "valid": false,
    "invalid_reason": "too_short"
}
```

---

## 5. Training Process

### 5.1. Training Parameters (Configurable)

| Parameter | Default | Description |
|---|---|---|
| `training_minutes` | 30 | How long to run the training (in minutes) |
| `training_period_months` | 6 | Historical data period for backtesting |
| `training_method` | `"ga_bayesian"` | `"random"` for quick testing, `"ga_bayesian"` for serious optimization |
| `min_conditions` | 8 | Minimum conditions per strategy |
| `max_conditions` | 16 | Maximum conditions per strategy |
| `min_threshold` | 0.5 | Minimum threshold (50%) |
| `max_threshold` | 0.7 | Maximum threshold (70%) |
| `min_sl` | 0.3 | Minimum stop-loss percentage |
| `max_sl` | 3.0 | Maximum stop-loss percentage |
| `min_rr` | 1.0 | Minimum risk-reward ratio |
| `max_rr` | 5.0 | Maximum risk-reward ratio |
| `min_trades_per_day` | 0.5 | If avg trades/day < this, apply penalty |
| `max_trades_per_day` | 10 | If avg trades/day > this, disqualify |
| `min_win_rate` | 0.35 | Minimum win rate to qualify |
| `max_drawdown` | 0.50 | Maximum drawdown to qualify (50%) |
| `cooldown_after_exit` | 4 | Number of candles to wait after exit before re-entering |
| `trading_fee_pct` | 0.1 | Trading fee per side (0.1% = Binance standard) |

### 5.2. GA + Bayesian Optimization Parameters

| Parameter | Default | Description |
|---|---|---|
| `ga_population_size` | 200 | Population per generation |
| `ga_generations` | 30 | Number of generations |
| `ga_elite_count` | 5 | Top strategies preserved unchanged each generation |
| `ga_mutation_prob` | 0.2 | Probability of mutation |
| `ga_crossover_prob` | 0.8 | Probability of crossover |
| `bayesian_n_trials` | 2000 | Number of Bayesian optimization trials |
| `bayesian_startup_trials` | 100 | Random trials before Bayesian model kicks in |

### 5.3. Training Methods

#### Mode 1: GA + Bayesian Optimization (Default)

```
Phase 1 — Genetic Algorithm (Global Exploration)
  Library: DEAP
  Population: 200 strategies per generation
  Generations: 20–30
  Total strategies tested: ~4,000–6,000
  Selection: Tournament selection (size 3)
  Crossover: See Section 5.3.1 for exact crossover logic
  Mutation: See Section 5.3.2 for exact mutation logic
  Elitism: Keep top 5 strategies unchanged in each generation
  Objective: Maximize score (RR/day × drawdown penalty)

Phase 2 — Bayesian Optimization (Local Refinement)
  Library: Optuna (Tree-structured Parzen Estimator / TPE)
  Trials: 1,000–2,000
  Seeding: Use top 10 strategies from Phase 1 as initial points
  Objective: Maximize the same score as Phase 1

Combined: ~6,000–8,000 strategies tested in 25–30 minutes
```

##### 5.3.1. Crossover Logic

When two parent strategies are crossed over (probability: `ga_crossover_prob = 0.8`):

1. **Conditions:** Take the first half of conditions from Parent 1 and the second half from Parent 2. If the parents have different numbers of conditions, split at the midpoint of the shorter list and pad with unique conditions from the longer list.
2. **Threshold:** Average the two parents' values: `child_threshold = (p1_threshold + p2_threshold) / 2`
3. **Stop-loss:** Average: `child_sl = (p1_sl + p2_sl) / 2`
4. **Risk-reward:** Average: `child_rr = (p1_rr + p2_rr) / 2`
5. **Direction:** Inherit from the parent with the higher score (or randomly if equal).

```python
def crossover(parent1, parent2):
    # Conditions: first half from p1, second half from p2
    mid = min(len(parent1['conditions']), len(parent2['conditions'])) // 2
    child_conditions = parent1['conditions'][:mid] + parent2['conditions'][mid:]
    # Remove duplicates, fill with random if needed
    child_conditions = list(dict.fromkeys(child_conditions))
    while len(child_conditions) < MIN_CONDITIONS:
        child_conditions.append(random_condition(parent1['direction']))

    return {
        'conditions': child_conditions,
        'threshold': (parent1['threshold'] + parent2['threshold']) / 2,
        'sl': (parent1['sl'] + parent2['sl']) / 2,
        'rr': (parent1['rr'] + parent2['rr']) / 2,
        'direction': parent1['direction'] if parent1['score'] >= parent2['score'] else parent2['direction'],
    }
```

##### 5.3.2. Mutation Logic

When a strategy is mutated (probability: `ga_mutation_prob = 0.2`), **randomly choose one** of the following four operations:

1. **Replace one condition:** Pick a random condition index and replace it with a new random condition from the appropriate pool (LONG or SHORT, matching the strategy's direction).
2. **Adjust threshold:** Add or subtract `±0.05` (clamped to `[min_threshold, max_threshold]`).
3. **Adjust stop-loss:** Add or subtract `±0.2%` (clamped to `[min_sl, max_sl]`).
4. **Adjust risk-reward:** Add or subtract `±0.5` (clamped to `[min_rr, max_rr]`).

```python
def mutate(strategy):
    if random.random() > GA_MUTATION_PROB:
        return strategy  # No mutation

    mutation_type = random.choice(['condition', 'threshold', 'sl', 'rr'])

    if mutation_type == 'condition':
        idx = random.randint(0, len(strategy['conditions']) - 1)
        pool = get_condition_pool(strategy['direction'])
        new_cond = random.choice([c for c in pool if c not in strategy['conditions']])
        strategy['conditions'][idx] = new_cond
    elif mutation_type == 'threshold':
        delta = random.choice([-0.05, 0.05])
        strategy['threshold'] = clamp(strategy['threshold'] + delta, MIN_THRESHOLD, MAX_THRESHOLD)
    elif mutation_type == 'sl':
        delta = random.choice([-0.2, 0.2])
        strategy['sl'] = clamp(strategy['sl'] + delta, MIN_SL, MAX_SL)
    elif mutation_type == 'rr':
        delta = random.choice([-0.5, 0.5])
        strategy['rr'] = clamp(strategy['rr'] + delta, MIN_RR, MAX_RR)

    return strategy
```

#### Mode 2: Pure Random Search (Quick Testing)

```
While timer < training_minutes:
  1. Generate a random strategy
  2. Run backtest
  3. Calculate score
  4. Track best

Use for: Quick smoke tests (2–5 min), debugging, comparison baseline.
```

### 5.4. Training Loop (GA + Bayesian)

```
1. Load historical data for the training period.
2. Compute all 53 conditions on the full dataset.
3. Drop rows with NaN (warmup period for longest indicator).
4. Initialize best_score = -inf, best_strategy = None.

Phase 1: Genetic Algorithm
  5. Create initial population of 200 random strategies.
  6. For each generation (1 to ga_generations):
     a. Evaluate all individuals (run backtest, calculate score).
     b. Apply disqualification rules (see 5.5).
     c. Select parents via tournament.
     d. Apply crossover and mutation.
     e. Preserve elite strategies.
     f. Log generation stats.
  7. Extract top 10 strategies from GA.

Phase 2: Bayesian Optimization
  8. Seed Optuna study with top 10 GA strategies.
  9. For each trial (1 to bayesian_n_trials):
     a. Suggest strategy parameters.
     b. Run backtest and calculate score.
     c. Apply disqualification rules.
     d. Track best score.
  10. Extract best strategy from Bayesian optimization.

11. Compare GA best vs Bayesian best → select overall winner.
12. Save best_strategy to models/best_strategy.json.
13. Save top 500 strategies to models/top_strategies.json.
14. Run efficiency analysis on all conditions used (see Section 6).
15. Generate and display efficiency report.
```

### 5.5. Disqualification Rules

Applied after each backtest:

| Rule | Threshold | Action |
|---|---|---|
| Too many trades/day | avg_trades_per_day > 10 | Disqualify (score = -inf) |
| Too few trades/day | avg_trades_per_day < 0.5 | Apply penalty |
| Low win rate | win_rate < 35% | Disqualify (score = -inf) |
| Excessive drawdown | max_drawdown > 50% | Disqualify (score = -inf) |

### 5.6. Scoring Formula

```
base_score = rr_per_day  (total_rr / trading_days)

Drawdown penalty:
  if max_drawdown < 0.15:  penalty = 1.0  (no penalty)
  if max_drawdown >= 0.15 and <= 0.50:
      penalty = 1.0 - ((max_drawdown - 0.15) / 0.35)
  if max_drawdown > 0.50:  disqualify

final_score = base_score × penalty
```

**Note:** Only RR/day is used (no Sharpe/Sortino ratio). Simplicity prioritized.

### 5.7. Trading Fees in Backtesting

Each trade incurs a **0.1% fee per side** (entry + exit = 0.2% round-trip):

```python
# Applied to each trade's P&L
fee_cost = entry_price * 0.001  # 0.1% of entry
# Deducted from trade profit or added to trade loss
net_pnl = gross_pnl - (2 * fee_cost)  # entry + exit
```

### 5.8. Progress Tracking During Training

Console output during training:

```
[2026-06-19 14:00:00] Training started. Method: GA + Bayesian | Period: 2025-12-19 to 2026-06-19
[2026-06-19 14:00:01] Phase 1: Genetic Algorithm (pop=200, gen=30)
[2026-06-19 14:00:05] Gen 1/30 | Best: 1.85 | Avg: 1.12 | New best! | Elapsed: 5s
[2026-06-19 14:00:35] Gen 5/30 | Best: 2.10 | Avg: 1.34 | Elapsed: 35s | ETA: 3m 15s
...
[2026-06-19 14:05:00] Gen 30/30 | Best: 2.45 | Avg: 1.54 | Elapsed: 5m 0s
[2026-06-19 14:05:01] Phase 2: Bayesian Optimization (trials=2000)
[2026-06-19 14:05:10] Trial 100/2000 | Best: 2.48 | Elapsed: 10s | ETA: 18m 50s
...
[2026-06-19 14:25:00] Trial 2000/2000 | Best: 2.68 | Elapsed: 20m 0s
[2026-06-19 14:25:01] Training finished.
   GA best score: 2.45
   Bayesian best score: 2.68 (trial #1847)
   Final best score: 2.68
   Total strategies tested: 7,842
   Time elapsed: 25m 1s
   Average: 5.22 strats/sec
   Best strategy saved to models/best_strategy.json
```

---

## 6. Efficiency Analysis and Alerts

After training finishes, analyze all conditions used across all tested strategies.

### 6.1. Per-Condition Statistics

```python
condition_stats[condition] = {
    "used_count": 0,              # total times used in all strategies
    "used_in_top_10_percent": 0,  # times used in strategies that scored in top 10%
    "avg_rr_per_day": 0.0,       # average RR/day of strategies that used this condition
    "avg_win_rate": 0.0,         # average win rate of strategies that used this condition
}
```

### 6.2. Efficiency Score

```
global_avg_rr_per_day = average of rr_per_day across all tested strategies
efficiency_score = avg_rr_per_day_condition / global_avg_rr_per_day
```

### 6.3. Alert Levels

| Efficiency Score | Alert Level | Action |
|---|---|---|
| < 0.3 | 🔴🔴 CRITICAL | **Automatically remove** from condition pool (see 6.3.1) |
| 0.3 – 0.5 | 🔴 ALERT | Mark as inefficient (lower priority in future generation) |
| 0.5 – 0.7 | ⚠️ WARNING | Keep but note in report |
| 0.7 – 1.3 | ✅ OK | Keep as normal |
| > 1.3 | ⭐ STRONG | Increase priority in future generation |

#### 6.3.1. Automatic Condition Removal

Conditions with an efficiency score < 0.3 are **automatically removed** from the condition pool at the end of each training session. This ensures future training runs don't waste time on consistently underperforming conditions.

**How it works:**
1. After the efficiency report is generated, scan all conditions.
2. If any condition has `efficiency_score < 0.3`, remove it from the active condition pool.
3. Log the removal:
   ```
   [EFFICIENCY] Condition 'stoch_k_lt_80' removed from pool (efficiency: 0.28).
   ```
4. Save the updated condition pool to `models/removed_conditions.json` for audit trail.
5. The removed condition is also excluded from the GA and Bayesian optimization search space in subsequent training runs.

**Manual restoration:**
The user can restore a removed condition by:
1. Editing `models/removed_conditions.json` to remove the entry, OR
2. Deleting the file entirely to reset all conditions to defaults.

On the next training run, the system checks `removed_conditions.json` and excludes those conditions from the pool.

### 6.4. Report Format

```
============================================================
EFFICIENCY REPORT - Training Session: 2026-06-19 14:30:00
============================================================
Period: 2025-12-19 to 2026-06-19
Method: GA + Bayesian Optimization
Strategies tested: 7,842
Best score: 2.68
Best strategy ID: strat_GA_BO_001

CRITICAL ALERTS (remove these conditions):

🔴🔴 'stoch_k_lt_80' (LONG)
Used in: 12 strategies
Avg RR/day: 0.42 (global: 1.85)
Avg win rate: 28%
→ This condition consistently underperforms. Remove.

ALERTS (consider removing):

🔴 'cci_14_gt_100' (SHORT)
Used in: 8 strategies
Avg RR/day: 0.68 (global: 1.85)
Avg win rate: 34%
→ This condition performs below average. Consider removing.

WARNINGS (insufficient data):

⚠️ 'volume_gt_sma_20_2_0'
Used in: 3 strategies
Avg RR/day: 1.72 (global: 1.85)
→ Insufficient data. Keep but monitor.

STRONG CONDITIONS (highly effective):

⭐ 'ema_9_gt_21' (LONG)
Used in: 234 strategies
Avg RR/day: 2.45 (global: 1.85)
Avg win rate: 52%
→ This condition consistently appears in top strategies.

⭐ 'adx_14_gt_25' (shared)
Used in: 189 strategies
Avg RR/day: 2.38 (global: 1.85)
Avg win rate: 50%
→ This condition consistently appears in top strategies.

============================================================
```

---

## 7. Validation (Full Backtest)

After training, the user can run a full backtest on the selected strategy:

```bash
python validation.py --symbol BTC/USDT --period 12
```

This runs the best strategy on the **validation period** (separate from training) and reports:

| Metric | Description |
|---|---|
| Total trades | Number of valid trades |
| Win rate (%) | Winning trades / total valid trades |
| Profit factor | Gross profit / gross loss |
| Max drawdown (%) | Peak-to-trough decline |
| Sharpe ratio | Risk-adjusted return |
| RR per day | Total RR / trading days |
| Exit breakdown | SL vs TP exits |
| Invalid trades | Trades with duration < 45 min |
| Time-closed trades | Trades closed at 48h limit |
| Fees paid | Total trading fees deducted |

### 7.1. Acceptance Criteria (Optional)

| Metric | Threshold |
|---|---|
| Win rate | ≥ 35% |
| Max drawdown | ≤ 20% |
| Profit factor | ≥ 1.3 |

If the strategy fails these criteria, the user can retrain or adjust parameters.

---

## 8. Live Signal Generator

### 8.1. CLI Command

```bash
python live_signal.py --symbol BTC/USDT
```

### 8.2. Startup Sequence

1. Load the best strategy from `models/best_strategy.json`.
2. Load the state from `state.json` (if exists).
3. Check for missed signals (see Section 8.6).
4. Enter the main loop.

### 8.3. Main Loop (Every 15 Minutes)

Executed at the **exact moment each candle closes** (`:00`, `:15`, `:30`, `:45`):

1. Fetch latest 500 candles (to ensure all indicators have valid values).
2. Compute all conditions for the strategy.
3. **If not in position:**
   a. Check cooldown — if `cooldown_remaining > 0`, decrement and skip.
   b. Check if entry conditions are met (`True / total >= threshold`).
   c. If signal: calculate entry price (close of current candle), SL, TP.
   d. Send Discord alert.
   e. Update `state.json`.
4. **If in position:**
   a. Check if SL or TP is hit (using candle high/low, conservative SL-first rule).
   b. If SL or TP hit: close position, send exit alert, **start 4-candle cooldown** (applies to ALL exit types: SL, TP, or timeout).
   c. If duration > 48 hours: send "consider closing manually" alert, close at current price, **start 4-candle cooldown**.
5. Sleep until next candle close.

### 8.4. Discord Alert Format

**Entry Signal:**
```
🟢 NEW SIGNAL DETECTED: BTC/USDT

Action: LONG
Entry: $67,200.00
Stop Loss: $66,384.00 (1.2%)
Take Profit: $68,880.00 (2.5 RR)
Risk-Reward: 2.5
Confidence: 72% (6/10 conditions met)

Strategy ID: strat_GA_BO_001
RR/day: 2.68
Win rate (historical): 52%

⚠️ This is a signal. Execute manually with your own position size.
```

**Exit Signal:**
```
🔴 POSITION CLOSED: BTC/USDT

Result: TP HIT ✅
Entry: $67,200.00
Exit: $68,880.00
Profit: +2.5% (+1.0 RR)
Duration: 2h 15m
```

**Cooldown Expired:**
```
⏳ COOLDOWN EXPIRED: BTC/USDT

Cooldown period (1 hour) has ended.
The bot is now actively monitoring for new signals.
```

**Missed Signal Recovery:**
```
📋 [RECOVERY] Missed signal detected while offline:

BTC/USDT LONG
Signal time: 2026-06-20 14:00:00
Entry: $67,200.00
Stop Loss: $66,384.00
Take Profit: $68,880.00
Risk-Reward: 2.5

ℹ️ This signal has expired. Reported for your information only.
```

### 8.5. State Management

State is stored in `state.json`:

```json
{
    "symbol": "BTC/USDT",
    "in_position": true,
    "entry_price": 67200.0,
    "entry_time": "2026-06-20T14:00:00Z",
    "sl": 66384.0,
    "tp": 68880.0,
    "direction": "LONG",
    "strategy_id": "strat_GA_BO_001",
    "cooldown_remaining": 0,
    "cooldown_expiry_time": null,
    "last_check_time": "2026-06-20T14:00:00Z"
}
```

This allows the bot to resume tracking a position across restarts.

### 8.6. Missed Signal Recovery

When the bot starts, it:

1. Reads `last_check_time` from `state.json`.
2. Fetches all candles between `last_check_time` and now.
3. For each missed candle, re-runs the strategy conditions on the data that **would have been available at that time**.
4. If a signal was generated, records it as a missed signal.
5. Sends a recovery notification with the **original** entry price, SL, TP, and timestamp.
6. **Does NOT adjust** entry price based on current market price.
7. **Does NOT suggest** trading the missed signal. It is informational only.
8. Marks the signal as "reported" so it is not sent again.

---

## 9. Logging

All logs are written to both **console (stdout)** and **file** in the `logs/` folder.

### 9.1. Training Log

```
[2026-06-19 14:00:00] Training started. Method: GA + Bayesian | Symbol: BTC/USDT | Period: 2025-12-19 to 2026-06-19
[2026-06-19 14:00:01] Phase 1: Genetic Algorithm (pop=200, gen=30, mutation=0.2, crossover=0.8)
[2026-06-19 14:00:05] Gen 1/30 | Best: 1.85 | Avg: 1.12 | New best! | Elapsed: 5s
[2026-06-19 14:05:00] Gen 30/30 | Best: 2.45 | Avg: 1.54 | Elapsed: 5m 0s
[2026-06-19 14:05:01] Phase 2: Bayesian Optimization (trials=2000, startup=100)
[2026-06-19 14:25:00] Trial 2000/2000 | Best: 2.68 | Elapsed: 20m 0s
[2026-06-19 14:25:01] Training finished.
[2026-06-19 14:25:01] Best strategy saved to models/best_strategy.json
[2026-06-19 14:25:01] Top 500 strategies saved to models/top_strategies.json
[2026-06-19 14:25:02] Efficiency report generated.
```

### 9.2. Live Signal Log

```
[2026-06-20 14:00:00] Signal: BTC/USDT LONG at $67,200.00, SL $66,384.00, TP $68,880.00
[2026-06-20 14:00:00] Discord alert sent.
[2026-06-20 15:15:00] Position still open. Duration: 1h 15m.
[2026-06-20 17:30:00] Position closed: TP hit at $68,880.00. Profit: +2.5% (1.0 RR)
[2026-06-20 17:30:00] Cooldown started: 4 candles (1h). Expires at 18:30.
[2026-06-20 18:30:00] Cooldown expired. Monitoring for new signals.
[2026-06-20 18:30:00] Discord cooldown alert sent.
```

---

## 10. File Structure

```
crypto_trading_bot/
├── data/
│   └── btc_usdt_15m.csv            # Cached OHLCV data
├── logs/
│   ├── training_2026-06-19.log
│   ├── live_2026-06-20.log
│   └── validation_2026-06-19.log
├── models/
│   ├── best_strategy.json           # The best strategy found
│   ├── top_strategies.json          # Top 500 strategies by score
│   └── condition_efficiency.json    # Efficiency report
├── state.json                       # Live state (position, cooldown, etc.)
├── .env                             # Discord webhook URL (gitignored)
├── config.py                        # All configuration parameters
├── conditions.py                    # All 53 conditions (LONG, SHORT, shared)
├── indicators.py                    # Indicator computation (TA-Lib / pandas_ta)
├── strategy.py                      # Strategy generation, scoring
├── backtest.py                      # Backtest engine
├── training.py                      # Training loop (random or GA+Bayesian)
├── genetic_optimizer.py             # GA implementation (DEAP)
├── bayesian_optimizer.py            # Bayesian optimization (Optuna)
├── validation.py                    # Full backtest validation
├── live_signal.py                   # Live signal generator
├── data_fetcher.py                  # OHLCV download from Binance (ccxt)
├── discord_bot.py                   # Discord webhook sender
├── efficiency.py                    # Condition efficiency analysis
├── requirements.txt
└── README.md
```

---

## 11. Configuration (`config.py`)

```python
import os
from dotenv import load_dotenv

load_dotenv()

# === Timeframe ===
TIMEFRAME = "15m"  # "5m", "15m", "1h"

# === Symbol ===
SYMBOL = "BTC/USDT"

# === Training ===
TRAINING_MINUTES = 30
TRAINING_PERIOD_MONTHS = 6
TRAINING_METHOD = "ga_bayesian"  # "random" or "ga_bayesian"

# === Strategy Generation ===
MIN_CONDITIONS = 8
MAX_CONDITIONS = 16
MIN_THRESHOLD = 0.5
MAX_THRESHOLD = 0.7
MIN_SL = 0.3   # percent
MAX_SL = 3.0   # percent
MIN_RR = 1.0
MAX_RR = 5.0

# === GA Parameters ===
GA_POPULATION_SIZE = 200
GA_GENERATIONS = 30
GA_ELITE_COUNT = 5
GA_MUTATION_PROB = 0.2
GA_CROSSOVER_PROB = 0.8

# === Bayesian Parameters ===
BAYESIAN_N_TRIALS = 2000
BAYESIAN_STARTUP_TRIALS = 100

# === Qualification / Disqualification ===
MIN_TRADES_PER_DAY = 0.5
MAX_TRADES_PER_DAY = 10
MIN_WIN_RATE = 0.35    # 35%
MAX_DRAWDOWN = 0.50    # 50%
DRAWDOWN_PENALTY_START = 0.15  # 15%
DRAWDOWN_PENALTY_END = 0.50   # 50%

# === Trade Parameters ===
MIN_TRADE_DURATION_MINUTES = 45
MAX_TRADE_DURATION_HOURS = 48
COOLDOWN_CANDLES = 4
TRADING_FEE_PCT = 0.1  # per side

# === Live Mode ===
LIVE_CHECK_INTERVAL_SECONDS = 900  # 15 minutes
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# === Paths ===
DATA_CACHE_DIR = "data/"
MODEL_DIR = "models/"
LOG_DIR = "logs/"
STATE_FILE = "state.json"

# === Indicator Library ===
# Auto-detected: TA-Lib preferred, pandas_ta fallback
```

---

## 12. Requirements (`requirements.txt`)

```
ccxt>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
python-dotenv>=1.0.0
deap>=1.4.0
optuna>=3.5.0
pandas_ta>=0.3.14
```

**Note:** TA-Lib requires a C compiler or pre-built wheel. If installation fails, the system falls back to pandas_ta automatically. On Windows, install TA-Lib from a `.whl` file or use `pip install pandas_ta` as the fallback.

---

## 13. User Workflow

### Setup
1. Clone repository.
2. Install dependencies: `pip install -r requirements.txt`
3. (Optional) Install TA-Lib for faster backtesting.
4. Create `.env` with `DISCORD_WEBHOOK_URL=<your_webhook_url>`.
5. Set config parameters in `config.py`.

### Train
```bash
# Serious training (GA + Bayesian, default)
python training.py --symbol BTC/USDT

# Quick smoke test (random search, 5 min)
python training.py --symbol BTC/USDT --method random --minutes 5
```
- Wait for training to finish.
- Review console logs and efficiency report.
- Best strategy saved to `models/best_strategy.json`.

### Validate
```bash
python validation.py --symbol BTC/USDT --period 12
```
- Runs full backtest on **validation period** (separate from training).
- Review metrics against acceptance criteria.
- If unsatisfied, retrain or adjust parameters.

### Live
```bash
python live_signal.py --symbol BTC/USDT
```
- Bot checks at every candle close (every 15 minutes).
- Sends Discord alerts when signals are generated.
- Sends cooldown-expired alerts.
- Handles missed signals on startup.

---

## 14. Key Design Decisions Summary

| Decision | Rationale |
|---|---|
| GA + Bayesian optimization | Dramatically better than random search for the same time budget |
| Separate LONG/SHORT conditions | Bearish conditions are genuinely bearish, not inverted bullish |
| Same threshold logic for LONG/SHORT | Conditions are already directional, no inversion needed |
| 0.1% trading fees per side | Realistic Binance spot fees |
| RR/day as sole scoring metric | Simple, intuitive, rewards consistent edge |
| Drawdown penalty | Punishes strategies with large drawdowns |
| Invalid trades: loss applied, metrics excluded | Honest equity tracking without inflating performance metrics |
| Conservative SL-first within candle | Avoids overestimating TP hits |
| Drop NaN rows (warmup) | Clean backtest start after indicator warmup |
| Non-overlapping train/validation | Prevents data leakage |
| TA-Lib primary, pandas_ta fallback | Speed for backtesting, portability for deployment |
| Top 500 strategies saved | Enough for analysis without excessive storage |
| Cooldown expiry notification | User knows when bot is actively monitoring again |
| Missed signals: report only, no adjustment | Informational, no false confidence in stale signals |
| Single symbol (BTC/USDT) | Simplicity for MVP |
| No paper trading | Signal-only model; backtest covers performance estimation |

---

## 15. Implementation Priority

| Phase | Description | Files |
|---|---|---|
| **Phase 1** | Data fetcher + conditions + indicators | `data_fetcher.py`, `conditions.py`, `indicators.py`, `config.py` |
| **Phase 2** | Backtest engine + strategy generation | `backtest.py`, `strategy.py` |
| **Phase 3** | Training loop (random mode first) | `training.py` |
| **Phase 4** | GA + Bayesian optimization | `genetic_optimizer.py`, `bayesian_optimizer.py` |
| **Phase 5** | Efficiency analysis + reports | `efficiency.py` |
| **Phase 6** | Validation (full backtest) | `validation.py` |
| **Phase 7** | Live signal generator + Discord | `live_signal.py`, `discord_bot.py` |
| **Phase 8** | Missed signal recovery + state management | Updates to `live_signal.py`, `state.json` |

---

## 16. Discord Integration

- **Method:** HTTP POST to Discord webhook URL via `requests` library.
- **Storage:** Webhook URL stored in `.env` file, loaded via `python-dotenv`.
- **Format:** Discord embeds with color-coded fields (green for entry, red for exit, yellow for cooldown).
- **Rate Limits:** Discord allows ~5 requests per 2 seconds. Handle `429` responses with `retry_after` backoff.
- **Colors:** Decimal integers (e.g., `0x00FF00` = `65280` for green).
- **Failure Handling:** If a Discord message fails to send (network error, 429 rate limit, timeout, etc.), **log the error and continue to the next cycle**. Do NOT retry immediately. The next signal will send a new notification. This prevents the bot from getting stuck on a failed send.

```python
def send_discord_alert(message, webhook_url):
    try:
        response = requests.post(webhook_url, json=message, timeout=10)
        if response.status_code == 204:
            logger.info("Discord alert sent.")
        elif response.status_code == 429:
            retry_after = response.json().get('retry_after', 5)
            logger.warning(f"Discord rate limited. Retry after {retry_after}s. Skipping this cycle.")
        else:
            logger.error(f"Discord send failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Discord send error: {e}. Continuing to next cycle.")
```
