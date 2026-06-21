# Bug Fixes Spec: Training & Validation Pipeline

**Created:** 2026-06-21
**Status:** Ready for implementation
**Priority order:** Issues 1, 2, 3, 5, 6, 7 (Issue 4 already fixed)

---

## Issue 1: RR/day Miscalculated in Validation (CRITICAL)

### Problem
In `validation.py`, a strategy with 1 trade in 180 days and Total RR = 7.38 reports RR/day = 7.38. This is because `backtest.py` counts `trading_days` as the number of unique days with at least one trade (1 day), not the total days in the period (180 days).

### Current Code (`backtest.py` lines 291-299)
```python
if trades:
    trade_dates = set()
    for t in trades:
        trade_date = datetime.fromisoformat(t["entry_time"]).date()
        trade_dates.add(trade_date)
    trading_days = max(len(trade_dates), 1)
else:
    trading_days = 1  # Avoid division by zero
rr_per_day = total_rr / trading_days
```

### Fix
**Scope:** Only change `validation.py`. Do NOT change `backtest.py` — during training, the `LOW_TRADES_PENALTY` (0.7 for avg_trades/day <= 1.5) already handles low-trade penalization in the score formula. Changing `backtest.py` would double-penalize during training.

**In `validation.py`:**
- After running `backtest_strategy()`, override `rr_per_day` with the correct formula:
  ```python
  # Recalculate RR/day using total period days (not just days with trades)
  total_days = (clean_df["timestamp"].iloc[-1] - clean_df["timestamp"].iloc[0]).days
  total_days = max(total_days, 1)  # Avoid division by zero
  rr_per_day = results["total_rr"] / total_days
  ```
- Use **calendar days** (first candle to last candle), not candle count / candles-per-day
- **Show only the period-adjusted RR/day** — do NOT show the old trading-days-based metric. The old metric is misleading and showing both would confuse users.
- Also fix `avg_trades_per_day` in the validation report using the same total_days divisor (consistency: both metrics use the same denominator)
  ```python
  avg_trades_per_day = results["valid_trades"] / total_days
  ```
- Update the report key name to `rr_per_day` (keep the same key for compatibility, but the value is now period-adjusted)
- Log the calculation clearly:
  ```
  RR/day: 0.0410  (total RR 7.38 / 180 calendar days)
  ```

### Files Modified
- `validation.py`: Override RR/day calculation after backtest

### Verification
- Run validation with a strategy that has very few trades
- Verify RR/day is now total_rr / total_period_days
- Verify avg_trades_per_day is also recalculated using total_period_days
- Verify ONLY the period-adjusted metrics are shown (no old trading-days metric)

---

## Issue 2: Best Strategy Overwritten Without Comparison (CRITICAL)

### Problem
At the end of each training run, `_save_results()` calls `save_strategy(best)` which unconditionally overwrites `models/best_strategy.json`. If the previous training run found a strategy with score 3.0 and the new run finds one with score 2.5, the better strategy is lost.

### Current Code (`training.py` `_save_results`)
```python
def _save_results(self, best: dict) -> None:
    if not best:
        logger.warning("No valid strategy found during training.")
        return
    # ...
    save_strategy(best)  # Unconditional overwrite
```

### Fix
**Scope:** Change `training.py` only. Keep `save_strategy()` in `strategy.py` as a simple overwrite (no business logic in the save function).

**In `_save_results()`:**
```python
# Compare with existing best before saving
saved_new = False
try:
    existing_best = load_strategy()
except Exception:
    existing_best = None

if existing_best is None:
    logger.info("No existing best strategy found. Saving new strategy.")
    save_strategy(best)
    saved_new = True
elif "score" not in existing_best:
    logger.warning("Existing strategy has no 'score' key. Overwriting with new strategy.")
    save_strategy(best)
    saved_new = True
elif existing_best["score"] >= best.get("score", float("-inf")):
    logger.info(
        f"Keeping existing best strategy (score: {existing_best['score']:.4f}) "
        f"-- new strategy score ({best.get('score', float('-inf')):.4f}) is not higher."
    )
else:
    logger.info(f"Saving new best strategy (score: {best.get('score', float('-inf')):.4f})")
    save_strategy(best)
    saved_new = True

# Save top 500 strategies (always overwrite — this is per-run, not cumulative)
# ... existing top_strategies code ...

return saved_new  # Return whether a new best was saved (for _log_finish)
```

**Also update `_log_finish()`:**
The finish log currently says "Best strategy saved to ..." unconditionally. Update it to reflect the actual outcome:
```python
if saved_new:
    logger.info(f"  New best strategy saved to {config.MODEL_DIR / 'best_strategy.json'}")
else:
    logger.info(f"  Existing best strategy retained (score: {existing_best['score']:.4f})")
```

**Note on `top_strategies.json`:** This file is a per-run snapshot of the top 500 strategies found in this training session. It should always be overwritten (not merged). The per-run top strategies list is useful for analysis but is separate from the "best strategy" question.

**Edge cases:**
- If `existing_best` is `None` (file doesn't exist): always save the new strategy
- If `existing_best` has no `score` key: treat as invalid, save the new strategy
- If the file is corrupted (JSON decode error): treat as None, save the new strategy
- If both scores are `-inf` (both disqualified): keep existing, log a warning

### Files Modified
- `training.py`: Add comparison logic in `_save_results()`
- Import `load_strategy` (already imported)

### Verification
- Run training twice with different `--minutes` values
- Verify the second run doesn't overwrite a better first-run strategy
- Verify the log message shows the comparison

---

## Issue 3: Training Time Not Used Correctly (HIGH)

### Problem
The GA is capped at 30 generations (`GA_GENERATIONS`) and Bayesian at 2000 trials (`BAYESIAN_N_TRIALS`). With a 60-minute budget, both hit their caps early (GA in ~2.5 min, Bayesian in ~15 min) and the remaining ~42 minutes are wasted.

### Current Code (`training.py` `_run_ga_bayesian`)
```python
ga_generations = min(config.GA_GENERATIONS, max_ga_gens)  # Cap at 30
n_trials = min(config.BAYESIAN_N_TRIALS, max_trials)       # Cap at 2000
```

### Fix
**Scope:** Change `training.py` (time allocation logic) and `config.py` (new parameters).

**New config parameters:**
```python
GA_MAX_GENERATIONS = 200          # Safety cap (rarely hit)
GA_TIME_BUDGET_PERCENT = 0.5      # Use 50% of training time for GA
BAYESIAN_MAX_TRIALS = 10000       # Safety cap (rarely hit)
BAYESIAN_MIN_TRIALS = 50          # Minimum trials to be useful
```

**GA phase changes:**
- `training.py` calculates the GA time budget (50% of total training time) and passes it to the GA
- `genetic_optimizer.py` `run()` receives a `time_limit_seconds` parameter
- Inside the GA loop, each generation checks `if time.time() - start_time >= time_limit_seconds: break`
- Also accepts `max_generations` as a safety cap (default 200) — whichever limit is hit first stops the GA
- Responsibility is clear: `training.py` defines the schedule, `genetic_optimizer.py` executes within it

```python
# In training.py:
ga_time_budget = self.training_minutes * 60 * config.GA_TIME_BUDGET_PERCENT
ga_best, ga_top_10 = ga.run(time_limit_seconds=ga_time_budget, max_generations=config.GA_MAX_GENERATIONS)

# In genetic_optimizer.py:
def run(self, time_limit_seconds=None, max_generations=200):
    import time as _time
    start_time = _time.time()
    for gen in range(max_generations):
        self._run_generation(gen)
        if time_limit_seconds and (_time.time() - start_time) >= time_limit_seconds:
            logger.info(f"GA: Time budget exhausted at generation {gen+1}. Stopping.")
            break
    else:
        logger.info(f"GA: Reached max generation cap ({max_generations}). Stopping.")
```

**GA logging update:** The current code logs `Gen {gen}/{self.generations}` which shows the fixed cap. With time-based stopping, the total is unknown at start. Change to:
```python
logger.info(f"GA Gen {gen+1} | Best score: {gen_best:.4f} | Avg score: {gen_avg:.4f} | Tested: {len(self.all_strategies)}{marker}")
```
(Remove the `/{self.generations}` suffix — the user sees the gen count incrementing and knows it stops when time runs out or cap is hit.)

**Bayesian phase changes:**
- Use Optuna's built-in `timeout` parameter instead of dynamically calculating n_trials:
  ```python
  # In training.py, calculate remaining time AFTER efficiency analysis completes
  remaining = total_time - (time.time() - self.start_time)
  
  # In bayesian_optimizer.py, add timeout parameter to run():
  def run(self, seed_strategies=None, timeout_seconds=None):
      study.optimize(
          objective,
          timeout=timeout_seconds,  # Optuna stops when time runs out
          n_trials=self.n_trials,   # Safety cap (default 10000)
          show_progress_bar=False,
      )
  ```
- Keep `BAYESIAN_MAX_TRIALS` as a safety cap (default 10000)
- **Fix startup_trials calculation:** Currently `startup_trials = n_trials // 5`. With n_trials=10000, this gives 2000 startup trials (way too many). Change to: `startup_trials = min(config.BAYESIAN_STARTUP_TRIALS, n_trials // 5)` where `BAYESIAN_STARTUP_TRIALS = 100` (keep existing value). This caps startup at 100 regardless of n_trials.
- Log the allocation:
  ```
  [INFO] Bayesian: Running with {remaining:.0f}s timeout, max {self.n_trials} trials, startup={self.startup_trials}
  ```
- **Important:** Calculate remaining time AFTER efficiency analysis completes (see Cross-cutting issue below)

### Files Modified
- `config.py`: Add `GA_MAX_GENERATIONS = 200`, `GA_TIME_BUDGET_PERCENT = 0.5`, `BAYESIAN_MAX_TRIALS = 10000`, `BAYESIAN_MIN_TRIALS = 50`. Remove or rename `GA_GENERATIONS` and `BAYESIAN_N_TRIALS`.
- `training.py`: Rewrite time allocation logic in `_run_ga_bayesian()`. Update `_log_finish()` to accept `saved_new` flag.
- `genetic_optimizer.py`: Add `time_limit_seconds` and `max_generations` parameters to `run()`. Update generation logging to remove fixed total.
- `bayesian_optimizer.py`: Add `timeout_seconds` parameter to `run()`. Fix `startup_trials` calculation.

### Verification
- Run with `--minutes 5` and `--minutes 15`
- Verify 15-minute run tests significantly more strategies than 5-minute run
- Verify GA generations scale with time (e.g., ~15 gens for 5 min, ~45 gens for 15 min)
- Verify Bayesian trials scale with remaining time

---

## Issue 4: Condition Removal Has No Effect (ALREADY FIXED)

**Status:** Fixed in previous conversation. Efficiency analysis now runs between GA and Bayesian phases. Conditions with efficiency < 0.3 are removed before Bayesian starts. Conditions with efficiency 0.3-0.5 get 0.5x selection weight.

**No further changes needed.**

---

## Issue 5: Training Results Don't Match README (MEDIUM)

### Problem
The README contains outdated information:
- Says "6 months" of training data (actual: 12 months per `TRAINING_PERIOD_MONTHS = 12`)
- Says `MAX_RR = 5.0` (actual: 8.0)
- Says `LOW_TRADES_PENALTY = 0.5` (actual: 0.7)
- Says `LOW_TRADES_THRESHOLD = 2.0` (actual: 1.5)
- Describes efficiency analysis as running "after all strategies are tested" (actual: between GA and Bayesian)
- Pipeline diagram doesn't show mid-training efficiency analysis

### Fix
**Scope:** Full audit of `README.md`. Update every parameter value and description to match current `config.py`.

**Specific changes:**

1. **Data preparation section**: Change "6 months" to "12 months"
2. **Score formula section**: Update `LOW_TRADES_THRESHOLD` to 1.5, `LOW_TRADES_PENALTY` to 0.7
3. **Configuration Reference table**: Update all mismatched values:
   | Parameter | Old | New |
   |---|---|---|
   | `TRAINING_PERIOD_MONTHS` | 6 | 12 |
   | `MAX_RR` | 5.0 | 8.0 |
   | `LOW_TRADES_THRESHOLD` | 2.0 | 1.5 |
   | `LOW_TRADES_PENALTY` | 0.5 | 0.7 |

4. **Efficiency Analysis section**: Rewrite to describe mid-training flow:
   ```
   **Efficiency Analysis (Mid-Training):**
   After the Genetic Algorithm phase, the system analyzes all conditions used
   in the GA strategies. Conditions with efficiency < 0.3 are removed from the
   pool before the Bayesian Optimization phase begins. Conditions with efficiency
   0.3-0.5 are kept but given 0.5x selection weight.
   
   This focuses the Bayesian search on the most promising conditions.
   Removed conditions are NOT persisted across training runs.
   ```

5. **Pipeline diagram**: Update to show efficiency between GA and Bayesian:
   ```
   Training Start
       |
       |-- Load 12 months of BTC/USDT data
       |-- Compute all 53 conditions (pre-cached for speed)
       |
       |-- PHASE 1: Genetic Algorithm (~50% of time)
       |   |-- Gen 0: 200 random strategies
       |   |-- Gen 1-N: Evolve via selection + crossover + mutation
       |   +-- Top 10 passed to Phase 2
       |
       |-- EFFICIENCY ANALYSIS (between phases)
       |   |-- Remove conditions with efficiency < 0.3
       |   |-- Weight conditions with efficiency 0.3-0.5 at 50%
       |   +-- Log removed/weighted conditions
       |
       |-- PHASE 2: Bayesian Optimization (~50% of time)
       |   |-- Seed with GA's top 10
       |   |-- Random trials (build initial model)
       |   +-- TPE-guided trials (smart search)
       |
       +-- Compare with existing best, save if better
   ```

6. **Future enhancements section**: Add note about Monte Carlo simulations

### Files Modified
- `README.md`: Full audit and update

### Verification
- Read through the updated README
- Verify every parameter matches `config.py`
- Verify the pipeline diagram matches the actual code flow

---

## Issue 6: Validation Period May Overlap with Training (MEDIUM)

### Problem
The README doesn't clearly explain that validation data is non-overlapping with training data. Users might worry about data leakage.

### Current Code (`data_fetcher.py` `get_validation_data`)
```python
# Validation ends where training begins
until = datetime.now(timezone.utc) - timedelta(days=config.TRAINING_PERIOD_MONTHS * 30)
since = until - timedelta(days=months * 30)
```

### Fix
**Scope:** Add clear logging to `validation.py`. Keep the dynamic approach (no hardcoded dates).

**Edge cases:**
- **Warmup gap:** `prepare_data()` drops NaN rows for indicator warmup (e.g., 200-candle EMA). So `clean_df` starts later than `raw_df`. The logged validation start should use `raw_df` (pre-warmup) to show the true fetched period, and note the warmup separately.
- **Timezone:** All datetimes must be normalized to UTC-aware before formatting. `raw_df["timestamp"]` is already UTC-aware, but `datetime.now(timezone.utc)` comparisons should use the same type.

**In `validation.py` `run_validation()`:**
- After fetching raw data (before warmup drop), log the periods:
  ```python
  from datetime import timedelta, timezone
  
  # Compute training period boundaries
  now = datetime.now(timezone.utc)
  train_end = now - timedelta(days=config.TRAINING_PERIOD_MONTHS * 30)
  train_start = train_end - timedelta(days=config.TRAINING_PERIOD_MONTHS * 30)
  
  # Validation period from raw data (pre-warmup)
  val_start = raw_df["timestamp"].iloc[0]
  val_end = raw_df["timestamp"].iloc[-1]
  
  logger.info(f"Training period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} ({config.TRAINING_PERIOD_MONTHS} months)")
  logger.info(f"Validation period: {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')} ({period_months} months, non-overlapping)")
  logger.info(f"  Note: {len(raw_df) - len(clean_df)} candles dropped for indicator warmup.")
  ```

**Note:** Do NOT claim "gap = 0 days" — the warmup drop means there's a small implicit gap. Instead, just state "non-overlapping" and let the dates speak for themselves.

### Files Modified
- `validation.py`: Add period logging after data fetch

### Verification
- Run validation and verify the period log lines appear
- Verify the training end date equals the validation start date (they should be contiguous in raw data)
- Verify the warmup note is shown

---

## Issue 8: Training Finish Log Missing Strategy Details (MEDIUM)

### Problem
The README promises the training finish output shows:
```
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

But in practice, Win rate, RR/day, Max drawdown, and Valid trades are **never shown**. The log silently skips them.

### Root Cause
`_run_ga_bayesian()` returns `best_strategy` which is the optimizer's return value (from `genetic_optimizer.py` or `bayesian_optimizer.py`). These dicts have `id`, `conditions`, `threshold`, `sl`, `rr`, `direction`, `score`, `method` — but **no `results` key**.

Meanwhile, `self.best_strategy` (tracked in `_eval_strategy`) DOES have `results`:
```python
# In _eval_strategy:
self.best_strategy["results"] = {
    "win_rate": results["win_rate"],
    "rr_per_day": results["rr_per_day"],
    "max_drawdown": results["max_drawdown"],
    "total_trades": results["total_trades"],
    "valid_trades": results["valid_trades"],
}
```

But `_log_finish(best)` receives the optimizer's dict (no `results`), so `if 'results' in best:` is `False`, and the metrics are silently skipped.

### Fix
**Option A (recommended):** Change `_run_ga_bayesian()` to return `self.best_strategy` instead of the optimizer's dict. This ensures the returned dict always has `results`.

```python
# At the end of _run_ga_bayesian(), instead of:
return best_strategy  # optimizer's dict (no results)

# Use:
return self.best_strategy or best_strategy  # tracked dict (has results)
```

**Option B:** Change `_log_finish()` to look up `self.best_strategy` for the results, falling back to the passed `best` for other fields.

**Recommended: Option A** — it's simpler and also fixes the same issue in `_save_results()` which saves the strategy to JSON (the saved JSON should also have `results` for the live signal reader).

**Also fix `_run_random_search()`** — it returns `self.best_strategy or {}` which is correct (already has `results`). No change needed for random search.

### Files Modified
- `training.py`: Change `_run_ga_bayesian()` return to use `self.best_strategy`

### Verification
- Run a short training (`--minutes 2 --method random` or `ga_bayesian`)
- Verify the finish log shows Win rate, RR/day, Max drawdown, Valid trades
- Verify the saved `best_strategy.json` contains a `results` key

---

## Issue 9: More Detailed Logging Throughout (MEDIUM)

### Problem
The training logs should provide more context at each phase so the user understands what's happening without reading the code. Several log messages are too terse or missing key details.

### Specific Improvements

**1. GA phase start log — add time budget info:**
```
Current:  GA: Starting | pop=200, gen=30, cx=0.8, mut=0.2, elite=5
Better:   GA: Starting | pop=200, time_budget=900s (15.0m), max_gen=200, cx=0.8, mut=0.2, elite=5
```

**2. GA generation log — remove fixed total (Issue 3), add elapsed time:**
```
Current:  GA Gen 5/30 | Best score: 2.10 | Avg score: 1.34 | Tested: 1200 [NEW BEST!]
Better:   GA Gen 5 | Best score: 2.10 | Avg score: 1.34 | Tested: 1200 | Elapsed: 45s [NEW BEST!]
```

**3. GA phase complete — add generation count and strategies/second:**
```
Current:  GA Phase complete | Elapsed: 273s | Best score: 3.97 | Strategies tested: 2165
Better:   GA Phase complete | 45 generations | Elapsed: 273s | Best score: 3.97 | Strategies tested: 2165 | Speed: 7.9 strats/s
```

**4. Efficiency analysis — already good, no changes needed.

**5. Bayesian phase start — add time budget info:**
```
Current:  Bayesian: Starting | trials=2000, startup=100
Better:   Bayesian: Starting | timeout=900s (15.0m), max_trials=10000, startup=100
```

**6. Bayesian phase complete — add trials/second:**
```
Current:  Bayesian: Finished | Best score: 2.50 | Total strategies tested: 335
Better:   Bayesian: Finished | Best score: 2.50 | 335 trials | Elapsed: 31s | Speed: 10.8 trials/s
```

**7. Training finish log — already detailed (after Issue 8 fix), add direction:**
```
Current:  Best strategy:       strat_abc123
Better:   Best strategy:       strat_abc123 (LONG)
```

**8. Data preparation — add training period dates:**
```
Current:  Raw data: 35040 candles
Better:   Raw data: 35040 candles (2025-06-21 to 2026-06-21, 12 months)
```

### Files Modified
- `training.py`: Update log messages in `_log_start()`, `_run_ga_bayesian()`, `_log_finish()`
- `genetic_optimizer.py`: Update log messages in `run()` (generation log, phase complete log)
- `bayesian_optimizer.py`: Update log messages in `run()` (start log, finish log)

### Verification
- Run a short training and verify all log messages show the improved format
- Verify time budgets, speeds, and elapsed times are shown
- Verify the training period dates appear in the data preparation log

---

## Issue 7: No Robustness Tests (LOW — Future Enhancement)

### Problem
No Monte Carlo simulations, sensitivity analysis, or walk-forward validation exists.

### Fix
**Document only. No code changes.**

**In `README.md`:**
- Add a "Future Enhancements" section at the bottom:
  ```markdown
  ## Future Enhancements
  
  The following features are planned but not yet implemented:
  
  - **Monte Carlo simulation**: Shuffle trade order to check if strategy performance is statistically significant
  - **Sensitivity analysis**: Test how strategy performance changes when parameters are slightly modified
  - **Walk-forward validation**: Rolling window validation to check strategy stability over time
  - **Out-of-sample testing**: Test on data from different market regimes (bull, bear, sideways)
  
  To contribute or request these features, open an issue on the repository.
  ```

**In `validation.py`:**
- Add a TODO comment near the acceptance criteria:
  ```python
  # TODO: Future enhancement — add Monte Carlo simulation here
  # Shuffle trade order N times, recalculate drawdown and Sharpe for each,
  # report confidence intervals. See README "Future Enhancements" section.
  ```

### Files Modified
- `README.md`: Add Future Enhancements section
- `validation.py`: Add TODO comment

---

## Implementation Order

| Priority | Issue | File(s) | Effort | Description |
|---|---|---|---|---|
| 1 | Issue 1 | `validation.py` | Small | Fix RR/day to use total period days |
| 2 | Issue 8 | `training.py` | Small | Fix missing Win rate/RR/day in finish log |
| 3 | Issue 2 | `training.py` | Small | Compare with existing best before saving |
| 4 | Issue 3 | `training.py`, `config.py`, `genetic_optimizer.py`, `bayesian_optimizer.py` | Medium | Make GA/Bayesian time-adaptive |
| 5 | Issue 9 | `training.py`, `genetic_optimizer.py`, `bayesian_optimizer.py` | Medium | Improve log detail throughout |
| 6 | Issue 5 | `README.md` | Medium | Full audit, update all values and descriptions |
| 7 | Issue 6 | `validation.py` | Small | Add period logging |
| 8 | Issue 7 | `README.md`, `validation.py` | Small | Document Monte Carlo as future enhancement |

**Total estimated effort:** ~3-4 hours

---

## Testing Strategy

1. **After Issue 1:** Run `python validation.py --symbol BTC/USDT --period 12` and verify RR/day is total_rr / calendar_days
2. **After Issue 2:** Run training twice, verify the second run doesn't overwrite a better strategy
3. **After Issue 3:** Run `python training.py --minutes 5` and `--minutes 15`, verify strategies tested scales with time
4. **After Issue 5:** Read through README, verify all values match `config.py`
5. **After Issue 6:** Run validation, verify period log lines appear with correct dates
6. **Final:** Run full smoke test: `python training.py --symbol BTC/USDT --method ga_bayesian --minutes 3`

---

## Cross-Cutting Concerns

### Time accounting: Efficiency analysis time
The efficiency analysis runs between GA and Bayesian phases (Issue 4, already fixed). This takes non-trivial time (could be seconds to minutes depending on result count). **The Bayesian time budget must be calculated AFTER efficiency analysis completes**, not after GA completes.

```python
# CORRECT: remaining time after efficiency analysis
remaining = self.training_minutes * 60 - (time.time() - self.start_time)

# WRONG: remaining time after GA (ignores efficiency analysis time)
remaining = self.training_minutes * 60 - ga_elapsed
```

### _log_finish update after Issue 2
After adding the best-strategy comparison in `_save_results()`, the `_log_finish()` method must be updated to reflect whether the new best was saved or the existing one was retained. This requires `_save_results()` to return a boolean or dict indicating the outcome.

### GA generation logging after Issue 3
The GA currently logs `Gen {gen}/{self.generations}`. With time-based stopping, the total generations is unknown at start. Remove the `/{self.generations}` suffix from the log format. The user sees the gen count incrementing and knows it stops when time runs out or cap is hit.

### Config parameter renaming (Issue 3 + Issue 5)
The old parameters `GA_GENERATIONS` and `BAYESIAN_N_TRIALS` must be either renamed or removed:
- `GA_GENERATIONS = 30` → `GA_MAX_GENERATIONS = 200` (safety cap)
- `BAYESIAN_N_TRIALS = 2000` → `BAYESIAN_MAX_TRIALS = 10000` (safety cap)
- Add `GA_TIME_BUDGET_PERCENT = 0.5` (new)
- Keep `BAYESIAN_STARTUP_TRIALS = 100` (unchanged, but fix the `n_trials // 5` calculation)

### top_strategies.json (Issue 2)
This file is a per-run snapshot — always overwrite, do not merge. It represents the top 500 strategies found in the current training session, not across all sessions.

---

## Dependencies

- Issue 4 is already fixed (no dependency)
- Issue 8 (finish log fix) should be done BEFORE Issue 2 (best strategy comparison) since both modify `_save_results` and `_log_finish`
- Issue 9 (logging improvements) modifies the same files as Issue 3 — implement together
- Issue 1 and Issue 6 both modify `validation.py` — implement together to avoid conflicts
- Issue 3 modifies `training.py`, `config.py`, `genetic_optimizer.py`, `bayesian_optimizer.py`
- Issue 2 modifies `training.py` — combine with Issue 3/8/9 changes to avoid merge conflicts
- Issue 5 depends on Issues 1, 2, 3, 8, 9 being done (README should reflect final state)
- Issue 7 is independent and can be done anytime

---

## Out of Scope

- Changing `backtest.py` RR/day calculation (would double-penalize during training)
- Hardcoded training/validation date ranges (dynamic approach is preferred)
- Monte Carlo implementation (documented as future enhancement)
- Walk-forward validation (documented as future enhancement)
