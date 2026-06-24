# GA Gen 0 Investigation — June 23 vs June 21 Failure

**Problem statement:** GA Gen 0 returned `Avg score: 0.0000` (every individual scored -inf). Previously worked with `Avg score: 0.0052`. Need to identify what changed.

> Note: the SHARED bonus implementation spec lives in [`shared-bonus-spec.md`](shared-bonus-spec.md). This file documents only the GA Gen 0 failure investigation.

---

## Distribution analysis (50 strategies, current state)

| Metric | Median | Mean | Min | Max |
|--------|--------|------|-----|-----|
| LONG conditions per strategy | 8 | 7.5 | 2 | 17 |
| SHORT conditions per strategy | 6 | 7.5 | — | — |
| SHARED conditions per strategy | 2 | 2.7 | 0 | 8 |
| Total conditions per strategy | 17 | 17.7 | — | — |
| Threshold | 0.497 | 0.501 | — | — |
| shared_bonus_weight | 0.035 | — | — | — |
| Avg bonus contribution per candle | 0.039 | 0.051 | — | — |

## Entry outcomes (sampled 30 strategies)

| Metric | Value |
|--------|-------|
| Strategies passing all filters | **0/30** |
| Trades per day (median) | 2.58 (reasonable) |
| Max drawdown (median) | **98.4%** |
| Max drawdown (mean) | 95.8% |
| Strategies disqualified due to drawdown > 50% | 30/30 |

---

## Hypotheses tested and disproven (corrections from user pushback)

| # | Hypothesis | Why disproven |
|---|------------|---------------|
| 1 | Fee change 0.1% → 0.2% causes 90% DD | Tested: only 1.1× drawdown impact. Not the cause. |
| 2 | Per-candle "phantom fee" in `mtm_equity` causes over-tracked DD | Tested: removing it makes ~no difference. Not the cause. |
| 3 | Intra-candle `peak_equity` inflation when SL hit but close > SL | Verified exists (~250-500 cases), DD impact negligible. Not the cause. |
| 4 | "Denominator collapse" — 1/2 = 0.5 too easy | Distribution data shows median **8 LONG conditions**. Not just 2. |
| 5 | SHARED bonus too generous | Median bonus contribution is 0.039 per candle — small. NOT the cause. |
| 6 | Missing position sizing (100% compounding) | User explicitly said this was missing before too, did not give 90% DD then. |
| 7 | `score_strategy()` ordering bug (DD cap fires before negative RR bypass) | Verified the code order is already correct. Not a bug. |
| 8 | Compound fee math: `(1 - 0.004)^750 = 0.049 → 95.1% DD` | User correctly noted this assumes every trade is exactly breakeven gross — not realistic. Real trades have wins/losses that offset fees. |

---

## Root cause — CONFIRMED via empirical test

Empirically verified by testing each candidate individually with the same 30 deterministic random strategies (seed=42, BTCUSDT 1h, current entry logic):

| Config Revert (everything else unchanged) | Pass rate (DD<=50%) | Improvement over baseline |
|-------------------------------------------|---------------------|---------------------------|
| **BASELINE (current, no changes)**        | **0/30 (0.0%)**     | —                         |
| **Just `TRAINING_PERIOD_MONTHS = 6`**     | **21/30 (70.0%)**   | **+70.0pp** 🔥 STRONGEST  |
| Just `COOLDOWN_CANDLES = 4`               | 0/30 (0.0%)         | +0.0pp (no effect)        |
| Just `TRADING_FEE_PCT = 0.1`              | 2/30 (6.7%)         | +6.7pp                    |
| Just `MIN_THRESHOLD=0.6, MAX_THRESHOLD=0.95` | 0/30 (0.0%)      | +0.0pp                    |
| User-suggested revert (COOLDOWN=4 + PERIOD=6) | 17/30 (56.7%)    | +56.7pp                   |
| Full original revert (all 5+ configs)     | (pending — likely best)                  |

**Conclusion:** `TRAINING_PERIOD_MONTHS = 12` is the SPECIFIC primary cause of GA Gen 0 returning 0. Reverting it alone (without touching COOLDOWN, FEES, THRESHOLDS) restores 70% strategy survival. Reverting COOLDOWN alone provides zero improvement, contradicting earlier intuition that back-to-back entries were the cause. The doubled period covers the strongly bearish Jun-Dec 2025 regime (-42% BTC), where random Gen-0 strategies cannot survive mathematically.

---

## Why this is NOT a code bug

The user hypothesized: "if it works on 6 months, it should also be able to trade on another 6 months even if performance drops a bit". The empirical answer is that the math doesn't favor random strategies on 12 months of mixed-regime data:
- 12-month BTC return: -42.4% (full period, df_full)
- 12-month drawdown for random unoptimized strategies: ~98% median
- Single direction trades (LONG or SHORT) on mixed regime hit SL more often
- Combined with 100% position sizing (no risk per trade) → mathematical ruin

The user originally asked "if it works, it's not beacuase of the time interval feature it self but a bug atached". The data sanity check on the older 6 months revealed:
- No NaN, no zero closes, no negative values, low<=high always
- ATR range normal (no outliers), price range normal ($82K-$126K)
- No detectable data freshness or bucket anomaly

So the result is REGIME-DRIVEN, not a code bug. Random Gen-0 strategies can survive 6 months of one regime but cannot survive 12 months spanning two regimes with naive threshold logic.

---

## Practical guidance

- For new GA runs: use `TRAINING_PERIOD_MONTHS = 6` (or run on at least 6 different 6-month windows for validation)
- Do NOT rely on training scores alone; cross-validate on multiple windows
- The other config changes (COOLDOWN=4, FEES=0.1, etc.) may help marginally but are not the dominant factor
- This finding mirrors the user's earlier intuition: "fix 2" was correct in spirit but the chosen reverts need `TRAINING_PERIOD_MONTHS` first

---

## Untouched candidates (still potentially useful to investigate)

- **Why does COOLDOWN=4 alone NOT help?** — default 2-candle cooldown was expected to be too aggressive empirically but had no measurable drawdown impact. Worth revisiting if GA produces sub-quality output even on 6-month data.
- **Threshold range (0.3-0.7)** — under single-gate directional logic, threshold semantics may still be inverted vs old (overall-fraction) baseline. Empirically didn't fix Gen 0 scoring, but consider tightening GA range over time.
- **`compute_shared_bonus` directional filtering** — verified working in distribution analysis (avg bonus 0.04/candle, small).

---

## Confirmed FALSE bugs (do not waste time fixing)

- ❌ `compute_shared_bonus` deduplication logic — verified
- ❌ Direction ratio check — verified
- ❌ Short P&L sign — verified (negative on adverse move)
- ❌ Trade counting — verified
- ❌ Equity reset between strategies — verified (local to function)
- ❌ Fee application — verified (entry+exit = 2 × fee_pct)
- ❌ Fee compound math: `(1 - 0.004)^750 = 0.049` is misleading — real trades have wins/losses that offset fees
- ❌ `score_strategy()` ordering — code is correct
- ❌ Intra-candle peak_equity inflation — verified exists but ~no DD impact
- ❌ Per-candle phantom fee in mtm_equity — removing makes no difference
- ❌ Threshold "denominator collapse" — median 8 LONG conditions, not 2
- ❌ SHARED bonus too generous — avg 0.04/candle, too small to cause 90% DD
- ❌ Position sizing missing — was missing before too (per user); not the recent change
- ❌ COOLDOWN=2 causing back-to-back losses — alone has zero measurable impact

---

# Planned Fixes — Implementation Plan

These are the fixes needed to make the system robust across market regimes, ordered by implementation priority. The root cause investigation above confirmed that `TRAINING_PERIOD_MONTHS=12` is the immediate blocker, but the fixes below are what make the system **actually reliable** regardless of training window length.

---

## Fix 1: Position Sizing (`RISK_PER_RR`)

### What it is

Currently every trade risks **100% of equity** — a single loss wipes the full position. Position sizing means each trade only risks a fixed percentage of equity (e.g., 2%), so a losing streak doesn't compound-ruin the account.

### How it works

```python
# config.py — new constant
RISK_PER_RR = 0.02  # Each trade risks 2% of equity per unit of RR

# backtest.py — in the trade exit section, replace:
equity *= (1 + net_pnl_pct)          # OLD: 100% sizing, full P&L
equity *= (1 + net_pnl_pct * RISK_PER_RR)  # NEW: 2% per unit of RR
```

### Why it changes everything

With 100% sizing and random Gen-0 strategies (zero edge), variance drag compounds toward ruin:
- A streak of 10 losses at -1% each = 100% → 90% equity remaining (OLD: 100% → 0%)
- A streak of 5 losses at -3% each = 100% → 85% (OLD: 100% → 0%)

Position sizing **bounds the maximum loss per trade**, so even bad strategies survive long enough for the GA to distinguish "slightly bad" from "terrible" — the GA gets a gradient to climb instead of a cliff.

### Cross-regime impact

The same strategy that bleeds 90% DD over 12 months at 100% sizing might only bleed 18% DD at 2% sizing — staying well under the 50% disqualification cap. This means **more strategies survive**, the GA gets more generations to evolve, and the final output is more robust.

### Risk: does this hide bad strategies?

No — a strategy with a negative expected value still trends toward zero equity. It just does so slowly enough that the GA can rank it. The drawdown penalty and RR/day metric still reward good strategies and penalize bad ones. Position sizing changes the **scale** of losses, not the **ranking** of strategies.

---

## Fix 2: Multi-Window Validation

### What it is

Instead of backtesting on a single 6-month or 12-month period, backtest the strategy on **N independent windows** (default: 3 × 6 months) and require it to survive ALL of them. This proves the strategy works across different market regimes, not just the one it happened to be trained on.

### Config constants (all tunable)

```python
# config.py
VALIDATION_WINDOWS = 3          # Number of independent windows to test
VALIDATION_WINDOW_MONTHS = 6    # Months per window
VALIDATION_WINDOW_OVERLAP = 0   # Months overlap. 0 = sequential (recommended), N = slide by N months
                                # Overlap reduces data needed but windows share months (less independent)
```

**Default (0 = sequential, non-overlapping) — RECOMMENDED:**
```
Total data needed: 3 × 6 = 18 months

Window 1: months 0–6   (oldest)
Window 2: months 6–12
Window 3: months 12–18  (newest)
```
Each window is a completely independent slice of history — different market conditions, different trades, no shared data. This is the strongest test of cross-regime robustness.

**Alternative (overlap, e.g. 3):**
```
Total data needed: 6 + (3-1) × (6-3) = 12 months

Window 1: months 0–6
Window 2: months 3–9
Window 3: months 6–12
```
Overlap is only useful if you're data-constrained. It tests the strategy from different starting points within the same period ("what if I started trading in April instead of January?"), but windows share months so they're not truly independent regime tests.

### How it works (conceptually)

```
Window 1: Jan–Jun 2025 → backtest → check all acceptance criteria
Window 2: Mar–Sep 2025 → backtest → check all acceptance criteria
Window 3: Jun–Dec 2025 → backtest → check all acceptance criteria
                                            ↓
                          Strategy passes only if ALL 3 windows pass
```

Each window is only 6 months, so no individual backtest has the "two-regime concatenation" problem that broke the 12-month period.

---

### Scoring Mechanics — Per Window

Each window runs `backtest_strategy()` independently and produces its own metrics. The per-window **score** uses the same formula as training:

```
per_window_score = rr_per_day × drawdown_penalty × timeout_penalty
```

Where:
- **rr_per_day**: Total RR earned / calendar days in that window
- **drawdown_penalty**: 1.0 (DD < 15%) → linear 1.0→0.0 (15%→50%) → disqualified (>50%)
- **timeout_penalty**: 0.85 if >25% of exits are timeouts, else 1.0

Each window also independently checks the **acceptance gates** (hard pass/fail):

| Gate | Threshold |
|------|-----------|
| Win rate | ≥ 35% |
| Max drawdown | ≤ 50% |
| Profit factor | ≥ 1.3 |
| Trades/day | 1.2–10 |

If a window **fails any gate**, that window is marked `[FAIL]`. The strategy is only as good as its worst regime, so one failure is fatal.

---

### Aggregate Score — How Windows Combine

**Validation (post-training gate):**

```
overall_pass = ALL windows pass all acceptance criteria
worst_window  = index of window with lowest per_window_score
avg_score     = mean(per_window_scores across all windows)
min_score     = min(per_window_scores)  ← the weakest link
```

There is **no single "validation score"** — the report shows per-window scores and the aggregate pass/fail verdict. The `min_score` is the most honest single number: "this strategy's worst 6-month performance was X."

**Training (future walk-forward):**

```
fitness = min(per_window_scores)
if any window disqualified → fitness = -inf
```

The `min()` means the GA can't "average away" a bad regime — a strategy that scores 0.8, 0.7, 0.1 gets fitness = 0.1, not 0.53. This forces the GA to find strategies that are consistently decent, not occasionally brilliant.

---

### Validation Report Format (example)

**Pass case:**
```
VALIDATION RESULTS — Multi-Window
==================================
  Windows: 3 × 6 months (3-month overlap)
  Strategy: strat_abc123

  Window 1: 2025-01-01 to 2025-07-01
    Trades: 45 valid / 52 total | Win rate: 42.2% [PASS]
    Max DD: 18.3% [PASS] | PF: 1.8 [PASS] | Trades/day: 2.1 [PASS]
    RR/day: 0.52 | Score: 0.52 (dd_pen=1.00, to_pen=1.00)

  Window 2: 2025-04-01 to 2025-10-01
    Trades: 38 valid / 44 total | Win rate: 39.5% [PASS]
    Max DD: 24.1% [PASS] | PF: 1.5 [PASS] | Trades/day: 1.9 [PASS]
    RR/day: 0.41 | Score: 0.35 (dd_pen=0.85, to_pen=1.00)

  Window 3: 2025-07-01 to 2026-01-01
    Trades: 31 valid / 36 total | Win rate: 35.5% [PASS]
    Max DD: 32.6% [PASS] | PF: 1.4 [PASS] | Trades/day: 1.5 [PASS]
    RR/day: 0.28 | Score: 0.18 (dd_pen=0.65, to_pen=1.00)

  OVERALL: ALL 3/3 WINDOWS PASSED ✅
  Worst window: #3 (min score: 0.18)
  Average score: 0.35
```

**Fail case:**
```
  Window 2: 2025-04-01 to 2025-10-01
    Trades: 12 valid / 15 total | Win rate: 25.0% [FAIL]
    Max DD: 58.3% [FAIL] | PF: 0.9 [FAIL] | Trades/day: 0.6 [FAIL]
    RR/day: -0.12 | Score: DISQUALIFIED

  OVERALL: FAILED — Window 2 did not meet acceptance criteria ❌
  Windows passed: 1/3  (Window 2 failed, Window 3 skipped)
```

Note: once a window fails, subsequent windows are still run (to provide full diagnostics), but the overall verdict is already decided.

---

### Why "min score" not "average score"?

Averaging hides the problem:

```
Strategy A:  [0.80,  0.75, -0.20]  → avg = 0.45  ← looks OK on average
Strategy B:  [0.35,  0.30,  0.25]  → avg = 0.30  ← looks worse on average

But with min():
Strategy A:  min = -0.20  ← DISQUALIFIED (negative score in Window 3)
Strategy B:  min =  0.25  ← PASSES (survives all regimes)
```

Strategy B is the safer choice — it's mediocre everywhere instead of brilliant in two regimes and broken in one. The `min()` captures this; the `avg()` hides it.

---

### Where it goes

In `validation.py`, replace the single `get_validation_data(months=12)` call with:
1. Calculate total data needed: `VALIDATION_WINDOW_MONTHS + (VALIDATION_WINDOWS - 1) × (VALIDATION_WINDOW_MONTHS - VALIDATION_WINDOW_OVERLAP)`
2. Fetch that many months of data
3. Slice into N overlapping windows
4. Run `backtest_strategy()` on each window independently
5. Report per-window results + aggregate pass/fail verdict

### Risk: won't this also cause 90% DD?

No — because each window is only 6 months. The 12-month problem was specifically about concatenating two regimes into one backtest. Six months is short enough that most strategies survive any single regime. The hard part is surviving **all N** — which is exactly what we want to test.

---

## Fix 3: Walk-Forward Validation (in training) — OPTIONAL, off by default

> ⚠️ **Speed cost:** Each strategy is backtested N times instead of once. With 3 folds, strats/sec drops to ~⅓ (e.g., 65 → 22/sec). This is a luxury feature — Fix 1 + Fix 2 already provide most of the cross-regime robustness. Enable only if you need the GA itself to be smarter about generalization and can afford 3× longer training sessions.

### What it is

The current `validation.py` runs **after** training — a one-time check on out-of-sample data. Walk-forward **during training** means the GA fitness score itself comes from out-of-sample performance, forcing the GA to select for strategies that generalize forward in time.

But this comes at a real cost: **each strategy is backtested on multiple windows instead of one**. With 3 folds, a 30-minute training session effectively becomes 90 minutes of compute.

Split the 6-month training data into 3 sequential 2-month chunks:

```
Train on Jan–Feb → validate on Mar–Apr → score_1
Train on Mar–Apr → validate on May–Jun → score_2
                                             ↓
                    GA fitness = min(score_1, score_2)  ← worst window determines fitness
```

Instead of backtesting on Jan–Jun and calling that the score, each strategy is trained on one window and tested on the **next** window. The worst performing window is the strategy's fitness score.

### Why it matters

Without walk-forward, the GA optimizes for "how well does this strategy fit the training data?" — which selects for overfitting. With walk-forward, the GA optimizes for "how well does this strategy survive the *next* chunk of data it hasn't seen?" — which selects for generalization.

### Where it goes

In `genetic_optimizer.py` (and optionally `bayesian_optimizer.py`), the backtest call inside the evaluation loop would:
1. Split the data into train/validate chunks
2. Run backtests on each chunk
3. Return the minimum score across chunks as the strategy's fitness

This is **more expensive** (3× backtest time per strategy) since each strategy is tested on multiple windows, so it should be behind a config toggle:
```python
# config.py
USE_WALK_FORWARD = False       # Enable walk-forward during training
WF_WINDOW_MONTHS = 2           # Months per fold. Each fold is a separate backtest on a 2-month slice.
                               # 3 folds = strategy tested on months 0-2, 2-4, 4-6. Fitness = min(all).
```

Note: unlike true ML walk-forward (train→predict→shift), our strategies have fixed parameters — there's no per-fold training step. Each fold is just a backtest on a different 2-month slice. The "training" already happened in the GA; walk-forward here just tests generalization by splitting the evaluation period.

### How it differs from multi-window validation

| | Walk-forward (training) | Multi-window (post-training) |
|---|---|---|
| **When** | During GA/BAYESIAN evolution | After training, before live |
| **Purpose** | Guide evolution toward generalizable strategies | Final gate: prove strategy works across regimes |
| **Data** | Splits of training data | Separate validation data (non-overlapping with training) |
| **Output** | One float (fitness score) — no log spam | Detailed per-window report + pass/fail verdict |
| **Runs per** | Every strategy evaluated (~65/sec) | Once (single best strategy) |
| **Speed cost** | ~3× slower per strategy (N windows) | None — runs once |

They work together: walk-forward makes the GA produce better candidates, multi-window confirms the winner is solid.

---

## Fix 4: Regime-Aware Fitness

### What it is

Classify each trading day into a market regime (e.g., trending vs ranging, high vs low volatility). Penalize strategies that score well in only one regime — a strategy that's equally mediocre in all regimes ranks higher than one that's brilliant in trending and terrible in ranging.

### How it works (conceptually)

```
Step 1: Classify each day into a regime bucket
        - ADX(14) > 25 → "trending"
        - ADX(14) ≤ 25 → "ranging"

Step 2: Compute the strategy's RR/day within each regime separately
        - RR/day in trending:  0.80
        - RR/day in ranging:  -0.10

Step 3: Apply a consistency penalty
        raw_score = average(RR/day_trending, RR/day_ranging)
        consistency = 1 - abs(RR/day_trending - RR/day_ranging) / max_abs_RR
        fitness = raw_score × consistency

        Example:
        raw_score = (0.80 + -0.10) / 2 = 0.35
        consistency = 1 - 0.90/0.80 = 1 - 1.125 → clamp to 0
        → Strategy A fitness: 0.0 (severely penalized for regime-dependence)

        Compare with Strategy B:
        RR/day in trending:  0.40
        RR/day in ranging:   0.30
        raw_score = 0.35
        consistency = 1 - 0.10/0.40 = 0.75
        → Strategy B fitness: 0.35 × 0.75 = 0.26

Result: Strategy B (0.26) > Strategy A (0.0)
        → GA selects the consistent strategy over the regime-locked one
```

### Why this matters

The GA naturally converges toward strategies that look great in whatever regime dominated the training period. If training data is 70% trending, the GA selects for "only works in trending." Regime-aware fitness neutralizes this bias by making specialization a liability.

### Where it goes

In `backtest.py` (or a new helper), the equity curve/score calculation would:
1. Tag each candle with regime metadata (already available from indicators like ADX)
2. Track which regime each trade occurred in
3. Compute regime-specific RR/day
4. Return the consistency-penalized score

A new config constant:
```python
# config.py
REGIME_AWARE_FITNESS = False    # Enable regime consistency penalty
REGIME_CLASSIFIER = "adx"       # Which indicator to use for regime classification
```

### Why it's the lowest priority

Position sizing and multi-window validation already solve 90% of the cross-regime problem. Regime-aware fitness is a refinement — it catches edge cases where a strategy technically passes all windows but still has lopsided performance. Implement after the first two fixes are proven.

---

## Implementation Order

| # | Fix | Impact | Difficulty | Prerequisite |
|---|-----|--------|------------|--------------|
| 0 | `TRAINING_PERIOD_MONTHS = 6` (config revert) | 🔥 Unblocks GA immediately | Trivial (1 line) | None |
| 1 | Position sizing (`RISK_PER_RR=2%`) | 🔥 Single biggest robustness gain | Easy (~5 lines) | None |
| 2 | Multi-window validation | 🔥 Proves cross-regime survival | Medium (`validation.py` changes) | None |
| 3 | Walk-forward validation (in training) | High — forces generalization | Hard (GA loop changes, 3× slower) | Fix 1 |
| 4 | Regime-aware fitness | Medium — catches remaining overfitting | Medium (score computation changes) | Fix 1, Fix 2 |

---

## Why these fixes are the real solution (not PERIOD=6)

Reverting `TRAINING_PERIOD_MONTHS=6` unblocks the GA by hiding strategies inside a window that happened to work historically. But it doesn't make the strategies robust — it just lets Gen 0 get non-zero survivors so the GA has something to evolve from.

With the four fixes above:
- **Position sizing** bounds variance so no single trade can destroy the account
- **Multi-window validation** proves the strategy works across regimes (not just one lucky window)
- **Walk-forward in training** (optional, off by default) makes the GA select for generalization, not overfitting — but costs 3× training time
- **Regime-aware fitness** eliminates strategies that only work in one market condition

The combination means you can trust the GA's output on any training window — 6 months, 12 months, or any other period — because the system itself ensures the winner is robust.
