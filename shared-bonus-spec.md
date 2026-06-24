# SHARED Conditions Bonus System — Spec

## Problem

SHARED conditions (volume spikes, breakouts, trend strength) are **rare triggers** (true on 5-15% of candles). When included in the entry strength denominator alongside LONG/SHORT conditions, they dilute the strength on most candles (when false), making it much harder to reach the entry threshold. This caused training scores to drop 10-20x.

At the same time, removing SHARED from entry logic entirely wastes useful signal information — when a volume spike or breakout DOES occur, it's a strong confirmation.

## Solution: SHARED as Bonus

SHARED conditions are **removed from the strength denominator**. Instead, they contribute a **GA-optimized bonus** to the final confidence score when active.

### Core Logic

```
base_strength = long_true / long_count          # pure LONG or SHORT conditions only
shared_bonus = number_of_true_shared × shared_bonus_weight
final_strength = min(base_strength + shared_bonus, 0.95)   # clamped at 95%

entry if: final_strength >= threshold AND long_pure > short_pure × DIRECTION_RATIO
```

### GA-Optimized Parameters (per strategy)

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `conditions` | list[str] | 4-34 keys | Core LONG/SHORT conditions for base strength |
| `shared_conditions` | list[str] | 0-9 keys | SHARED conditions that contribute bonus |
| `shared_bonus_weight` | float | 0.0 - 0.15 | Bonus per true shared condition |
| `threshold` | float | 0.30 - 0.70 | Entry threshold (applied to final_strength) |
| `sl_atr_mult` | float | 1.0 - 3.0 | ATR multiplier for stop loss |
| `rr` | float | 1.0 - 5.0 | Risk-reward ratio |

### How It Works

1. **Base strength**: Computed from LONG-only or SHORT-only conditions (whichever direction is being evaluated). This is the same as before, but SHARED conditions are excluded from both numerator and denominator.

2. **Shared bonus**: Count how many of the strategy's `shared_conditions` are true, multiply by `shared_bonus_weight`. This is a flat addition to the base strength.

3. **Clamp**: Final strength is clamped to 95% max. This prevents unrealistic confidence.

4. **Entry check**: `final_strength >= threshold` AND `long_pure > short_pure × DIRECTION_RATIO`.

### Hierarchical Deduplication

Some SHARED conditions are strict subsets of others. When the stronger condition is true, the weaker one is automatically true too. To prevent double-counting:

| Stronger condition | Weaker condition | Rule |
|---|---|---|
| `volume_gt_sma_20_2_0` (Vol > 2.0x) | `volume_gt_sma_20_1_5` (Vol > 1.5x) | If 2.0x is true, ignore 1.5x |
| `adx_14_gt_30` (ADX > 30) | `adx_14_gt_25` (ADX > 25) | If >30 is true, ignore >25 |

This is deterministic (not GA-optimized) because the hierarchy is a mathematical fact. The GA can still include both conditions in a strategy, but only the stronger one contributes to the bonus when both are true.

### Example

Strategy: 5 LONG, 5 SHORT, 2 SHARED conditions, threshold=0.50, shared_bonus_weight=0.08

Candle where: 3 LONG true, 0 SHORT true, 1 SHARED true:
- base_strength = 3/5 = 0.60
- shared_bonus = 1 × 0.08 = 0.08
- final_strength = 0.60 + 0.08 = 0.68
- 0.68 >= 0.50 ✓
- long_pure = 3/5 = 0.60, short_pure = 0/5 = 0.00
- 0.60 > 0.00 × 1.3 = 0 ✓
- → LONG entry at 68% confidence

Candle where: 2 LONG true, 0 SHORT true, 0 SHARED true:
- base_strength = 2/5 = 0.40
- shared_bonus = 0 × 0.08 = 0.00
- final_strength = 0.40 + 0.00 = 0.40
- 0.40 >= 0.50 ✗
- → No entry (threshold not met)

### Two Special Conditions (Directional Breakouts)

These conditions are currently in the SHARED pool but are **naturally directional**:

| Condition | True Direction | Description |
|-----------|---------------|-------------|
| `price_gt_high_20_1_03` | LONG only | Close > 20-day high × 1.03 (bullish breakout) |
| `price_lt_low_20_0_97` | SHORT only | Close < 20-day low × 0.97 (bearish breakdown) |

**Decision**: Keep them in SHARED pool for backward compatibility, but when computing the bonus:
- `price_gt_high_20_1_03` only contributes to the LONG bonus
- `price_lt_low_20_0_97` only contributes to the SHORT bonus

This is **hardcoded** (the direction is obvious from the condition's nature). The bonus weight itself is still GA-optimized via `shared_bonus_weight`.

### Cross-Mode Consistency

The bonus calculation must be **identical** across all 3 modes. If backtest calculates a 68% confidence entry, live must calculate the same 68% for the same conditions.

| Mode | File | Entry Function | Bonus Applied? |
|------|------|----------------|----------------|
| **Training** | `backtest.py` | `backtest_strategy()` inner loop | ✅ Yes — determines if/when strategy enters |
| **Validation** | `backtest.py` | Same `backtest_strategy()` | ✅ Yes — same code path as training |
| **Live** | `live_signal.py` | `_check_entry()` | ✅ Yes — determines live entry signals |
| **Live (recovery)** | `live_signal.py` | `_check_missed_signals()` | ✅ Yes — must match `_check_entry` exactly |

**Shared helper function:** Extract the bonus calculation into a reusable function (e.g., `compute_shared_bonus()`) to ensure consistency. Both `backtest.py` and `live_signal.py` call the same function.

```python
def compute_shared_bonus(
    conditions_df_row,           # current candle's condition values
    shared_conditions: list[str], # strategy's shared condition keys
    shared_bonus_weight: float,  # GA-optimized weight
) -> float:
    """Compute the SHARED bonus for a single candle.

    Applies hierarchical deduplication:
    - If volume_gt_sma_20_2_0 is true, ignore volume_gt_sma_20_1_5
    - If adx_14_gt_30 is true, ignore adx_14_gt_25

    Returns:
        Bonus value (0.0 to shared_bonus_weight × num_shared_conditions).
    """
```

This function handles:
1. Counting true SHARED conditions
2. Applying hierarchical deduplication (vol 2.0x > 1.5x, ADX 30 > 25)
3. Directional filtering (price_gt_high → LONG only, price_lt_low → SHORT only)
4. Multiplying count × shared_bonus_weight
5. Returning the bonus value

### Potential Gaps (things to verify)

| Area | Status | Notes |
|------|--------|-------|
| Training backtest | ✅ Covered | `backtest_strategy()` inner loop |
| Validation backtest | ✅ Covered | Same code path as training |
| Live `_check_entry` | ✅ Covered | Uses `compute_shared_bonus()` |
| Live `_check_missed_signals` | ✅ Covered | Uses `compute_shared_bonus()` |
| Confidence display (Discord) | ⚠️ Needs update | Show `base_confidence + bonus` in alert message |
| Efficiency analysis | ⚠️ Needs update | Track which SHARED conditions contribute to winning trades |
| Coverage seeds | ✅ OK | Seeds can include SHARED in shared_conditions field |
| Strategy JSON format | ⚠️ Needs update | Add `shared_conditions` and `shared_bonus_weight` fields |
| GA Individual class | ⚠️ Needs update | Add new fields to `__slots__`, `to_strategy()`, `copy()` |
| Bayesian param space | ⚠️ Needs update | Add `shared_bonus_weight` to TPE-suggested parameters |
| Crossover/mutation | ⚠️ Needs update | Handle new fields in `_mate()` and `_mutate()` |

### Requirements for the 4 Gaps

#### 1. Discord Confidence Display

When sending entry alerts via `discord_bot.py`, the confidence message should show the breakdown:

```
Confidence 68% (60% base + 8% bonus from 1 SHARED)
```

If no SHARED conditions are active (bonus = 0), show the old format:
```
Confidence 60% (6/10 LONG)
```

The `send_entry_signal()` function needs `base_confidence` and `shared_bonus` as separate parameters. The caller (`_check_entry` and `_check_missed_signals`) computes both and passes them.

#### 2. Efficiency Analysis

In `efficiency.py`, the condition usage tracking should also track SHARED conditions separately:
- Which SHARED conditions appear in strategies with positive scores
- Which SHARED conditions correlate with higher win rates
- Log a separate section: "SHARED BONUS CONDITIONS" showing each SHARED condition's contribution

This does NOT affect removal logic (SHARED conditions are never in the denominator, so they can't be "inefficient" in the traditional sense). It's purely informational — helps the user understand which SHARED conditions are worth including.

#### 3. GA Individual Class Updates

In `genetic_optimizer.py`, update the `Individual` class:

```python
class Individual:
    __slots__ = (
        "conditions",        # core LONG/SHORT conditions
        "shared_conditions", # SHARED conditions for bonus
        "shared_bonus_weight", # GA-optimized bonus weight
        "threshold",
        "sl_atr_mult",
        "rr",
        "fitness",
    )
```

Update these methods:
- `__init__()` — accept new params
- `to_strategy()` — include new fields in output dict
- `copy()` — copy new fields
- `copy_without_fitness()` — copy new fields, reset fitness

#### 4. Crossover and Mutation

In `genetic_optimizer.py`:

**Crossover (`_mate`):**
- `shared_conditions`: first half from parent 1, second half from parent 2 (same as core conditions)
- `shared_bonus_weight`: average of two parents
- Deduplicate shared_conditions (same as core conditions)

**Mutation (`_mutate`):**
- New mutation type `"shared_condition"`: swap one shared condition for another from the SHARED pool
- New mutation type `"shared_bonus_weight"`: nudge by ±0.02 (same pattern as threshold ±0.05)
- Mutation types are chosen with equal probability: `["condition", "shared_condition", "threshold", "sl_atr_mult", "rr", "shared_bonus_weight"]`

**Bayesian (`bayesian_optimizer.py`):**
- Add `shared_bonus_weight` to `trial.suggest_float()` in `_trial_to_strategy()`
- Add `shared_conditions` to `_strategy_to_params()` for seeding
- Light mutation: optionally swap 0-1 shared conditions per trial

### `generate_random_strategy()` Updates

When generating random strategies, the function must also:

1. Randomly select 0-4 SHARED conditions from the SHARED pool for `shared_conditions`
2. Set `shared_bonus_weight = random.uniform(MIN_SHARED_BONUS_WEIGHT, MAX_SHARED_BONUS_WEIGHT)`
3. Keep LONG/SHORT conditions in the existing `conditions` field (unchanged)

The `shared_conditions` list is independent of the `conditions` list. A strategy can have 0 SHARED conditions (no bonus) or up to all 9 (maximum bonus when all are true).

### `_build_coverage_seeds()` Updates

In `training.py`, the `_build_coverage_seeds()` method creates `Individual` objects directly. It must also:
- Populate `shared_conditions` with SHARED conditions that need coverage
- Set `shared_bonus_weight` to a mid-range default (e.g., 0.07)
- Or generate them randomly like `generate_random_strategy()` does

### `conditions.py` Pool Changes

`price_gt_high_20_1_03` and `price_lt_low_20_0_97` stay in `CONDITIONS_SHARED` (not moved). No pool changes needed. They are SHARED conditions but with directional filtering during bonus computation.

### Files to Modify

| File | Change |
|------|--------|
| `conditions.py` | No pool changes. Add `SHARED_DIRECTIONAL_MAP` dict mapping directional SHARED conditions to their allowed direction. |
| `config.py` | Add `MIN_SHARED_BONUS_WEIGHT = 0.0`, `MAX_SHARED_BONUS_WEIGHT = 0.15` |
| `strategy.py` | Add `shared_conditions` and `shared_bonus_weight` to strategy generation, mutation, crossover |
| `backtest.py` | Rewrite entry logic: base strength from LONG/SHORT only, add shared bonus, clamp at 95% |
| `live_signal.py` | Same entry logic change in `_check_entry` and `_check_missed_signals` |
| `genetic_optimizer.py` | Update `_mate`, `_mutate`, `Individual` to handle new fields |
| `bayesian_optimizer.py` | Add `shared_bonus_weight` to optimized parameters |
| `training.py` | Update `_eval_strategy` if needed |
| `efficiency.py` | No change needed (still analyzes condition usage) |
| `validation.py` | No change needed (uses backtest_engine) |
| `README.md` | Document the bonus system |

### Strategy JSON Format & Backward Compatibility

Existing strategies in `best_strategy.json` and `top_strategies.json` don't have `shared_conditions` or `shared_bonus_weight`. The code must handle missing fields gracefully:

- If `shared_conditions` is missing → empty list (no bonus)
- If `shared_bonus_weight` is missing → 0.0 (no bonus)
- Existing strategies work exactly as before until retrained

All JSON serialization functions (`save_strategy`, `load_strategy`, `save_top_strategies`) handle new fields automatically since they work with dicts. No special migration needed.

When loading an old strategy for live trading or validation, the missing fields default to no-bonus behavior — this is safe and backward compatible.

### Shared Conditions Count Range

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| `shared_conditions` count | 0 | 9 | No minimum — strategy can have zero SHARED conditions |

A strategy with 0 `shared_conditions` gets zero bonus (equivalent to the current behavior without the bonus system). The GA will naturally discover how many SHARED conditions to include based on their contribution to score.

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Bonus too generous (trades triggered by SHARED alone) | Low | Clamp at 95%, threshold still applies |
| Bonus too small (SHARED still irrelevant) | Medium | GA will increase weight if it helps |
| New GA parameters slow convergence | Low | Only 1-2 new continuous params |
| Directional breakout conditions misclassified | None | Hardcoded direction assignment |

### What This Does NOT Change

- DIRECTION_RATIO check (still uses pure LONG/SHORT)
- Disqualification criteria (trades/day, win rate, drawdown)
- Score calculation (rr_per_day × drawdown_penalty)
- Exit logic (SL/TP/timeout unchanged)
- Indicator computation (all pre-computed as before)

---

## See also

The investigation into why GA Gen 0 stopped producing survivors starting June 23 (root cause: `TRAINING_PERIOD_MONTHS=12`) lives in [`gen0-investigation-spec.md`](gen0-investigation-spec.md).

