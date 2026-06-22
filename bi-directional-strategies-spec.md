# Bi-Directional Strategies Spec

## Overview

**Goal:** Allow strategies to use conditions from both LONG and SHORT pools, deciding direction dynamically at entry time based on which conditions dominate.

**Breaking change:** Old single-direction strategies (with a `direction` field) are incompatible. Users must retrain after this change.

**Scope:** Core files only (config, conditions, strategy, backtest, GA, Bayesian, training). Live/validation/discord handled in a separate follow-up.

---

## 1. New Configuration Parameters (config.py)

```python
# === Dynamic Direction Thresholds ===
MIN_DIRECTION_STRENGTH = 0.60   # Minimum strength for dominant direction (60% of that direction's conditions must be true)
DIRECTION_RATIO = 1.3           # Dominant direction must be at least 1.3× stronger than opposite direction
```

These are **fixed config values**, not per-strategy parameters. They define the minimum clarity required for a trade — a system-wide safety rule.

Also update:
```python
MAX_CONDITION_PERCENTAGE = 0.65  # Reduced from 0.90 — with 53 conditions, 65% = 34 conditions
```

Remove redundant hard cap (0.65 × 53 = 34, which is already reasonable):
```python
# MAX_CONDITIONS_ABSOLUTE removed — redundant with MAX_CONDITION_PERCENTAGE=0.65
```

### Rationale for fixed config values

- Evolving MIN_DIRECTION_STRENGTH/DIRECTION_RATIO per strategy would allow the GA to set its own entry thresholds, leading to overfitting and ambiguous entries (e.g., DIRECTION_RATIO = 1.0 = enter on any small majority).
- Fixed values ensure consistency across all strategies and make the system more predictable.
- Users can adjust them globally in config.py if needed.

---

## 2. Condition Balance Rules

Every strategy **must** have:
- **At least 2 LONG conditions**
- **At least 2 SHORT conditions**

This ensures all strategies are genuinely mixed-direction and capable of making a balanced decision.

| Total conditions | Must have | Can have |
|---|---|---|
| 4 (minimum) | 2 LONG + 2 SHORT | 0 SHARED |
| 5+ | 2 LONG + 2 SHORT | SHARED or additional LONG/SHORT |

### Rationale

| Situation | With 1 of each | With 2 of each |
|---|---|---|
| 3 LONG, 1 SHORT | Direction is almost always LONG (1 SHORT doesn't provide balance) | With 3 LONG and 2 SHORT, there's a real decision |
| 4 conditions | Could be 3 LONG + 1 SHORT (effectively LONG) | Must be 2 LONG + 2 SHORT (balanced) |

---

## 3. Entry Logic (backtest.py)

Replace the current direction-fixed entry logic with dynamic direction decision.

### Current logic
```python
if satisfies >= threshold:
    if direction == "LONG":
        # enter LONG
    else:
        # enter SHORT
```

### Pre-compute category masks (before the loop)

```python
# Pre-compute which conditions belong to which category
from conditions import get_direction_for_condition

conditions = strategy["conditions"]
long_indices = [i for i, c in enumerate(conditions) if get_direction_for_condition(c) == "LONG"]
short_indices = [i for i, c in enumerate(conditions) if get_direction_for_condition(c) == "SHORT"]
shared_indices = [i for i, c in enumerate(conditions) if get_direction_for_condition(c) == "SHARED"]

total_long = len(long_indices)
total_short = len(short_indices)
total_conditions = len(conditions)
```

### New entry logic (inside the loop)

```python
if not in_position and cooldown_remaining == 0:
    # Overall signal strength check
    total_true = int(satisfies[i] * total_conditions)  # Or use pre-computed sum
    if total_true / total_conditions >= threshold:
        # Calculate per-direction strength
        long_true = sum(1 for idx in long_indices if conditions_df.iloc[i, idx])
        short_true = sum(1 for idx in short_indices if conditions_df.iloc[i, idx])
        
        long_strength = long_true / total_long if total_long > 0 else 0
        short_strength = short_true / total_short if total_short > 0 else 0
        
        # Determine direction
        if long_strength >= config.MIN_DIRECTION_STRENGTH and long_strength > short_strength * config.DIRECTION_RATIO:
            direction = "LONG"
        elif short_strength >= config.MIN_DIRECTION_STRENGTH and short_strength > long_strength * config.DIRECTION_RATIO:
            direction = "SHORT"
        else:
            direction = None  # HOLD — ambiguous
        
        if direction is not None:
            in_position = True
            entry_price = closes[i]
            entry_time = ts
            entry_idx = i
            atr_at_entry = atr_values[i]
            sl_distance = atr_at_entry * sl_atr_mult
            
            if direction == "LONG":
                sl_price = entry_price - sl_distance
                tp_price = entry_price + sl_distance * rr_ratio
            else:  # SHORT
                sl_price = entry_price + sl_distance
                tp_price = entry_price - sl_distance * rr_ratio
```

**Key fix:** The `direction` variable is computed dynamically per-candle. The old code had `direction = strategy["direction"]` which would KeyError.

### Key rule: No mid-trade flipping

Once in a position, **ignore all signals** (both LONG and SHORT) until exit + cooldown. The dynamic direction logic only applies at entry time. If the market reverses, the SL will capture it.

This is consistent with the earlier decision to reject signal reversal.

### Condition categories

Compute on-the-fly using `get_direction_for_condition()` from conditions.py. **Do not store categories in the strategy dict** — avoids duplication and sync issues.

---

## 4. Strategy Generation (strategy.py)

### Changes to `generate_random_strategy()`

```python
def generate_random_strategy(direction: Optional[str] = None) -> dict:
    """Generate a random strategy with mixed-direction conditions.
    
    Args:
        direction: DEPRECATED. If provided, restricts to that direction's pool.
                   If None (default), picks from all 53 conditions.
    """
    # If deprecated direction param is used, filter pool
    if direction is not None:
        pool = get_condition_pool(direction)  # Old behavior
    else:
        # Mixed-direction: pick from all pools
        pool = list(ALL_CONDITIONS.keys())
    
    # Remove excluded conditions
    removed = _load_removed_conditions()
    pool = [c for c in pool if c not in removed]
    
    # Apply low-efficiency weighting
    weights = _load_condition_weights()
    if weights:
        weighted_pool = [c for c in pool if weights.get(c, 1.0) >= 1.0 or random.random() < weights[c]]
        if weighted_pool:
            pool = weighted_pool
    
    # Determine total conditions
    min_count, max_count = get_condition_count_range(len(pool))
    num_conditions = random.randint(min_count, max_count)
    
    # Ensure at least 2 LONG and 2 SHORT conditions
    long_pool = [c for c in pool if get_direction_for_condition(c) == "LONG"]
    short_pool = [c for c in pool if get_direction_for_condition(c) == "SHORT"]
    
    if len(long_pool) < 2 or len(short_pool) < 2:
        # Can't satisfy balance requirement — fall back to old behavior
        return _generate_single_direction_strategy()
    
    long_conditions = random.sample(long_pool, 2)
    short_conditions = random.sample(short_pool, 2)
    
    # Fill remaining slots from any pool
    remaining = num_conditions - 4
    already_selected = set(long_conditions + short_conditions)
    available = [c for c in pool if c not in already_selected]
    
    if remaining > 0 and available:
        extra_conditions = random.sample(available, min(remaining, len(available)))
    else:
        extra_conditions = []
    
    conditions = long_conditions + short_conditions + extra_conditions
    
    # Generate numeric parameters
    threshold = round(random.uniform(config.MIN_THRESHOLD, config.MAX_THRESHOLD), 4)
    sl_atr_mult = round(random.uniform(config.MIN_SL_ATR_MULT, config.MAX_SL_ATR_MULT), 2)
    rr = round(random.uniform(config.MIN_RR, config.MAX_RR), 2)
    
    return {
        "id": f"strat_{uuid.uuid4().hex[:8]}",
        "conditions": conditions,
        "threshold": threshold,
        "sl_atr_mult": sl_atr_mult,
        "rr": rr,
        # No "direction" field — direction is dynamic
    }
```

### Strategy dict format (new)

```python
{
    "id": "strat_xxx",
    "conditions": ["ema_9_gt_21", "rsi_14_gt_50", "adx_14_gt_25", ...],
    "threshold": 0.5,
    "sl_atr_mult": 1.8,
    "rr": 3.0,
    # No "direction" field
}
```

---

## 5. Genetic Optimizer Changes (genetic_optimizer.py)

### Individual class

Remove `direction` from `__slots__` and `__init__`:

```python
class Individual:
    __slots__ = ("conditions", "threshold", "sl_atr_mult", "rr", "fitness")
    
    def __init__(self, conditions, threshold, sl_atr_mult, rr):
        self.conditions = list(conditions)
        self.threshold = threshold
        self.sl_atr_mult = sl_atr_mult
        self.rr = rr
        self.fitness: Optional[float] = None
    
    def to_strategy(self) -> dict:
        return {
            "id": "",
            "conditions": list(self.conditions),
            "threshold": self.threshold,
            "sl_atr_mult": self.sl_atr_mult,
            "rr": self.rr,
        }
```

### Crossover

Combine conditions from two parents. No direction inheritance.

```python
def _mate(ind1, ind2):
    # Swap condition subsets
    mid1 = len(ind1.conditions) // 2
    mid2 = len(ind2.conditions) // 2
    child1_conds = ind1.conditions[:mid1] + ind2.conditions[mid2:]
    child2_conds = ind2.conditions[:mid2] + ind1.conditions[mid1:]
    
    # Deduplicate
    for child_conds in [child1_conds, child2_conds]:
        seen = set()
        child_conds[:] = [c for c in child_conds if not (c in seen or seen.add(c))]
    
    # Ensure balance: at least 2 LONG + 2 SHORT
    _ensure_balance(child1_conds)
    _ensure_balance(child2_conds)
    
    # Enforce max conditions cap
    max_count = config.get_condition_count_range(len(ALL_CONDITIONS))[1]
    child1_conds[:] = child1_conds[:max_count]
    child2_conds[:] = child2_conds[:max_count]
    
    # Numeric parameters: average
    child_threshold = round((ind1.threshold + ind2.threshold) / 2, 4)
    child_sl = round((ind1.sl_atr_mult + ind2.sl_atr_mult) / 2, 2)
    child_rr = round((ind1.rr + ind2.rr) / 2, 2)
    
    return (
        Individual(child1_conds, child_threshold, child_sl, child_rr),
        Individual(child2_conds, child_threshold, child_sl, child_rr),
    )
```

### Mutation

Pick from any pool (LONG, SHORT, or SHARED):

```python
def _mutate(ind):
    mutation_type = rng.choice(["condition", "threshold", "sl_atr_mult", "rr"])
    
    if mutation_type == "condition":
        # Replace a random condition with one from any pool
        all_pool = list(ALL_CONDITIONS.keys())
        removed = _load_removed()
        available = [c for c in all_pool if c not in ind.conditions and c not in removed]
        if available:
            idx = rng.randint(0, len(ind.conditions) - 1)
            ind.conditions[idx] = rng.choice(available)
        # Ensure balance after mutation
        _ensure_balance(ind.conditions)
```

### Balance enforcement

```python
def _ensure_balance(conditions):
    """Ensure at least 2 LONG and 2 SHORT conditions."""
    long_count = sum(1 for c in conditions if get_direction_for_condition(c) == "LONG")
    short_count = sum(1 for c in conditions if get_direction_for_condition(c) == "SHORT")
    
    all_long = [c for c in ALL_CONDITIONS.keys() if get_direction_for_condition(c) == "LONG"]
    all_short = [c for c in ALL_CONDITIONS.keys() if get_direction_for_condition(c) == "SHORT"]
    
    available_long = [c for c in all_long if c not in conditions]
    available_short = [c for c in all_short if c not in conditions]
    
    while long_count < 2 and available_long:
        extra = rng.choice(available_long)
        conditions.append(extra)
        available_long.remove(extra)
        long_count += 1
    
    while short_count < 2 and available_short:
        extra = rng.choice(available_short)
        conditions.append(extra)
        available_short.remove(extra)
        short_count += 1
```

---

## 6. Bayesian Optimizer Changes (bayesian_optimizer.py)

### Remove direction from `_trial_to_strategy()`

```python
def _trial_to_strategy(self, trial: optuna.Trial) -> dict:
    # Remove: direction = trial.suggest_categorical("direction", ["LONG", "SHORT"])
    # Remove: pool = get_condition_pool(direction)
    
    # Use full pool instead
    pool = list(ALL_CONDITIONS.keys())
    removed = self._cached_removed or set()
    pool = [c for c in pool if c not in removed]
    
    # ... rest of condition selection stays the same ...
    
    return {
        "id": f"strat_{uuid.uuid4().hex[:8]}",
        "conditions": conditions,
        "threshold": threshold,
        "sl_atr_mult": sl_atr_mult,
        "rr": rr,
        # No "direction" field
    }
```

### Remove direction from `_strategy_to_params()`

```python
def _strategy_to_params(self, strategy: dict) -> dict:
    # Remove direction extraction
    # Remove pool filtering by direction
    
    pool = list(ALL_CONDITIONS.keys())
    removed = self._cached_removed or set()
    pool = [c for c in pool if c not in removed]
    min_count, max_count = get_condition_count_range(len(pool))
    num = min(max(len(strategy["conditions"]), min_count), max_count)
    
    params = {
        # Remove: "direction": direction,
        "num_conditions": num,
        "threshold": strategy["threshold"],
        "sl_atr_mult": strategy.get("sl_atr_mult", strategy.get("sl", 1.5)),
        "rr": strategy["rr"],
    }
    
    for i, cond in enumerate(strategy["conditions"]):
        params[f"cond_{i}"] = cond
    
    return params
```

### Update fallback dict

```python
# Old fallback:
return {"id": "none", "conditions": [], "threshold": 0.5, "sl_atr_mult": 1.5, "rr": 2.0, "direction": "LONG", ...}

# New fallback:
return {"id": "none", "conditions": [], "threshold": 0.5, "sl_atr_mult": 1.5, "rr": 2.0, ...}
```

---

## 7. Condition Pool Changes (conditions.py)

### Add new function

```python
def get_all_condition_pools() -> list:
    """Return a combined list of all condition names from LONG, SHORT, and SHARED pools."""
    return list(CONDITIONS_LONG.keys()) + list(CONDITIONS_SHORT.keys()) + list(CONDITIONS_SHARED.keys())
```

### Update `get_condition_count_range()`

```python
def get_condition_count_range(pool_size: int) -> tuple:
    import config
    min_count = config.MIN_CONDITIONS_ABSOLUTE  # 4
    max_count = min(pool_size, int(pool_size * config.MAX_CONDITION_PERCENTAGE))  # 0.65
    min_count = min(min_count, max_count)
    return min_count, max_count
```

---

## 7. Training Logging (training.py)

### Update `_log_finish()` to show direction distribution

```python
# Show direction mix of best strategy
if best.get('conditions'):
    from conditions import get_direction_for_condition
    long_conds = sum(1 for c in best['conditions'] if get_direction_for_condition(c) == 'LONG')
    short_conds = sum(1 for c in best['conditions'] if get_direction_for_condition(c) == 'SHORT')
    shared_conds = sum(1 for c in best['conditions'] if get_direction_for_condition(c) == 'SHARED')
    logger.info(f"  Best strategy:       {best.get('id', 'N/A')} (LONG:{long_conds} SHORT:{short_conds} SHARED:{shared_conds})")
else:
    logger.info(f"  Best strategy:       {best.get('id', 'N/A')}")
```

---

## 8. Backward Compatibility

**No backward compatibility.** Old strategies with a `direction` field are incompatible.

### Handling old strategies

In `backtest.py`, `live_signal.py`, and `validation.py`:
- If a strategy has a `direction` field → log a warning and skip/fail
- Document in README that this is a breaking change and users must retrain

### File cleanup

Delete `models/best_strategy.json` and `models/top_strategies.json` before first training run with new code.

---

## 9. Edge Cases

| Case | Behavior |
|---|---|
| Strategy with only LONG conditions (no SHORT) | Rejected by balance check (requires 2 SHORT) |
| Strategy with only SHARED conditions | Rejected by balance check (requires 2 LONG + 2 SHORT) |
| Strategy with 4 conditions | Must be exactly 2 LONG + 2 SHORT (0 SHARED) |
| Long strength = 0.65, Short strength = 0.50 | LONG wins (0.65 >= 0.60 AND 0.65 > 0.50 * 1.3 = 0.65) — edge case, enters LONG |
| Long strength = 0.60, Short strength = 0.55 | HOLD (0.60 >= 0.60 but 0.60 <= 0.55 * 1.3 = 0.715) — ratio not met |
| Both strengths below 0.60 | HOLD — neither meets minimum strength |

---

## 10. Testing Strategy

After implementation:
1. **Smoke test:** Run training with `--minutes 3` and verify it completes without errors
2. **Direction distribution:** Check best strategy's mix — should show LONG/SHORT/SHARED counts
3. **Entry logic test:** Manually verify that 3 LONG true + 1 SHORT true enters LONG (if MIN_DIRECTION_STRENGTH=0.60 and RATIO=1.3)
4. **Trade frequency test:** Monitor trades/day — should increase because strategies can act in both directions
5. **Balance check:** Verify no strategy has < 2 LONG or < 2 SHORT conditions

---

## 11. Implementation Order

1. Add new config parameters (config.py)
2. Add `get_all_condition_pools()` and update `get_condition_count_range()` (conditions.py)
3. Modify `generate_random_strategy()` to use mixed pools (strategy.py)
4. Remove `direction` from Individual class and update crossover/mutation (genetic_optimizer.py)
5. Update Bayesian optimizer to remove direction (bayesian_optimizer.py)
6. Modify entry logic in backtest.py to use dynamic direction
7. Update training.py logging
8. Run smoke test
9. Run full training and verify direction distribution

---

## 12. Potential Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Overfitting due to more flexibility | MIN_DIRECTION_STRENGTH=0.60 and DIRECTION_RATIO=1.3 ensure clear consensus |
| Increased search space | MAX_CONDITIONS_ABSOLUTE=35 caps complexity |
| Strategies might never enter | MIN_TRADES_PER_DAY=1.2 disqualifies non-trading strategies |
| Shared conditions dilute direction signal | Shared conditions count toward total strength but not toward direction; this is intentional |
| Backtest slower (per-category computation) | Accept initially, optimize with numpy if needed |
| GA produces degenerate strategies | Balance check ensures at least 2 LONG + 2 SHORT conditions |
