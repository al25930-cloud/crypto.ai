"""
Technical conditions for strategy generation.

Conditions are organized into three pools:
- CONDITIONS_LONG: 22 bullish conditions
- CONDITIONS_SHORT: 22 bearish conditions
- CONDITIONS_SHARED: 9 direction-neutral conditions

Total: 53 unique conditions.
"""

from typing import Dict, List, Optional

# === LONG-Only Conditions (22) ===
CONDITIONS_LONG: Dict[str, str] = {
    # Trend (bullish)
    "ema_9_gt_21": "EMA(9) > EMA(21)",
    "ema_12_gt_26": "EMA(12) > EMA(26)",
    "ema_20_gt_50": "EMA(20) > SMA(50)",
    "sma_20_gt_50": "SMA(20) > SMA(50)",
    "price_gt_sma_50": "Close > SMA(50)",
    "price_gt_sma_200": "Close > SMA(200)",
    "macd_gt_signal": "MACD line > Signal line",
    "macd_hist_gt_0": "MACD histogram > 0",
    # Momentum (bullish / oversold reversal)
    "rsi_14_gt_50": "RSI(14) > 50",
    "rsi_14_lt_30": "RSI(14) < 30 (oversold reversal)",
    "rsi_21_gt_50": "RSI(21) > 50",
    "rsi_21_lt_30": "RSI(21) < 30 (oversold reversal)",
    "stoch_k_gt_20": "Stochastic %K > 20",
    "stoch_k_lt_80": "Stochastic %K < 80",
    "cci_14_gt_-100": "CCI(14) > -100",
    "cci_14_lt_100": "CCI(14) < 100",
    "williams_gt_-80": "Williams %R > -80",
    "williams_lt_-20": "Williams %R < -20",
    # Volatility (bullish)
    "price_lt_bb_lower_20_2": "Close < BB lower (20,2) — oversold bounce",
    "price_gt_bb_upper_20_2": "Close > BB upper (20,2) — breakout",
    "price_lt_bb_lower_20_1_5": "Close < BB lower (20,1.5) — oversold bounce",
    "price_gt_bb_upper_20_1_5": "Close > BB upper (20,1.5) — breakout",
}

# === SHORT-Only Conditions (22) ===
CONDITIONS_SHORT: Dict[str, str] = {
    # Trend (bearish)
    "ema_9_lt_21": "EMA(9) < EMA(21)",
    "ema_12_lt_26": "EMA(12) < EMA(26)",
    "ema_20_lt_50": "EMA(20) < SMA(50)",
    "sma_20_lt_50": "SMA(20) < SMA(50)",
    "price_lt_sma_50": "Close < SMA(50)",
    "price_lt_sma_200": "Close < SMA(200)",
    "macd_lt_signal": "MACD line < Signal line",
    "macd_hist_lt_0": "MACD histogram < 0",
    # Momentum (bearish / overbought reversal)
    "rsi_14_lt_50": "RSI(14) < 50",
    "rsi_14_gt_70": "RSI(14) > 70 (overbought reversal)",
    "rsi_21_lt_50": "RSI(21) < 50",
    "rsi_21_gt_70": "RSI(21) > 70 (overbought reversal)",
    "stoch_k_gt_80": "Stochastic %K > 80 (overbought)",
    "stoch_k_lt_20": "Stochastic %K < 20 (oversold breakdown)",
    "cci_14_gt_100": "CCI(14) > 100 (overbought)",
    "cci_14_lt_-100": "CCI(14) < -100 (breakdown)",
    "williams_gt_-20": "Williams %R > -20 (overbought)",
    "williams_lt_-80": "Williams %R < -80 (breakdown)",
    # Volatility (bearish)
    "price_gt_bb_upper_20_2_s": "Close > BB upper (20,2) — overbought reversal",
    "price_lt_bb_lower_20_2_s": "Close < BB lower (20,2) — breakdown",
    "price_gt_bb_upper_20_1_5_s": "Close > BB upper (20,1.5) — overbought reversal",
    "price_lt_bb_lower_20_1_5_s": "Close < BB lower (20,1.5) — breakdown",
}

# === Shared Conditions (9) — Used by Both LONG and SHORT ===
CONDITIONS_SHARED: Dict[str, str] = {
    "atr_gt_sma_atr_20": "ATR(14) > SMA(ATR, 20) — high volatility",
    "volume_gt_sma_20_1_5": "Volume > SMA(Volume, 20) × 1.5",
    "volume_gt_sma_20_2_0": "Volume > SMA(Volume, 20) × 2.0",
    "obv_gt_sma_obv_20": "OBV > SMA(OBV, 20)",
    "adx_14_gt_25": "ADX(14) > 25 — trending",
    "adx_14_gt_30": "ADX(14) > 30 — strong trend",
    "adx_14_lt_20": "ADX(14) < 20 — ranging/no trend",
    "price_gt_high_20_1_02": "Close > Highest(High, 20) × 1.03 — breakout",
    "price_lt_low_20_0_98": "Close < Lowest(Low, 20) × 0.97 — breakdown",
}

# === Combined Pools ===
# For LONG strategies: pick from LONG + SHARED (31 conditions)
CONDITIONS_LONG_POOL: Dict[str, str] = {**CONDITIONS_LONG, **CONDITIONS_SHARED}

# For SHORT strategies: pick from SHORT + SHARED (31 conditions)
CONDITIONS_SHORT_POOL: Dict[str, str] = {**CONDITIONS_SHORT, **CONDITIONS_SHARED}

# All conditions combined (53 unique)
ALL_CONDITIONS: Dict[str, str] = {
    **CONDITIONS_LONG,
    **CONDITIONS_SHORT,
    **CONDITIONS_SHARED,
}


def get_condition_count_range(pool_size: int) -> tuple:
    """Calculate the min/max absolute condition counts from the percentage config.

    Args:
        pool_size: Number of available conditions in the pool.

    Returns:
        Tuple of (min_count, max_count).
    """
    import config
    min_count = config.MIN_CONDITIONS_ABSOLUTE  # Hard floor: always 4
    max_count = min(pool_size, int(pool_size * config.MAX_CONDITION_PERCENTAGE))
    # Ensure min <= max
    min_count = min(min_count, max_count)
    return min_count, max_count


def get_condition_pool(direction: str) -> List[str]:
    """Get the list of condition keys for a given direction.

    Args:
        direction: "LONG" or "SHORT"

    Returns:
        List of condition key strings.

    Raises:
        ValueError: If direction is not "LONG" or "SHORT".
    """
    if direction == "LONG":
        return list(CONDITIONS_LONG_POOL.keys())
    elif direction == "SHORT":
        return list(CONDITIONS_SHORT_POOL.keys())
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'LONG' or 'SHORT'.")


def get_condition_description(condition_key: str) -> str:
    """Get the human-readable description of a condition.

    Args:
        condition_key: The condition key string.

    Returns:
        Description string, or "Unknown condition" if not found.
    """
    return ALL_CONDITIONS.get(condition_key, "Unknown condition")


def get_all_condition_pools() -> list[str]:
    """Return a combined list of all condition names from LONG, SHORT, and SHARED pools."""
    return list(ALL_CONDITIONS.keys())


# === Shared Bonus Configuration ===
# Directional SHARED conditions: only apply bonus to one direction
SHARED_DIRECTIONAL_MAP: Dict[str, str] = {
    "price_gt_high_20_1_02": "LONG",   # bullish breakout → LONG bonus only
    "price_lt_low_20_0_98": "SHORT",   # bearish breakdown → SHORT bonus only
}

# Hierarchical deduplication: if the stronger condition is true, ignore the weaker
SHARED_DEDUPLICATION: Dict[str, str] = {
    "volume_gt_sma_20_1_5": "volume_gt_sma_20_2_0",   # 1.5x is weaker than 2.0x
    "adx_14_gt_25": "adx_14_gt_30",                    # >25 is weaker than >30
}


def compute_shared_bonus(
    cond_row,                            # pandas Series or dict of condition values
    shared_conditions: list[str],         # strategy's shared condition keys
    shared_bonus_weight: float,           # GA-optimized weight per condition
    direction: str = "LONG",             # which direction we're evaluating
) -> float:
    """Compute the SHARED bonus for a single candle.

    Applies:
    1. Directional filtering (price_gt_high → LONG only, price_lt_low → SHORT only)
    2. Hierarchical deduplication (vol 2.0x > 1.5x, ADX 30 > 25)
    3. Counts true conditions × shared_bonus_weight

    Args:
        cond_row: Row from conditions DataFrame (Series with bool values).
        shared_conditions: List of SHARED condition keys in the strategy.
        shared_bonus_weight: Bonus weight per true condition.
        direction: "LONG" or "SHORT" — for directional filtering.

    Returns:
        Bonus value (0.0 to shared_bonus_weight × len(shared_conditions)).
    """
    if not shared_conditions or shared_bonus_weight <= 0:
        return 0.0

    # 1. Directional filtering: skip conditions that don't apply to this direction
    eligible = []
    for cond in shared_conditions:
        dir_filter = SHARED_DIRECTIONAL_MAP.get(cond)
        if dir_filter is not None and dir_filter != direction:
            continue  # skip — this condition only applies to the other direction
        eligible.append(cond)

    # 2. Evaluate which eligible conditions are true
    true_conds = set()
    for cond in eligible:
        val = cond_row[cond] if hasattr(cond_row, '__getitem__') else getattr(cond_row, cond, False)
        if val:
            true_conds.add(cond)

    # 3. Hierarchical deduplication: remove weaker conditions if stronger is true
    deduped = set(true_conds)
    for weaker, stronger in SHARED_DEDUPLICATION.items():
        if weaker in deduped and stronger in deduped:
            deduped.discard(weaker)  # remove weaker, keep stronger

    return len(deduped) * shared_bonus_weight


def get_direction_for_condition(condition_key: str) -> str:
    """Determine which direction(s) a condition belongs to.

    Args:
        condition_key: The condition key string.

    Returns:
        "LONG", "SHORT", "SHARED", or "UNKNOWN".
    """
    if condition_key in CONDITIONS_LONG:
        return "LONG"
    elif condition_key in CONDITIONS_SHORT:
        return "SHORT"
    elif condition_key in CONDITIONS_SHARED:
        return "SHARED"
    return "UNKNOWN"
