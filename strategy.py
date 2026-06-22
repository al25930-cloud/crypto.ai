"""
Strategy generation and scoring.

Handles:
- Random strategy generation for training
- Scoring (RR/day with drawdown penalty)
- Disqualification checks
- Strategy serialization (to/from dict/JSON)
"""

import json
import logging
import random
import uuid
from pathlib import Path
from typing import Optional

import config
from conditions import (
    get_condition_pool,
    get_condition_count_range,
    get_direction_for_condition,
    get_all_condition_pools,
    ALL_CONDITIONS,
)

logger = logging.getLogger(__name__)


def generate_random_strategy(direction: Optional[str] = None) -> dict:
    """Generate a random strategy with mixed-direction conditions.

    Args:
        direction: DEPRECATED. If provided, restricts to that direction's pool
            for backward compatibility. If None (default), picks from all pools.

    Returns:
        Strategy dict with keys: id, conditions, threshold, sl_atr_mult, rr.
        No 'direction' field — direction is decided dynamically at entry time.
    """
    if direction is not None:
        # Backward-compat path (for random search mode or external callers)
        pool = get_condition_pool(direction)
    else:
        # Mixed-direction: pick from all 53 conditions
        pool = get_all_condition_pools()

    # Load removed conditions and exclude them
    removed = _load_removed_conditions()
    pool = [c for c in pool if c not in removed]

    # Apply low-efficiency weighting: include with 50% probability
    weights = _load_condition_weights()
    if weights:
        weighted_pool = [
            c for c in pool
            if weights.get(c, 1.0) >= 1.0 or random.random() < weights[c]
        ]
        if weighted_pool:
            pool = weighted_pool

    min_count, max_count = get_condition_count_range(len(pool))
    num_conditions = random.randint(min_count, max_count)

    # Ensure at least 2 LONG and 2 SHORT conditions
    long_pool = [c for c in pool if get_direction_for_condition(c) == "LONG"]
    short_pool = [c for c in pool if get_direction_for_condition(c) == "SHORT"]

    if len(long_pool) < 2 or len(short_pool) < 2:
        # Can't satisfy balance requirement — fall back to old behavior
        direction = random.choice(["LONG", "SHORT"])
        pool = get_condition_pool(direction)
        pool = [c for c in pool if c not in removed]
        num_conditions = random.randint(min_count, max_count)
        conditions = random.sample(pool, min(num_conditions, len(pool)))
    else:
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


def score_strategy(results: dict) -> float:
    """Calculate the score for a backtest result.

    Score = rr_per_day * drawdown_penalty.
    Returns -inf if the strategy is disqualified.

    Args:
        results: Backtest results dict from backtest_strategy().
            Must have keys: rr_per_day, max_drawdown, win_rate, avg_trades_per_day.

    Returns:
        Float score, or float('-inf') if disqualified.
    """
    # Disqualification checks
    if results["avg_trades_per_day"] < config.MIN_TRADES_PER_DAY:
        return float("-inf")
    if results["avg_trades_per_day"] > config.MAX_TRADES_PER_DAY:
        return float("-inf")
    if results["win_rate"] < config.MIN_WIN_RATE:
        return float("-inf")
    if results["max_drawdown"] > config.MAX_DRAWDOWN:
        return float("-inf")

    rr_per_day = results["rr_per_day"]
    max_drawdown = results["max_drawdown"]

    # Drawdown penalty
    if max_drawdown < config.DRAWDOWN_PENALTY_START:
        penalty = 1.0
    else:
        penalty = 1.0 - (
            (max_drawdown - config.DRAWDOWN_PENALTY_START)
            / (config.DRAWDOWN_PENALTY_END - config.DRAWDOWN_PENALTY_START)
        )
        penalty = max(0.0, penalty)  # Clamp to 0

    score = rr_per_day * penalty

    # Timeout penalty: reduce score if too many exits are timeouts
    total_exits = results.get("exit_sl_count", 0) + results.get("exit_tp_count", 0) + results.get("exit_timeout_count", 0)
    if total_exits > 0:
        timeout_ratio = results.get("exit_timeout_count", 0) / total_exits
        if timeout_ratio > config.TIMEOUT_PENALTY_THRESHOLD:
            score *= (1.0 - config.TIMEOUT_PENALTY)

    return score


def is_disqualified(results: dict) -> tuple[bool, str]:
    """Check if a strategy is disqualified.

    Args:
        results: Backtest results dict.

    Returns:
        Tuple of (is_disqualified: bool, reason: str).
    """
    if results["avg_trades_per_day"] < config.MIN_TRADES_PER_DAY:
        return True, f"Too few trades/day: {results['avg_trades_per_day']:.2f} < {config.MIN_TRADES_PER_DAY} minimum"
    if results["avg_trades_per_day"] > config.MAX_TRADES_PER_DAY:
        return True, f"Too many trades/day: {results['avg_trades_per_day']:.1f} > {config.MAX_TRADES_PER_DAY}"
    if results["win_rate"] < config.MIN_WIN_RATE:
        return True, f"Win rate too low: {results['win_rate']:.1%} < {config.MIN_WIN_RATE:.0%}"
    if results["max_drawdown"] > config.MAX_DRAWDOWN:
        return True, f"Drawdown too high: {results['max_drawdown']:.1%} > {config.MAX_DRAWDOWN:.0%}"
    return False, ""


def save_strategy(strategy: dict, path: Optional[Path] = None) -> None:
    """Save a strategy dict to JSON file.

    Args:
        strategy: Strategy dict.
        path: File path. Defaults to models/best_strategy.json.
    """
    path = path or config.MODEL_DIR / "best_strategy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(strategy, f, indent=2)
    logger.info(f"Strategy saved to {path}")


def load_strategy(path: Optional[Path] = None) -> Optional[dict]:
    """Load a strategy from JSON file.

    Args:
        path: File path. Defaults to models/best_strategy.json.

    Returns:
        Strategy dict, or None if file doesn't exist.
    """
    path = path or config.MODEL_DIR / "best_strategy.json"
    if not path.exists():
        logger.warning(f"Strategy file not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_top_strategies(strategies: list[dict], path: Optional[Path] = None) -> None:
    """Save top strategies to JSON file.

    Args:
        strategies: List of strategy dicts (already sorted by score, best first).
        path: File path. Defaults to models/top_strategies.json.
    """
    path = path or config.MODEL_DIR / "top_strategies.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(strategies[:config.TOP_STRATEGIES_COUNT], f, indent=2)
    logger.info(f"Top {min(len(strategies), config.TOP_STRATEGIES_COUNT)} strategies saved to {path}")


def _load_removed_conditions() -> set:
    """Load the set of removed conditions from models/removed_conditions.json.

    Returns:
        Set of removed condition key strings.
    """
    path = config.REMOVED_CONDITIONS_FILE
    if not path.exists():
        return set()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return set(data.get("removed", []))
    except Exception:
        return set()


def _load_condition_weights() -> dict:
    """Load condition weights from models/removed_conditions.json.

    Low-efficiency conditions (efficiency 0.3-0.5) get a weight of 0.5,
    meaning they have a 50% chance of being included in the pool per strategy.

    Returns:
        Dict mapping condition_key -> weight (0.5 for low-efficiency).
        Conditions not in the dict have implicit weight 1.0.
    """
    path = config.REMOVED_CONDITIONS_FILE
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        low_eff = set(data.get("low_efficiency", []))
        return {c: 0.5 for c in low_eff}
    except Exception:
        return {}


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))
