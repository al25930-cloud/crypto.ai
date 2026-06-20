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
from conditions import get_condition_pool, get_condition_count_range

logger = logging.getLogger(__name__)


def generate_random_strategy(direction: Optional[str] = None) -> dict:
    """Generate a random strategy with random conditions, threshold, SL, and RR.

    Args:
        direction: "LONG" or "SHORT". If None, chosen randomly with 50/50 odds.

    Returns:
        Strategy dict with keys: id, conditions, threshold, sl, rr, direction.
    """
    if direction is None:
        direction = random.choice(["LONG", "SHORT"])

    pool = get_condition_pool(direction)

    # Load removed conditions and exclude them
    removed = _load_removed_conditions()
    pool = [c for c in pool if c not in removed]

    min_count, max_count = get_condition_count_range(len(pool))
    num_conditions = random.randint(min_count, max_count)
    conditions = random.sample(pool, min(num_conditions, len(pool)))

    threshold = round(random.uniform(config.MIN_THRESHOLD, config.MAX_THRESHOLD), 4)
    sl = round(random.uniform(config.MIN_SL, config.MAX_SL), 2)
    rr = round(random.uniform(config.MIN_RR, config.MAX_RR), 2)

    return {
        "id": f"strat_{uuid.uuid4().hex[:8]}",
        "conditions": conditions,
        "threshold": threshold,
        "sl": sl,
        "rr": rr,
        "direction": direction,
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
    if results["avg_trades_per_day"] > config.MAX_TRADES_PER_DAY:
        return float("-inf")
    if results["win_rate"] < config.MIN_WIN_RATE:
        return float("-inf")
    if results["max_drawdown"] > config.MAX_DRAWDOWN:
        return float("-inf")

    rr_per_day = results["rr_per_day"]
    max_drawdown = results["max_drawdown"]
    avg_trades = results["avg_trades_per_day"]

    # Low trade frequency penalty
    if avg_trades <= config.LOW_TRADES_THRESHOLD:
        rr_per_day *= config.LOW_TRADES_PENALTY

    # Drawdown penalty
    if max_drawdown < config.DRAWDOWN_PENALTY_START:
        penalty = 1.0
    else:
        penalty = 1.0 - (
            (max_drawdown - config.DRAWDOWN_PENALTY_START)
            / (config.DRAWDOWN_PENALTY_END - config.DRAWDOWN_PENALTY_START)
        )
        penalty = max(0.0, penalty)  # Clamp to 0

    return rr_per_day * penalty


def is_disqualified(results: dict) -> tuple[bool, str]:
    """Check if a strategy is disqualified.

    Args:
        results: Backtest results dict.

    Returns:
        Tuple of (is_disqualified: bool, reason: str).
    """
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


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))
