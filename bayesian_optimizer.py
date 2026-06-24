"""
Bayesian Optimization for trading strategies using Optuna.

Phase 2 of the GA+Bayesian training pipeline.
Refines promising strategy regions found by the GA.

Design: The GA excels at combinatorics (which conditions work together).
Bayesian optimization excels at continuous tuning. So we split responsibilities:
- GA provides the best condition sets (top strategies)
- Bayesian picks a GA seed as base, then optimizes threshold, sl_atr_mult, rr
- Light condition mutation (0-2 swaps) adds exploration without derailing TPE
"""

import logging
import random as rng
import uuid
from typing import Callable, Optional

import config
from conditions import ALL_CONDITIONS, CONDITIONS_SHARED, get_condition_count_range, get_direction_for_condition
from strategy import _load_removed_conditions, _load_condition_weights

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("[WARNING] Optuna not installed. Bayesian optimizer unavailable.")


# Type for the evaluation function
EvalFunc = Callable[[dict], float]


# Maximum number of conditions to randomly swap per trial (light mutation)
_MAX_CONDITION_SWAPS = 2


class BayesianOptimizer:
    """Bayesian Optimization using Optuna's TPE sampler.

    Seeds initial trials from GA's top strategies, then uses
    Tree-structured Parzen Estimator to optimize continuous parameters
    (threshold, sl_atr_mult, rr) while using light condition mutation.
    """

    def __init__(
        self,
        eval_func: EvalFunc,
        n_trials: int = config.BAYESIAN_MAX_TRIALS,
        startup_trials: int = config.BAYESIAN_STARTUP_TRIALS,
    ):
        """Initialize the Bayesian optimizer.

        Args:
            eval_func: Function that takes a strategy dict and returns a float score.
            n_trials: Total number of trials to run.
            startup_trials: Number of random trials before Bayesian model kicks in.
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is not installed. Install with: pip install optuna")

        self.eval_func = eval_func
        self.n_trials = n_trials
        self.startup_trials = startup_trials
        self.all_strategies: list[dict] = []
        self.best_score: float = float("-inf")
        self.best_strategy: Optional[dict] = None
        # Cached file I/O (loaded once per run, not per trial)
        self._cached_removed: Optional[set] = None
        self._cached_weights: Optional[dict] = None
        # Seed strategies from GA (stored for base_idx selection)
        self._seed_strategies: list[dict] = []
        # Pool of available conditions for mutation (cached per run)
        self._available_long: list[str] = []
        self._available_short: list[str] = []
        self._available_shared: list[str] = []

    def run(self, seed_strategies: Optional[list[dict]] = None, timeout_seconds: Optional[float] = None) -> dict:
        """Run Bayesian optimization.

        Args:
            seed_strategies: Optional list of strategy dicts from GA to seed initial trials.
                These are used as base strategies — TPE optimizes continuous params on top.
            timeout_seconds: Optional time budget in seconds. Optuna stops when exceeded.

        Returns:
            Best strategy dict found.
        """
        import time as _time
        start_time = _time.time()

        timeout_str = f"{timeout_seconds:.0f}s ({timeout_seconds / 60:.1f}m)" if timeout_seconds else "unlimited"
        logger.info(
            f"Bayesian: Starting | timeout={timeout_str}, "
            f"max_trials={self.n_trials}, startup={self.startup_trials}"
        )

        # Store seed strategies for base_idx selection
        self._seed_strategies = seed_strategies or []

        # Cache file I/O reads ONCE before the loop (not per trial)
        self._cached_removed = _load_removed_conditions()
        self._cached_weights = _load_condition_weights()

        # Build available condition pools for mutation (excluding removed conditions)
        removed = self._cached_removed or set()
        all_conds = [c for c in ALL_CONDITIONS.keys() if c not in removed]
        self._available_long = [c for c in all_conds if get_direction_for_condition(c) == "LONG"]
        self._available_short = [c for c in all_conds if get_direction_for_condition(c) == "SHORT"]
        self._available_shared = [c for c in all_conds if get_direction_for_condition(c) == "SHARED"]

        if self._seed_strategies:
            logger.info(
                f"Bayesian: Using {len(self._seed_strategies)} GA seed strategies as bases. "
                f"TPE optimizes threshold, SL ATR mult, and RR. "
                f"Conditions are lightly mutated (0-{_MAX_CONDITION_SWAPS} swaps per trial)."
            )
        else:
            logger.info("Bayesian: No seed strategies provided. Generating random bases.")

        # Dynamic startup trials: fewer when we have good seeds
        effective_startup = min(
            self.startup_trials,
            max(10, len(self._seed_strategies) * 2),
        ) if self._seed_strategies else self.startup_trials

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=effective_startup,
            ),
        )

        # Enqueue seed strategies as the first trials (with their original continuous params)
        if self._seed_strategies:
            for i, strat in enumerate(self._seed_strategies):
                params = self._strategy_to_params(strat, i)
                study.enqueue_trial(params)
            logger.info(f"Bayesian: Enqueued {len(self._seed_strategies)} seed trials.")

        # Manual timeout callback as safety net.
        _deadline = start_time + timeout_seconds if timeout_seconds else None

        def _timeout_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            if _deadline and _time.time() >= _deadline:
                logger.info("Bayesian: Manual timeout safety net triggered. Stopping.")
                study.stop()

        # Objective function wrapping eval_func
        def objective(trial: optuna.Trial) -> float:
            # Check deadline INSIDE objective so we stop immediately
            if _deadline and _time.time() >= _deadline:
                logger.info("Bayesian: Deadline reached inside objective. Stopping.")
                study.stop()
                raise optuna.TrialPruned()
            strategy = self._trial_to_strategy(trial)
            score = self.eval_func(strategy)
            self.all_strategies.append({"strategy": strategy, "score": score})
            if score > self.best_score:
                self.best_score = score
                self.best_strategy = strategy
            return score

        callbacks = [_timeout_callback] if _deadline else []

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=timeout_seconds,
            callbacks=callbacks,
            show_progress_bar=False,
        )

        # Build best strategy
        if self.best_strategy is None:
            logger.warning("Bayesian: No valid strategy found.")
            return {"id": "none", "conditions": [], "shared_conditions": [], "shared_bonus_weight": 0.0, "threshold": 0.5, "sl_atr_mult": 1.5, "rr": 2.0, "score": float("-inf"), "method": "bayesian"}

        best = self.best_strategy.copy()
        best["score"] = self.best_score
        best["method"] = "bayesian"

        total_elapsed = _time.time() - start_time
        speed = len(self.all_strategies) / total_elapsed if total_elapsed > 0 else 0
        logger.info(
            f"Bayesian: Finished | Best score: {self.best_score:.4f} | "
            f"{len(self.all_strategies)} trials | Elapsed: {total_elapsed:.0f}s | "
            f"Speed: {speed:.1f} trials/s"
        )
        logger.info(
            "  Score = rr_per_day * drawdown_penalty * low_trades_penalty. "
            "Higher = better."
        )

        return best

    def _trial_to_strategy(self, trial: optuna.Trial) -> dict:
        """Convert an Optuna trial into a strategy dict.

        Instead of generating conditions from scratch (which creates an impossibly
        large search space for TPE), this method:
        1. Picks a base strategy from the GA seeds via base_idx
        2. Copies its conditions + shared_conditions
        3. Applies light mutation (0-2 condition swaps)
        4. Lets TPE optimize the continuous parameters (threshold, sl_atr_mult, rr, shared_bonus_weight)

        Args:
            trial: Optuna trial object.

        Returns:
            Strategy dict (no 'direction' field — direction is dynamic).
        """
        # Step 1: Pick a base strategy from GA seeds
        if self._seed_strategies:
            base_idx = trial.suggest_categorical("base_idx", list(range(len(self._seed_strategies))))
            base = self._seed_strategies[base_idx]
            conditions = list(base["conditions"])
            shared_conditions = list(base.get("shared_conditions", []))
        else:
            # Fallback: generate random conditions if no seeds
            conditions = self._generate_random_conditions()
            shared_conditions = []

        # Step 2: Light condition mutation (0-2 swaps on core conditions)
        num_swaps = rng.randint(0, _MAX_CONDITION_SWAPS)
        for _ in range(num_swaps):
            conditions = self._swap_one_condition(conditions)

        # Ensure balance after mutation, then trim preserving balance
        self._ensure_balance(conditions)

        # Cap at max condition count while preserving at least 2 LONG + 2 SHORT
        _, max_count = get_condition_count_range(len(ALL_CONDITIONS))
        if len(conditions) > max_count:
            self._trim_preserving_balance(conditions, max_count)

        # Step 3: TPE optimizes the continuous parameters (including shared_bonus_weight)
        threshold = round(
            trial.suggest_float("threshold", config.MIN_THRESHOLD, config.MAX_THRESHOLD), 4
        )
        sl_atr_mult = round(
            trial.suggest_float("sl_atr_mult", config.MIN_SL_ATR_MULT, config.MAX_SL_ATR_MULT), 2
        )
        rr = round(
            trial.suggest_float("rr", config.MIN_RR, config.MAX_RR), 2
        )
        shared_bonus_weight = round(
            trial.suggest_float("shared_bonus_weight", config.MIN_SHARED_BONUS_WEIGHT, config.MAX_SHARED_BONUS_WEIGHT), 4
        )

        return {
            "id": f"strat_{uuid.uuid4().hex[:8]}",
            "conditions": conditions,
            "shared_conditions": shared_conditions,
            "shared_bonus_weight": shared_bonus_weight,
            "threshold": threshold,
            "sl_atr_mult": sl_atr_mult,
            "rr": rr,
            # No "direction" field — direction is dynamic
        }

    def _swap_one_condition(self, conditions: list[str]) -> list[str]:
        """Swap one random condition for another of the same direction.

        Note: SHARED conditions are NOT in the core conditions list — they are in
        the separate shared_conditions field. So this only swaps LONG/SHORT conditions.

        Args:
            conditions: Current condition list (core LONG/SHORT only).

        Returns:
            New condition list with one swap applied (or unchanged if no swap possible).
        """
        if not conditions:
            return conditions

        conditions = list(conditions)

        # Pick a random condition to swap
        idx = rng.randint(0, len(conditions) - 1)
        old_cond = conditions[idx]
        old_direction = get_direction_for_condition(old_cond)

        # Pick a replacement from the same direction pool
        if old_direction == "LONG":
            pool = self._available_long
        elif old_direction == "SHORT":
            pool = self._available_short
        else:
            # SHARED shouldn't be in core conditions, but fallback just in case
            pool = self._available_shared

        available = [c for c in pool if c not in conditions]
        if not available:
            return conditions  # No swap possible

        conditions[idx] = rng.choice(available)
        return conditions

    def _ensure_balance(self, conditions: list[str]) -> None:
        """Ensure at least 2 LONG and 2 SHORT conditions (in-place).

        Args:
            conditions: Condition list to enforce balance on.
        """
        long_count = sum(1 for c in conditions if get_direction_for_condition(c) == "LONG")
        short_count = sum(1 for c in conditions if get_direction_for_condition(c) == "SHORT")

        available_long = [c for c in self._available_long if c not in conditions]
        available_short = [c for c in self._available_short if c not in conditions]

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

    def _trim_preserving_balance(self, conditions: list[str], max_count: int) -> None:
        """Trim conditions to max_count while preserving at least 2 LONG and 2 SHORT (in-place)."""
        if len(conditions) <= max_count:
            return

        to_remove = len(conditions) - max_count

        shared = [c for c in conditions if get_direction_for_condition(c) == "SHARED"]
        long_conds = [c for c in conditions if get_direction_for_condition(c) == "LONG"]
        short_conds = [c for c in conditions if get_direction_for_condition(c) == "SHORT"]

        removable = []
        removable.extend(shared)
        removable.extend(long_conds[2:])
        removable.extend(short_conds[2:])

        rng.shuffle(removable)
        removed = set(removable[:to_remove])
        conditions[:] = [c for c in conditions if c not in removed]

    def _generate_random_conditions(self) -> list[str]:
        """Generate random conditions as a fallback when no seeds are available.

        Returns:
            List of condition keys with balanced LONG/SHORT mix.
        """
        removed = self._cached_removed or set()
        pool = [c for c in ALL_CONDITIONS.keys() if c not in removed]

        min_count, max_count = get_condition_count_range(len(pool))
        num_conditions = rng.randint(min_count, max_count)

        long_pool = [c for c in pool if get_direction_for_condition(c) == "LONG"]
        short_pool = [c for c in pool if get_direction_for_condition(c) == "SHORT"]

        long_conditions = rng.sample(long_pool, min(2, len(long_pool)))
        short_conditions = rng.sample(short_pool, min(2, len(short_pool)))

        already_selected = set(long_conditions + short_conditions)
        available = [c for c in pool if c not in already_selected]
        remaining = num_conditions - len(long_conditions) - len(short_conditions)

        if remaining > 0 and available:
            extra = rng.sample(available, min(remaining, len(available)))
        else:
            extra = []

        return long_conditions + short_conditions + extra

    def _strategy_to_params(self, strategy: dict, base_idx: int = 0) -> dict:
        """Convert a strategy dict to Optuna parameters for seeding.

        Args:
            strategy: Strategy dict.
            base_idx: Index of this strategy in the seed list.

        Returns:
            Dict of parameters compatible with study.enqueue_trial().
        """
        return {
            "base_idx": base_idx,
            "threshold": strategy["threshold"],
            "sl_atr_mult": strategy.get("sl_atr_mult", strategy.get("sl", 1.5)),
            "rr": strategy["rr"],
            "shared_bonus_weight": strategy.get("shared_bonus_weight", 0.0),
        }
