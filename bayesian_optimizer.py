"""
Bayesian Optimization for trading strategies using Optuna.

Phase 2 of the GA+Bayesian training pipeline.
Refines promising strategy regions found by the GA.
"""

import logging
import random as rng
import uuid
from typing import Callable, Optional

import config
from conditions import ALL_CONDITIONS, get_condition_pool, get_condition_count_range
from strategy import _load_removed_conditions

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


class BayesianOptimizer:
    """Bayesian Optimization using Optuna's TPE sampler.

    Seeds initial trials from GA's top strategies, then uses
    Tree-structured Parzen Estimator to explore the search space.
    """

    def __init__(
        self,
        eval_func: EvalFunc,
        n_trials: int = config.BAYESIAN_N_TRIALS,
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

    def run(self, seed_strategies: Optional[list[dict]] = None) -> dict:
        """Run Bayesian optimization.

        Args:
            seed_strategies: Optional list of strategy dicts from GA to seed initial trials.
                These are enqueued as the first trials in the Optuna study.

        Returns:
            Best strategy dict found.
        """
        logger.info(
            f"Bayesian: Starting | trials={self.n_trials}, "
            f"startup={self.startup_trials}"
        )
        logger.info(
            "  Each trial generates a strategy with random conditions, threshold, "
            "stop-loss, and risk-reward. The TPE model learns which parameter "
            "combinations score highest and focuses the search there."
        )

        self.all_cond_names = sorted(ALL_CONDITIONS.keys())

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=self.startup_trials,
            ),
        )

        # Enqueue seed strategies from GA
        if seed_strategies:
            for strat in seed_strategies:
                params = self._strategy_to_params(strat)
                study.enqueue_trial(params)
            logger.info(f"Bayesian: Seeded {len(seed_strategies)} strategies from GA.")

        # Objective function wrapping eval_func
        def objective(trial: optuna.Trial) -> float:
            strategy = self._trial_to_strategy(trial)
            score = self.eval_func(strategy)
            self.all_strategies.append({"strategy": strategy, "score": score})
            if score > self.best_score:
                self.best_score = score
                self.best_strategy = strategy
            return score

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        # Build best strategy
        if self.best_strategy is None:
            logger.warning("Bayesian: No valid strategy found.")
            return {"id": "none", "conditions": [], "threshold": 0.5, "sl": 1.0, "rr": 2.0, "direction": "LONG", "score": float("-inf"), "method": "bayesian"}

        best = self.best_strategy.copy()
        best["score"] = self.best_score
        best["method"] = "bayesian"

        logger.info(
            f"Bayesian: Finished | Best score: {self.best_score:.4f} | "
            f"Total strategies tested: {len(self.all_strategies)}"
        )
        logger.info(
            "  Score = rr_per_day * drawdown_penalty * low_trades_penalty. "
            "Higher = better."
        )

        return best

    def _trial_to_strategy(self, trial: optuna.Trial) -> dict:
        """Convert an Optuna trial into a strategy dict.

        Args:
            trial: Optuna trial object.

        Returns:
            Strategy dict.
        """
        direction = trial.suggest_categorical("direction", ["LONG", "SHORT"])
        pool = get_condition_pool(direction)
        removed = _load_removed_conditions()
        pool = [c for c in pool if c not in removed]
        pool_set = set(pool)

        min_count, max_count = get_condition_count_range(len(pool))

        if len(pool) < min_count:
            # Not enough conditions to form a valid strategy
            return {
                "id": f"strat_{uuid.uuid4().hex[:8]}",
                "conditions": pool[:min_count],
                "threshold": 0.5,
                "sl": 1.0,
                "rr": 2.0,
                "direction": direction,
            }

        num_conditions = trial.suggest_int(
            "num_conditions", min_count, max_count
        )

        # Use condition names as categorical values (pool-size invariant).
        # This avoids index-out-of-range errors when the pool changes size
        # between GA seeding and Bayesian execution (e.g. removed conditions).
        selected_conditions = []
        for i in range(max_count):
            name = trial.suggest_categorical(f"cond_{i}", self.all_cond_names)
            # Accept only if in current pool and not already selected
            if name in pool_set and name not in selected_conditions:
                selected_conditions.append(name)
            if len(selected_conditions) >= num_conditions:
                break

        # Safety fallback: if no valid conditions (shouldn't happen), pick randomly
        if not selected_conditions:
            selected_conditions = list(rng.sample(pool, min(num_conditions, len(pool))))

        conditions = selected_conditions

        threshold = round(
            trial.suggest_float("threshold", config.MIN_THRESHOLD, config.MAX_THRESHOLD), 4
        )
        sl = round(
            trial.suggest_float("sl", config.MIN_SL, config.MAX_SL), 2
        )
        rr = round(
            trial.suggest_float("rr", config.MIN_RR, config.MAX_RR), 2
        )

        return {
            "id": f"strat_{uuid.uuid4().hex[:8]}",
            "conditions": conditions,
            "threshold": threshold,
            "sl": sl,
            "rr": rr,
            "direction": direction,
        }

    def _strategy_to_params(self, strategy: dict) -> dict:
        """Convert a strategy dict to Optuna parameters for seeding.

        Args:
            strategy: Strategy dict.

        Returns:
            Dict of parameters compatible with study.enqueue_trial().
        """
        direction = strategy["direction"]

        # Clamp num_conditions to the valid range for the current pool
        pool = get_condition_pool(direction)
        removed = _load_removed_conditions()
        pool = [c for c in pool if c not in removed]
        min_count, max_count = get_condition_count_range(len(pool))
        num = min(max(len(strategy["conditions"]), min_count), max_count)

        params = {
            "direction": direction,
            "num_conditions": num,
            "threshold": strategy["threshold"],
            "sl": strategy["sl"],
            "rr": strategy["rr"],
        }

        # Store condition names directly (not indices) so seeded values
        # remain valid even if the pool changes size.
        for i, cond in enumerate(strategy["conditions"]):
            params[f"cond_{i}"] = cond

        return params
