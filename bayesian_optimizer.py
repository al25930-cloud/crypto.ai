"""
Bayesian Optimization for trading strategies using Optuna.

Phase 2 of the GA+Bayesian training pipeline.
Refines promising strategy regions found by the GA.
"""

import logging
import uuid
from typing import Callable, Optional

import config
from conditions import get_condition_pool
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

        # Build the search space based on direction
        # We pick the direction from seed strategies or random
        directions = ["LONG", "SHORT"]

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

        if len(pool) < config.MIN_CONDITIONS:
            # Not enough conditions to form a valid strategy
            return {
                "id": f"strat_{uuid.uuid4().hex[:8]}",
                "conditions": pool[:config.MIN_CONDITIONS],
                "threshold": 0.5,
                "sl": 1.0,
                "rr": 2.0,
                "direction": direction,
            }

        num_conditions = trial.suggest_int(
            "num_conditions", config.MIN_CONDITIONS, config.MAX_CONDITIONS
        )

        # Use full pool indices for a fixed categorical space (Optuna requirement)
        # Deduplicate by wrapping around if collision occurs
        full_pool_indices = list(range(len(pool)))
        condition_indices = []
        for i in range(config.MAX_CONDITIONS):
            idx = trial.suggest_categorical(f"cond_{i}", full_pool_indices)
            # Resolve collisions deterministically
            original_idx = idx
            while idx in condition_indices:
                idx = (idx + 1) % len(pool)
                if idx == original_idx:
                    break  # Pool exhausted
            if idx not in condition_indices:
                condition_indices.append(idx)
            if len(condition_indices) >= num_conditions:
                break

        conditions = [pool[i] for i in condition_indices]

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
        pool = get_condition_pool(direction)
        removed = _load_removed_conditions()
        pool = [c for c in pool if c not in removed]

        params = {
            "direction": direction,
            "num_conditions": len(strategy["conditions"]),
            "threshold": strategy["threshold"],
            "sl": strategy["sl"],
            "rr": strategy["rr"],
        }

        # Map conditions to indices
        for i, cond in enumerate(strategy["conditions"]):
            if cond in pool:
                params[f"cond_{i}"] = pool.index(cond)
            else:
                # Condition not in pool, use first available
                params[f"cond_{i}"] = 0

        return params
