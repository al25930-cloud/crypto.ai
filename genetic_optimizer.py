"""
Genetic Algorithm optimizer for trading strategies.

Phase 1 of the GA+Bayesian training pipeline.
Explores the global search space to find promising strategy regions.
"""

import logging
import random as rng
from typing import Callable, Optional

import config
from conditions import (
    get_condition_count_range,
    get_direction_for_condition,
    ALL_CONDITIONS,
    CONDITIONS_SHARED,
)
from strategy import generate_random_strategy

logger = logging.getLogger(__name__)

# Type for the evaluation function: takes a strategy dict, returns float
EvalFunc = Callable[[dict], float]


class Individual:
    """A single individual in the GA population.

    Wraps strategy parameters and tracks fitness. Uses None as the
    unevaluated sentinel (distinct from float('-inf') which means
    evaluated-but-disqualified).
    """

    __slots__ = ("conditions", "shared_conditions", "shared_bonus_weight", "threshold", "sl_atr_mult", "rr", "fitness")

    def __init__(self, conditions: list, threshold: float, sl_atr_mult: float, rr: float,
                 shared_conditions: Optional[list] = None, shared_bonus_weight: float = 0.0):
        self.conditions = list(conditions)  # always a fresh copy
        self.shared_conditions = list(shared_conditions) if shared_conditions else []
        self.shared_bonus_weight = shared_bonus_weight
        self.threshold = threshold
        self.sl_atr_mult = sl_atr_mult
        self.rr = rr
        self.fitness: Optional[float] = None  # None = not yet evaluated

    def to_strategy(self) -> dict:
        """Convert to a strategy dict."""
        return {
            "id": "",
            "conditions": list(self.conditions),
            "shared_conditions": list(self.shared_conditions),
            "shared_bonus_weight": self.shared_bonus_weight,
            "threshold": self.threshold,
            "sl_atr_mult": self.sl_atr_mult,
            "rr": self.rr,
        }

    def copy(self) -> "Individual":
        """Create a deep copy preserving fitness."""
        ind = Individual(
            list(self.conditions),
            self.threshold, self.sl_atr_mult, self.rr,
            shared_conditions=list(self.shared_conditions),
            shared_bonus_weight=self.shared_bonus_weight,
        )
        ind.fitness = self.fitness
        return ind

    def copy_without_fitness(self) -> "Individual":
        """Create a deep copy with fitness reset to None (needs re-evaluation)."""
        return Individual(
            list(self.conditions), self.threshold, self.sl_atr_mult, self.rr,
            shared_conditions=list(self.shared_conditions),
            shared_bonus_weight=self.shared_bonus_weight,
        )


def _create_random_individual() -> Individual:
    """Create a random individual from a random strategy."""
    strat = generate_random_strategy()
    return Individual(
        conditions=strat["conditions"],
        threshold=strat["threshold"],
        sl_atr_mult=strat["sl_atr_mult"],
        rr=strat["rr"],
        shared_conditions=strat.get("shared_conditions", []),
        shared_bonus_weight=strat.get("shared_bonus_weight", 0.0),
    )


def _ensure_balance(conditions: list) -> None:
    """Ensure at least 2 LONG and 2 SHORT conditions in the list (in-place).

    Adds random conditions from the appropriate pool if needed.
    """
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


def _trim_preserving_balance(conditions: list, max_count: int) -> None:
    """Trim conditions to max_count while preserving at least 2 LONG and 2 SHORT (in-place).

    Removes non-essential conditions first (SHARED, then excess LONG/SHORT beyond 2 each).
    """
    if len(conditions) <= max_count:
        return

    to_remove = len(conditions) - max_count

    # Categorize conditions by direction
    shared = [c for c in conditions if get_direction_for_condition(c) == "SHARED"]
    long_conds = [c for c in conditions if get_direction_for_condition(c) == "LONG"]
    short_conds = [c for c in conditions if get_direction_for_condition(c) == "SHORT"]

    # Build removal candidates: SHARED first, then excess LONG (beyond 2), then excess SHORT (beyond 2)
    removable = []
    removable.extend(shared)  # SHARED conditions are least essential for balance
    removable.extend(long_conds[2:])  # Keep at least 2 LONG
    removable.extend(short_conds[2:])  # Keep at least 2 SHORT

    # Remove up to `to_remove` candidates (randomly selected)
    rng.shuffle(removable)
    removed = set(removable[:to_remove])
    conditions[:] = [c for c in conditions if c not in removed]


def _mate(ind1: Individual, ind2: Individual) -> tuple[Individual, Individual]:
    """Crossover two individuals to produce two distinct offspring.

    Crossover logic:
    - Conditions: first half from parent 1, second half from parent 2 (and vice versa)
    - Shared conditions: union of both parents, deduplicated
    - Threshold, SL, RR, shared_bonus_weight: average of parents
    - No direction inheritance — direction emerges from condition mix at entry time
    """
    conds1 = list(ind1.conditions)
    conds2 = list(ind2.conditions)
    mid1 = len(conds1) // 2
    mid2 = len(conds2) // 2

    child1_conds = conds1[:mid1] + conds2[mid2:]
    child2_conds = conds2[:mid2] + conds1[mid1:]

    # Deduplicate each child
    for child_conds in [child1_conds, child2_conds]:
        seen = set()
        child_conds[:] = [c for c in child_conds if not (c in seen or seen.add(c))]

    # Enforce max conditions cap, preserving balance if possible
    pool_size = len(ALL_CONDITIONS)
    _, max_count = get_condition_count_range(pool_size)
    for child_conds in [child1_conds, child2_conds]:
        _ensure_balance(child_conds)
        if len(child_conds) > max_count:
            # Remove excess conditions while preserving at least 2 LONG + 2 SHORT
            _trim_preserving_balance(child_conds, max_count)

    # Shared conditions: union of both parents (deduplicated)
    child1_shared = list(set(ind1.shared_conditions) | set(ind2.shared_conditions))
    child2_shared = list(set(ind2.shared_conditions) | set(ind1.shared_conditions))

    # Numeric parameters: average
    child_threshold = round((ind1.threshold + ind2.threshold) / 2, 4)
    child_sl_atr_mult = round((ind1.sl_atr_mult + ind2.sl_atr_mult) / 2, 2)
    child_rr = round((ind1.rr + ind2.rr) / 2, 2)
    child_shared_bonus = round((ind1.shared_bonus_weight + ind2.shared_bonus_weight) / 2, 4)

    # New individuals have fitness=None (need evaluation)
    new_ind1 = Individual(child1_conds, child_threshold, child_sl_atr_mult, child_rr,
                          shared_conditions=child1_shared, shared_bonus_weight=child_shared_bonus)
    new_ind2 = Individual(child2_conds, child_threshold, child_sl_atr_mult, child_rr,
                          shared_conditions=child2_shared, shared_bonus_weight=child_shared_bonus)
    return new_ind1, new_ind2


def _mutate(ind: Individual, mutation_prob: float = config.GA_MUTATION_PROB) -> Individual:
    """Mutate an individual, returning a new Individual.

    If mutation occurs, fitness is reset to None (must be re-evaluated).
    If no mutation occurs, the copy retains the parent's fitness.
    """
    if rng.random() > mutation_prob:
        # No mutation — safe to keep parent's fitness
        return ind.copy()

    # Apply mutation — create a copy WITHOUT fitness (must re-evaluate)
    new_ind = ind.copy_without_fitness()

    mutation_type = rng.choice(["condition", "shared_condition", "shared_bonus_weight", "threshold", "sl_atr_mult", "rr"])

    if mutation_type == "condition":
        if len(new_ind.conditions) > 0:
            # Only swap with non-SHARED conditions (SHARED are in shared_conditions)
            all_pool = [c for c in ALL_CONDITIONS.keys() if c not in CONDITIONS_SHARED]
            removed = _load_removed()
            available = [c for c in all_pool if c not in new_ind.conditions and c not in removed]
            if available:
                idx = rng.randint(0, len(new_ind.conditions) - 1)
                new_ind.conditions[idx] = rng.choice(available)
            # Ensure balance after mutation, then trim preserving balance
            _ensure_balance(new_ind.conditions)
            _, max_count = get_condition_count_range(len(ALL_CONDITIONS))
            if len(new_ind.conditions) > max_count:
                _trim_preserving_balance(new_ind.conditions, max_count)
    elif mutation_type == "shared_condition":
        # Mutate a shared condition: swap one out, or add/remove
        shared_pool = list(CONDITIONS_SHARED.keys())
        removed = _load_removed()
        available = [c for c in shared_pool if c not in new_ind.shared_conditions and c not in removed]
        if not new_ind.shared_conditions and available:
            # Add a shared condition
            new_ind.shared_conditions.append(rng.choice(available))
        elif new_ind.shared_conditions:
            action = rng.choice(["swap", "remove"])
            if action == "swap" and available:
                idx = rng.randint(0, len(new_ind.shared_conditions) - 1)
                new_ind.shared_conditions[idx] = rng.choice(available)
            elif action == "remove":
                idx = rng.randint(0, len(new_ind.shared_conditions) - 1)
                new_ind.shared_conditions.pop(idx)
    elif mutation_type == "shared_bonus_weight":
        delta = rng.choice([-0.02, 0.02])
        new_ind.shared_bonus_weight = round(
            max(config.MIN_SHARED_BONUS_WEIGHT, min(config.MAX_SHARED_BONUS_WEIGHT, new_ind.shared_bonus_weight + delta)), 4
        )
    elif mutation_type == "threshold":
        delta = rng.choice([-0.05, 0.05])
        new_ind.threshold = round(
            max(config.MIN_THRESHOLD, min(config.MAX_THRESHOLD, new_ind.threshold + delta)), 4
        )
    elif mutation_type == "sl_atr_mult":
        delta = rng.choice([-0.2, 0.2])
        new_ind.sl_atr_mult = round(
            max(config.MIN_SL_ATR_MULT, min(config.MAX_SL_ATR_MULT, new_ind.sl_atr_mult + delta)), 2
        )
    elif mutation_type == "rr":
        delta = rng.choice([-0.5, 0.5])
        new_ind.rr = round(
            max(config.MIN_RR, min(config.MAX_RR, new_ind.rr + delta)), 2
        )

    return new_ind


_REMOVED_CACHE: set = set()
_REMOVED_LOADED = False


def _load_removed() -> set:
    """Load removed conditions (cached after first read)."""
    global _REMOVED_CACHE, _REMOVED_LOADED
    if _REMOVED_LOADED:
        return _REMOVED_CACHE
    import json
    path = config.REMOVED_CONDITIONS_FILE
    if not path.exists():
        _REMOVED_LOADED = True
        return _REMOVED_CACHE
    try:
        with open(path) as f:
            _REMOVED_CACHE = set(json.load(f).get("removed", []))
    except Exception:
        pass
    _REMOVED_LOADED = True
    return _REMOVED_CACHE


class GeneticOptimizer:
    """Genetic Algorithm optimizer.

    Evolves a population of strategies over multiple generations,
    using tournament selection, crossover, mutation, and elitism.
    """

    def __init__(
        self,
        eval_func: EvalFunc,
        population_size: int = config.GA_POPULATION_SIZE,
        generations: int = config.GA_MAX_GENERATIONS,
        elite_count: int = config.GA_ELITE_COUNT,
        crossover_prob: float = config.GA_CROSSOVER_PROB,
        mutation_prob: float = config.GA_MUTATION_PROB,
    ):
        self.eval_func = eval_func
        self.population_size = population_size
        self.generations = generations
        self.elite_count = elite_count
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # Tracking
        self.all_strategies: list[dict] = []
        self.best_individual: Optional[Individual] = None
        self.best_score: float = float("-inf")

    def run(self, time_limit_seconds: Optional[float] = None, max_generations: int = config.GA_MAX_GENERATIONS, seed_individuals: Optional[list] = None) -> tuple[dict, list[dict]]:
        """Run the genetic algorithm.

        Args:
            time_limit_seconds: Optional time budget in seconds. GA stops when exceeded.
            max_generations: Safety cap on generations (default: config.GA_MAX_GENERATIONS).
                Whichever limit (time or generations) is hit first stops the GA.
            seed_individuals: Optional list of Individual objects to inject into the
                initial population (e.g., coverage strategies). These replace the
                first N random individuals.

        Returns:
            Tuple of (best_strategy_dict, top_10_strategies_list).
            Returns an empty strategy dict if no valid strategy was found.
        """
        import time as _time
        start_time = _time.time()

        time_str = f"{time_limit_seconds:.0f}s ({time_limit_seconds / 60:.1f}m)" if time_limit_seconds else "unlimited"
        logger.info(
            f"GA: Starting | pop={self.population_size}, "
            f"time_budget={time_str}, max_gen={max_generations}, "
            f"cx={self.crossover_prob}, mut={self.mutation_prob}, elite={self.elite_count}"
        )
        logger.info(
            "  Score = rr_per_day * drawdown_penalty * low_trades_penalty. "
            "Higher = better. Negative or -inf means disqualified."
        )

        # Create initial population, seeding with provided individuals first
        population: list[Individual] = []
        if seed_individuals:
            population.extend(seed_individuals[:self.population_size])
            logger.info(f"GA: Seeded {len(population)} individuals into initial population.")
        remaining = self.population_size - len(population)
        population.extend(_create_random_individual() for _ in range(remaining))

        # Evaluate initial population
        self._evaluate_population(population)
        population.sort(key=_fitness_key, reverse=True)

        # Track best
        if population[0].fitness is not None and population[0].fitness > self.best_score:
            self.best_score = population[0].fitness
            self.best_individual = population[0].copy()

        gen_elapsed = _time.time() - start_time
        logger.info(
            f"GA Gen 0 | "
            f"Best score: {self.best_score:.4f} | "
            f"Avg score: {self._avg_score(population):.4f} | "
            f"Pop: {len(population)} | Elapsed: {gen_elapsed:.0f}s"
        )

        # Evolve
        for gen in range(1, max_generations + 1):
            # Elitism: preserve top N (copies with fitness preserved)
            elite = [population[i].copy() for i in range(min(self.elite_count, len(population)))]

            # Selection: tournament
            offspring = self._tournament_select(population, len(population) - self.elite_count)

            # Crossover (produces new Individuals with fitness=None)
            for i in range(0, len(offspring) - 1, 2):
                if rng.random() < self.crossover_prob:
                    offspring[i], offspring[i + 1] = _mate(offspring[i], offspring[i + 1])

            # Mutation (resets fitness to None if mutation occurs)
            for i in range(len(offspring)):
                offspring[i] = _mutate(offspring[i], self.mutation_prob)

            # Combine elite + offspring
            population = elite + offspring

            # Evaluate only individuals that need it (fitness is None)
            self._evaluate_population(population)

            # Sort by fitness (None treated as -inf for sorting)
            population.sort(key=_fitness_key, reverse=True)

            # Track best
            gen_best = _fitness_key(population[0])
            gen_avg = self._avg_score(population)
            new_best = False
            if gen_best > self.best_score + 1e-5:  # epsilon to avoid floating-point false positives
                self.best_score = gen_best
                self.best_individual = population[0].copy()
                new_best = True

            marker = " [NEW BEST!]" if new_best else ""
            gen_elapsed = _time.time() - start_time
            logger.info(
                f"GA Gen {gen} | "
                f"Best score: {gen_best:.4f} | Avg score: {gen_avg:.4f} | "
                f"Tested: {len(self.all_strategies)} | Elapsed: {gen_elapsed:.0f}s{marker}"
            )

            # Check time limit
            if time_limit_seconds and (_time.time() - start_time) >= time_limit_seconds:
                logger.info(f"GA: Time budget exhausted at generation {gen}. Stopping.")
                break
        else:
            if max_generations > 0:
                logger.info(f"GA: Reached max generation cap ({max_generations}). Stopping.")

        # Handle case where no valid strategy was found
        if self.best_individual is None:
            logger.warning("GA: No valid strategy found during training.")
            return {
                "id": "none", "conditions": [], "shared_conditions": [], "shared_bonus_weight": 0.0,
                "threshold": 0.5, "sl_atr_mult": 1.5, "rr": 2.0,
                "score": float("-inf"), "method": "ga",
            }, []

        # Extract top 10 strategies
        top_10 = []
        for ind in population[:10]:
            strat = ind.to_strategy()
            strat["score"] = _fitness_key(ind)
            top_10.append(strat)

        best_strat = self.best_individual.to_strategy()
        best_strat["score"] = self.best_score
        best_strat["method"] = "ga"

        total_elapsed = _time.time() - start_time
        speed = len(self.all_strategies) / total_elapsed if total_elapsed > 0 else 0
        logger.info(
            f"GA: Finished | Best score: {self.best_score:.4f} | "
            f"{gen} generations | Elapsed: {total_elapsed:.0f}s | "
            f"Strategies tested: {len(self.all_strategies)} | Speed: {speed:.1f} strats/s | "
            f"Passing top 10 to Bayesian optimizer."
        )

        return best_strat, top_10

    def _evaluate_population(self, population: list) -> None:
        """Evaluate fitness for individuals that haven't been evaluated yet.

        Uses None as the sentinel for unevaluated individuals.
        """
        for ind in population:
            if ind.fitness is None:
                strat = ind.to_strategy()
                score = self.eval_func(strat)
                ind.fitness = score
                self.all_strategies.append({
                    "strategy": strat,
                    "score": score,
                })

    def _tournament_select(self, population: list, n: int, tournsize: int = 3) -> list:
        """Tournament selection — returns fresh copies of winners."""
        selected = []
        for _ in range(n):
            aspirants = rng.sample(population, min(tournsize, len(population)))
            winner = max(aspirants, key=_fitness_key)
            selected.append(winner.copy())
        return selected

    def _avg_score(self, population: list) -> float:
        """Calculate average score of a population (excluding -inf and None)."""
        scores = [_fitness_key(ind) for ind in population if _fitness_key(ind) > float("-inf")]
        return sum(scores) / len(scores) if scores else 0.0


def _fitness_key(ind: Individual) -> float:
    """Get fitness value for sorting/comparison. None maps to -inf."""
    return ind.fitness if ind.fitness is not None else float("-inf")
