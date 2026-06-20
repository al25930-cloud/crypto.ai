"""
Genetic Algorithm optimizer for trading strategies.

Phase 1 of the GA+Bayesian training pipeline.
Explores the global search space to find promising strategy regions.
"""

import logging
import random as rng
from typing import Callable, Optional

import config
from conditions import get_condition_count_range
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

    __slots__ = ("direction", "conditions", "threshold", "sl", "rr", "fitness")

    def __init__(self, direction: str, conditions: list, threshold: float, sl: float, rr: float):
        self.direction = direction
        self.conditions = list(conditions)  # always a fresh copy
        self.threshold = threshold
        self.sl = sl
        self.rr = rr
        self.fitness: Optional[float] = None  # None = not yet evaluated

    def to_strategy(self) -> dict:
        """Convert to a strategy dict."""
        return {
            "id": "",
            "direction": self.direction,
            "conditions": list(self.conditions),
            "threshold": self.threshold,
            "sl": self.sl,
            "rr": self.rr,
        }

    def copy(self) -> "Individual":
        """Create a deep copy preserving fitness."""
        ind = Individual(
            self.direction, list(self.conditions),
            self.threshold, self.sl, self.rr,
        )
        ind.fitness = self.fitness
        return ind

    def copy_without_fitness(self) -> "Individual":
        """Create a deep copy with fitness reset to None (needs re-evaluation)."""
        return Individual(
            self.direction, list(self.conditions),
            self.threshold, self.sl, self.rr,
        )


def _create_random_individual(direction: Optional[str] = None) -> Individual:
    """Create a random individual from a random strategy."""
    strat = generate_random_strategy(direction)
    return Individual(
        direction=strat["direction"],
        conditions=strat["conditions"],
        threshold=strat["threshold"],
        sl=strat["sl"],
        rr=strat["rr"],
    )


def _mate(ind1: Individual, ind2: Individual) -> tuple[Individual, Individual]:
    """Crossover two individuals to produce two distinct offspring.

    Crossover logic (from spec section 5.3.1):
    - Conditions: first half from parent 1, second half from parent 2 (and vice versa)
    - Threshold, SL, RR: average of parents
    - Direction: inherit from the respective parent
    """
    conds1 = list(ind1.conditions)
    conds2 = list(ind2.conditions)
    mid1 = len(conds1) // 2
    mid2 = len(conds2) // 2

    child1_conds = conds1[:mid1] + conds2[mid2:]
    child2_conds = conds2[:mid2] + conds1[mid1:]

    # Deduplicate and pad each child
    from conditions import get_condition_pool
    for child_conds, direction in [(child1_conds, ind1.direction), (child2_conds, ind2.direction)]:
        seen = set()
        child_conds[:] = [c for c in child_conds if not (c in seen or seen.add(c))]
        pool = get_condition_pool(direction)
        removed = _load_removed()
        pool = [c for c in pool if c not in removed]
        min_count, max_count = get_condition_count_range(len(pool))
        while len(child_conds) < min_count:
            extra = rng.choice(pool)
            if extra not in child_conds:
                child_conds.append(extra)
        child_conds[:] = child_conds[:max_count]

    # Numeric parameters: average
    child_threshold = round((ind1.threshold + ind2.threshold) / 2, 4)
    child_sl = round((ind1.sl + ind2.sl) / 2, 2)
    child_rr = round((ind1.rr + ind2.rr) / 2, 2)

    # New individuals have fitness=None (need evaluation)
    new_ind1 = Individual(ind1.direction, child1_conds, child_threshold, child_sl, child_rr)
    new_ind2 = Individual(ind2.direction, child2_conds, child_threshold, child_sl, child_rr)
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

    mutation_type = rng.choice(["condition", "threshold", "sl", "rr"])

    if mutation_type == "condition":
        if len(new_ind.conditions) > 0:
            from conditions import get_condition_pool
            pool = get_condition_pool(new_ind.direction)
            removed = _load_removed()
            pool = [c for c in pool if c not in removed]
            available = [c for c in pool if c not in new_ind.conditions]
            if available:
                idx = rng.randint(0, len(new_ind.conditions) - 1)
                new_ind.conditions[idx] = rng.choice(available)
    elif mutation_type == "threshold":
        delta = rng.choice([-0.05, 0.05])
        new_ind.threshold = round(
            max(config.MIN_THRESHOLD, min(config.MAX_THRESHOLD, new_ind.threshold + delta)), 4
        )
    elif mutation_type == "sl":
        delta = rng.choice([-0.2, 0.2])
        new_ind.sl = round(
            max(config.MIN_SL, min(config.MAX_SL, new_ind.sl + delta)), 2
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
        generations: int = config.GA_GENERATIONS,
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

    def run(self) -> tuple[dict, list[dict]]:
        """Run the genetic algorithm.

        Returns:
            Tuple of (best_strategy_dict, top_10_strategies_list).
            Returns an empty strategy dict if no valid strategy was found.
        """
        logger.info(
            f"GA: Starting | pop={self.population_size}, gen={self.generations}, "
            f"cx={self.crossover_prob}, mut={self.mutation_prob}, elite={self.elite_count}"
        )
        logger.info(
            "  Score = rr_per_day * drawdown_penalty * low_trades_penalty. "
            "Higher = better. Negative or -inf means disqualified."
        )

        # Create initial population
        population = [_create_random_individual() for _ in range(self.population_size)]

        # Evaluate initial population
        self._evaluate_population(population)
        population.sort(key=_fitness_key, reverse=True)

        # Track best
        if population[0].fitness is not None and population[0].fitness > self.best_score:
            self.best_score = population[0].fitness
            self.best_individual = population[0].copy()

        logger.info(
            f"GA Gen 0/{self.generations} | "
            f"Best score: {self.best_score:.4f} | "
            f"Avg score: {self._avg_score(population):.4f} | "
            f"Pop: {len(population)}"
        )

        # Evolve
        for gen in range(1, self.generations + 1):
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
            if gen_best > self.best_score:
                self.best_score = gen_best
                self.best_individual = population[0].copy()
                new_best = True

            marker = " [NEW BEST!]" if new_best else ""
            logger.info(
                f"GA Gen {gen}/{self.generations} | "
                f"Best score: {gen_best:.4f} | Avg score: {gen_avg:.4f} | "
                f"Tested: {len(self.all_strategies)}{marker}"
            )

        # Handle case where no valid strategy was found
        if self.best_individual is None:
            logger.warning("GA: No valid strategy found during training.")
            return {
                "id": "none", "conditions": [], "threshold": 0.5,
                "sl": 1.0, "rr": 2.0, "direction": "LONG",
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

        logger.info(
            f"GA: Finished | Best score: {self.best_score:.4f} | "
            f"Total strategies tested: {len(self.all_strategies)} | "
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
