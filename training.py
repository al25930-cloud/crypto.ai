"""
Training loop for the crypto trading strategy optimizer.

Default method: GA + Bayesian Optimization (ga_bayesian).
Optional method: Pure random search (--method random, for quick testing).

Usage:
    python training.py --symbol BTC/USDT                     # GA + Bayesian (default)
    python training.py --symbol BTC/USDT --method random     # Random search
    python training.py --symbol BTC/USDT --minutes 5 --method random  # Quick smoke test
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

import config
from backtest import backtest_strategy, prepare_data
from conditions import ALL_CONDITIONS, CONDITIONS_SHARED, get_direction_for_condition
from data_fetcher import fetch_ohlcv, get_training_data
from efficiency import analyze_conditions
from indicators import compute_all_conditions
from strategy import (
    generate_random_strategy,
    is_disqualified,
    load_strategy,
    save_strategy,
    save_top_strategies,
    score_strategy,
)

logger = logging.getLogger(__name__)


class TrainingSession:
    """Manages a full training session with progress tracking.

    Orchestrates the GA + Bayesian pipeline or random search,
    tracks all tested strategies, and saves results.
    """

    def __init__(
        self,
        symbol: str = config.SYMBOL,
        timeframe: str = config.TIMEFRAME,
        training_minutes: int = config.TRAINING_MINUTES,
        method: str = config.TRAINING_METHOD,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.training_minutes = training_minutes
        self.method = method

        # Results tracking
        self.all_results: list[dict] = []  # [{strategy, results, score}, ...]
        self.best_score: float = float("-inf")
        self.best_strategy: Optional[dict] = None
        self.start_time: float = 0.0
        self.strategies_tested: int = 0

        # Pre-computed data
        self.clean_df: Optional[pd.DataFrame] = None
        self.all_conditions_df: Optional[pd.DataFrame] = None

    def run(self) -> dict:
        """Run the full training session.

        Returns:
            Best strategy dict found.
        """
        self.start_time = time.time()
        self._setup_logging()
        self._reset_removed_conditions()
        self._log_start()
        self._load_and_prepare_data()

        if self.method == "ga_bayesian":
            best = self._run_ga_bayesian()
        else:
            best = self._run_random_search()

        saved_new = self._save_results(best)
        self._log_finish(best, saved_new=saved_new)
        return best

    def _setup_logging(self) -> None:
        """Set up file logging for this training session."""
        config.setup_logging()
        log_file = config.LOG_DIR_TRAINING / f"training_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.DATE_FORMAT))
        logging.getLogger().addHandler(fh)
        logger.info(f"Training log: {log_file}")

    def _reset_removed_conditions(self) -> None:
        """Clear the removed conditions file at the start of each training run.

        This ensures conditions are re-evaluated with fresh market data
        instead of being permanently excluded based on a previous regime.
        """
        path = config.REMOVED_CONDITIONS_FILE
        if path.exists():
            path.unlink()
            logger.info("Cleared removed_conditions.json -- all 53 conditions available for this run.")

    def _log_start(self) -> None:
        """Log training session start."""
        logger.info(
            f"Training started | Method: {self.method} | "
            f"Symbol: {self.symbol} | Timeframe: {self.timeframe} | "
            f"Duration: {self.training_minutes} min"
        )
        logger.info(
            "  Objective: Find the strategy with the highest score. "
            "Score = rr_per_day x drawdown_penalty x low_trades_penalty."
        )
        logger.info(
            f"  Disqualified if: trades/day < {config.MIN_TRADES_PER_DAY}, win_rate < 35%, "
            "max_drawdown > 50%, or avg_trades/day outside 0.5-10."
        )

    def _load_and_prepare_data(self) -> None:
        """Load historical data, compute all indicators and conditions."""
        logger.info("Loading historical data...")
        raw_df = get_training_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            months=config.TRAINING_PERIOD_MONTHS,
        )
        # Show training period dates
        if not raw_df.empty:
            period_start = raw_df["timestamp"].iloc[0].strftime("%Y-%m-%d")
            period_end = raw_df["timestamp"].iloc[-1].strftime("%Y-%m-%d")
            logger.info(f"Raw data: {len(raw_df)} candles ({period_start} to {period_end}, {config.TRAINING_PERIOD_MONTHS} months)")
        else:
            logger.info(f"Raw data: {len(raw_df)} candles")

        # Compute all indicators and drop NaN warmup rows
        logger.info("Computing indicators...")
        self.clean_df, _ = prepare_data(raw_df)
        logger.info(f"Clean data: {len(self.clean_df)} candles after warmup")

        # Pre-compute ALL condition columns for speed
        # (avoids recomputing per-strategy during GA/Bayesian)
        all_cond_keys = list(ALL_CONDITIONS.keys())
        logger.info(f"Pre-computing {len(all_cond_keys)} conditions...")
        self.all_conditions_df = compute_all_conditions(self.clean_df, all_cond_keys)
        logger.info("Data preparation complete.")

    def _eval_strategy(self, strategy: dict) -> float:
        """Evaluate a single strategy by running a backtest.

        Args:
            strategy: Strategy dict.

        Returns:
            Float score (or -inf if disqualified).
        """
        # Select the relevant condition columns from pre-computed DataFrame
        # Include both core conditions and shared conditions
        cond_keys = strategy["conditions"]
        shared_keys = strategy.get("shared_conditions", [])
        all_keys = list(set(cond_keys + shared_keys))
        # Only use conditions that exist in the pre-computed DataFrame
        valid_keys = [k for k in all_keys if k in self.all_conditions_df.columns]
        if not valid_keys:
            return float("-inf")

        conditions_df = self.all_conditions_df[valid_keys]
        results = backtest_strategy(self.clean_df, strategy, conditions_df)
        score = score_strategy(results)

        self.strategies_tested += 1
        # Strip heavyweight data (equity_curve, trades) before storing.
        # These are only needed for the best strategy, not the full history.
        # Storing them for 5000+ strategies causes memory exhaustion, GC pauses,
        # and OS swapping that cripples the Bayesian phase (36x slower + timeout failure).
        lite_results = {k: v for k, v in results.items() if k not in ("equity_curve", "trades")}
        self.all_results.append({
            "strategy": strategy,
            "results": lite_results,
            "score": score,
        })

        # Track best
        if score > self.best_score:
            self.best_score = score
            self.best_strategy = strategy.copy()
            self.best_strategy["score"] = score
            self.best_strategy["results"] = {
                "win_rate": results["win_rate"],
                "rr_per_day": results["rr_per_day"],
                "max_drawdown": results["max_drawdown"],
                "total_trades": results["total_trades"],
                "valid_trades": results["valid_trades"],
            }
            return score

        return score

    def _build_coverage_seeds(self) -> list:
        """Generate seed strategies guaranteeing every condition gets tested enough.

        Without coverage injection, the GA converges on a narrow set of conditions,
        leaving many conditions with zero evaluations. This makes the efficiency
        analysis unreliable and can permanently eliminate useful conditions.

        Returns:
            List of Individual objects to inject into the GA's initial population.
        """
        from genetic_optimizer import Individual

        min_evals = config.COVERAGE_EVALS_PER_CONDITION

        # Count how many times each condition appeared in already-evaluated strategies
        # Count both core and shared conditions separately
        counts = {c: 0 for c in ALL_CONDITIONS}
        for r in self.all_results:
            for c in r["strategy"].get("conditions", []):
                if c in counts:
                    counts[c] += 1
            for c in r["strategy"].get("shared_conditions", []):
                if c in counts:
                    counts[c] += 1

        # Deficit: how many more evaluations each condition still needs
        deficit = {c: max(0, min_evals - n) for c, n in counts.items()}
        deficit = {c: n for c, n in deficit.items() if n > 0}

        if not deficit:
            return []

        logger.info(
            f"[COVERAGE] {len(deficit)} conditions need more evaluations "
            f"(target: {min_evals} each). Generating seed strategies."
        )

        seeds = []
        max_seeds = min(config.GA_POPULATION_SIZE, sum(deficit.values()))
        remaining = dict(deficit)
        max_iterations = max_seeds * 2  # Safety guard against theoretical infinite loop
        iteration = 0

        while remaining and len(seeds) < max_seeds and iteration < max_iterations:
            iteration += 1
            # Generate a random strategy as the base
            strat = generate_random_strategy()
            conds = list(strat["conditions"])
            shared_conds = list(strat.get("shared_conditions", []))

            # Try to inject underrepresented conditions
            for cond_key in list(remaining.keys()):
                if remaining[cond_key] <= 0:
                    remaining.pop(cond_key, None)
                    continue

                # Check if it's already in core or shared
                if cond_key in conds or cond_key in shared_conds:
                    remaining[cond_key] -= 1
                    if remaining[cond_key] <= 0:
                        remaining.pop(cond_key, None)
                    continue

                direction = get_direction_for_condition(cond_key)

                if direction == "SHARED":
                    # SHARED conditions go into shared_conditions list
                    shared_conds.append(cond_key)
                    remaining[cond_key] -= 1
                    if remaining[cond_key] <= 0:
                        remaining.pop(cond_key, None)
                else:
                    # LONG/SHORT conditions go into core conditions
                    # Find a slot to replace: same direction
                    swappable = [
                        i for i, c in enumerate(conds)
                        if get_direction_for_condition(c) == direction
                        and c != cond_key
                    ]

                    if swappable:
                        conds[swappable[0]] = cond_key
                        remaining[cond_key] -= 1
                        if remaining[cond_key] <= 0:
                            remaining.pop(cond_key, None)
                    elif len(conds) < int(len(ALL_CONDITIONS) * config.MAX_CONDITION_PERCENTAGE):
                        # Room to append
                        conds.append(cond_key)
                        remaining[cond_key] -= 1
                        if remaining[cond_key] <= 0:
                            remaining.pop(cond_key, None)

            # Deduplicate both lists
            seen = set()
            conds = [c for c in conds if not (c in seen or seen.add(c))]
            seen2 = set()
            shared_conds = [c for c in shared_conds if not (c in seen2 or seen2.add(c))]

            seeds.append(Individual(
                conditions=conds,
                threshold=strat["threshold"],
                sl_atr_mult=strat["sl_atr_mult"],
                rr=strat["rr"],
                shared_conditions=shared_conds,
                shared_bonus_weight=strat.get("shared_bonus_weight", 0.0),
            ))

        logger.info(f"[COVERAGE] Generated {len(seeds)} seed strategies for condition coverage.")
        return seeds

    # =========================================================================
    # GA + Bayesian Pipeline (Default Method)
    # =========================================================================

    def _run_ga_bayesian(self) -> dict:
        """Run the full GA + Bayesian optimization pipeline.

        Phase 1: Genetic Algorithm (global exploration)
        Phase 2: Bayesian Optimization (local refinement)

        Returns:
            Best strategy dict.
        """
        from genetic_optimizer import GeneticOptimizer
        from bayesian_optimizer import BayesianOptimizer

        best_strategy = None
        best_score = float("-inf")

        # --- Phase 1: Genetic Algorithm ---
        logger.info("=" * 60)
        logger.info("Phase 1: Genetic Algorithm (global search -- evolve strategies over generations)")
        logger.info("=" * 60)

        # Allocate ~50% of time to GA, ~50% to Bayesian
        ga_start = time.time()
        ga_time_budget = self.training_minutes * 60 * config.GA_TIME_BUDGET_PERCENT

        # Build coverage seeds: guarantee every condition gets tested
        coverage_seeds = self._build_coverage_seeds()

        ga = GeneticOptimizer(
            eval_func=self._eval_strategy,
            population_size=config.GA_POPULATION_SIZE,
            elite_count=config.GA_ELITE_COUNT,
            crossover_prob=config.GA_CROSSOVER_PROB,
            mutation_prob=config.GA_MUTATION_PROB,
        )

        ga_best, ga_top_10 = ga.run(
            time_limit_seconds=ga_time_budget,
            max_generations=config.GA_MAX_GENERATIONS,
            seed_individuals=coverage_seeds,
        )
        ga_elapsed = time.time() - ga_start

        if ga_best["score"] > best_score:
            best_score = ga_best["score"]
            best_strategy = ga_best

        # --- Efficiency Analysis (between GA and Bayesian) ---
        logger.info("")
        logger.info("[EFFICIENCY] GA phase complete. Analyzing condition efficiency...")
        try:
            eff_result = analyze_conditions(self.all_results, remove=True)
            removed = eff_result.get("removed", [])
            low_eff = eff_result.get("low_efficiency", [])
            if removed:
                logger.info(f"[EFFICIENCY] Removing {len(removed)} conditions (efficiency < {config.EFFICIENCY_CRITICAL}) before Bayesian.")
                logger.info(f"[EFFICIENCY] Removed: {removed}")
            if low_eff:
                logger.info(f"[EFFICIENCY] {len(low_eff)} conditions flagged low-efficiency (0.5x weight, eff {config.EFFICIENCY_CRITICAL:.1f}-{config.EFFICIENCY_ALERT:.1f}): {low_eff}")
            if not removed and not low_eff:
                logger.info("[EFFICIENCY] All conditions performing well. No removals.")
        except Exception as e:
            logger.warning(f"Efficiency analysis failed: {e}")
        logger.info("")

        # Check remaining time AFTER efficiency analysis (it takes non-trivial time)
        remaining_seconds = self.training_minutes * 60 - (time.time() - self.start_time)
        if remaining_seconds < 30:
            logger.info("No time remaining for Bayesian optimization. Skipping.")
            return self.best_strategy or best_strategy

        # --- Phase 2: Bayesian Optimization ---
        logger.info("=" * 60)
        logger.info("Phase 2: Bayesian Optimization (local refinement -- focus on promising regions)")
        logger.info("=" * 60)

        bayesian_start = time.time()

        bo = BayesianOptimizer(
            eval_func=self._eval_strategy,
            n_trials=config.BAYESIAN_MAX_TRIALS,
            startup_trials=config.BAYESIAN_STARTUP_TRIALS,
        )

        # Seed with top GA strategies, run with timeout
        bo_best = bo.run(seed_strategies=ga_top_10, timeout_seconds=remaining_seconds)
        bayesian_elapsed = time.time() - bayesian_start

        if bo_best.get("score", float("-inf")) > best_score:
            best_score = bo_best["score"]
            best_strategy = bo_best

        return self.best_strategy or best_strategy

    # =========================================================================
    # Random Search (Quick Testing Mode)
    # =========================================================================

    def _run_random_search(self) -> dict:
        """Run pure random search for the specified duration.

        Quick testing mode for debugging and comparison.
        """
        logger.info("=" * 60)
        logger.info("Random Search Mode")
        logger.info("=" * 60)

        end_time = self.start_time + (self.training_minutes * 60)
        count = 0

        while time.time() < end_time:
            strategy = generate_random_strategy()
            self._eval_strategy(strategy)
            count += 1

            if count % 100 == 0:
                elapsed = time.time() - self.start_time
                rate = self.strategies_tested / elapsed if elapsed > 0 else 0
                remaining = (end_time - time.time()) / 60
                logger.info(
                    f"Random #{self.strategies_tested} | "
                    f"Best: {self.best_score:.4f} | "
                    f"Rate: {rate:.1f} strats/s | "
                    f"Remaining: {remaining:.1f}m"
                )

        return self.best_strategy or {}

    # =========================================================================
    # Results Saving
    # =========================================================================

    def _save_results(self, best: dict) -> bool:
        """Save best strategy, top strategies, and efficiency report.

        Returns:
            True if a new best strategy was saved, False if existing was retained.
        """
        saved_new = False

        if not best:
            logger.warning("No valid strategy found during training.")
            return saved_new

        # Add training metadata
        best["training_time"] = datetime.now().isoformat()
        best["training_method"] = self.method
        best["symbol"] = self.symbol
        best["strategies_tested"] = self.strategies_tested

        # Compare with existing best before saving (Issue 2)
        try:
            existing_best = load_strategy()
        except Exception:
            existing_best = None

        if existing_best is None:
            logger.info("No existing best strategy found. Saving new strategy.")
            save_strategy(best)
            saved_new = True
        elif "score" not in existing_best:
            logger.warning("Existing strategy has no 'score' key. Overwriting with new strategy.")
            save_strategy(best)
            saved_new = True
        elif existing_best["score"] >= best.get("score", float("-inf")):
            logger.info(
                f"Keeping existing best strategy (score: {existing_best['score']:.4f}) "
                f"-- new strategy score ({best.get('score', float('-inf')):.4f}) is not higher."
            )
        else:
            logger.info(f"Saving new best strategy (score: {best.get('score', float('-inf')):.4f})")
            save_strategy(best)
            saved_new = True

        # Save top 500 strategies (always overwrite -- per-run snapshot, not cumulative)
        scored = [r for r in self.all_results if r["score"] > float("-inf")]
        scored.sort(key=lambda r: r["score"], reverse=True)
        top_strategies = []
        for r in scored[:config.TOP_STRATEGIES_COUNT]:
            s = r["strategy"].copy()
            s["score"] = r["score"]
            if "results" in r:
                s["win_rate"] = r["results"]["win_rate"]
                s["rr_per_day"] = r["results"]["rr_per_day"]
                s["max_drawdown"] = r["results"]["max_drawdown"]
            top_strategies.append(s)
        save_top_strategies(top_strategies)

        # Efficiency analysis: for ga_bayesian, it already ran between GA and Bayesian.
        # For other methods (e.g. random), run it now as a final report.
        if self.method != "ga_bayesian":
            try:
                analyze_conditions(self.all_results, remove=False)
            except Exception as e:
                logger.warning(f"Efficiency analysis failed: {e}")

        return saved_new

    def _log_finish(self, best: dict, saved_new: bool = False) -> None:
        """Log training session summary."""
        elapsed = time.time() - self.start_time
        rate = self.strategies_tested / elapsed if elapsed > 0 else 0

        logger.info("=" * 60)
        logger.info("Training finished.")
        logger.info("")

        score = best.get('score', float('-inf'))
        if score > float('-inf'):
            logger.info(f"  Best score:          {score:.4f}")
            # Show direction mix instead of fixed direction
            conds = best.get('conditions', [])
            shared_conds_list = best.get('shared_conditions', [])
            shared_bonus_weight = best.get('shared_bonus_weight', 0.0)
            if conds:
                from conditions import get_direction_for_condition
                long_conds = sum(1 for c in conds if get_direction_for_condition(c) == 'LONG')
                short_conds = sum(1 for c in conds if get_direction_for_condition(c) == 'SHORT')
                logger.info(f"  Best strategy:       {best.get('id', 'N/A')} (LONG:{long_conds} SHORT:{short_conds} SHARED(bonus):{len(shared_conds_list)} weight:{shared_bonus_weight:.4f})")
            else:
                logger.info(f"  Best strategy:       {best.get('id', 'N/A')}")
            if 'results' in best:
                r = best['results']
                logger.info(f"    Win rate:          {r.get('win_rate', 0):.1%}")
                logger.info(f"    RR/day:            {r.get('rr_per_day', 0):.4f}")
                logger.info(f"    Max drawdown:      {r.get('max_drawdown', 0):.1%}")
                logger.info(f"    Valid trades:      {r.get('valid_trades', 0)}")
        else:
            logger.info("  No valid strategy found (all disqualified).")
            logger.info("  Consider: increase training time, relax criteria in config.py, or try a different symbol.")

        logger.info("")
        logger.info(f"  Total strategies tested: {self.strategies_tested}")
        logger.info(f"  Time elapsed:        {elapsed:.0f}s ({elapsed / 60:.1f}m)")
        logger.info(f"  Average:             {rate:.1f} strats/sec")
        if score > float('-inf'):
            if saved_new:
                logger.info(f"  New best strategy saved to {config.MODEL_DIR / 'best_strategy.json'}")
            else:
                logger.info(f"  Existing best strategy retained (score: {score:.4f})")
            logger.info("")
            logger.info("  Next steps:")
            logger.info("    1. Review efficiency report above -- which conditions helped/hurt")
            logger.info(f"    2. Run validation: python validation.py --symbol {self.symbol}")
            logger.info("    3. If validation passes, go live: python live_signal.py")
        logger.info("=" * 60)


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train a crypto trading strategy optimizer.")
    parser.add_argument("--symbol", default=config.SYMBOL, help="Trading pair (e.g., BTC/USDT)")
    parser.add_argument("--timeframe", default=config.TIMEFRAME, help="Candle timeframe (e.g., 15m)")
    parser.add_argument("--minutes", type=int, default=config.TRAINING_MINUTES, help="Training duration in minutes")
    parser.add_argument("--method", choices=["ga_bayesian", "random"], default=config.TRAINING_METHOD,
                        help="Training method: ga_bayesian (default) or random (quick test)")
    args = parser.parse_args()

    session = TrainingSession(
        symbol=args.symbol,
        timeframe=args.timeframe,
        training_minutes=args.minutes,
        method=args.method,
    )
    session.run()


if __name__ == "__main__":
    main()
