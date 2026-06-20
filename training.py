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
from conditions import ALL_CONDITIONS
from data_fetcher import fetch_ohlcv, get_training_data
from efficiency import analyze_conditions
from indicators import compute_all_conditions
from strategy import (
    generate_random_strategy,
    is_disqualified,
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

        self._save_results(best)
        self._log_finish(best)
        return best

    def _setup_logging(self) -> None:
        """Set up file logging for this training session."""
        config.setup_logging()
        log_file = config.LOG_DIR / f"training_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"
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
            "  Disqualified if: win_rate < 35%, max_drawdown > 50%, "
            "or avg_trades/day outside 0.5-10."
        )

    def _load_and_prepare_data(self) -> None:
        """Load historical data, compute all indicators and conditions."""
        logger.info("Loading historical data...")
        raw_df = get_training_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            months=config.TRAINING_PERIOD_MONTHS,
        )
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
        cond_keys = strategy["conditions"]
        # Only use conditions that exist in the pre-computed DataFrame
        valid_keys = [k for k in cond_keys if k in self.all_conditions_df.columns]
        if not valid_keys:
            return float("-inf")

        conditions_df = self.all_conditions_df[valid_keys]
        results = backtest_strategy(self.clean_df, strategy, conditions_df)
        score = score_strategy(results)

        self.strategies_tested += 1
        self.all_results.append({
            "strategy": strategy,
            "results": results,
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
        ga_minutes = self.training_minutes * 0.5

        # Adjust GA generations based on time budget
        estimated_secs_per_gen = 5  # Rough estimate
        max_ga_gens = max(5, int((ga_minutes * 60) / estimated_secs_per_gen))
        ga_generations = min(config.GA_GENERATIONS, max_ga_gens)

        ga = GeneticOptimizer(
            eval_func=self._eval_strategy,
            population_size=config.GA_POPULATION_SIZE,
            generations=ga_generations,
            elite_count=config.GA_ELITE_COUNT,
            crossover_prob=config.GA_CROSSOVER_PROB,
            mutation_prob=config.GA_MUTATION_PROB,
        )

        ga_best, ga_top_10 = ga.run()
        ga_elapsed = time.time() - ga_start

        if ga_best["score"] > best_score:
            best_score = ga_best["score"]
            best_strategy = ga_best

        logger.info(
            f"GA Phase complete | Elapsed: {ga_elapsed:.0f}s | "
            f"Best score: {ga_best['score']:.4f} | "
            f"Strategies tested: {self.strategies_tested}"
        )

        # Check if we still have time for Bayesian
        remaining_minutes = self.training_minutes - (ga_elapsed / 60)
        if remaining_minutes < 0.5:
            logger.info("No time remaining for Bayesian optimization. Skipping.")
            return best_strategy

        # --- Phase 2: Bayesian Optimization ---
        logger.info("=" * 60)
        logger.info("Phase 2: Bayesian Optimization (local refinement -- focus on promising regions)")
        logger.info("=" * 60)

        bayesian_start = time.time()

        # Adjust trial count based on remaining time
        estimated_secs_per_trial = 0.5
        max_trials = max(50, int((remaining_minutes * 60) / estimated_secs_per_trial))
        n_trials = min(config.BAYESIAN_N_TRIALS, max_trials)

        bo = BayesianOptimizer(
            eval_func=self._eval_strategy,
            n_trials=n_trials,
            startup_trials=min(config.BAYESIAN_STARTUP_TRIALS, n_trials // 5),
        )

        # Seed with top GA strategies
        bo_best = bo.run(seed_strategies=ga_top_10)
        bayesian_elapsed = time.time() - bayesian_start

        if bo_best.get("score", float("-inf")) > best_score:
            best_score = bo_best["score"]
            best_strategy = bo_best

        logger.info(
            f"Bayesian Phase complete | Elapsed: {bayesian_elapsed:.0f}s | "
            f"Best score: {bo_best.get('score', float('-inf')):.4f} | "
            f"Strategies tested: {self.strategies_tested}"
        )

        return best_strategy

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

    def _save_results(self, best: dict) -> None:
        """Save best strategy, top strategies, and efficiency report."""
        if not best:
            logger.warning("No valid strategy found during training.")
            return

        # Add training metadata
        best["training_time"] = datetime.now().isoformat()
        best["training_method"] = self.method
        best["symbol"] = self.symbol
        best["strategies_tested"] = self.strategies_tested

        # Save best strategy
        save_strategy(best)

        # Save top 500 strategies (sorted by score)
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

        # Run efficiency analysis
        try:
            analyze_conditions(self.all_results)
        except Exception as e:
            logger.warning(f"Efficiency analysis failed: {e}")

    def _log_finish(self, best: dict) -> None:
        """Log training session summary."""
        elapsed = time.time() - self.start_time
        rate = self.strategies_tested / elapsed if elapsed > 0 else 0

        logger.info("=" * 60)
        logger.info("Training finished.")
        logger.info("")

        score = best.get('score', float('-inf'))
        if score > float('-inf'):
            logger.info(f"  Best score:          {score:.4f}")
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
            logger.info(f"  Best strategy saved to {config.MODEL_DIR / 'best_strategy.json'}")
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
