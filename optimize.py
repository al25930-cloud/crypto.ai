"""
Crypto Trading Bot — Parameter Optimizer.

Runs Optuna (Bayesian TPE) to find the best combination of the 7 tunable
strategy parameters, validates them on held-out test data, and reports results.

Usage:
    python optimize.py                         # Full optimization (BTC + ETH)
    python optimize.py --symbol BTC/USDT       # Single symbol
    python optimize.py --trials 400            # Override trial count
    python optimize.py --redownload            # Force fresh data download

Phases:
    0 — Data validation & integrity checks
    3 — Optuna optimization on 70 % training set
    4 — Out-of-sample validation on 30 % test set
    5 — Summary report  (deployment is manual — edit config.py)
"""

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtest import run_backtest
from config import (
    SYMBOLS,
    START_DATE,
    END_DATE,
    OPTIMIZE_PARAMS,
)

# ============================================================================
# Logging
# ============================================================================

_results_dir = Path("results")
_results_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_results_dir / "optimize.log", encoding="utf-8"),
    ],
)
logging.Formatter.converter = time.gmtime
logger = logging.getLogger("optimize")

# ============================================================================
# Phase 0 — Data Validation
# ============================================================================


def validate_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Run data-quality checks and return a clean DataFrame.

    Checks:
      * Required columns present (lowercase).
      * No NaN in OHLCV.
      * No duplicate timestamps.
      * Gaps > 2 h logged as warnings (not forward-filled).
      * Timezone is UTC.
      * At least 12 months of data.
    """
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    actual = set(df.columns)
    missing = required - actual
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    df = df.copy()

    # Timestamp handling
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    # Drop NaN in OHLCV
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    before = len(df)
    df.dropna(subset=ohlcv_cols, inplace=True)
    if len(df) < before:
        logger.warning("%s: dropped %d rows with NaN OHLCV.", symbol, before - len(df))

    # Drop duplicate timestamps
    before = len(df)
    df.drop_duplicates(subset="timestamp", inplace=True)
    if len(df) < before:
        logger.warning("%s: dropped %d duplicate timestamps.", symbol, before - len(df))

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Gap detection
    if len(df) >= 2:
        td = df["timestamp"].diff()
        gap_mask = td > pd.Timedelta(hours=2)
        gap_count = gap_mask.sum()
        if gap_count:
            gap_examples = df["timestamp"][gap_mask].head(3)
            logger.warning(
                "%s: %d gaps > 2 h detected (examples: %s). Not forward-filled.",
                symbol, gap_count, list(gap_examples),
            )

    # Date span
    span_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
    if span_days < 365:
        logger.warning(
            "%s: only %d days of data (< 12 months). Optimization may be unreliable.",
            symbol, span_days,
        )

    logger.info(
        "%s: validated — %d rows, %s → %s, %d gaps.",
        symbol, len(df),
        df["timestamp"].iloc[0].date(), df["timestamp"].iloc[-1].date(),
        gap_count if len(df) >= 2 else 0,
    )
    return df


# ============================================================================
# Phase 3 — Optuna Optimization
# ============================================================================


def run_optimization(
    symbol: str,
    train_df: pd.DataFrame,
    n_trials: int = 200,
    study_name: str = "crypto_optimization",
) -> dict:
    """Run Optuna TPE optimization on the training set.

    Args:
        symbol: Trading pair name.
        train_df: Training portion of data (DatetimeIndex, capitalized cols).
        n_trials: Number of Optuna trials.
        study_name: Optuna study name for storage.

    Returns:
        Dict with keys: best_params, best_pf, study, n_trials.
    """
    import optuna

    train_start = str(train_df.index[0].date())
    train_end = str(train_df.index[-1].date())

    logger.info(
        "%s: starting Optuna optimization — %d trials on %s → %s (%d candles).",
        symbol, n_trials, train_start, train_end, len(train_df),
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "voting_threshold": trial.suggest_categorical("voting_threshold", [1, 2]),
            "ema_fast": trial.suggest_categorical("ema_fast", [5, 9, 12]),
            "ema_slow": trial.suggest_categorical("ema_slow", [20, 21, 26]),
            "zscore_threshold": trial.suggest_categorical("zscore_threshold", [1.5, 2.0, 2.5]),
            "volume_multiplier": trial.suggest_categorical("volume_multiplier", [1.5, 2.0, 2.5]),
            "atr_stop_mult": trial.suggest_categorical("atr_stop_mult", [1.5, 2.0, 2.5]),
            "atr_tp_mult": trial.suggest_categorical("atr_tp_mult", [2.5, 3.0, 4.0]),
        }

        # Constraints
        if params["ema_slow"] <= params["ema_fast"]:
            return 0.0
        if params["atr_tp_mult"] <= params["atr_stop_mult"]:
            return 0.0

        try:
            stats = run_backtest(
                symbol=symbol,
                df=train_df,
                start_date=train_start,
                end_date=train_end,
                params=params,
            )
        except Exception as exc:
            logger.debug("Trial failed: %s", exc)
            return 0.0

        # Log secondary metrics
        trial.set_user_attr("sharpe_ratio", stats["sharpe_ratio"])
        trial.set_user_attr("win_rate", stats["win_rate"])
        trial.set_user_attr("total_trades", stats["total_trades"])

        return stats["profit_factor"]

    # Suppress Optuna's own logging below WARNING during trials
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=study_name,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_pf = study.best_value
    best_params = study.best_params

    logger.info(
        "%s: optimization complete — best PF = %.4f with %s",
        symbol, best_pf, best_params,
    )

    # Save study
    safe = symbol.replace("/", "_").lower()
    pkl_path = _results_dir / f"optuna_{safe}.pkl"
    csv_path = _results_dir / f"trials_{safe}.csv"
    import joblib
    joblib.dump(study, pkl_path)
    study.trials_dataframe().to_csv(csv_path, index=False)
    logger.info("%s: study saved → %s  |  trials CSV → %s", symbol, pkl_path, csv_path)

    return {
        "best_params": best_params,
        "best_pf": best_pf,
        "study": study,
        "n_trials": n_trials,
    }


# ============================================================================
# Phase 4 — Validation
# ============================================================================


def validate_best_params(
    symbol: str,
    test_df: pd.DataFrame,
    best_params: dict,
    train_pf: float,
) -> dict:
    """Run the best parameters on the held-out test set.

    Returns a dict with validation results and acceptance flags.
    """
    test_start = str(test_df.index[0].date())
    test_end = str(test_df.index[-1].date())

    logger.info("%s: validating on test set %s → %s (%d candles) ...",
                symbol, test_start, test_end, len(test_df))

    stats = run_backtest(
        symbol=symbol,
        df=test_df,
        start_date=test_start,
        end_date=test_end,
        params=best_params,
    )

    pf_test = stats["profit_factor"]
    win_test = stats["win_rate"]
    trades_test = stats["total_trades"]
    sharpe_test = stats["sharpe_ratio"]
    dd_test = stats["max_drawdown"]

    # Acceptance criteria
    accept_pf = pf_test >= 1.2
    accept_consistency = pf_test >= 0.8 * train_pf if train_pf > 0 else False
    accept_win = win_test >= 40.0
    accept_trades = trades_test >= 200
    accepted = all([accept_pf, accept_consistency, accept_win, accept_trades])

    result = {
        "symbol": symbol,
        "pf_test": pf_test,
        "pf_train": train_pf,
        "win_test": win_test,
        "trades_test": trades_test,
        "sharpe_test": sharpe_test,
        "dd_test": dd_test,
        "accepted": accepted,
        "checks": {
            "pf >= 1.2": (accept_pf, pf_test),
            "pf_test >= 0.8 * pf_train": (accept_consistency, pf_test),
            "win_rate >= 40%": (accept_win, win_test),
            "trades >= 200": (accept_trades, trades_test),
        },
        "best_params": best_params,
    }

    logger.info("%s: validation %s (PF_test=%.4f, Win=%.1f%%, Trades=%d)",
                symbol, "PASSED" if accepted else "FAILED",
                pf_test, win_test, trades_test)

    return result


# ============================================================================
# Phase 4b — Fallback
# ============================================================================


def run_fallback(
    symbol: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pf: float,
) -> Optional[dict]:
    """Reduced-parameter-space fallback if validation fails.

    Fixes zscore_threshold=2.0, volume_multiplier=2.0, atr_tp_mult=3.0
    and optimises only voting_threshold, ema_fast, ema_slow, atr_stop_mult.
    """
    import optuna

    logger.info("%s: running FALLBACK optimisation (4 params) ...", symbol)

    train_start = str(train_df.index[0].date())
    train_end = str(train_df.index[-1].date())

    def objective(trial: optuna.Trial) -> float:
        params = {
            "voting_threshold": trial.suggest_categorical("voting_threshold", [1, 2]),
            "ema_fast": trial.suggest_categorical("ema_fast", [5, 9, 12]),
            "ema_slow": trial.suggest_categorical("ema_slow", [20, 21, 26]),
            "zscore_threshold": 2.0,          # fixed
            "volume_multiplier": 2.0,         # fixed
            "atr_stop_mult": trial.suggest_categorical("atr_stop_mult", [1.5, 2.0, 2.5]),
            "atr_tp_mult": 3.0,               # fixed
        }
        if params["ema_slow"] <= params["ema_fast"]:
            return 0.0
        if params["atr_tp_mult"] <= params["atr_stop_mult"]:
            return 0.0
        try:
            stats = run_backtest(symbol, train_df, train_start, train_end, params)
        except Exception:
            return 0.0
        trial.set_user_attr("sharpe_ratio", stats["sharpe_ratio"])
        trial.set_user_attr("win_rate", stats["win_rate"])
        trial.set_user_attr("total_trades", stats["total_trades"])
        return stats["profit_factor"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=99))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        study.optimize(objective, n_trials=150, show_progress_bar=True)

    fallback_params = study.best_params
    fallback_params["zscore_threshold"] = 2.0
    fallback_params["volume_multiplier"] = 2.0
    fallback_params["atr_tp_mult"] = 3.0

    logger.info("%s: fallback best PF = %.4f with %s", symbol, study.best_value, fallback_params)

    return validate_best_params(symbol, test_df, fallback_params, study.best_value)


# ============================================================================
# Report
# ============================================================================


def _write_report(results: list[dict]) -> None:
    """Write validation_report.md to results/."""
    lines = [
        "# Optimization Validation Report",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Symbol | PF Train | PF Test | Win Rate | Trades | Sharpe | Max DD | Accepted |",
        "|--------|----------|---------|----------|--------|--------|--------|----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['symbol']} | {r['pf_train']:.4f} | {r['pf_test']:.4f} "
            f"| {r['win_test']:.1f}% | {r['trades_test']} "
            f"| {r['sharpe_test']:.2f} | {r['dd_test']:.1f}% "
            f"| {'✅' if r['accepted'] else '❌'} |"
        )

    lines.append("")
    for r in results:
        lines.append(f"## {r['symbol']} — Best Parameters")
        lines.append("```json")
        import json
        lines.append(json.dumps(r["best_params"], indent=4))
        lines.append("```")
        lines.append("")
        lines.append("### Acceptance Checks")
        for check, (passed, value) in r["checks"].items():
            lines.append(f"- {'✅' if passed else '❌'} {check}: {value:.4f}")
        lines.append("")

    report_path = _results_dir / "validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", report_path)


# ============================================================================
# Main CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto Trading Bot — Optimizer")
    parser.add_argument("--symbol", choices=SYMBOLS, default=None,
                        help="Optimize a single symbol (default: all).")
    parser.add_argument("--trials", type=int, default=200,
                        help="Number of Optuna trials (default: 200).")
    parser.add_argument("--redownload", action="store_true",
                        help="Force fresh Binance download.")
    parser.add_argument("--skip-optimize", action="store_true",
                        help="Skip Optuna; validate current config.py params only.")
    args = parser.parse_args()

    symbols_to_run = [args.symbol] if args.symbol else SYMBOLS

    logger.info("=== Crypto Optimization Pipeline ===")
    logger.info("Symbols: %s | Trials: %d | Skip-Optimize: %s",
                symbols_to_run, args.trials, args.skip_optimize)

    all_results: list[dict] = []

    for symbol in symbols_to_run:
        logger.info("--- %s ---", symbol)

        # Phase 0 — load & validate (read cached CSV directly for full date range)
        safe = symbol.replace("/", "_").lower()
        csv_path = Path("data") / f"{safe}_1h.csv"

        # If CSV missing or --redownload, trigger a fetch via backtest
        if not csv_path.exists() or args.redownload:
            logger.info("%s: fetching data from Binance ...", symbol)
            from backtest import load_or_fetch_data
            load_or_fetch_data(symbol, redownload=True)

        raw_df = pd.read_csv(csv_path)
        clean = validate_data(raw_df, symbol)

        # Prepare for backtest (DatetimeIndex + capitalised columns, no date filter)
        bt_df = clean.copy()
        bt_df.set_index("timestamp", inplace=True)
        bt_df.index = pd.DatetimeIndex(bt_df.index, tz="UTC")
        rename_map = {"open": "Open", "high": "High", "low": "Low",
                      "close": "Close", "volume": "Volume"}
        existing = [c for c in rename_map if c in bt_df.columns]
        if existing:
            bt_df.rename(columns={c: rename_map[c] for c in existing}, inplace=True)

        # Phase 1 — train / test split (70 / 30 chronological)
        split_idx = int(0.7 * len(bt_df))
        train_df = bt_df.iloc[:split_idx]
        test_df = bt_df.iloc[split_idx:]
        logger.info("%s: split %d train / %d test candles.",
                    symbol, len(train_df), len(test_df))

        # Phase 3 — optimize (or use current config defaults)
        if args.skip_optimize:
            from config import (VOTING_THRESHOLD, EMA_FAST, EMA_SLOW,
                                ZSCORE_THRESHOLD, VOLUME_MULTIPLIER,
                                ATR_STOP_MULT, ATR_TP_MULT)
            best_params = {
                "voting_threshold": VOTING_THRESHOLD,
                "ema_fast": EMA_FAST,
                "ema_slow": EMA_SLOW,
                "zscore_threshold": ZSCORE_THRESHOLD,
                "volume_multiplier": VOLUME_MULTIPLIER,
                "atr_stop_mult": ATR_STOP_MULT,
                "atr_tp_mult": ATR_TP_MULT,
            }
            # Compute train PF for comparison
            train_stats = run_backtest(
                symbol, train_df,
                str(train_df.index[0].date()), str(train_df.index[-1].date()),
                best_params,
            )
            train_pf = train_stats["profit_factor"]
        else:
            opt_result = run_optimization(symbol, train_df, n_trials=args.trials)
            best_params = opt_result["best_params"]
            train_pf = opt_result["best_pf"]

        # Phase 4 — validate
        validation = validate_best_params(symbol, test_df, best_params, train_pf)

        # Fallback if needed
        if not validation["accepted"] and not args.skip_optimize:
            logger.info("%s: validation FAILED — trying fallback ...", symbol)
            fb = run_fallback(symbol, train_df, test_df, train_pf)
            if fb and fb["accepted"]:
                validation = fb
            else:
                logger.warning("%s: fallback also FAILED.", symbol)

        all_results.append(validation)

    # Phase 5 — report
    _write_report(all_results)

    print("\n" + "=" * 60)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 60)
    for r in all_results:
        status = "PASSED" if r["accepted"] else "FAILED"
        print(f"  {r['symbol']}: {status}  "
              f"(PF_train={r['pf_train']:.4f}, PF_test={r['pf_test']:.4f}, "
              f"Win={r['win_test']:.1f}%, Trades={r['trades_test']})")
        print(f"    Best params: {r['best_params']}")
    print("=" * 60)
    print(f"\nFull report: results/validation_report.md")
    print("To deploy, edit config.py with the best parameters above.\n")


if __name__ == "__main__":
    main()
