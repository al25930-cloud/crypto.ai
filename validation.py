"""
Validation module — run a full backtest on the best strategy using the validation period.

Computes comprehensive metrics: win rate, profit factor, Sharpe ratio, max drawdown,
RR/day, exit breakdown, invalid trades, and checks against acceptance criteria.

Usage:
    python validation.py --symbol BTC/USDT --period 12
"""

import argparse
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from backtest import backtest_strategy, prepare_data
from data_fetcher import fetch_ohlcv, get_validation_data
from strategy import load_strategy

logger = logging.getLogger(__name__)


def compute_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio from a list of per-trade returns.

    Uses sqrt(total_trades) as the annualization factor, which is the correct
    scaling when working with per-trade returns (not daily returns).

    Args:
        returns: List of net PnL percentages per trade.
        risk_free_rate: Annualized risk-free rate (default 0).

    Returns:
        Sharpe ratio (annualized).
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    mean_ret = np.mean(arr) - risk_free_rate / 365
    std_ret = np.std(arr, ddof=1)
    if std_ret == 0:
        return 0.0
    n_trades = len(returns)
    return float((mean_ret / std_ret) * math.sqrt(n_trades))


def compute_profit_factor(trades: list[dict]) -> float:
    """Compute profit factor = gross profit / gross loss.

    Args:
        trades: List of trade dicts with 'net_pnl_pct' and 'valid'.

    Returns:
        Profit factor (inf if no losses).
    """
    gross_profit = sum(t["net_pnl_pct"] for t in trades if t["valid"] and t["net_pnl_pct"] > 0)
    gross_loss = abs(sum(t["net_pnl_pct"] for t in trades if t["valid"] and t["net_pnl_pct"] < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _compute_window_score(results: dict, profit_factor: float) -> dict:
    """Compute score + acceptance for a single window's backtest results.

    Uses same formula as training: rr_per_day × drawdown_penalty × timeout_penalty.
    """
    total_days = results.get("total_period_days", 1)
    rr_per_day = results["total_rr"] / total_days if total_days > 0 else 0.0

    # Drawdown penalty
    dd = results["max_drawdown"]
    if dd < config.DRAWDOWN_PENALTY_START:
        dd_penalty = 1.0
    else:
        dd_penalty = 1.0 - (dd - config.DRAWDOWN_PENALTY_START) / (config.DRAWDOWN_PENALTY_END - config.DRAWDOWN_PENALTY_START)
        dd_penalty = max(0.0, dd_penalty)

    # Timeout penalty
    total_exits = results["exit_sl_count"] + results["exit_tp_count"] + results["exit_timeout_count"] + results.get("exit_data_end_count", 0)
    timeout_ratio = results["exit_timeout_count"] / total_exits if total_exits > 0 else 0.0
    to_penalty = config.TIMEOUT_PENALTY if timeout_ratio > config.TIMEOUT_PENALTY_THRESHOLD else 1.0

    # Score (disqualified if DD > cap)
    if dd > config.MAX_DRAWDOWN:
        score = -float("inf")
    else:
        score = rr_per_day * dd_penalty * to_penalty

    # Acceptance gates
    trades_per_day = results["valid_trades"] / total_days if total_days > 0 else 0.0
    acceptance = {
        "win_rate_pass": results["win_rate"] >= config.MIN_WIN_RATE,
        "max_drawdown_pass": dd <= config.MAX_DRAWDOWN,
        "profit_factor_pass": profit_factor >= 1.3,
        "trades_per_day_pass": config.MIN_TRADES_PER_DAY <= trades_per_day <= config.MAX_TRADES_PER_DAY,
    }
    acceptance["all_pass"] = all(acceptance.values())

    return {
        "rr_per_day": round(rr_per_day, 4),
        "score": round(score, 4) if not math.isinf(score) else "-inf",
        "score_raw": score,
        "dd_penalty": round(dd_penalty, 4),
        "to_penalty": round(to_penalty, 4),
        "acceptance": acceptance,
    }


def _backtest_window(
    window_raw: pd.DataFrame,
    strategy: dict,
    window_label: str,
) -> Optional[dict]:
    """Run backtest on a single window and return metrics.

    Returns None if window has insufficient data.
    """
    clean_df, conditions_df = prepare_data(window_raw, strategy["conditions"])
    if len(clean_df) < 100:
        logger.warning(f"{window_label}: too few candles after warmup ({len(clean_df)}).")
        return None

    results = backtest_strategy(clean_df, strategy, conditions_df)
    valid_trades = [t for t in results["trades"] if t["valid"]]
    all_trades = results["trades"]

    # Date range
    if clean_df["timestamp"].dt.tz is not None:
        start_date = clean_df["timestamp"].iloc[0].strftime("%Y-%m-%d")
        end_date = clean_df["timestamp"].iloc[-1].strftime("%Y-%m-%d")
    else:
        start_date = str(clean_df["timestamp"].iloc[0].date())
        end_date = str(clean_df["timestamp"].iloc[-1].date())

    profit_factor = compute_profit_factor(all_trades)
    score_info = _compute_window_score(results, profit_factor)
    returns_list = [t["net_pnl_pct"] for t in valid_trades]
    sharpe = compute_sharpe_ratio(returns_list)

    total_days = results.get("total_period_days", 1)
    timeout_trades = [t for t in all_trades if t.get("result") == "timeout"]
    data_end_trades = [t for t in all_trades if t.get("result") == "data_end"]

    return {
        "period": f"{start_date} to {end_date}",
        "total_trades": results["total_trades"],
        "valid_trades": results["valid_trades"],
        "invalid_trades": results["invalid_trades"],
        "timeout_trades": results["timeout_trades"],
        "win_rate": round(results["win_rate"], 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(results["max_drawdown"], 4),
        "total_rr": round(results["total_rr"], 4),
        "total_days": total_days,
        "avg_trades_per_day": round(results["valid_trades"] / total_days, 4) if total_days > 0 else 0.0,
        "exit_sl_count": results["exit_sl_count"],
        "exit_tp_count": results["exit_tp_count"],
        "exit_timeout_count": results["exit_timeout_count"],
        "data_end_trades": len(data_end_trades),
        "total_fees_pct": round(results["total_fees"], 6),
        "rr_per_day": score_info["rr_per_day"],
        "score": score_info["score"],
        "score_raw": score_info["score_raw"],
        "acceptance": score_info["acceptance"],
    }


def run_validation(
    symbol: str = config.SYMBOL,
    timeframe: str = config.TIMEFRAME,
    period_months: int = 12,
    strategy_path: Optional[Path] = None,
) -> dict:
    """Run a full validation backtest on the best strategy.

    Supports single-window (VALIDATION_WINDOWS=1) and multi-window mode.

    Args:
        symbol: Trading pair.
        timeframe: Candle timeframe.
        period_months: Number of months for the validation period (single-window mode only).
        strategy_path: Path to strategy JSON. Defaults to models/best_strategy.json.

    Returns:
        Dict with all validation metrics and acceptance check results.
    """
    config.setup_logging()
    _setup_validation_file_logging()
    logger.info("=" * 60)
    logger.info("VALIDATION BACKTEST")
    logger.info("=" * 60)

    # Load strategy
    strategy = load_strategy(strategy_path)
    if strategy is None:
        logger.error("No best strategy found. Run training first.")
        return {"error": "No best strategy found"}

    logger.info(f"Strategy: {strategy['id']}")
    # Show direction mix instead of fixed direction
    from conditions import get_direction_for_condition
    conds = strategy['conditions']
    long_conds = sum(1 for c in conds if get_direction_for_condition(c) == 'LONG')
    short_conds = sum(1 for c in conds if get_direction_for_condition(c) == 'SHORT')
    shared_conds = sum(1 for c in conds if get_direction_for_condition(c) == 'SHARED')
    logger.info(f"  Direction mix: LONG:{long_conds} SHORT:{short_conds} SHARED:{shared_conds}")
    logger.info(f"  Conditions ({len(conds)}): {conds}")
    logger.info(f"  Threshold: {strategy['threshold']}")
    logger.info(f"  SL_ATR_MULT: {strategy.get('sl_atr_mult', strategy.get('sl', 'N/A'))}, RR: {strategy['rr']}")

    n_windows = getattr(config, 'VALIDATION_WINDOWS', 1)

    if n_windows <= 1:
        return _run_single_window_validation(symbol, timeframe, period_months, strategy, conds, long_conds, short_conds, shared_conds)
    else:
        return _run_multi_window_validation(symbol, timeframe, strategy, conds, long_conds, short_conds, shared_conds)


def _run_single_window_validation(
    symbol: str,
    timeframe: str,
    period_months: int,
    strategy: dict,
    conds: list,
    long_conds: int,
    short_conds: int,
    shared_conds: int,
) -> dict:
    """Original single-window validation (backward compatible)."""
    # Fetch validation data
    logger.info(f"Fetching validation data: {symbol} {timeframe}, non-overlapping {period_months} months")
    raw_df = get_validation_data(symbol=symbol, timeframe=timeframe, months=period_months)

    if raw_df.empty:
        logger.error("No data fetched. Check connection and symbol.")
        return {"error": "No data fetched"}

    # Log period information
    now = datetime.now(timezone.utc)
    train_end = now - timedelta(days=config.TRAINING_PERIOD_MONTHS * 30)
    train_start = train_end - timedelta(days=config.TRAINING_PERIOD_MONTHS * 30)
    val_start = raw_df["timestamp"].iloc[0]
    val_end = raw_df["timestamp"].iloc[-1]
    logger.info(f"Raw data: {len(raw_df)} candles ({val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}, {period_months} months)")
    logger.info(f"Training period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} ({config.TRAINING_PERIOD_MONTHS} months)")
    logger.info(f"Validation period: {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')} (non-overlapping)")

    # Run backtest on the single window
    window_metrics = _backtest_window(raw_df, strategy, "Single window")
    if window_metrics is None:
        logger.error("Insufficient data for validation.")
        return {"error": "Insufficient data"}

    # Log report (single-window format)
    report = {
        "strategy_id": strategy.get("id", "unknown"),
        "symbol": symbol,
        "timeframe": timeframe,
        "multi_window": False,
        **window_metrics,
        "validated_at": datetime.now().isoformat(),
    }
    _log_single_window_report(report)

    # Save report
    report_path = config.MODEL_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Validation report saved to {report_path}")

    return report


def _run_multi_window_validation(
    symbol: str,
    timeframe: str,
    strategy: dict,
    conds: list,
    long_conds: int,
    short_conds: int,
    shared_conds: int,
) -> dict:
    """Multi-window validation: test strategy on N independent windows."""
    n_windows = config.VALIDATION_WINDOWS
    window_months = config.VALIDATION_WINDOW_MONTHS
    overlap = config.VALIDATION_WINDOW_OVERLAP
    total_months = window_months + (n_windows - 1) * (window_months - overlap)
    step_days = (window_months - overlap) * 30

    logger.info(f"Multi-window validation: {n_windows} × {window_months} months (overlap={overlap})")
    logger.info(f"Total data needed: {total_months} months")

    # Fetch all data
    raw_df = get_validation_data(symbol=symbol, timeframe=timeframe, months=total_months)
    if raw_df.empty:
        logger.error("No data fetched.")
        return {"error": "No data fetched"}

    logger.info(f"Raw data: {len(raw_df)} candles")

    # Run each window
    window_reports = []
    for w in range(n_windows):
        since = raw_df["timestamp"].iloc[0] + timedelta(days=w * step_days)
        until = since + timedelta(days=window_months * 30)
        window_raw = raw_df[(raw_df["timestamp"] >= since) & (raw_df["timestamp"] < until)].copy()

        if len(window_raw) < 50:
            logger.warning(f"  Window {w+1}: too few raw candles ({len(window_raw)}). Skipping.")
            window_reports.append({"error": "insufficient_data", "window": w + 1})
            continue

        logger.info(f"  Window {w+1}: {len(window_raw)} raw candles")
        wm = _backtest_window(window_raw, strategy, f"Window {w+1}")
        if wm is not None:
            wm["window"] = w + 1
            window_reports.append(wm)
        else:
            window_reports.append({"error": "insufficient_data_after_warmup", "window": w + 1})

    # Aggregate
    valid_windows = [w for w in window_reports if "acceptance" in w]
    n_valid = len(valid_windows)

    if n_valid == 0:
        logger.error("No windows had sufficient data.")
        return {"error": "No valid windows", "windows": window_reports}

    overall_pass = all(w["acceptance"]["all_pass"] for w in valid_windows)
    scores = [w["score_raw"] for w in valid_windows]
    min_score = min(scores)
    avg_score = sum(scores) / n_valid if n_valid > 0 else 0.0
    worst_idx = scores.index(min_score) if scores else -1

    # Log multi-window report
    _log_multi_window_report(window_reports, valid_windows, overall_pass, min_score, avg_score, worst_idx, n_windows)

    # Build final report
    report = {
        "strategy_id": strategy.get("id", "unknown"),
        "symbol": symbol,
        "timeframe": timeframe,
        "multi_window": True,
        "n_windows": n_windows,
        "window_months": window_months,
        "overlap": overlap,
        "overall_pass": overall_pass,
        "worst_window_score": round(min_score, 4) if not math.isinf(min_score) else "-inf",
        "avg_score": round(avg_score, 4),
        "windows": window_reports,
        "validated_at": datetime.now().isoformat(),
    }

    # Save report
    report_path = config.MODEL_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        # Convert non-serializable values
        clean_report = json.loads(json.dumps(report, default=str))
        json.dump(clean_report, f, indent=2)
    logger.info(f"Validation report saved to {report_path}")

    return report


def _log_single_window_report(report: dict) -> None:
    """Log the single-window validation report."""
    acc = report.get("acceptance", {})
    pass_icon = lambda v: "[PASS]" if v else "[FAIL]"

    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Period:          {report.get('period', 'N/A')}")
    logger.info(f"  Strategy:        {report['strategy_id']}")
    logger.info("")
    logger.info(f"  Total trades:    {report.get('total_trades', 0)}")
    logger.info(f"  Valid trades:    {report.get('valid_trades', 0)}")
    logger.info(f"  Invalid trades:  {report.get('invalid_trades', 0)} (too short)")
    logger.info(f"  Timeout trades:  {report.get('timeout_trades', 0)} (limit: {config.MAX_TRADE_DURATION_HOURS}h)")
    logger.info(f"  Data-end closes: {report.get('data_end_trades', 0)}")
    logger.info("")
    logger.info(f"  Win rate:        {report.get('win_rate', 0):.1%}  {pass_icon(acc.get('win_rate_pass', False))} (threshold: ≥{config.MIN_WIN_RATE:.0%})")
    logger.info(f"  Profit factor:   {report.get('profit_factor', 'N/A')}  {pass_icon(acc.get('profit_factor_pass', False))} (threshold: ≥1.3)")
    logger.info(f"  Sharpe ratio:    {report.get('sharpe_ratio', 0):.4f}")
    logger.info(f"  Max drawdown:    {report.get('max_drawdown', 0):.1%}  {pass_icon(acc.get('max_drawdown_pass', False))} (threshold: ≤{config.MAX_DRAWDOWN:.0%})")
    logger.info(f"  RR/day:          {report.get('rr_per_day', 0):.4f}")
    logger.info(f"  Total RR:        {report.get('total_rr', 0):.4f}")
    logger.info("")
    logger.info(f"  Exit breakdown:")
    logger.info(f"    SL hits:       {report.get('exit_sl_count', 0)}")
    logger.info(f"    TP hits:       {report.get('exit_tp_count', 0)}")
    logger.info(f"    Timeouts:      {report.get('exit_timeout_count', 0)}")
    logger.info(f"    Fees paid:     {report.get('total_fees_pct', 0):.4%} (rate: {config.TRADING_FEE_PCT}% per side)")
    logger.info("")
    if acc.get("all_pass", False):
        logger.info("  ALL ACCEPTANCE CRITERIA PASSED")
    else:
        logger.info("  SOME ACCEPTANCE CRITERIA FAILED -- consider retraining")
    logger.info("=" * 60)


def _log_multi_window_report(
    window_reports: list,
    valid_windows: list,
    overall_pass: bool,
    min_score: float,
    avg_score: float,
    worst_idx: int,
    total_windows: int,
) -> None:
    """Log the multi-window validation report."""
    pass_icon = lambda v: "[PASS]" if v else "[FAIL]"

    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS — Multi-Window")
    logger.info("=" * 60)
    logger.info(f"  Windows: {total_windows} × {config.VALIDATION_WINDOW_MONTHS} months (overlap={config.VALIDATION_WINDOW_OVERLAP})")
    logger.info(f"  Strategy: {valid_windows[0].get('strategy_id', 'N/A') if valid_windows else 'N/A'}")
    logger.info("")

    for i, wr in enumerate(window_reports):
        w_num = wr.get("window", i + 1)
        if "error" in wr:
            status = "SKIPPED" if wr["error"] == "insufficient_data" else "FAILED"
            logger.info(f"  Window {w_num}: {status} — {wr['error']}")
        else:
            acc = wr.get("acceptance", {})
            score_str = str(wr.get('score', 'N/A'))
            logger.info(f"  Window {w_num}: {wr.get('period', 'N/A')}")
            logger.info(f"    Trades: {wr.get('valid_trades', 0)} valid / {wr.get('total_trades', 0)} total | Win rate: {wr.get('win_rate', 0):.1%} {pass_icon(acc.get('win_rate_pass', False))}")
            logger.info(f"    Max DD: {wr.get('max_drawdown', 0):.1%} {pass_icon(acc.get('max_drawdown_pass', False))} | PF: {wr.get('profit_factor', 'N/A')} {pass_icon(acc.get('profit_factor_pass', False))}")
            logger.info(f"    Trades/day: {wr.get('avg_trades_per_day', 0):.1f} {pass_icon(acc.get('trades_per_day_pass', True))} | RR/day: {wr.get('rr_per_day', 0):.4f} | Score: {score_str}")
        logger.info("")

    n_passed = sum(1 for w in valid_windows if w["acceptance"]["all_pass"])
    if overall_pass:
        logger.info(f"  OVERALL: ALL {n_passed}/{total_windows} WINDOWS PASSED ✅")
    else:
        logger.info(f"  OVERALL: FAILED — {n_passed}/{total_windows} windows passed ❌")

    logger.info(f"  Worst window: #{worst_idx + 1 if worst_idx >= 0 else '?'} (min score: {round(min_score, 4) if not math.isinf(min_score) else '-inf'})")
    logger.info(f"  Average score: {round(avg_score, 4)}")
    logger.info("=" * 60)


def _setup_validation_file_logging() -> None:
    """Set up file logging for validation."""
    log_file = config.LOG_DIR_VALIDATION / f"validation_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.DATE_FORMAT))
    logging.getLogger().addHandler(fh)
    logger.info(f"Validation log: {log_file}")


def main():
    """CLI entry point for validation."""
    parser = argparse.ArgumentParser(description="Run validation backtest on the best strategy.")
    parser.add_argument("--symbol", default=config.SYMBOL, help="Trading pair")
    parser.add_argument("--timeframe", default=config.TIMEFRAME, help="Candle timeframe")
    parser.add_argument("--period", type=int, default=12, help="Validation period in months")
    parser.add_argument("--strategy", type=str, default=None, help="Path to strategy JSON file")
    args = parser.parse_args()

    strategy_path = Path(args.strategy) if args.strategy else None
    run_validation(
        symbol=args.symbol,
        timeframe=args.timeframe,
        period_months=args.period,
        strategy_path=strategy_path,
    )


if __name__ == "__main__":
    main()
