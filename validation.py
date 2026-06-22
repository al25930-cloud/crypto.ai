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

    Args:
        returns: List of net PnL percentages per trade.
        risk_free_rate: Annualized risk-free rate (default 0).

    Returns:
        Sharpe ratio (annualized assuming ~250 trading days per year).
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    mean_ret = np.mean(arr) - risk_free_rate / 250
    std_ret = np.std(arr, ddof=1)
    if std_ret == 0:
        return 0.0
    return float((mean_ret / std_ret) * math.sqrt(365))  # 365 for 24/7 crypto markets


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


def run_validation(
    symbol: str = config.SYMBOL,
    timeframe: str = config.TIMEFRAME,
    period_months: int = 12,
    strategy_path: Optional[Path] = None,
) -> dict:
    """Run a full validation backtest on the best strategy.

    Args:
        symbol: Trading pair.
        timeframe: Candle timeframe.
        period_months: Number of months for the validation period.
        strategy_path: Path to strategy JSON. Defaults to models/best_strategy.json.

    Returns:
        Dict with all validation metrics and acceptance check results.
    """
    config.setup_logging()
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

    # Fetch validation data
    logger.info(f"Fetching validation data: {symbol} {timeframe}, non-overlapping {period_months} months")
    raw_df = get_validation_data(symbol=symbol, timeframe=timeframe, months=period_months)

    if raw_df.empty:
        logger.error("No data fetched. Check connection and symbol.")
        return {"error": "No data fetched"}

    # Log period information (Issue 6)
    now = datetime.now(timezone.utc)
    train_end = now - timedelta(days=config.TRAINING_PERIOD_MONTHS * 30)
    train_start = train_end - timedelta(days=config.TRAINING_PERIOD_MONTHS * 30)
    val_start = raw_df["timestamp"].iloc[0]
    val_end = raw_df["timestamp"].iloc[-1]
    logger.info(f"Raw data: {len(raw_df)} candles ({val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}, {period_months} months)")
    logger.info(f"Training period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} ({config.TRAINING_PERIOD_MONTHS} months)")
    logger.info(f"Validation period: {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')} (non-overlapping)")

    # Prepare data
    clean_df, conditions_df = prepare_data(raw_df, strategy["conditions"])
    logger.info(f"Clean data: {len(clean_df)} candles after warmup")
    logger.info(f"  Note: {len(raw_df) - len(clean_df)} candles dropped for indicator warmup.")

    if len(clean_df) < 100:
        logger.warning("Very few candles after warmup. Results may be unreliable.")

    # Run backtest
    logger.info("Running backtest...")
    results = backtest_strategy(clean_df, strategy, conditions_df)

    # Extract metrics
    valid_trades = [t for t in results["trades"] if t["valid"]]
    all_trades = results["trades"]
    returns = [t["net_pnl_pct"] for t in valid_trades]

    # Compute additional metrics
    sharpe = compute_sharpe_ratio(returns)
    profit_factor = compute_profit_factor(all_trades)

    # Date range
    if clean_df["timestamp"].dt.tz is not None:
        start_date = clean_df["timestamp"].iloc[0].strftime("%Y-%m-%d")
        end_date = clean_df["timestamp"].iloc[-1].strftime("%Y-%m-%d")
    else:
        start_date = str(clean_df["timestamp"].iloc[0].date())
        end_date = str(clean_df["timestamp"].iloc[-1].date())

    # --- Issue 1: Recalculate RR/day using total period days ---
    total_days = (clean_df["timestamp"].iloc[-1] - clean_df["timestamp"].iloc[0]).days
    total_days = max(total_days, 1)  # Avoid division by zero
    rr_per_day = results["total_rr"] / total_days
    avg_trades_per_day = results["valid_trades"] / total_days
    logger.info(f"RR/day: {rr_per_day:.4f}  (total RR {results['total_rr']:.2f} / {total_days} calendar days)")

    # Acceptance criteria check
    acceptance = {
        "win_rate_pass": results["win_rate"] >= 0.35,
        "max_drawdown_pass": results["max_drawdown"] <= config.MAX_DRAWDOWN,
        "profit_factor_pass": profit_factor >= 1.3,
    }
    acceptance["all_pass"] = all(acceptance.values())

    # Compute timeout-specific metrics
    timeout_trades = [t for t in all_trades if t.get("result") == "timeout"]
    data_end_trades = [t for t in all_trades if t.get("result") == "data_end"]
    timeout_rr_values = [t["rr"] for t in timeout_trades]
    avg_timeout_rr = float(np.mean(timeout_rr_values)) if timeout_rr_values else 0.0
    total_exits = results["exit_sl_count"] + results["exit_tp_count"] + results["exit_timeout_count"] + results.get("exit_data_end_count", 0)
    timeout_ratio = results["exit_timeout_count"] / total_exits if total_exits > 0 else 0.0

    # Build validation report
    # TODO: Future enhancement — add Monte Carlo simulation here
    # Shuffle trade order N times, recalculate drawdown and Sharpe for each,
    # report confidence intervals. See README "Future Enhancements" section.
    report = {
        "strategy_id": strategy.get("id", "unknown"),
        "symbol": symbol,
        "timeframe": timeframe,
        "period": f"{start_date} to {end_date}",
        "period_months": period_months,
        "total_trades": results["total_trades"],
        "valid_trades": results["valid_trades"],
        "invalid_trades": results["invalid_trades"],
        "timeout_trades": results["timeout_trades"],
        "win_rate": round(results["win_rate"], 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(results["max_drawdown"], 4),
        "rr_per_day": round(rr_per_day, 4),
        "total_rr": round(results["total_rr"], 4),
        "total_days": total_days,
        "avg_trades_per_day": round(avg_trades_per_day, 4),
        "exit_sl_count": results["exit_sl_count"],
        "exit_tp_count": results["exit_tp_count"],
        "exit_timeout_count": results["exit_timeout_count"],
        "data_end_trades": len(data_end_trades),
        "timeout_ratio": round(timeout_ratio, 4),
        "avg_timeout_rr": round(avg_timeout_rr, 4),
        "total_fees_pct": round(results["total_fees"], 6),
        "acceptance": acceptance,
        "validated_at": datetime.now().isoformat(),
    }

    # Log report
    _log_validation_report(report)

    # Save report
    report_path = config.MODEL_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Validation report saved to {report_path}")

    return report


def _log_validation_report(report: dict) -> None:
    """Log the validation report in a formatted table."""
    acc = report["acceptance"]
    pass_icon = lambda v: "[PASS]" if v else "[FAIL]"

    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Period:          {report['period']}")
    logger.info(f"  Strategy:        {report['strategy_id']}")
    logger.info("")
    logger.info(f"  Total trades:    {report['total_trades']}")
    logger.info(f"  Valid trades:    {report['valid_trades']}")
    logger.info(f"  Invalid trades:  {report['invalid_trades']} (too short)")
    logger.info(f"  Timeout trades:  {report['timeout_trades']} (limit: {config.MAX_TRADE_DURATION_HOURS}h, {report['timeout_ratio']:.0%} of exits, avg RR: {report['avg_timeout_rr']:+.2f})")
    logger.info(f"  Data-end closes: {report.get('data_end_trades', 0)} (backtest period ended while in position)")
    logger.info("")
    logger.info(f"  Win rate:        {report['win_rate']:.1%}  {pass_icon(acc['win_rate_pass'])} (threshold: ≥35%)")
    logger.info(f"  Profit factor:   {report['profit_factor']}  {pass_icon(acc['profit_factor_pass'])} (threshold: ≥1.3)")
    logger.info(f"  Sharpe ratio:    {report['sharpe_ratio']:.4f}")
    logger.info(f"  Max drawdown:    {report['max_drawdown']:.1%}  {pass_icon(acc['max_drawdown_pass'])} (threshold: ≤50%)")
    logger.info(f"  RR/day:          {report['rr_per_day']:.4f}")
    logger.info(f"  Total RR:        {report['total_rr']:.4f}")
    logger.info("")
    logger.info(f"  Exit breakdown:")
    logger.info(f"    SL hits:       {report['exit_sl_count']}")
    logger.info(f"    TP hits:       {report['exit_tp_count']}")
    logger.info(f"    Timeouts:      {report['exit_timeout_count']} (limit: {config.MAX_TRADE_DURATION_HOURS}h, avg RR: {report['avg_timeout_rr']:+.2f})")
    logger.info(f"    Fees paid:     {report['total_fees_pct']:.4%} (rate: {config.TRADING_FEE_PCT}% per side)")
    logger.info("")
    if acc["all_pass"]:
        logger.info("  ALL ACCEPTANCE CRITERIA PASSED")
    else:
        logger.info("  SOME ACCEPTANCE CRITERIA FAILED -- consider retraining")
    logger.info("=" * 60)


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
