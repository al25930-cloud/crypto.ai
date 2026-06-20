"""
Condition efficiency analysis.

Analyzes which conditions are underperforming across all tested strategies.
Run after training to identify conditions to remove or promote.
"""

import json
import logging
from typing import Optional

import config
from conditions import ALL_CONDITIONS, get_direction_for_condition, get_condition_pool

logger = logging.getLogger(__name__)


def analyze_conditions(all_results: list[dict]) -> dict:
    """Analyze condition efficiency across all tested strategies.

    For each condition, computes:
    - How many times it was used
    - How many times it appeared in top 10% strategies
    - Average RR/day and win rate of strategies that used it
    - Efficiency score vs global average

    Args:
        all_results: List of dicts with keys 'strategy', 'results', 'score'.
            Each 'strategy' has 'conditions' list, each 'results' has metrics.

    Returns:
        Dict mapping condition_key -> stats dict.
    """
    if not all_results:
        logger.warning("No results to analyze.")
        return {}

    # Filter to strategies with valid scores
    valid = [r for r in all_results if r["score"] > float("-inf")]
    if not valid:
        logger.warning("No valid strategies to analyze.")
        return {}

    # Sort by score for top-10% calculation
    valid.sort(key=lambda r: r["score"], reverse=True)
    top_10_cutoff = max(1, len(valid) // 10)
    top_10_set = set(id(r) for r in valid[:top_10_cutoff])

    # Global averages
    global_rr = sum(r["results"]["rr_per_day"] for r in valid) / len(valid)

    # Per-condition stats
    condition_stats: dict[str, dict] = {}
    for cond_key in ALL_CONDITIONS:
        condition_stats[cond_key] = {
            "used_count": 0,
            "used_in_top_10_percent": 0,
            "rr_per_day_sum": 0.0,
            "win_rate_sum": 0.0,
        }

    for result in valid:
        strategy = result["strategy"]
        results = result["results"]
        conds = strategy.get("conditions", [])
        is_top_10 = id(result) in top_10_set

        for cond in conds:
            if cond in condition_stats:
                condition_stats[cond]["used_count"] += 1
                condition_stats[cond]["rr_per_day_sum"] += results["rr_per_day"]
                condition_stats[cond]["win_rate_sum"] += results["win_rate"]
                if is_top_10:
                    condition_stats[cond]["used_in_top_10_percent"] += 1

    # Compute averages and efficiency scores
    removed_conditions = []
    report_lines = []

    for cond_key, stats in condition_stats.items():
        used = stats["used_count"]
        if used > 0:
            stats["avg_rr_per_day"] = stats["rr_per_day_sum"] / used
            stats["avg_win_rate"] = stats["win_rate_sum"] / used
            stats["efficiency_score"] = stats["avg_rr_per_day"] / global_rr if global_rr > 0 else 0.0
        else:
            stats["avg_rr_per_day"] = 0.0
            stats["avg_win_rate"] = 0.0
            stats["efficiency_score"] = 0.0

        # Determine alert level
        eff = stats["efficiency_score"]
        if used == 0:
            stats["alert_level"] = "NO_DATA"
        elif eff < config.EFFICIENCY_CRITICAL:
            stats["alert_level"] = "CRITICAL"
            removed_conditions.append(cond_key)
        elif eff < config.EFFICIENCY_ALERT:
            stats["alert_level"] = "ALERT"
        elif eff < config.EFFICIENCY_WARNING:
            stats["alert_level"] = "WARNING"
        elif eff <= config.EFFICIENCY_STRONG:
            stats["alert_level"] = "OK"
        else:
            stats["alert_level"] = "STRONG"

        del stats["rr_per_day_sum"]
        del stats["win_rate_sum"]

    # Log the report
    _log_report(condition_stats, global_rr, len(valid), removed_conditions)

    # Save efficiency report
    _save_report(condition_stats, len(valid), global_rr)

    # Auto-remove CRITICAL conditions (with pool size floor safeguard)
    if removed_conditions:
        _remove_conditions(removed_conditions)

    return condition_stats


def _log_report(stats: dict, global_rr: float, total_strategies: int, removed: list) -> None:
    """Log the efficiency report."""
    logger.info("=" * 60)
    logger.info("EFFICIENCY REPORT")
    logger.info("=" * 60)
    logger.info(f"Strategies analyzed: {total_strategies}")
    logger.info(f"Global avg RR/day: {global_rr:.4f}")
    logger.info("")

    for level, emoji in [
        ("CRITICAL", "[CRITICAL]"), ("ALERT", "[ALERT]"), ("WARNING", "[WARNING]"),
        ("STRONG", "[STRONG]"), ("OK", "[OK]"), ("NO_DATA", "[NO DATA]"),
    ]:
        conds = [(k, v) for k, v in stats.items() if v["alert_level"] == level]
        if not conds:
            continue
        label = {
            "CRITICAL": "CRITICAL ALERTS (auto-removed):",
            "ALERT": "ALERTS (consider removing):",
            "WARNING": "WARNINGS (insufficient data):",
            "STRONG": "STRONG CONDITIONS:",
            "OK": "OK CONDITIONS:",
            "NO_DATA": "UNUSED CONDITIONS:",
        }[level]
        logger.info(f"{emoji} {label}")
        for key, s in conds:
            direction = get_direction_for_condition(key)
            desc = ALL_CONDITIONS.get(key, "")
            logger.info(
                f"  '{key}' ({direction}) | "
                f"Used: {s['used_count']} | "
                f"Top10%: {s['used_in_top_10_percent']} | "
                f"Avg RR/day: {s['avg_rr_per_day']:.2f} (global: {global_rr:.2f}) | "
                f"WR: {s['avg_win_rate']:.0%} | "
                f"Eff: {s['efficiency_score']:.2f}"
            )
        logger.info("")

    if removed:
        logger.info(f"Auto-removed {len(removed)} conditions: {removed}")

    logger.info("=" * 60)


def _save_report(stats: dict, total_strategies: int, global_rr: float) -> None:
    """Save efficiency report to JSON."""
    import datetime as _dt
    report = {
        "timestamp": _dt.datetime.now().isoformat(),
        "strategies_analyzed": total_strategies,
        "global_avg_rr_per_day": global_rr,
        "conditions": stats,
    }
    path = config.MODEL_DIR / "condition_efficiency.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Efficiency report saved to {path}")


def _remove_conditions(condition_keys: list) -> None:
    """Add conditions to the removed list, respecting the per-direction pool size floor.

    Checks how many conditions would remain in each direction (LONG, SHORT)
    after removal. If a direction's pool would drop below MIN_POOL_SIZE,
    conditions from that direction are skipped.

    Args:
        condition_keys: List of condition key strings to remove.
    """
    path = config.REMOVED_CONDITIONS_FILE
    existing = set()
    if path.exists():
        try:
            with open(path) as f:
                existing = set(json.load(f).get("removed", []))
        except Exception:
            pass

    # Count current active conditions per direction
    long_pool = set(get_condition_pool("LONG"))
    short_pool = set(get_condition_pool("SHORT"))
    already_removed = existing

    long_active = len(long_pool - already_removed)
    short_active = len(short_pool - already_removed)

    actually_removed = []
    skipped = []

    for key in condition_keys:
        if key in existing:
            continue  # Already removed, skip

        direction = get_direction_for_condition(key)

        # Count how many from each pool we've already committed to removing
        long_removed_count = len([k for k in actually_removed if k in long_pool])
        short_removed_count = len([k for k in actually_removed if k in short_pool])
        long_remaining = long_active - long_removed_count
        short_remaining = short_active - short_removed_count

        # Check pool size floor for the affected direction
        if direction == "LONG" and long_remaining <= config.MIN_POOL_SIZE:
            skipped.append(key)
            logger.info(
                f"[EFFICIENCY] Skipping removal of '{key}' -- LONG pool would drop below {config.MIN_POOL_SIZE}"
            )
            continue
        elif direction == "SHORT" and short_remaining <= config.MIN_POOL_SIZE:
            skipped.append(key)
            logger.info(
                f"[EFFICIENCY] Skipping removal of '{key}' -- SHORT pool would drop below {config.MIN_POOL_SIZE}"
            )
            continue
        elif direction == "SHARED":
            if long_remaining <= config.MIN_POOL_SIZE or short_remaining <= config.MIN_POOL_SIZE:
                skipped.append(key)
                logger.info(
                    f"[EFFICIENCY] Skipping removal of '{key}' -- SHARED condition, pool would drop below {config.MIN_POOL_SIZE}"
                )
                continue

        actually_removed.append(key)

    if not actually_removed:
        if skipped:
            logger.info(f"[EFFICIENCY] All {len(skipped)} CRITICAL conditions kept due to pool size floor ({config.MIN_POOL_SIZE}).")
        return

    all_removed = sorted(existing | set(actually_removed))

    import datetime as _dt
    with open(path, "w") as f:
        json.dump({"removed": all_removed, "updated": _dt.datetime.now().isoformat()}, f, indent=2)

    for key in actually_removed:
        logger.info(f"[EFFICIENCY] Condition '{key}' removed from pool (efficiency < {config.EFFICIENCY_CRITICAL}).")

    if skipped:
        logger.info(
            f"[EFFICIENCY] {len(skipped)} condition(s) kept due to pool size floor: {skipped}"
        )


