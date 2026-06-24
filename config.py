"""
Configuration for Crypto Trading Strategy Optimizer.

All parameters are centralized here for easy tuning.
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# === Logging Setup ===
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application. Call once at startup."""
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )


logger = logging.getLogger(__name__)

# === Timeframe ===
TIMEFRAME = "15m"  # "5m", "15m", "1h"

# === Symbol ===
SYMBOL = "BTC/USDT"

# === Training ===
TRAINING_MINUTES = 30
TRAINING_PERIOD_MONTHS = 6
TRAINING_METHOD = "ga_bayesian"  # "random" or "ga_bayesian"

# === Strategy Generation ===
MIN_CONDITION_PERCENTAGE = 0.25  # INACTIVE (superseded by MIN_CONDITIONS_ABSOLUTE)
MAX_CONDITION_PERCENTAGE = 0.65  # Maximum conditions as fraction of pool (ACTIVE)
MIN_CONDITIONS_ABSOLUTE = 4      # ACTIVE — Hard floor (never go below 4 conditions)
MIN_THRESHOLD = 0.3
MAX_THRESHOLD = 0.7
MIN_SL_ATR_MULT = 1.0   # Minimum ATR multiplier for stop loss
MAX_SL_ATR_MULT = 3.0   # Maximum ATR multiplier for stop loss
MIN_RR = 1.0
MAX_RR = 5.0

# === Dynamic Direction Thresholds ===
# Single-gate entry: the dominant direction's strength (true/total for that direction's
# conditions) must clear the strategy's threshold AND be at least DIRECTION_RATIO× stronger
# than the opposite direction. No overall "all conditions" satisfaction gate — this avoided
# false negatives on clear one-sided signals (e.g., 4 LONG true / 0 SHORT would fail Gate 1
# even though it's a strong LONG signal).
DIRECTION_RATIO = 1.3           # Dominant direction must be this many times stronger than opposite

# === GA Parameters ===
GA_POPULATION_SIZE = 200
GA_MAX_GENERATIONS = 200         # Safety cap for time-based GA (rarely hit)
GA_TIME_BUDGET_PERCENT = 0.5     # Fraction of training time allocated to GA
GA_ELITE_COUNT = 5
GA_MUTATION_PROB = 0.2
GA_CROSSOVER_PROB = 0.8

# === Bayesian Parameters ===
BAYESIAN_MAX_TRIALS = 10000      # Safety cap for timeout-based Bayesian (rarely hit)
BAYESIAN_STARTUP_TRIALS = 20     # Random trials before TPE model kicks in (low because GA seeds are strong)

# === Qualification / Disqualification ===
MIN_TRADES_PER_DAY = 1.2
MAX_TRADES_PER_DAY = 10
MIN_WIN_RATE = 0.35
MAX_DRAWDOWN = 0.50
DRAWDOWN_PENALTY_START = 0.15
DRAWDOWN_PENALTY_END = 0.50

# === Timeout Penalty ===
TIMEOUT_PENALTY_THRESHOLD = 0.25   # Fraction of exits that are timeouts before penalty kicks in
TIMEOUT_PENALTY = 0.15             # Score multiplier reduction when timeout threshold is exceeded

# === Trade Parameters ===
MIN_TRADE_DURATION_MINUTES = 45
MAX_TRADE_DURATION_HOURS = 24
COOLDOWN_CANDLES = 4
TRADING_FEE_PCT = 0.1  # per side

# === Live Mode ===
LIVE_CHECK_INTERVAL_SECONDS = 900  # Seconds between live signal checks
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# === Paths ===
BASE_DIR = Path(__file__).parent
DATA_CACHE_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR_TRAINING = BASE_DIR / "logs" / "training"
LOG_DIR_VALIDATION = BASE_DIR / "logs" / "validation"
LOG_DIR_LIVE = BASE_DIR / "logs" / "live"
STATE_FILE = BASE_DIR / "state.json"
REMOVED_CONDITIONS_FILE = MODEL_DIR / "removed_conditions.json"

# Ensure directories exist
DATA_CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR_TRAINING.mkdir(parents=True, exist_ok=True)
LOG_DIR_VALIDATION.mkdir(parents=True, exist_ok=True)
LOG_DIR_LIVE.mkdir(parents=True, exist_ok=True)

# === Indicator Library ===
# Auto-detected: TA-Lib preferred, pandas_ta fallback
# See indicators.py for detection logic

# === Position Sizing ===
RISK_PER_RR = 0.02  # Fraction of equity risked per unit of RR

# === Validation Parameters ===
VALIDATION_WINDOWS = 3          # Number of independent windows to test (1 = single, like before)
VALIDATION_WINDOW_MONTHS = 6    # Months per window
VALIDATION_WINDOW_OVERLAP = 0   # 0 = sequential non-overlapping (recommended), N = slide by N months

# === Walk-Forward Validation (Training) ===
# NOT YET IMPLEMENTED — config toggles only. Each fold backtests on a different slice.
# Enabling this would make training ~3× slower (each strategy tested N times).
USE_WALK_FORWARD = False        # Enable walk-forward fitness during training
WF_WINDOW_MONTHS = 2            # Months per fold

# === Top Strategies ===
TOP_STRATEGIES_COUNT = 500

# === Efficiency Thresholds ===
EFFICIENCY_CRITICAL = 0.3
MIN_POOL_SIZE = 18  # Minimum conditions per direction; refuse removals below this floor
EFFICIENCY_ALERT = 0.5
EFFICIENCY_WARNING = 0.7
EFFICIENCY_STRONG = 1.3
MIN_EVALS_FOR_REMOVAL = 10  # Don't flag conditions for removal/low-eff unless tested >= N times
COVERAGE_EVALS_PER_CONDITION = 5  # Guarantee each condition is tested at least this many times

# === Shared Bonus Weight ===
# SHARED conditions contribute a bonus to confidence (numerator only, not denominator).
# The GA/Bayesian optimizer picks a weight per strategy within this range.
MIN_SHARED_BONUS_WEIGHT = 0.0   # No bonus (SHARED conditions ignored)
MAX_SHARED_BONUS_WEIGHT = 0.15  # Maximum bonus fraction per true SHARED condition
