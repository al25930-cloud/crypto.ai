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
TRAINING_PERIOD_MONTHS = 12
TRAINING_METHOD = "ga_bayesian"  # "random" or "ga_bayesian"

# === Strategy Generation (percentage-based) ===
MIN_CONDITION_PERCENTAGE = 0.25  # 25% of the pool
MAX_CONDITION_PERCENTAGE = 0.90  # 90% of the pool
MIN_CONDITIONS_ABSOLUTE = 3      # Safety floor (never go below 3 conditions)
MIN_THRESHOLD = 0.5
MAX_THRESHOLD = 0.7
MIN_SL = 0.3  # percent
MAX_SL = 3.0  # percent
MIN_RR = 1.0
MAX_RR = 8.0

# === GA Parameters ===
GA_POPULATION_SIZE = 200
GA_GENERATIONS = 30
GA_ELITE_COUNT = 5
GA_MUTATION_PROB = 0.2
GA_CROSSOVER_PROB = 0.8

# === Bayesian Parameters ===
BAYESIAN_N_TRIALS = 2000
BAYESIAN_STARTUP_TRIALS = 100

# === Qualification / Disqualification ===
MIN_TRADES_PER_DAY = 0.5
MAX_TRADES_PER_DAY = 10
LOW_TRADES_THRESHOLD = 1.5       # If avg trades/day <= this, apply penalty
LOW_TRADES_PENALTY = 0.7        # 30% score penalty for low trade frequency
MIN_WIN_RATE = 0.35  # 35%
MAX_DRAWDOWN = 0.50  # 50%
DRAWDOWN_PENALTY_START = 0.15  # 15%
DRAWDOWN_PENALTY_END = 0.50  # 50%

# === Trade Parameters ===
MIN_TRADE_DURATION_MINUTES = 45
MAX_TRADE_DURATION_HOURS = 48
COOLDOWN_CANDLES = 4
TRADING_FEE_PCT = 0.1  # per side

# === Live Mode ===
LIVE_CHECK_INTERVAL_SECONDS = 900  # 15 minutes
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# === Paths ===
BASE_DIR = Path(__file__).parent
DATA_CACHE_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / "state.json"
REMOVED_CONDITIONS_FILE = MODEL_DIR / "removed_conditions.json"

# Ensure directories exist
DATA_CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# === Indicator Library ===
# Auto-detected: TA-Lib preferred, pandas_ta fallback
# See indicators.py for detection logic

# === Top Strategies ===
TOP_STRATEGIES_COUNT = 500

# === Efficiency Thresholds ===
EFFICIENCY_CRITICAL = 0.3
MIN_POOL_SIZE = 20  # Minimum conditions per direction; refuse removals below this floor
EFFICIENCY_ALERT = 0.5
EFFICIENCY_WARNING = 0.7
EFFICIENCY_STRONG = 1.3
