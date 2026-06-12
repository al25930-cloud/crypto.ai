"""
Central configuration for the Crypto Trading Bot.

All tunable strategy parameters, fixed constants, and optimization metadata
live here.  signals.py, backtest.py, live_signal.py, and optimize.py all
import from this module so there is a single source of truth.

To deploy optimized parameters: edit the values below (or let optimize.py
overwrite them via Phase 5).
"""

# =============================================================================
# Strategy Parameters (tunable via optimization)
# =============================================================================

VOTING_THRESHOLD = 1       # total >= this  -> LONG,  total <= -this -> SHORT

# Signal A – Trend Following
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14

# Signal B – Mean Reversion
ZSCORE_PERIOD = 20
ZSCORE_THRESHOLD = 2.0

# Signal C – Volume Breakout
BB_PERIOD = 20
BB_STD = 2.0
VOLUME_PERIOD = 20
VOLUME_MULTIPLIER = 1.5

# Risk Management
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5
ATR_TP_MULT = 2.5

# =============================================================================
# Fixed Constants (NOT optimized)
# =============================================================================

COMMISSION = 0.0009       # 0.09% (0.04% Binance taker + 0.05% slippage)
POSITION_SIZE = 0.5       # 50% of equity per trade (margin buffer)
INITIAL_CAPITAL = 500_000
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

# =============================================================================
# Optimization Metadata
# =============================================================================

# Ordered list of parameter keys that optimize.py will tune.
# Must match the keys used in Strategy.__init__'s params dict.
OPTIMIZE_PARAMS = [
    "voting_threshold",
    "ema_fast",
    "ema_slow",
    "zscore_threshold",
    "volume_multiplier",
    "atr_stop_mult",
    "atr_tp_mult",
]

# --- column-name helpers (used by signals.py & live_signal.py) ------------

def ema_fast_col() -> str:
    return f"EMA_{EMA_FAST}"

def ema_slow_col() -> str:
    return f"EMA_{EMA_SLOW}"

def rsi_col() -> str:
    return f"RSI_{RSI_PERIOD}"

def sma_col() -> str:
    return f"SMA_{ZSCORE_PERIOD}"

def stdev_col() -> str:
    return f"STDEV_{ZSCORE_PERIOD}"

def atr_col() -> str:
    return f"ATR_{ATR_PERIOD}"

def vol_sma_col() -> str:
    return f"VOLUME_SMA_{VOLUME_PERIOD}"
