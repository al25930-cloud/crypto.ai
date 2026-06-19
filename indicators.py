"""
Technical indicator computation.

Uses TA-Lib as primary library (fast, C-based).
Falls back to pandas_ta if TA-Lib is not installed.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Detect indicator library
try:
    import talib

    USE_TA_LIB = True
    logger.info("TA-Lib loaded successfully. Using TA-Lib for indicators.")
except ImportError:
    try:
        import pandas_ta as ta

        USE_TA_LIB = False
        logger.warning("[WARNING] TA-Lib not found. Using pandas_ta fallback.")
    except ImportError:
        raise ImportError(
            "Neither TA-Lib nor pandas_ta is installed. "
            "Install one: pip install TA-Lib  OR  pip install pandas_ta"
        )


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators needed for the 53 conditions.

    Adds columns to the DataFrame for every indicator referenced by the
    condition pool. After calling this, any row with NaN in an indicator
    column represents the warmup period and should be dropped.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume].
            Index should be datetime or integer. Columns are case-insensitive
            and will be normalized to lowercase.

    Returns:
        DataFrame with all indicator columns added (modified in-place and returned).
    """
    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values.astype(float)

    if USE_TA_LIB:
        _compute_indicators_talib(df, close, high, low, volume)
    else:
        _compute_indicators_pandas_ta(df)

    return df


def _compute_indicators_talib(
    df: pd.DataFrame,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
) -> None:
    """Compute indicators using TA-Lib."""
    # === Trend: EMAs ===
    df["ema_9"] = talib.EMA(close, timeperiod=9)
    df["ema_12"] = talib.EMA(close, timeperiod=12)
    df["ema_20"] = talib.EMA(close, timeperiod=20)
    df["ema_21"] = talib.EMA(close, timeperiod=21)
    df["ema_26"] = talib.EMA(close, timeperiod=26)
    df["sma_20"] = talib.SMA(close, timeperiod=20)
    df["sma_50"] = talib.SMA(close, timeperiod=50)
    df["sma_200"] = talib.SMA(close, timeperiod=200)

    # === MACD ===
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # === Momentum: RSI ===
    df["rsi_14"] = talib.RSI(close, timeperiod=14)
    df["rsi_21"] = talib.RSI(close, timeperiod=21)

    # === Momentum: Stochastic ===
    stoch_k, stoch_d = talib.STOCH(
        high, low, close,
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0,
    )
    df["stoch_k"] = stoch_k

    # === Momentum: CCI ===
    df["cci_14"] = talib.CCI(high, low, close, timeperiod=14)

    # === Momentum: Williams %R ===
    df["williams_r"] = talib.WILLR(high, low, close, timeperiod=14)

    # === Volatility: Bollinger Bands (20, 2.0) ===
    bb_upper_2, bb_middle_2, bb_lower_2 = talib.BBANDS(
        close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
    )
    df["bb_upper_20_2"] = bb_upper_2
    df["bb_lower_20_2"] = bb_lower_2

    # === Volatility: Bollinger Bands (20, 1.5) ===
    bb_upper_1_5, _, bb_lower_1_5 = talib.BBANDS(
        close, timeperiod=20, nbdevup=1.5, nbdevdn=1.5, matype=0
    )
    df["bb_upper_20_1_5"] = bb_upper_1_5
    df["bb_lower_20_1_5"] = bb_lower_1_5

    # === Volatility: ATR ===
    df["atr_14"] = talib.ATR(high, low, close, timeperiod=14)
    df["sma_atr_20"] = talib.SMA(df["atr_14"].values, timeperiod=20)

    # === Volume ===
    df["sma_volume_20"] = talib.SMA(volume, timeperiod=20)

    # === Volume: OBV ===
    df["obv"] = talib.OBV(close, volume)
    df["sma_obv_20"] = talib.SMA(df["obv"].values, timeperiod=20)

    # === ADX ===
    df["adx_14"] = talib.ADX(high, low, close, timeperiod=14)

    # === Price Action: Highest High / Lowest Low (20 periods) ===
    df["highest_high_20"] = talib.MAX(high, timeperiod=20)
    df["lowest_low_20"] = talib.MIN(low, timeperiod=20)

    logger.debug("All indicators computed via TA-Lib.")


def _compute_indicators_pandas_ta(df: pd.DataFrame) -> None:
    """Compute indicators using pandas_ta."""
    import pandas_ta as ta

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)

    # === Trend: EMAs ===
    df["ema_9"] = ta.ema(close, length=9)
    df["ema_12"] = ta.ema(close, length=12)
    df["ema_20"] = ta.ema(close, length=20)
    df["ema_21"] = ta.ema(close, length=21)
    df["ema_26"] = ta.ema(close, length=26)
    df["sma_20"] = ta.sma(close, length=20)
    df["sma_50"] = ta.sma(close, length=50)
    df["sma_200"] = ta.sma(close, length=200)

    # === MACD ===
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        df["macd"] = macd_df.iloc[:, 0]  # MACD line
        df["macd_signal"] = macd_df.iloc[:, 2]  # Signal line
        df["macd_hist"] = macd_df.iloc[:, 1]  # Histogram

    # === Momentum: RSI ===
    df["rsi_14"] = ta.rsi(close, length=14)
    df["rsi_21"] = ta.rsi(close, length=21)

    # === Momentum: Stochastic ===
    stoch_df = ta.stoch(high=high, low=low, close=close, k=14, d=3, smooth_k=3)
    if stoch_df is not None and not stoch_df.empty:
        df["stoch_k"] = stoch_df.iloc[:, 0]

    # === Momentum: CCI ===
    df["cci_14"] = ta.cci(high=high, low=low, close=close, length=14)

    # === Momentum: Williams %R ===
    df["williams_r"] = ta.willr(high=high, low=low, close=close, length=14)

    # === Volatility: Bollinger Bands (20, 2.0) ===
    bb_2 = ta.bbands(close, length=20, std=2.0)
    if bb_2 is not None and not bb_2.empty:
        df["bb_lower_20_2"] = bb_2.iloc[:, 0]
        df["bb_upper_20_2"] = bb_2.iloc[:, 2]

    # === Volatility: Bollinger Bands (20, 1.5) ===
    bb_1_5 = ta.bbands(close, length=20, std=1.5)
    if bb_1_5 is not None and not bb_1_5.empty:
        df["bb_lower_20_1_5"] = bb_1_5.iloc[:, 0]
        df["bb_upper_20_1_5"] = bb_1_5.iloc[:, 2]

    # === Volatility: ATR ===
    df["atr_14"] = ta.atr(high=high, low=low, close=close, length=14)
    df["sma_atr_20"] = ta.sma(df["atr_14"], length=20)

    # === Volume ===
    df["sma_volume_20"] = ta.sma(volume, length=20)

    # === Volume: OBV ===
    df["obv"] = ta.obv(close, volume)
    df["sma_obv_20"] = ta.sma(df["obv"], length=20)

    # === ADX ===
    adx_df = ta.adx(high=high, low=low, close=close, length=14)
    if adx_df is not None and not adx_df.empty:
        df["adx_14"] = adx_df.iloc[:, 0]

    # === Price Action: Highest High / Lowest Low (20 periods) ===
    df["highest_high_20"] = high.rolling(window=20).max()
    df["lowest_low_20"] = low.rolling(window=20).min()

    logger.debug("All indicators computed via pandas_ta.")


def compute_condition(
    df: pd.DataFrame, condition_key: str, row_index: Optional[int] = None
) -> pd.Series:
    """Evaluate a single condition across the DataFrame (or at a specific row).

    Args:
        df: DataFrame with all indicator columns already computed.
        condition_key: The condition key string (e.g., "ema_9_gt_21").
        row_index: If provided, evaluate only at this row index. Otherwise evaluate all rows.

    Returns:
        Boolean Series (or single bool if row_index is provided).
        NaN indicator values result in False (condition not met).
    """
    if row_index is not None:
        row = df.iloc[row_index]
        return _evaluate_single(condition_key, row)

    return df.apply(lambda row: _evaluate_single(condition_key, row), axis=1)


def compute_all_conditions(
    df: pd.DataFrame, condition_keys: list[str]
) -> pd.DataFrame:
    """Evaluate multiple conditions and return a boolean DataFrame.

    Uses vectorized pandas operations for speed (critical during backtesting
    where thousands of strategies are tested on large DataFrames).

    Args:
        df: DataFrame with all indicator columns already computed.
        condition_keys: List of condition key strings to evaluate.

    Returns:
        DataFrame with boolean columns named after each condition_key.
    """
    result = pd.DataFrame(index=df.index)
    for key in condition_keys:
        result[key] = _evaluate_vectorized(key, df)
    return result


def _evaluate_vectorized(condition_key: str, df: pd.DataFrame) -> pd.Series:
    """Evaluate a condition across all rows using vectorized pandas operations.

    Args:
        condition_key: The condition key string.
        df: DataFrame with indicator columns.

    Returns:
        Boolean Series. NaN indicator values result in False.
    """
    try:
        result = _get_vectorized_result(condition_key, df)
        if result is None:
            return pd.Series(False, index=df.index)
        return result.fillna(False).astype(bool)
    except (KeyError, TypeError) as e:
        logger.warning(f"Vectorized eval failed for '{condition_key}': {e}")
        return pd.Series(False, index=df.index)


# Module-level vectorized condition map (created once, reused for all evaluations)
_VECTORIZED_MAP: dict[str, callable] = {
        # Trend: EMA / SMA crosses
        "ema_9_gt_21": lambda d: d["ema_9"] > d["ema_21"],
        "ema_9_lt_21": lambda d: d["ema_9"] < d["ema_21"],
        "ema_12_gt_26": lambda d: d["ema_12"] > d["ema_26"],
        "ema_12_lt_26": lambda d: d["ema_12"] < d["ema_26"],
        "ema_20_gt_50": lambda d: d["ema_20"] > d["sma_50"],
        "ema_20_lt_50": lambda d: d["ema_20"] < d["sma_50"],
        "sma_20_gt_50": lambda d: d["sma_20"] > d["sma_50"],
        "sma_20_lt_50": lambda d: d["sma_20"] < d["sma_50"],
        "price_gt_sma_50": lambda d: d["close"] > d["sma_50"],
        "price_lt_sma_50": lambda d: d["close"] < d["sma_50"],
        "price_gt_sma_200": lambda d: d["close"] > d["sma_200"],
        "price_lt_sma_200": lambda d: d["close"] < d["sma_200"],
        # MACD
        "macd_gt_signal": lambda d: d["macd"] > d["macd_signal"],
        "macd_lt_signal": lambda d: d["macd"] < d["macd_signal"],
        "macd_hist_gt_0": lambda d: d["macd_hist"] > 0,
        "macd_hist_lt_0": lambda d: d["macd_hist"] < 0,
        # RSI
        "rsi_14_gt_50": lambda d: d["rsi_14"] > 50,
        "rsi_14_lt_50": lambda d: d["rsi_14"] < 50,
        "rsi_14_lt_30": lambda d: d["rsi_14"] < 30,
        "rsi_14_gt_70": lambda d: d["rsi_14"] > 70,
        "rsi_21_gt_50": lambda d: d["rsi_21"] > 50,
        "rsi_21_lt_50": lambda d: d["rsi_21"] < 50,
        "rsi_21_lt_30": lambda d: d["rsi_21"] < 30,
        "rsi_21_gt_70": lambda d: d["rsi_21"] > 70,
        # Stochastic
        "stoch_k_gt_20": lambda d: d["stoch_k"] > 20,
        "stoch_k_lt_80": lambda d: d["stoch_k"] < 80,
        "stoch_k_gt_80": lambda d: d["stoch_k"] > 80,
        "stoch_k_lt_20": lambda d: d["stoch_k"] < 20,
        # CCI
        "cci_14_gt_-100": lambda d: d["cci_14"] > -100,
        "cci_14_lt_100": lambda d: d["cci_14"] < 100,
        "cci_14_gt_100": lambda d: d["cci_14"] > 100,
        "cci_14_lt_-100": lambda d: d["cci_14"] < -100,
        # Williams %R
        "williams_gt_-80": lambda d: d["williams_r"] > -80,
        "williams_lt_-20": lambda d: d["williams_r"] < -20,
        "williams_gt_-20": lambda d: d["williams_r"] > -20,
        "williams_lt_-80": lambda d: d["williams_r"] < -80,
        # Bollinger Bands
        "price_lt_bb_lower_20_2": lambda d: d["close"] < d["bb_lower_20_2"],
        "price_gt_bb_upper_20_2": lambda d: d["close"] > d["bb_upper_20_2"],
        "price_lt_bb_lower_20_1_5": lambda d: d["close"] < d["bb_lower_20_1_5"],
        "price_gt_bb_upper_20_1_5": lambda d: d["close"] > d["bb_upper_20_1_5"],
        "price_gt_bb_upper_20_2_s": lambda d: d["close"] > d["bb_upper_20_2"],
        "price_lt_bb_lower_20_2_s": lambda d: d["close"] < d["bb_lower_20_2"],
        "price_gt_bb_upper_20_1_5_s": lambda d: d["close"] > d["bb_upper_20_1_5"],
        "price_lt_bb_lower_20_1_5_s": lambda d: d["close"] < d["bb_lower_20_1_5"],
        # ATR
        "atr_gt_sma_atr_20": lambda d: d["atr_14"] > d["sma_atr_20"],
        # Volume
        "volume_gt_sma_20_1_5": lambda d: d["volume"] > (d["sma_volume_20"] * 1.5),
        "volume_gt_sma_20_2_0": lambda d: d["volume"] > (d["sma_volume_20"] * 2.0),
        # OBV
        "obv_gt_sma_obv_20": lambda d: d["obv"] > d["sma_obv_20"],
        # ADX
        "adx_14_gt_25": lambda d: d["adx_14"] > 25,
        "adx_14_gt_30": lambda d: d["adx_14"] > 30,
        "adx_14_lt_20": lambda d: d["adx_14"] < 20,
        # Price Action
        "price_gt_high_20_1_02": lambda d: d["close"] > (d["highest_high_20"] * 1.02),
        "price_lt_low_20_0_98": lambda d: d["close"] < (d["lowest_low_20"] * 0.98),
}


def _get_vectorized_result(condition_key: str, df: pd.DataFrame) -> Optional[pd.Series]:
    """Get the vectorized boolean result of a condition.

    Args:
        condition_key: The condition key string.
        df: DataFrame with indicator columns.

    Returns:
        Boolean Series if known, None if unknown key.
    """
    fn = _VECTORIZED_MAP.get(condition_key)
    if fn is None:
        logger.warning(f"Unknown condition key: {condition_key}")
        return None
    return fn(df)


def _evaluate_single(condition_key: str, row: pd.Series) -> bool:
    """Evaluate a single condition on a single row of data.

    Args:
        condition_key: The condition key string.
        row: A single row (Series) with indicator values.

    Returns:
        True if the condition is met, False otherwise.
        Returns False if any required indicator value is NaN.
    """
    try:
        # Map condition keys to their evaluation logic
        val = _get_indicator_value(condition_key, row)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return False
        return bool(val)
    except (KeyError, TypeError):
        return False


def _get_indicator_value(condition_key: str, row: pd.Series) -> Optional[bool]:
    """Get the boolean result of a condition for a single row.

    Args:
        condition_key: The condition key string.
        row: A single row with indicator values.

    Returns:
        True/False if the condition can be evaluated, None if key is unknown.
    """
    # Helper to safely get values
    def _v(col: str) -> float:
        return float(row.get(col, np.nan))

    # === Trend: EMA / SMA crosses ===
    if condition_key == "ema_9_gt_21":
        return _v("ema_9") > _v("ema_21")
    elif condition_key == "ema_9_lt_21":
        return _v("ema_9") < _v("ema_21")
    elif condition_key == "ema_12_gt_26":
        return _v("ema_12") > _v("ema_26")
    elif condition_key == "ema_12_lt_26":
        return _v("ema_12") < _v("ema_26")
    elif condition_key == "ema_20_gt_50":
        return _v("ema_20") > _v("sma_50")
    elif condition_key == "ema_20_lt_50":
        return _v("ema_20") < _v("sma_50")
    elif condition_key == "sma_20_gt_50":
        return _v("sma_20") > _v("sma_50")
    elif condition_key == "sma_20_lt_50":
        return _v("sma_20") < _v("sma_50")
    elif condition_key == "price_gt_sma_50":
        return _v("close") > _v("sma_50")
    elif condition_key == "price_lt_sma_50":
        return _v("close") < _v("sma_50")
    elif condition_key == "price_gt_sma_200":
        return _v("close") > _v("sma_200")
    elif condition_key == "price_lt_sma_200":
        return _v("close") < _v("sma_200")

    # === MACD ===
    elif condition_key == "macd_gt_signal":
        return _v("macd") > _v("macd_signal")
    elif condition_key == "macd_lt_signal":
        return _v("macd") < _v("macd_signal")
    elif condition_key == "macd_hist_gt_0":
        return _v("macd_hist") > 0
    elif condition_key == "macd_hist_lt_0":
        return _v("macd_hist") < 0

    # === RSI ===
    elif condition_key == "rsi_14_gt_50":
        return _v("rsi_14") > 50
    elif condition_key == "rsi_14_lt_50":
        return _v("rsi_14") < 50
    elif condition_key == "rsi_14_lt_30":
        return _v("rsi_14") < 30
    elif condition_key == "rsi_14_gt_70":
        return _v("rsi_14") > 70
    elif condition_key == "rsi_21_gt_50":
        return _v("rsi_21") > 50
    elif condition_key == "rsi_21_lt_50":
        return _v("rsi_21") < 50
    elif condition_key == "rsi_21_lt_30":
        return _v("rsi_21") < 30
    elif condition_key == "rsi_21_gt_70":
        return _v("rsi_21") > 70

    # === Stochastic ===
    elif condition_key == "stoch_k_gt_20":
        return _v("stoch_k") > 20
    elif condition_key == "stoch_k_lt_80":
        return _v("stoch_k") < 80
    elif condition_key == "stoch_k_gt_80":
        return _v("stoch_k") > 80
    elif condition_key == "stoch_k_lt_20":
        return _v("stoch_k") < 20

    # === CCI ===
    elif condition_key == "cci_14_gt_-100":
        return _v("cci_14") > -100
    elif condition_key == "cci_14_lt_100":
        return _v("cci_14") < 100
    elif condition_key == "cci_14_gt_100":
        return _v("cci_14") > 100
    elif condition_key == "cci_14_lt_-100":
        return _v("cci_14") < -100

    # === Williams %R ===
    elif condition_key == "williams_gt_-80":
        return _v("williams_r") > -80
    elif condition_key == "williams_lt_-20":
        return _v("williams_r") < -20
    elif condition_key == "williams_gt_-20":
        return _v("williams_r") > -20
    elif condition_key == "williams_lt_-80":
        return _v("williams_r") < -80

    # === Bollinger Bands ===
    elif condition_key == "price_lt_bb_lower_20_2":
        return _v("close") < _v("bb_lower_20_2")
    elif condition_key == "price_gt_bb_upper_20_2":
        return _v("close") > _v("bb_upper_20_2")
    elif condition_key == "price_lt_bb_lower_20_1_5":
        return _v("close") < _v("bb_lower_20_1_5")
    elif condition_key == "price_gt_bb_upper_20_1_5":
        return _v("close") > _v("bb_upper_20_1_5")
    # SHORT BB variants (same logic, different naming)
    elif condition_key == "price_gt_bb_upper_20_2_s":
        return _v("close") > _v("bb_upper_20_2")
    elif condition_key == "price_lt_bb_lower_20_2_s":
        return _v("close") < _v("bb_lower_20_2")
    elif condition_key == "price_gt_bb_upper_20_1_5_s":
        return _v("close") > _v("bb_upper_20_1_5")
    elif condition_key == "price_lt_bb_lower_20_1_5_s":
        return _v("close") < _v("bb_lower_20_1_5")

    # === ATR / Volatility ===
    elif condition_key == "atr_gt_sma_atr_20":
        return _v("atr_14") > _v("sma_atr_20")

    # === Volume ===
    elif condition_key == "volume_gt_sma_20_1_5":
        return _v("volume") > (_v("sma_volume_20") * 1.5)
    elif condition_key == "volume_gt_sma_20_2_0":
        return _v("volume") > (_v("sma_volume_20") * 2.0)

    # === OBV ===
    elif condition_key == "obv_gt_sma_obv_20":
        return _v("obv") > _v("sma_obv_20")

    # === ADX ===
    elif condition_key == "adx_14_gt_25":
        return _v("adx_14") > 25
    elif condition_key == "adx_14_gt_30":
        return _v("adx_14") > 30
    elif condition_key == "adx_14_lt_20":
        return _v("adx_14") < 20

    # === Price Action: Breakout / Breakdown ===
    elif condition_key == "price_gt_high_20_1_02":
        return _v("close") > (_v("highest_high_20") * 1.02)
    elif condition_key == "price_lt_low_20_0_98":
        return _v("close") < (_v("lowest_low_20") * 0.98)

    else:
        logger.warning(f"Unknown condition key: {condition_key}")
        return None


def get_required_indicators(condition_keys: list[str]) -> list[str]:
    """Get the list of indicator column names required by a set of conditions.

    Useful for determining which columns to check for NaN values.

    Args:
        condition_keys: List of condition key strings.

    Returns:
        Sorted list of unique indicator column names.
    """
    indicators = set()
    for key in condition_keys:
        indicators.update(_CONDITION_INDICATOR_MAP.get(key, []))
    return sorted(indicators)


# Mapping from condition keys to the indicator columns they reference
_CONDITION_INDICATOR_MAP: dict[str, list[str]] = {
    # Trend
    "ema_9_gt_21": ["ema_9", "ema_21"],
    "ema_9_lt_21": ["ema_9", "ema_21"],
    "ema_12_gt_26": ["ema_12", "ema_26"],
    "ema_12_lt_26": ["ema_12", "ema_26"],
    "ema_20_gt_50": ["ema_20", "sma_50"],
    "ema_20_lt_50": ["ema_20", "sma_50"],
    "sma_20_gt_50": ["sma_20", "sma_50"],
    "sma_20_lt_50": ["sma_20", "sma_50"],
    "price_gt_sma_50": ["close", "sma_50"],
    "price_lt_sma_50": ["close", "sma_50"],
    "price_gt_sma_200": ["close", "sma_200"],
    "price_lt_sma_200": ["close", "sma_200"],
    # MACD
    "macd_gt_signal": ["macd", "macd_signal"],
    "macd_lt_signal": ["macd", "macd_signal"],
    "macd_hist_gt_0": ["macd_hist"],
    "macd_hist_lt_0": ["macd_hist"],
    # RSI
    "rsi_14_gt_50": ["rsi_14"],
    "rsi_14_lt_50": ["rsi_14"],
    "rsi_14_lt_30": ["rsi_14"],
    "rsi_14_gt_70": ["rsi_14"],
    "rsi_21_gt_50": ["rsi_21"],
    "rsi_21_lt_50": ["rsi_21"],
    "rsi_21_lt_30": ["rsi_21"],
    "rsi_21_gt_70": ["rsi_21"],
    # Stochastic
    "stoch_k_gt_20": ["stoch_k"],
    "stoch_k_lt_80": ["stoch_k"],
    "stoch_k_gt_80": ["stoch_k"],
    "stoch_k_lt_20": ["stoch_k"],
    # CCI
    "cci_14_gt_-100": ["cci_14"],
    "cci_14_lt_100": ["cci_14"],
    "cci_14_gt_100": ["cci_14"],
    "cci_14_lt_-100": ["cci_14"],
    # Williams
    "williams_gt_-80": ["williams_r"],
    "williams_lt_-20": ["williams_r"],
    "williams_gt_-20": ["williams_r"],
    "williams_lt_-80": ["williams_r"],
    # Bollinger Bands
    "price_lt_bb_lower_20_2": ["close", "bb_lower_20_2"],
    "price_gt_bb_upper_20_2": ["close", "bb_upper_20_2"],
    "price_lt_bb_lower_20_1_5": ["close", "bb_lower_20_1_5"],
    "price_gt_bb_upper_20_1_5": ["close", "bb_upper_20_1_5"],
    "price_gt_bb_upper_20_2_s": ["close", "bb_upper_20_2"],
    "price_lt_bb_lower_20_2_s": ["close", "bb_lower_20_2"],
    "price_gt_bb_upper_20_1_5_s": ["close", "bb_upper_20_1_5"],
    "price_lt_bb_lower_20_1_5_s": ["close", "bb_lower_20_1_5"],
    # ATR
    "atr_gt_sma_atr_20": ["atr_14", "sma_atr_20"],
    # Volume
    "volume_gt_sma_20_1_5": ["volume", "sma_volume_20"],
    "volume_gt_sma_20_2_0": ["volume", "sma_volume_20"],
    # OBV
    "obv_gt_sma_obv_20": ["obv", "sma_obv_20"],
    # ADX
    "adx_14_gt_25": ["adx_14"],
    "adx_14_gt_30": ["adx_14"],
    "adx_14_lt_20": ["adx_14"],
    # Price Action
    "price_gt_high_20_1_02": ["close", "highest_high_20"],
    "price_lt_low_20_0_98": ["close", "lowest_low_20"],
}
