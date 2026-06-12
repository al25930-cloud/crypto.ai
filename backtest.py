"""
Crypto Trading Bot — Backtest Module.

Runs the strategy on historical data for BTC/USDT and ETH/USDT using the
backtesting.py library.  Fetches OHLCV data from Binance via ccxt and caches
it to CSV for faster subsequent runs.

Usage:
    python backtest.py                          # Run for both symbols
    python backtest.py --symbol BTC/USDT        # Single symbol
    python backtest.py --redownload             # Force fresh download
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy

from config import (
    EMA_FAST,
    EMA_SLOW,
    RSI_PERIOD,
    ZSCORE_PERIOD,
    ZSCORE_THRESHOLD,
    BB_PERIOD,
    BB_STD,
    VOLUME_PERIOD,
    VOLUME_MULTIPLIER,
    ATR_PERIOD,
    ATR_STOP_MULT,
    ATR_TP_MULT,
    VOTING_THRESHOLD,
    COMMISSION,
    POSITION_SIZE,
    INITIAL_CAPITAL,
    SYMBOLS,
    TIMEFRAME,
    START_DATE,
    END_DATE,
)

from signals import (
    signal_a_trend,
    signal_b_mean_reversion,
    signal_c_volume_breakout,
    voting_system,
    calculate_risk,
)

# =============================================================================
# Logging Setup
# =============================================================================

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            log_dir / "backtest.log", encoding="utf-8"
        ),
    ],
)
logging.Formatter.converter = time.gmtime  # UTC timestamps
logger = logging.getLogger(__name__)


# =============================================================================
# Data Fetching & Caching
# =============================================================================


def fetch_ohlcv(
    symbol: str, timeframe: str, start: str, end: str
) -> pd.DataFrame:
    """Download OHLCV data from Binance via ccxt with pagination.

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT'.
        timeframe: Candle timeframe, e.g. '1h'.
        start: Start date string 'YYYY-MM-DD'.
        end: End date string 'YYYY-MM-DD'.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = exchange.parse8601(f"{start}T00:00:00Z")
    end_ms = exchange.parse8601(f"{end}T23:59:59Z")

    all_candles = []
    current_since = since_ms

    logger.info(
        "Downloading %s %s from %s to %s ...",
        symbol,
        timeframe,
        start,
        end,
    )

    while current_since < end_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe, current_since, limit=1000
            )
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.error("CCXT error: %s. Retrying in 5s ...", e)
            time.sleep(5)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        # Advance beyond the last fetched timestamp
        current_since = candles[-1][0] + 1

    if not all_candles:
        raise RuntimeError(f"No data returned for {symbol}.")

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Downloaded %d candles for %s.", len(df), symbol)
    return df


def load_or_fetch_data(
    symbol: str, redownload: bool = False
) -> pd.DataFrame:
    """Load OHLCV data from CSV cache or fetch from Binance.

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT'.
        redownload: If True, force a fresh download even if cache exists.

    Returns:
        DataFrame ready for backtesting (DatetimeIndex, capitalized columns).
    """
    safe_name = symbol.replace("/", "_").lower()  # btc_usdt
    csv_path = Path("data") / f"{safe_name}_1h.csv"
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    if csv_path.exists() and not redownload:
        logger.info("Loading cached data from %s", csv_path)
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        return _prepare_for_backtest(df)

    # Fetch and cache
    df = fetch_ohlcv(symbol, TIMEFRAME, START_DATE, END_DATE)
    df.to_csv(csv_path, index=False)
    logger.info("Cached data to %s", csv_path)

    return _prepare_for_backtest(df)


def _prepare_for_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for backtesting.py.

    - Sets timestamp as DatetimeIndex (UTC).
    - Capitalizes OHLCV column names to match backtesting.py expectations.
    - Filters to the configured date range.
    """
    df = df.copy()
    df.set_index("timestamp", inplace=True)
    df.index = pd.DatetimeIndex(df.index, tz="UTC")

    # Filter date range
    start = pd.Timestamp(START_DATE, tz="UTC")
    end = pd.Timestamp(END_DATE, tz="UTC")
    df = df.loc[start:end]

    # Capitalize for backtesting.py (only if lowercase columns exist —
    # cached CSVs may already have capitalized names from a prior run).
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    existing_lower = [c for c in rename_map if c in df.columns]
    if existing_lower:
        df.rename(columns={c: rename_map[c] for c in existing_lower}, inplace=True)

    return df


# =============================================================================
# Strategy parameter injection
# =============================================================================

# backtesting.py instantiates Strategy(broker, data) with exactly 2 positional
# args, so we cannot pass ``params`` through the constructor directly.
# Instead, the Strategy reads this module-level variable in __init__.
_CURRENT_PARAMS: Optional[dict] = None


class CryptoStrategy(Strategy):
    """Trend-following + mean-reversion + volume-breakout strategy.

    Uses backtesting.py's built-in stop-loss / take-profit (absolute price
    levels) and handles commission + slippage automatically.

    Parameters can be overridden via the ``params`` dict passed to the
    constructor, enabling optimization without monkey-patching globals.
    """

    # ── instance defaults (set in __init__) ─────────────────────────────
    ema_fast: int
    ema_slow: int
    rsi_period: int
    zscore_period: int
    zscore_threshold: float
    bb_period: int
    bb_std: float
    volume_period: int
    volume_multiplier: float
    atr_period: int
    atr_stop_mult: float
    atr_tp_mult: float
    voting_threshold: int

    def __init__(self, broker, data, params: Optional[dict] = None) -> None:  # noqa: D107
        super().__init__(broker, data, params)

        # Read from module-level _CURRENT_PARAMS if set (used by optimize.py),
        # otherwise fall back to the explicit ``params`` arg (unused by backtesting.py)
        # or config.py defaults.
        p = _CURRENT_PARAMS or params or {}
        self.ema_fast = p.get("ema_fast", EMA_FAST)
        self.ema_slow = p.get("ema_slow", EMA_SLOW)
        self.rsi_period = p.get("rsi_period", RSI_PERIOD)
        self.zscore_period = p.get("zscore_period", ZSCORE_PERIOD)
        self.zscore_threshold = p.get("zscore_threshold", ZSCORE_THRESHOLD)
        self.bb_period = p.get("bb_period", BB_PERIOD)
        self.bb_std = p.get("bb_std", BB_STD)
        self.volume_period = p.get("volume_period", VOLUME_PERIOD)
        self.volume_multiplier = p.get("volume_multiplier", VOLUME_MULTIPLIER)
        self.atr_period = p.get("atr_period", ATR_PERIOD)
        self.atr_stop_mult = p.get("atr_stop_mult", ATR_STOP_MULT)
        self.atr_tp_mult = p.get("atr_tp_mult", ATR_TP_MULT)
        self.voting_threshold = p.get("voting_threshold", VOTING_THRESHOLD)

    def init(self):  # noqa: D102
        # backtesting.py's self.I() passes numpy arrays to indicator
        # functions, but pandas_ta expects pandas Series.  Every wrapper
        # below converts the numpy array(s) to Series before calling the
        # pandas_ta function and returns a numpy array of the same length.
        # All indicator periods come from instance attributes so that they
        # reflect any parameter overrides.

        # ── helpers ──────────────────────────────────────────────────────
        def _ema(arr: np.ndarray, length: int) -> np.ndarray:
            return ta.ema(pd.Series(arr), length=length).to_numpy()

        def _rsi(arr: np.ndarray, length: int) -> np.ndarray:
            return ta.rsi(pd.Series(arr), length=length).to_numpy()

        def _sma(arr: np.ndarray, length: int) -> np.ndarray:
            return ta.sma(pd.Series(arr), length=length).to_numpy()

        def _atr(
            high_arr: np.ndarray,
            low_arr: np.ndarray,
            close_arr: np.ndarray,
            length: int,
        ) -> np.ndarray:
            return ta.atr(
                pd.Series(high_arr),
                pd.Series(low_arr),
                pd.Series(close_arr),
                length=length,
            ).to_numpy()

        def _zscore(arr: np.ndarray, length: int) -> np.ndarray:
            s = pd.Series(arr)
            sma = ta.sma(s, length=length)
            std = ta.stdev(s, length=length)
            std = std.replace(0, np.nan)
            return ((s - sma) / std).to_numpy()

        def _bbu(arr: np.ndarray, length: int, std: float) -> np.ndarray:
            bb = ta.bbands(pd.Series(arr), length=length, std=std)
            col = next(c for c in bb.columns if c.startswith("BBU_"))
            return bb[col].to_numpy()

        def _bbl(arr: np.ndarray, length: int, std: float) -> np.ndarray:
            bb = ta.bbands(pd.Series(arr), length=length, std=std)
            col = next(c for c in bb.columns if c.startswith("BBL_"))
            return bb[col].to_numpy()

        # ── register indicators ─────────────────────────────────────────
        self.ema9 = self.I(_ema, self.data.Close, length=self.ema_fast)
        self.ema21 = self.I(_ema, self.data.Close, length=self.ema_slow)
        self.rsi = self.I(_rsi, self.data.Close, length=self.rsi_period)
        self.zscore = self.I(_zscore, self.data.Close,
                             length=self.zscore_period)
        self.bb_upper = self.I(_bbu, self.data.Close,
                               length=self.bb_period, std=self.bb_std)
        self.bb_lower = self.I(_bbl, self.data.Close,
                               length=self.bb_period, std=self.bb_std)
        self.vol_sma = self.I(_sma, self.data.Volume,
                              length=self.volume_period)
        self.atr = self.I(_atr, self.data.High, self.data.Low,
                          self.data.Close, length=self.atr_period)

    def next(self):  # noqa: D102
        # Ensure enough bars have passed for all indicators to warm up
        min_bars = max(
            self.ema_slow, self.atr_period, self.zscore_period,
            self.bb_period, self.volume_period,
        )
        if len(self.data.Close) < min_bars + 2:
            return

        # Current and previous values
        ema9 = self.ema9[-1]
        ema21 = self.ema21[-1]
        ema9_prev = self.ema9[-2]
        ema21_prev = self.ema21[-2]
        rsi = self.rsi[-1]
        zscore = self.zscore[-1]
        close = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        atr = self.atr[-1]

        # Skip if any indicator is NaN
        if any(
            pd.isna(x)
            for x in [
                ema9,
                ema21,
                ema9_prev,
                ema21_prev,
                rsi,
                zscore,
                vol,
                vol_sma,
                bb_upper,
                bb_lower,
                atr,
            ]
        ):
            return

        # Compute signals
        sig_a = signal_a_trend(ema9, ema21, ema9_prev, ema21_prev, rsi)
        sig_b = signal_b_mean_reversion(zscore)
        sig_c = signal_c_volume_breakout(
            close, vol, vol_sma, bb_upper, bb_lower
        )
        action, _total = voting_system(sig_a, sig_b, sig_c)

        # Enter trades
        if action == "LONG" and not self.position.is_long:
            sl, tp = calculate_risk(close, atr, "LONG")
            self.buy(sl=sl, tp=tp, size=POSITION_SIZE)
        elif action == "SHORT" and not self.position.is_short:
            sl, tp = calculate_risk(close, atr, "SHORT")
            self.sell(sl=sl, tp=tp, size=POSITION_SIZE)


# =============================================================================
# Backtest Runner (importable by optimize.py)
# =============================================================================


def run_backtest(
    symbol: str,
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    params: Optional[dict] = None,
) -> dict:
    """Run the strategy backtest on a single symbol.

    Args:
        symbol: Trading pair name (for display only).
        df: Full OHLCV DataFrame with DatetimeIndex and capitalized columns.
        start_date: Subset start ('YYYY-MM-DD').  If None, uses first date.
        end_date: Subset end ('YYYY-MM-DD').  If None, uses last date.
        params: Dict of strategy parameter overrides.  If None, uses
                config.py defaults (via CryptoStrategy defaults).

    Returns:
        Dictionary of backtest statistics with keys:
        profit_factor, win_rate, total_trades, sharpe_ratio,
        max_drawdown, return_pct, final_equity,
        and raw ``_trades`` DataFrame.
    """
    # Subset the data
    sub = df.copy()
    if start_date is not None:
        sub = sub.loc[pd.Timestamp(start_date, tz="UTC"):]
    if end_date is not None:
        sub = sub.loc[:pd.Timestamp(end_date, tz="UTC")]

    if len(sub) < 500:
        logger.warning(
            "%s: only %d candles after date filter — backtest may be unreliable.",
            symbol, len(sub),
        )

    logger.info("Running backtest for %s (%d candles) ...", symbol, len(sub))

    bt = Backtest(
        sub,
        CryptoStrategy,
        cash=INITIAL_CAPITAL,
        commission=COMMISSION,
        exclusive_orders=True,
        finalize_trades=True,
    )
    # Inject params via module-level variable (backtesting.py only passes
    # broker/data to the constructor, so we cannot pass params directly).
    global _CURRENT_PARAMS
    _CURRENT_PARAMS = params
    try:
        stats = bt.run()
    finally:
        _CURRENT_PARAMS = None

    # Extract the standard metrics we return
    return {
        "profit_factor": float(stats["Profit Factor"]),
        "win_rate": float(stats["Win Rate [%]"]),
        "total_trades": int(stats["# Trades"]),
        "sharpe_ratio": float(stats["Sharpe Ratio"]),
        "max_drawdown": float(stats["Max. Drawdown [%]"]),
        "return_pct": float(stats["Return [%]"]),
        "final_equity": float(stats["Equity Final [$]"]),
        "_trades": stats["_trades"],
    }


# =============================================================================
# Pretty-printer (used by CLI)
# =============================================================================


def _print_results(symbol: str, stats: dict) -> None:
    """Print backtest results to stdout."""
    sub_start = START_DATE
    sub_end = END_DATE
    print(f"\n{'='*60}")
    print(f"  Backtest Results: {symbol}")
    print(f"{'='*60}")
    print(f"  Period:            {sub_start} -> {sub_end}")
    print(f"  Timeframe:         {TIMEFRAME}")
    print(f"  Starting Capital:  ${INITIAL_CAPITAL:,.2f}")
    print(f"  Commission+Slippage:{COMMISSION*100:.2f}%")
    print(f"{'-'*60}")
    print(f"  Total Trades:      {stats['total_trades']}")
    print(f"  Win Rate:          {stats['win_rate']:.2f}%")
    print(f"  Profit Factor:     {stats['profit_factor']:.2f}")
    print(f"  Max Drawdown:      {stats['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio:      {stats['sharpe_ratio']:.2f}")
    print(f"  Final Equity:      ${stats['final_equity']:,.2f}")
    print(f"  Return:            {stats['return_pct']:.2f}%")
    print(f"{'='*60}\n")


# =============================================================================
# CLI Entry Point  (backward-compatible: python backtest.py [--symbol X])
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot — Backtester"
    )
    parser.add_argument(
        "--symbol",
        choices=SYMBOLS,
        default=None,
        help="Run backtest for a single symbol (default: all).",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Force re-download of historical data from Binance.",
    )
    args = parser.parse_args()

    symbols_to_test = [args.symbol] if args.symbol else SYMBOLS

    logger.info("=== Crypto Trading Bot Backtest ===")
    logger.info(
        "Symbols: %s | Period: %s -> %s | Timeframe: %s",
        symbols_to_test, START_DATE, END_DATE, TIMEFRAME,
    )

    for symbol in symbols_to_test:
        try:
            df = load_or_fetch_data(symbol, redownload=args.redownload)
            stats = run_backtest(symbol, df)
            _print_results(symbol, stats)

            # Save trade log CSV
            trades = stats["_trades"]
            if len(trades) > 0:
                safe_name = symbol.replace("/", "_").lower()
                trades_path = Path("data") / f"{safe_name}_trades.csv"
                trades.to_csv(trades_path)
                logger.info("Trade log saved to %s (%d trades)",
                            trades_path, len(trades))
        except Exception as e:
            logger.error("Failed to backtest %s: %s", symbol, e, exc_info=True)

    logger.info("Backtest complete.")


if __name__ == "__main__":
    main()
