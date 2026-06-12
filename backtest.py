"""
Crypto Trading Bot — Backtest Module.

Runs the strategy on historical data for BTC/USDT and ETH/USDT using the
backtesting.py library. Fetches OHLCV data from Binance via ccxt and caches
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

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy

from signals import (
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
    signal_a_trend,
    signal_b_mean_reversion,
    signal_c_volume_breakout,
    voting_system,
    calculate_risk,
)

# =============================================================================
# Configuration
# =============================================================================

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"
INITIAL_CAPITAL = 1000  # Simulated USD
# Combined: 0.04% taker fee + 0.05% slippage = 0.09% total per-trade cost.
# backtesting.py has no built-in slippage param, so we fold it into commission.
COMMISSION = 0.0009  # 0.09% (0.04% fee + 0.05% slippage)

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
# Backtesting Strategy Class
# =============================================================================


class CryptoStrategy(Strategy):
    """Trend-following + mean-reversion + volume-breakout strategy.

    Uses backtesting.py's built-in stop-loss / take-profit (absolute price
    levels) and handles commission + slippage automatically.
    """

    def init(self):  # noqa: D102
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        vol = self.data.Volume

        # --- EMAs ---
        self.ema9 = self.I(ta.ema, close, EMA_FAST)
        self.ema21 = self.I(ta.ema, close, EMA_SLOW)

        # --- RSI ---
        self.rsi = self.I(ta.rsi, close, RSI_PERIOD)

        # --- Z-Score (custom indicator) ---
        def _zscore(c: pd.Series, length: int = ZSCORE_PERIOD) -> pd.Series:
            sma = ta.sma(c, length=length)
            std = ta.stdev(c, length=length)
            std = std.replace(0, np.nan)
            return (c - sma) / std

        self.zscore = self.I(_zscore, close)

        # --- Bollinger Bands ---
        def _bbu(c: pd.Series) -> pd.Series:
            bb = ta.bbands(c, length=BB_PERIOD, std=BB_STD)
            return bb[f"BBU_{BB_PERIOD}_{BB_STD}"]

        def _bbl(c: pd.Series) -> pd.Series:
            bb = ta.bbands(c, length=BB_PERIOD, std=BB_STD)
            return bb[f"BBL_{BB_PERIOD}_{BB_STD}"]

        self.bb_upper = self.I(_bbu, close)
        self.bb_lower = self.I(_bbl, close)

        # --- Volume SMA ---
        self.vol_sma = self.I(ta.sma, vol, VOLUME_PERIOD)

        # --- ATR ---
        self.atr = self.I(ta.atr, high, low, close, ATR_PERIOD)

    def next(self):  # noqa: D102
        # Ensure enough bars have passed for all indicators to warm up
        min_bars = max(
            EMA_SLOW, ATR_PERIOD, ZSCORE_PERIOD, BB_PERIOD, VOLUME_PERIOD
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
            self.buy(sl=sl, tp=tp)
        elif action == "SHORT" and not self.position.is_short:
            sl, tp = calculate_risk(close, atr, "SHORT")
            self.sell(sl=sl, tp=tp)


# =============================================================================
# Backtest Runner
# =============================================================================


def run_backtest(symbol: str, df: pd.DataFrame) -> dict:
    """Run the strategy backtest on a single symbol.

    Args:
        symbol: Trading pair name (for display only).
        df: Prepared DataFrame with DatetimeIndex and OHLCV columns.

    Returns:
        Dictionary of backtest statistics.
    """
    logger.info("Running backtest for %s (%d candles) ...", symbol, len(df))

    bt = Backtest(
        df,
        CryptoStrategy,
        cash=INITIAL_CAPITAL,
        commission=COMMISSION,
        exclusive_orders=True,
    )

    stats = bt.run()

    # Print results
    print(f"\n{'='*60}")
    print(f"  Backtest Results: {symbol}")
    print(f"{'='*60}")
    print(f"  Period:            {START_DATE} → {END_DATE}")
    print(f"  Timeframe:         {TIMEFRAME}")
    print(f"  Starting Capital:  ${INITIAL_CAPITAL:,.2f}")
    print(f"  Commission+Slippage:{COMMISSION*100:.2f}%")
    print(f"{'─'*60}")
    print(f"  Total Trades:      {stats['# Trades']}")
    print(f"  Win Rate:          {stats['Win Rate [%]']:.2f}%")
    print(f"  Profit Factor:     {stats['Profit Factor']:.2f}")
    print(f"  Max Drawdown:      {stats['Max. Drawdown [%]']:.2f}%")
    print(f"  Sharpe Ratio:      {stats['Sharpe Ratio']:.2f}")
    print(f"  Final Equity:      ${stats['Equity Final [$]']:,.2f}")
    print(f"  Return:            {stats['Return [%]']:.2f}%")
    print(f"  Buy & Hold Return: {stats['Buy & Hold Return [%]']:.2f}%")
    print(f"{'='*60}\n")

    # Save trade log CSV
    trades = stats["_trades"]
    if len(trades) > 0:
        safe_name = symbol.replace("/", "_").lower()
        trades_path = Path("data") / f"{safe_name}_trades.csv"
        trades.to_csv(trades_path)
        logger.info("Trade log saved to %s (%d trades)", trades_path, len(trades))

    return stats


# =============================================================================
# CLI Entry Point
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
        "Symbols: %s | Period: %s → %s | Timeframe: %s",
        symbols_to_test,
        START_DATE,
        END_DATE,
        TIMEFRAME,
    )

    for symbol in symbols_to_test:
        try:
            df = load_or_fetch_data(symbol, redownload=args.redownload)
            if len(df) < 500:
                logger.warning(
                    "%s: only %d candles — backtest may be unreliable.",
                    symbol,
                    len(df),
                )
            run_backtest(symbol, df)
        except Exception as e:
            logger.error("Failed to backtest %s: %s", symbol, e, exc_info=True)

    logger.info("Backtest complete.")


if __name__ == "__main__":
    main()
