"""
OHLCV data fetcher from Binance via ccxt.

Features:
- Paginated fetching (500 candles per request)
- Incremental CSV caching (only fetch new data)
- Exponential backoff on rate limits and network errors
- Configurable symbol, timeframe, and date range
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

import config

logger = logging.getLogger(__name__)

# CCXT timeframe to timedelta mapping
TIMEFRAME_DELTA = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


def create_exchange() -> ccxt.binance:
    """Create and configure a Binance exchange instance.

    Returns:
        Configured ccxt.binance exchange object.
    """
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
        },
    })
    return exchange


def fetch_ohlcv(
    symbol: str = None,
    timeframe: str = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data with pagination and caching.

    If a cached CSV exists and use_cache is True, only fetches data newer
    than the last cached candle (incremental update).

    Args:
        symbol: Trading pair (e.g., "BTC/USDT"). Defaults to config.SYMBOL.
        timeframe: Candle timeframe (e.g., "15m"). Defaults to config.TIMEFRAME.
        since: Start datetime (UTC). Defaults to training period start.
        until: End datetime (UTC). Defaults to now.
        use_cache: Whether to use cached CSV data.

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume].
        timestamp is datetime64[ns, UTC]. Sorted ascending by time.
    """
    symbol = symbol or config.SYMBOL
    timeframe = timeframe or config.TIMEFRAME
    until = until or datetime.now(timezone.utc)

    if since is None:
        since = until - timedelta(days=config.TRAINING_PERIOD_MONTHS * 30)

    cache_path = _get_cache_path(symbol, timeframe)
    cached_df = None

    # Load cached data if available
    if use_cache and cache_path.exists():
        cached_df = _load_cache(cache_path)
        if cached_df is not None and not cached_df.empty:
            last_cached = cached_df["timestamp"].max()
            # Convert to timezone-aware if needed
            if last_cached.tzinfo is None:
                last_cached = last_cached.tz_localize("UTC")
            # Only fetch from where cache ends
            since_fetch = last_cached + TIMEFRAME_DELTA[timeframe]
            logger.info(
                f"Cache found: {len(cached_df)} candles up to {last_cached}. "
                f"Fetching from {since_fetch}."
            )
            since = max(since, since_fetch)

    # Fetch new data
    new_df = _fetch_paginated(symbol, timeframe, since, until)

    # Merge with cache
    if cached_df is not None and not cached_df.empty and not new_df.empty:
        df = pd.concat([cached_df, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        df = df.sort_values("timestamp").reset_index(drop=True)
    elif not new_df.empty:
        df = new_df
    elif cached_df is not None and not cached_df.empty:
        df = cached_df
    else:
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Save to cache
    if use_cache and not df.empty:
        _save_cache(df, cache_path)

    logger.info(f"Total candles available: {len(df)}")
    return df


def _fetch_paginated(
    symbol: str,
    timeframe: str,
    since: datetime,
    until: datetime,
) -> pd.DataFrame:
    """Fetch OHLCV data with pagination.

    Args:
        symbol: Trading pair.
        timeframe: Candle timeframe.
        since: Start datetime (UTC).
        until: End datetime (UTC).

    Returns:
        DataFrame with fetched OHLCV data.
    """
    exchange = create_exchange()

    since_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000)
    limit = 500

    all_candles = []
    current_since = since_ms
    request_count = 0
    max_retries = 5

    logger.info(
        f"Fetching {symbol} {timeframe} from "
        f"{since.strftime('%Y-%m-%d %H:%M')} to "
        f"{until.strftime('%Y-%m-%d %H:%M')}"
    )

    while current_since < until_ms:
        retries = 0
        while retries < max_retries:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_since, limit=limit
                )
                break
            except ccxt.RateLimitExceeded as e:
                retries += 1
                wait_time = 2 ** retries
                logger.warning(
                    f"Rate limit hit. Retry {retries}/{max_retries} in {wait_time}s."
                )
                time.sleep(wait_time)
            except ccxt.NetworkError as e:
                retries += 1
                wait_time = 2 ** retries
                logger.warning(
                    f"Network error: {e}. Retry {retries}/{max_retries} in {wait_time}s."
                )
                time.sleep(wait_time)
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                raise
        else:
            logger.error(f"Max retries exceeded at {current_since}. Stopping fetch.")
            break

        if not candles:
            break

        all_candles.extend(candles)
        request_count += 1

        # Move to next page
        last_timestamp = candles[-1][0]
        current_since = last_timestamp + 1  # +1ms to avoid duplicate

        # Log progress
        if request_count % 10 == 0:
            fetched_dt = datetime.fromtimestamp(
                last_timestamp / 1000, tz=timezone.utc
            )
            logger.info(
                f"  Fetched {len(all_candles)} candles... "
                f"up to {fetched_dt.strftime('%Y-%m-%d %H:%M')}"
            )

        # Respect rate limits
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        logger.warning("No candles fetched.")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = _candles_to_dataframe(all_candles)

    # Filter to requested range
    df = df[(df["timestamp"] >= since) & (df["timestamp"] <= until)]
    df = df.reset_index(drop=True)

    logger.info(f"Fetched {len(df)} new candles in {request_count} requests.")
    return df


def _candles_to_dataframe(candles: list) -> pd.DataFrame:
    """Convert raw ccxt candle data to a DataFrame.

    Args:
        candles: List of [timestamp_ms, open, high, low, close, volume].

    Returns:
        DataFrame with proper datetime column.
    """
    df = pd.DataFrame(candles, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])
    # Reorder columns
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df


def _get_cache_path(symbol: str, timeframe: str) -> Path:
    """Get the file path for cached OHLCV data.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT").
        timeframe: Candle timeframe (e.g., "15m").

    Returns:
        Path to the CSV cache file.
    """
    safe_symbol = symbol.replace("/", "_").lower()
    filename = f"{safe_symbol}_{timeframe}.csv"
    return config.DATA_CACHE_DIR / filename


def _load_cache(cache_path: Path) -> Optional[pd.DataFrame]:
    """Load cached OHLCV data from CSV.

    Args:
        cache_path: Path to the CSV file.

    Returns:
        DataFrame or None if file is empty/invalid.
    """
    try:
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        if df.empty:
            return None
        # Ensure timezone-aware timestamps
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        return df
    except Exception as e:
        logger.warning(f"Failed to load cache from {cache_path}: {e}")
        return None


def _save_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save OHLCV data to CSV cache.

    Args:
        df: DataFrame to save.
        cache_path: Path to the CSV file.
    """
    try:
        df.to_csv(cache_path, index=False)
        logger.debug(f"Cache saved: {cache_path} ({len(df)} candles)")
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_path}: {e}")


def get_training_data(
    symbol: str = None,
    timeframe: str = None,
    months: int = None,
) -> pd.DataFrame:
    """Convenience function to fetch training period data.

    Args:
        symbol: Trading pair. Defaults to config.SYMBOL.
        timeframe: Candle timeframe. Defaults to config.TIMEFRAME.
        months: Number of months of history. Defaults to config.TRAINING_PERIOD_MONTHS.

    Returns:
        DataFrame with OHLCV data for the training period.
    """
    months = months or config.TRAINING_PERIOD_MONTHS
    until = datetime.now(timezone.utc)
    since = until - timedelta(days=months * 30)
    return fetch_ohlcv(symbol, timeframe, since, until)


def get_validation_data(
    symbol: str = None,
    timeframe: str = None,
    months: int = 12,
) -> pd.DataFrame:
    """Convenience function to fetch validation period data.

    The validation period is strictly before the training period
    to prevent data leakage.

    Args:
        symbol: Trading pair. Defaults to config.SYMBOL.
        timeframe: Candle timeframe. Defaults to config.TIMEFRAME.
        months: Number of months of validation history.

    Returns:
        DataFrame with OHLCV data for the validation period.
    """
    symbol = symbol or config.SYMBOL
    timeframe = timeframe or config.TIMEFRAME

    # Validation ends where training begins
    until = datetime.now(timezone.utc) - timedelta(
        days=config.TRAINING_PERIOD_MONTHS * 30
    )
    since = until - timedelta(days=months * 30)
    return fetch_ohlcv(symbol, timeframe, since, until)


def get_latest_candles(
    symbol: str = None,
    timeframe: str = None,
    count: int = 500,
) -> pd.DataFrame:
    """Fetch the latest N candles (for live mode).

    Args:
        symbol: Trading pair. Defaults to config.SYMBOL.
        timeframe: Candle timeframe. Defaults to config.TIMEFRAME.
        count: Number of candles to fetch.

    Returns:
        DataFrame with the latest OHLCV candles.
    """
    symbol = symbol or config.SYMBOL
    timeframe = timeframe or config.TIMEFRAME

    delta = TIMEFRAME_DELTA.get(timeframe, timedelta(minutes=15))
    until = datetime.now(timezone.utc)
    since = until - (delta * count)

    return fetch_ohlcv(symbol, timeframe, since, until, use_cache=False)
