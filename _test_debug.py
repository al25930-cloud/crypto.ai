"""Debug: run strategy logic on cached data to find why backtest produces 0 trades."""
import pandas as pd
import pandas_ta as ta
import numpy as np
from signals import (
    signal_a_trend, signal_b_mean_reversion, signal_c_volume_breakout,
    voting_system, calculate_risk,
    EMA_FAST, EMA_SLOW, RSI_PERIOD, ZSCORE_PERIOD, ZSCORE_THRESHOLD,
    BB_PERIOD, BB_STD, VOLUME_PERIOD, VOLUME_MULTIPLIER, ATR_PERIOD,
)

# Load cached data
df = pd.read_csv("data/btc_usdt_1h.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
print(f"Loaded {len(df)} candles: {df.index[0]} -> {df.index[-1]}")

# Rename to capitalized for consistency with backtest.py
df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)

close = df["Close"].to_numpy()
high = df["High"].to_numpy()
low = df["Low"].to_numpy()
vol = df["Volume"].to_numpy()

# Same wrapper functions as in backtest.py's init()
def _ema(arr, length):
    return ta.ema(pd.Series(arr), length=length).to_numpy()

def _rsi(arr, length):
    return ta.rsi(pd.Series(arr), length=length).to_numpy()

def _sma(arr, length):
    return ta.sma(pd.Series(arr), length=length).to_numpy()

def _atr(high_arr, low_arr, close_arr, length):
    return ta.atr(pd.Series(high_arr), pd.Series(low_arr), pd.Series(close_arr), length=length).to_numpy()

def _zscore(arr, length=ZSCORE_PERIOD):
    s = pd.Series(arr)
    sma = ta.sma(s, length=length)
    std = ta.stdev(s, length=length)
    std = std.replace(0, np.nan)
    return ((s - sma) / std).to_numpy()

def _bbu(arr, length=BB_PERIOD, std=BB_STD):
    bb = ta.bbands(pd.Series(arr), length=length, std=std)
    col = next(c for c in bb.columns if c.startswith("BBU_"))
    return bb[col].to_numpy()

def _bbl(arr, length=BB_PERIOD, std=BB_STD):
    bb = ta.bbands(pd.Series(arr), length=length, std=std)
    col = next(c for c in bb.columns if c.startswith("BBL_"))
    return bb[col].to_numpy()

# Compute all indicators over the full dataset (simulating what self.I() produces)
print("Computing indicators...")
ema9 = _ema(close, EMA_FAST)
ema21 = _ema(close, EMA_SLOW)
rsi = _rsi(close, RSI_PERIOD)
zscore = _zscore(close, ZSCORE_PERIOD)
bb_upper = _bbu(close, BB_PERIOD, BB_STD)
bb_lower = _bbl(close, BB_PERIOD, BB_STD)
vol_sma = _sma(vol, VOLUME_PERIOD)
atr = _atr(high, low, close, ATR_PERIOD)

print(f"ema9: {ema9.shape}, NaN={np.isnan(ema9).sum()}, last={ema9[-1]:.2f}")
print(f"ema21: {ema21.shape}, NaN={np.isnan(ema21).sum()}, last={ema21[-1]:.2f}")
print(f"rsi: {rsi.shape}, NaN={np.isnan(rsi).sum()}, last={rsi[-1]:.2f}")
print(f"zscore: {zscore.shape}, NaN={np.isnan(zscore).sum()}, last={zscore[-1]:.3f}")
print(f"bb_upper: {bb_upper.shape}, NaN={np.isnan(bb_upper).sum()}, last={bb_upper[-1]:.2f}")
print(f"bb_lower: {bb_lower.shape}, NaN={np.isnan(bb_lower).sum()}, last={bb_lower[-1]:.2f}")
print(f"vol_sma: {vol_sma.shape}, NaN={np.isnan(vol_sma).sum()}, last={vol_sma[-1]:.2f}")
print(f"atr: {atr.shape}, NaN={np.isnan(atr).sum()}, last={atr[-1]:.2f}")

# Now iterate like next() does
min_bars = max(EMA_SLOW, ATR_PERIOD, ZSCORE_PERIOD, BB_PERIOD, VOLUME_PERIOD)
print(f"\nmin_bars = {min_bars}, starting iteration from bar {min_bars + 1}")

nan_skips = 0
trades_long = 0
trades_short = 0
single_signals = {"A_LONG": 0, "A_SHORT": 0, "B_LONG": 0, "B_SHORT": 0, "C_LONG": 0, "C_SHORT": 0}

for i in range(min_bars + 2, len(close)):
    # Skip if any indicator is NaN (same as next())
    vals = [ema9[i], ema21[i], ema9[i-1], ema21[i-1], rsi[i], zscore[i],
            vol[i], vol_sma[i], bb_upper[i], bb_lower[i], atr[i]]
    if any(np.isnan(v) for v in vals):
        nan_skips += 1
        continue

    sig_a = signal_a_trend(ema9[i], ema21[i], ema9[i-1], ema21[i-1], rsi[i])
    sig_b = signal_b_mean_reversion(zscore[i])
    sig_c = signal_c_volume_breakout(close[i], vol[i], vol_sma[i], bb_upper[i], bb_lower[i])

    # Track individual signals
    if sig_a == 1: single_signals["A_LONG"] += 1
    if sig_a == -1: single_signals["A_SHORT"] += 1
    if sig_b == 1: single_signals["B_LONG"] += 1
    if sig_b == -1: single_signals["B_SHORT"] += 1
    if sig_c == 1: single_signals["C_LONG"] += 1
    if sig_c == -1: single_signals["C_SHORT"] += 1

    action, total = voting_system(sig_a, sig_b, sig_c)
    if action == "LONG":
        trades_long += 1
        if trades_long <= 3:
            print(f"  LONG at bar {i} ({df.index[i]}): A={sig_a} B={sig_b} C={sig_c} total={total} price={close[i]:.2f}")
    elif action == "SHORT":
        trades_short += 1
        if trades_short <= 3:
            print(f"  SHORT at bar {i} ({df.index[i]}): A={sig_a} B={sig_b} C={sig_c} total={total} price={close[i]:.2f}")

print(f"\nNaN skips: {nan_skips}")
print(f"Individual signals: {single_signals}")
print(f"TOTAL: LONG={trades_long}, SHORT={trades_short}")
print(f"Total bars checked: {len(close) - min_bars - 2}")
print("Done.")
