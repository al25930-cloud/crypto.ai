"""Quick debug: verify pandas_ta wrapper functions work with numpy arrays."""
import pandas as pd
import pandas_ta as ta
import numpy as np

np.random.seed(42)
n = 300
close = np.random.randn(n).cumsum() + 100
high = close + np.abs(np.random.randn(n))
low = close - np.abs(np.random.randn(n))
vol = np.abs(np.random.randn(n) * 1000) + 5000


def _ema(arr, length):
    return ta.ema(pd.Series(arr), length=length).to_numpy()


def _rsi(arr, length):
    return ta.rsi(pd.Series(arr), length=length).to_numpy()


def _sma(arr, length):
    return ta.sma(pd.Series(arr), length=length).to_numpy()


def _atr(h, l, c, length):
    return ta.atr(pd.Series(h), pd.Series(l), pd.Series(c), length=length).to_numpy()


def _zscore(arr, length=20):
    s = pd.Series(arr)
    sma = ta.sma(s, length=length)
    std = ta.stdev(s, length=length)
    std = std.replace(0, np.nan)
    return ((s - sma) / std).to_numpy()


def _bbu(arr, length=20, std=2.0):
    bb = ta.bbands(pd.Series(arr), length=length, std=std)
    col = next(c for c in bb.columns if c.startswith("BBU_"))
    return bb[col].to_numpy()


# Test final values
print("--- Final values (bar 299) ---")
print("ema9:", _ema(close, 9)[-1])
print("ema9 NaN count:", np.isnan(_ema(close, 9)).sum())
print("rsi14:", _rsi(close, 14)[-1])
print("atr14:", _atr(high, low, close, 14)[-1])
print("zscore:", _zscore(close)[-1])
print("bbu:", _bbu(close)[-1])
print("vol_sma:", _sma(vol, 20)[-1])

# Test progressive bars
print("\n--- Progressive bars (warmup check) ---")
for i in [23, 24, 25, 26, 50, 100, 200]:
    e9 = _ema(close[: i + 1], 9)[-1]
    e21 = _ema(close[: i + 1], 21)[-1]
    r = _rsi(close[: i + 1], 14)[-1]
    z = _zscore(close[: i + 1])[-1]
    ok = not any(np.isnan([e9, e21, r, z]))
    print(f"  bar {i}: ema9={e9:.4f} ema21={e21:.4f} rsi={r:.2f} z={z:.3f} ok={ok}")

# Test signal functions
from signals import signal_a_trend, signal_b_mean_reversion, signal_c_volume_breakout, voting_system

print("\n--- Signal tests ---")
full_close = close
full_ema9 = _ema(full_close, 9)
full_ema21 = _ema(full_close, 21)
full_rsi = _rsi(full_close, 14)
full_zscore = _zscore(full_close)
full_bbu = _bbu(full_close, 20, 2.0)
bbl = next(c for c in ta.bbands(pd.Series(full_close), length=20, std=2.0).columns if c.startswith("BBL_"))
full_bbl = ta.bbands(pd.Series(full_close), length=20, std=2.0)[bbl].to_numpy()
full_vol_sma = _sma(vol, 20)

trigger_count = 0
for i in range(50, n):
    sig_a = signal_a_trend(full_ema9[i], full_ema21[i], full_ema9[i - 1], full_ema21[i - 1], full_rsi[i])
    sig_b = signal_b_mean_reversion(full_zscore[i])
    sig_c = signal_c_volume_breakout(full_close[i], vol[i], full_vol_sma[i], full_bbu[i], full_bbl[i])
    action, total = voting_system(sig_a, sig_b, sig_c)
    if action != "HOLD":
        trigger_count += 1
        if trigger_count <= 5:
            print(f"  bar {i}: A={sig_a} B={sig_b} C={sig_c} -> {action} (total={total})")

print(f"\nTotal triggers (LONG/SHORT) after warmup: {trigger_count}/{n - 50}")
print(f"Test complete.")
