"""
ETH/USDT specific parameter overrides.

ETH has higher volatility than BTC — wider stops prevent premature exits,
and stricter thresholds filter more noise.  Imported by backtest.py when
running the ETH/USDT symbol.
"""

ETH_PARAMS = {
    "atr_stop_mult": 2.0,         # wider stop (was 1.5) — ETH volatility needs room
    "zscore_threshold": 3.0,      # ±3.0 sigma (was 2.5) — stronger mean-reversion signals
    "volume_multiplier": 2.5,     # 2.5× avg volume (was 2.0) — filter more volume noise
}
