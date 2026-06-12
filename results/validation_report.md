# Optimization Validation Report
Generated: 2026-06-12T19:19:48.514285+00:00

| Symbol | PF Train | PF Test | Win Rate | Trades | Sharpe | Max DD | Accepted |
|--------|----------|---------|----------|--------|--------|--------|----------|
| BTC/USDT | 0.8446 | 0.9269 | 28.7% | 108 | -0.18 | -7.9% | ❌ |

## BTC/USDT — Best Parameters
```json
{
    "ema_fast": 9,
    "ema_slow": 21,
    "zscore_threshold": 3.0,
    "volume_multiplier": 2.0,
    "atr_stop_mult": 1.5,
    "atr_tp_mult": 3.5
}
```

### Acceptance Checks
- ❌ pf >= 1.2: 0.9269
- ✅ pf_test >= 0.8 * pf_train: 0.9269
- ❌ win_rate >= 40%: 28.7037
- ❌ trades >= 200: 108.0000
