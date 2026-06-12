# Optimization Validation Report
Generated: 2026-06-12T18:45:21.229300+00:00

| Symbol | PF Train | PF Test | Win Rate | Trades | Sharpe | Max DD | Accepted |
|--------|----------|---------|----------|--------|--------|--------|----------|
| BTC/USDT | 0.8424 | 0.7690 | 43.9% | 312 | -2.74 | -16.4% | ❌ |

## BTC/USDT — Best Parameters
```json
{
    "voting_threshold": 2,
    "ema_fast": 12,
    "ema_slow": 26,
    "zscore_threshold": 2.5,
    "volume_multiplier": 2.0,
    "atr_stop_mult": 2.0,
    "atr_tp_mult": 3.0
}
```

### Acceptance Checks
- ❌ pf >= 1.2: 0.7690
- ✅ pf_test >= 0.8 * pf_train: 0.7690
- ✅ win_rate >= 40%: 43.9103
- ✅ trades >= 200: 312.0000
