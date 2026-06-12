# Optimization Validation Report
Generated: 2026-06-12T18:58:18.366242+00:00

| Symbol | PF Train | PF Test | Win Rate | Trades | Sharpe | Max DD | Accepted |
|--------|----------|---------|----------|--------|--------|--------|----------|
| BTC/USDT | 1.2443 | 1.0482 | 47.4% | 19 | 0.06 | -2.4% | ❌ |

## BTC/USDT — Best Parameters
```json
{
    "ema_fast": 12,
    "ema_slow": 26,
    "atr_stop_mult": 2.0,
    "atr_tp_mult": 2.5
}
```

### Acceptance Checks
- ❌ pf >= 1.2: 1.0482
- ✅ pf_test >= 0.8 * pf_train: 1.0482
- ✅ win_rate >= 40%: 47.3684
- ❌ trades >= 200: 19.0000
