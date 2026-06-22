# Trade Frequency Optimization Spec

**Goal:** Achieve ≥1.5 trades/day (≥270 trades over 180-day backtest) while maintaining
strategy quality and statistical significance.

**Problem:** Current system achieves only ~0.5–0.7 trades/day. The bottleneck is NOT entry
signals (we get ~65/day) but **position blocking** — while a trade is open, no new entries
can fire, and trades with extreme SL/TP values can sit open for 2 full days.

---

## Agreed Changes

### 1. MAX_RR = 5.0 ✅ DONE
- **Before:** `MAX_RR = 8.0` (TP could be 24% away with MAX_SL=3.0 — never hits)
- **After:** `MAX_RR = 5.0` (TP max 15% away — achievable in hours)
- **Impact:** Eliminates the most extreme "lottery ticket" strategies that trade once and timeout

### 2. MIN_CONDITIONS_ABSOLUTE = 4 ✅ DONE
- **File:** config.py, conditions.py (get_condition_count_range)
- **Before:** `MIN_CONDITIONS_ABSOLUTE = 3` (but effectively 7 due to 25% of 31 pool)
- **After:** `MIN_CONDITIONS_ABSOLUTE = 4` (hard floor, not percentage-based)
- **Impact:** Allows lean 4-condition strategies that fire entry signals much more frequently.
  With 4 conditions at threshold 0.3, you need 2/4 True = ~68% of candles generate signals.

### 3. MIN_THRESHOLD = 0.3 ✅ DONE
- **File:** config.py
- **Before:** `MIN_THRESHOLD = 0.5` (need 50%+ conditions True to enter)
- **After:** `MIN_THRESHOLD = 0.3` (need 30%+ conditions True to enter)
- **Impact:** Combined with MIN_CONDITIONS=4, dramatically increases entry signal frequency.
  The optimizer can explore aggressive lean strategies alongside conservative fat ones.

### 4. ATR-Based SL/TP ✅ DONE
- **Files:** config.py, strategy.py, backtest.py
- **Before:** Fixed percentage SL (`sl_pct` 0.3%–3.0%). Doesn't adapt to market volatility.
  In quiet markets, 2% SL takes days to hit. In volatile markets, 2% SL gets stopped out instantly.
- **After:** ATR-multiplier SL (`sl_atr_mult` 1.0–3.0 × ATR(14)). Self-adapting:
  - Quiet market (ATR=0.2%): SL at 0.3%–0.6% → exits in minutes
  - Normal market (ATR=0.5%): SL at 0.5%–1.5% → exits in 1-4 hours
  - Volatile market (ATR=1.0%): SL at 1.0%–3.0% → exits in 2-8 hours

#### Implementation Details

**config.py changes:**
```
# Remove:
MIN_SL = 0.3
MAX_SL = 3.0

# Add:
MIN_SL_ATR_MULT = 1.0   # Minimum ATR multiplier for stop loss
MAX_SL_ATR_MULT = 3.0   # Maximum ATR multiplier for stop loss
```

**strategy.py changes:**
- `generate_random_strategy()`: Pick `sl_atr_mult` instead of `sl`:
  ```python
  sl_atr_mult = round(random.uniform(config.MIN_SL_ATR_MULT, config.MAX_SL_ATR_MULT), 2)
  ```
- Strategy dict: `{"sl_atr_mult": 1.8, "rr": 3.0, ...}` instead of `{"sl": 1.5, "rr": 3.0}`

**backtest.py changes:**
- On entry, read the ATR(14) value at the entry candle
- Compute SL/TP using ATR instead of percentage:
  ```python
  atr_value = df["atr_14"].iloc[i]
  if direction == "LONG":
      sl_price = entry_price - (atr_value * sl_atr_mult)
      tp_price = entry_price + (atr_value * sl_atr_mult * rr_ratio)
  else:
      sl_price = entry_price + (atr_value * sl_atr_mult)
      tp_price = entry_price - (atr_value * sl_atr_mult * rr_ratio)
  ```
- ATR column already exists (computed by indicators.py via talib/pandas_ta)
- The `atr_14` column must be passed through to backtest_strategy or read from df

**Expected trade duration with ATR-based SL:**
| ATR multiplier | Typical exit time | Trades/day potential |
|---|---|---|
| 1.0× ATR | 15–60 min | 5–10+ |
| 1.5× ATR | 1–4 hours | 2–4 |
| 2.0× ATR | 2–8 hours | 1–2 |
| 3.0× ATR | 4–16 hours | 0.5–1 |

**Sweet spot for ≥1.5 trades/day:** 1.5×–2.5× ATR multiplier

---

## Rejected Changes

### 5. Frequency Bonus in Score ❌ REJECTED
- **Why:** `rr_per_day` already naturally rewards frequency — more trades = more RR
  accumulated per day. Adding an extra frequency multiplier is redundant and could
  over-bias the optimizer toward quantity over quality.

### 6. Early Signal Reversal ❌ REJECTED
- **Why:** Tested previously — doesn't work. The bot doesn't learn to trade in the
  trend and just keeps flipping signals back and forth, resulting in poor performance.
  The position blocking is better solved by faster exits via ATR-based SL/TP.

---

## Pending Changes

### 7. Reduce MAX_TRADE_DURATION_HOURS 💡 PENDING
- **File:** config.py
- **Before:** `MAX_TRADE_DURATION_HOURS = 48`
- **After:** `MAX_TRADE_DURATION_HOURS = 12` (or 6)
- **Why:** Safety net. Even with ATR-based SL, a trade might sit in a narrow range
  and not hit either SL or TP. 12h is generous — if a trade hasn't moved enough
  in 12h on a 15m chart, it's dead money. Cut it and free the position.
- **Priority:** Low — ATR-based SL/TP (#4) should already resolve most stuck trades
  by adapting SL to actual volatility. This is a fallback only.

---

## Summary: Full Change List

| # | Change | File(s) | Status | Impact |
|---|---|---|---|---|
| 1 | MAX_RR = 5.0 | config.py | ✅ Done | Eliminates extreme TP targets |
| 2 | MIN_CONDITIONS_ABSOLUTE = 4 | config.py, conditions.py | ✅ Done | More entry signals |
| 3 | MIN_THRESHOLD = 0.3 | config.py | ✅ Done | Easier entry trigger |
| 4 | ATR-based SL/TP | config.py, strategy.py, backtest.py, bayesian_optimizer.py, genetic_optimizer.py, live_signal.py | ✅ Done | Self-adapting, faster exits |
| 5 | Frequency bonus in score | — | ❌ Rejected | `rr_per_day` already rewards frequency |
| 6 | Early signal reversal | — | ❌ Rejected | Bot flips signals instead of learning trends |
| 7 | MAX_TRADE_DURATION = 12h | config.py | 💡 Pending | Safety net (low priority) |

**Estimated combined impact:**
- Current: ~0.5 trades/day
- After #1-4: ~1.0–1.5 trades/day
- After #1-4 + #7: ~1.5–2.5 trades/day

---

## Existing Changes Already Applied (From Previous Work)

These were implemented before this spec and are already in the codebase:

1. **MIN_TRADES_PER_DAY = 1.5** (config.py) — Hard disqualification for strategies below threshold
2. **RR/day uses calendar days** (backtest.py) — Fixed inflated metric from dividing by trading-days
3. **Equity curve stripped from all_results** (training.py) — Memory leak fix
4. **File I/O caching in Bayesian optimizer** (bayesian_optimizer.py) — 18x speed improvement
5. **Manual timeout callback** (bayesian_optimizer.py) — Optuna timeout safety net
6. **MIN_VALID_TRADES removed** (config.py, strategy.py) — Replaced by MIN_TRADES_PER_DAY check
7. **LOW_TRADES_PENALTY removed** (config.py, strategy.py) — Now a hard disqualification instead

**Note:** MIN_TRADES_PER_DAY is set to 1.2 as a starting point. Once the ATR-based changes
prove they can achieve higher frequency, this can be increased toward 1.5+.
