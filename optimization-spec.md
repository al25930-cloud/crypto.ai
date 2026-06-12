# Crypto Trading Bot — Optimization Specification

**Version:** 1.0
**Date:** 2026-06-12
**Status:** Ready for Implementation

---

## 1. Overview

Optimize the existing 3-signal voting strategy (trend + mean-reversion + volume breakout) by:
- Refactoring `backtest.py` into a callable function accepting date ranges and parameter overrides
- Creating a centralized `config.py` for all tunable parameters
- Running Optuna (Bayesian optimization) on 70% of historical data
- Validating the best parameters on the held-out 30%
- Deploying validated parameters to `live_signal.py`

**Target:** Profit factor ≥ 1.3 (current: 0.85) with win rate ≥ 40% on out-of-sample data.

---

## 2. New File Structure

```
crypto_bot/
├── config.py               # NEW — central parameter store
├── optimize.py             # NEW — optimization runner
├── signals.py              # MODIFIED — import from config.py, dynamic columns
├── backtest.py             # MODIFIED — refactored run_backtest(), CryptoStrategy with params
├── live_signal.py          # MODIFIED — read params from config.py
├── requirements.txt        # MODIFIED — add optuna
├── .env
├── .gitignore
├── results/                # NEW — logs, study files, reports
│   ├── optuna_btc.pkl
│   ├── optuna_eth.pkl
│   ├── trials_btc.csv
│   ├── trials_eth.csv
│   └── validation_report.md
├── data/
├── logs/
└── last_signal.json
```

---

## 3. Phase 0 — Data & Environment Validation

### 3.1 Data Quality Checks

Run once before optimization, in `optimize.py`:

1. Load data via existing `load_or_fetch_data()` (uses CSV cache; `--redownload` to refresh).
2. Verify columns: `timestamp, open, high, low, close, volume`.
3. Drop rows with NaN in any OHLCV column.
4. Drop duplicate timestamps (already handled by existing code).
5. **Gap detection:** Check for missing hourly candles. If gaps > 2 consecutive hours, log a warning (`WARNING: 5 consecutive hours missing at 2024-03-15T12:00`). Do NOT forward-fill — backtesting.py naturally skips missing timestamps. Forward-filling creates artificial flat candles that bias indicators.
6. Confirm timezone is UTC on all timestamps.
7. Verify data span covers at least 12 months.

### 3.2 Commission & Slippage

Keep existing values in `config.py`:
```python
COMMISSION = 0.0009  # 0.09% (0.04% Binance taker + 0.05% slippage)
POSITION_SIZE = 0.5  # 50% of equity per trade (margin buffer)
INITIAL_CAPITAL = 500_000
```

Do NOT optimize these.

### 3.3 Output

- Verified dataset info logged (rows, date range, gap warnings).
- `results/` folder created if not present.

---

## 4. Phase 1 — Refactored `config.py` & `run_backtest()`

### 4.1 `config.py` — Central Parameter Store

```python
# Strategy parameters (tunable)
VOTING_THRESHOLD = 1       # >= this value → LONG; <= -this → SHORT
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ZSCORE_PERIOD = 20
ZSCORE_THRESHOLD = 2.0
BB_PERIOD = 20
BB_STD = 2.0
VOLUME_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5
ATR_TP_MULT = 2.5

# Fixed (not tunable)
COMMISSION = 0.0009
POSITION_SIZE = 0.5
INITIAL_CAPITAL = 500_000
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1h"
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

# Optimization (not used during normal operation)
OPTIMIZE_PARAMS = [
    "voting_threshold",
    "ema_fast",
    "ema_slow",
    "zscore_threshold",
    "volume_multiplier",
    "atr_stop_mult",
    "atr_tp_mult",
]
```

`signals.py` imports from `config.py` instead of defining its own constants.
`backtest.py` imports from `config.py`.
`live_signal.py` imports from `config.py`.

### 4.2 `CryptoStrategy` — Parameter Override via Constructor

The Strategy class accepts an optional `params` dict. Instance attributes override `config.py` defaults:

```python
class CryptoStrategy(Strategy):
    def __init__(self, broker, data, params=None):
        super().__init__(broker, data)
        self.ema_fast = params.get("ema_fast", EMA_FAST) if params else EMA_FAST
        self.ema_slow = params.get("ema_slow", EMA_SLOW) if params else EMA_SLOW
        # ... all 13 parameters
```

`init()` and `next()` use `self.ema_fast`, `self.ema_slow`, etc. instead of module constants.

**Dynamic indicator column names:** `init()` builds column names dynamically:
```python
self.ema9 = self.I(_ema, self.data.Close, length=self.ema_fast)
self.ema21 = self.I(_ema, self.data.Close, length=self.ema_slow)
```
For `live_signal.py`'s `compute_indicators()`, column names use f-strings:
```python
df[f"EMA_{EMA_FAST}"] = ta.ema(df["close"], length=EMA_FAST)
df[f"EMA_{EMA_SLOW}"] = ta.ema(df["close"], length=EMA_SLOW)
df[f"RSI_{RSI_PERIOD}"] = ta.rsi(df["close"], length=RSI_PERIOD)
```

### 4.3 Refactored `run_backtest()` Signature

```python
def run_backtest(
    symbol: str,
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    params: dict | None = None,
) -> dict:
    """
    Args:
        symbol: Trading pair name.
        df: Full OHLCV DataFrame with DatetimeIndex.
        start_date, end_date: Subset range. If None, uses full df range.
        params: Dict of strategy parameter overrides. If None, uses config.py defaults.

    Returns:
        Dict with keys: profit_factor, win_rate, total_trades, sharpe_ratio,
        max_drawdown, return_pct, final_equity, _trades (DataFrame).
    """
```

**Key behaviors:**
- Filters `df` to `[start_date, end_date]` before backtest.
- Passes `params` to `CryptoStrategy`.
- When called without params/date range, behaves identically to current `run_backtest()`.
- Existing CLI (`python backtest.py --symbol BTC`) still works — `main()` calls `run_backtest()` with defaults.
- Prints metrics as before; also returns the stats dict for programmatic use.

---

## 5. Phase 2 — Parameter Search Space

### 5.1 Tunable Parameters (7 total)

Only these 7 are optimized. Period lengths (RSI_PERIOD, BB_PERIOD, etc.) are kept at their standard defaults.

| Parameter | config.py key | Values | Notes |
|---|---|---|---|
| Voting threshold | `voting_threshold` | `[1, 2]` | 1 = any signal triggers; 2 = majority |
| EMA fast | `ema_fast` | `[5, 9, 12]` | |
| EMA slow | `ema_slow` | `[20, 21, 26]` | Must remain > `ema_fast` |
| Z-score threshold | `zscore_threshold` | `[1.5, 2.0, 2.5]` | |
| Volume multiplier | `volume_multiplier` | `[1.5, 2.0, 2.5]` | |
| ATR stop multiplier | `atr_stop_mult` | `[1.5, 2.0, 2.5]` | |
| ATR take-profit multiplier | `atr_tp_mult` | `[2.5, 3.0, 4.0]` | Must remain > `atr_stop_mult` |

### 5.2 Constraint

`ema_slow > ema_fast` and `atr_tp_mult > atr_stop_mult` must hold. If a trial violates these, return a penalty score (profit_factor = 0) rather than crashing.

### 5.3 Fixed Parameters (NOT optimized)

- `RSI_PERIOD = 14`
- `ZSCORE_PERIOD = 20`
- `BB_PERIOD = 20`
- `BB_STD = 2.0`
- `VOLUME_PERIOD = 20`
- `ATR_PERIOD = 14`
- `COMMISSION = 0.0009`
- `POSITION_SIZE = 0.5`
- `INITIAL_CAPITAL = 500_000`

---

## 6. Phase 3 — Optuna Optimization (Training Set)

### 6.1 Train/Test Split

- **Split:** 70% train, 30% test (chronological — no shuffle).
- **Split index:** `split_idx = int(0.7 * len(data))`
- `train_data = data.iloc[:split_idx]`
- `test_data = data.iloc[split_idx:]`

### 6.2 Optuna Setup

```python
import optuna

def objective(trial):
    params = {
        "voting_threshold": trial.suggest_categorical("voting_threshold", [1, 2]),
        "ema_fast": trial.suggest_categorical("ema_fast", [5, 9, 12]),
        "ema_slow": trial.suggest_categorical("ema_slow", [20, 21, 26]),
        "zscore_threshold": trial.suggest_categorical("zscore_threshold", [1.5, 2.0, 2.5]),
        "volume_multiplier": trial.suggest_categorical("volume_multiplier", [1.5, 2.0, 2.5]),
        "atr_stop_mult": trial.suggest_categorical("atr_stop_mult", [1.5, 2.0, 2.5]),
        "atr_tp_mult": trial.suggest_categorical("atr_tp_mult", [2.5, 3.0, 4.0]),
    }

    # Constraint: ema_slow must be > ema_fast
    if params["ema_slow"] <= params["ema_fast"]:
        return 0.0  # penalty

    # Constraint: atr_tp_mult must be > atr_stop_mult
    if params["atr_tp_mult"] <= params["atr_stop_mult"]:
        return 0.0

    stats = run_backtest(
        symbol=symbol,
        df=train_data,
        start_date=str(train_data.index[0].date()),
        end_date=str(train_data.index[-1].date()),
        params=params,
    )

    # Log Sharpe ratio as user attribute for later analysis
    trial.set_user_attr("sharpe_ratio", stats["sharpe_ratio"])
    trial.set_user_attr("win_rate", stats["win_rate"])
    trial.set_user_attr("total_trades", stats["total_trades"])

    return stats["profit_factor"]
```

### 6.3 Execution

```python
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=200, show_progress_bar=True)
```

- **200 trials** per symbol. TPE sampler converges efficiently — 200 covers the most relevant regions of the 2,187-combination space.
- If validation fails, re-run with 400 trials before trying fallback.
- Run separately for BTC/USDT and ETH/USDT (different `study` objects).

### 6.4 Output

- `results/optuna_btc.pkl` — full Optuna study (BTC).
- `results/optuna_eth.pkl` — full Optuna study (ETH).
- `results/trials_btc.csv` — all trial results exported as CSV.
- `results/trials_eth.csv` — same for ETH.
- Console output: best params, best profit factor, number of completed trials.

---

## 7. Phase 4 — Validation on Test Set

### 7.1 Procedure

1. Take `best_params` from Phase 3.
2. Run `run_backtest()` on `test_data` (last 30%) with those params.
3. Obtain test metrics: `pf_test`, `win_rate_test`, `total_trades_test`, `sharpe_test`.

### 7.2 Acceptance Criteria

| Metric | Threshold |
|---|---|
| Profit factor (test) | `pf_test ≥ 1.2` |
| Profit factor consistency | `pf_test ≥ 0.8 × pf_train` |
| Win rate | `win_rate_test ≥ 40%` |
| Trade count | `total_trades_test ≥ 200` (statistically meaningful) |

All four must pass. If any fails, proceed to fallback.

### 7.3 Fallback

**Step 1 — Reduce parameter space (as in original plan):**
- Fix `zscore_threshold = 2.0`, `volume_multiplier = 2.0`, `atr_tp_mult = 3.0`.
- Optimize only `voting_threshold`, `ema_fast`, `ema_slow`, `atr_stop_mult` (4 params).
- Run 150 trials.

**Step 2 — If still failing, try 80/20 split:**
- `split_idx = int(0.8 * len(data))`
- Re-run full optimization (7 params, 200 trials) on 80% train data.
- Do NOT look at test results before deciding the split — decide beforehand to avoid data leakage.

**Step 3 — If both fail:**
- Try optimizing for Sharpe ratio instead of profit factor.
- Or check data quality and extend the date range.

### 7.4 Output

- `results/validation_report.md` — comparison table and acceptance status.

---

## 8. Phase 5 — Deploy to Live Signal

### 8.1 Update `config.py`

Replace default parameter values with the validated best parameters from Phase 4.

### 8.2 Verify `live_signal.py`

1. `live_signal.py` already imports constants from `signals.py` (which will now import from `config.py`).
2. Run `python live_signal.py` briefly to confirm:
   - Fetches data without errors.
   - Evaluates signals using new parameters.
   - Discord alerts format correctly.
3. No code changes needed in `live_signal.py` — it reads parameters automatically.

### 8.3 Git Commit

```bash
git add config.py optimize.py signals.py backtest.py results/
git commit -m "feat: optimize strategy parameters via Optuna walk-forward (PF: X.XX)"
```

---

## 9. `optimize.py` — CLI Entry Point

```bash
python optimize.py                    # Full optimization (BTC + ETH)
python optimize.py --symbol BTC/USDT  # Single symbol
python optimize.py --trials 400       # Override trial count
python optimize.py --validate-only    # Skip optimization, just validate current params
python optimize.py --redownload       # Force fresh data download
```

### Flow

1. Parse CLI args.
2. Phase 0: validate data.
3. For each symbol:
   a. Split data 70/30.
   b. Run Optuna on training set (Phase 3).
   c. Validate best params on test set (Phase 4).
   d. If validation passes: print result, save to `results/`.
   e. If validation fails: run fallback sequence.
4. Print summary table of all results.
5. Prompt user to run Phase 5 manually (to avoid auto-deploying).

---

## 10. `signals.py` Modifications

### 10.1 Import from config.py

```python
from config import (
    EMA_FAST, EMA_SLOW, RSI_PERIOD,
    ZSCORE_PERIOD, ZSCORE_THRESHOLD,
    BB_PERIOD, BB_STD,
    VOLUME_PERIOD, VOLUME_MULTIPLIER,
    ATR_PERIOD, ATR_STOP_MULT, ATR_TP_MULT,
    VOTING_THRESHOLD,
)
```

Remove the hardcoded constants currently at the top of `signals.py`.

### 10.2 Dynamic Column Names in `compute_indicators()`

```python
df[f"EMA_{EMA_FAST}"] = ta.ema(df["close"], length=EMA_FAST)
df[f"EMA_{EMA_SLOW}"] = ta.ema(df["close"], length=EMA_SLOW)
df[f"RSI_{RSI_PERIOD}"] = ta.rsi(df["close"], length=RSI_PERIOD)
df[f"SMA_{ZSCORE_PERIOD}"] = ta.sma(df["close"], length=ZSCORE_PERIOD)
df[f"STDEV_{ZSCORE_PERIOD}"] = ta.stdev(df["close"], length=ZSCORE_PERIOD)
df["Z_SCORE"] = (df["close"] - df[f"SMA_{ZSCORE_PERIOD}"]) / df[f"STDEV_{ZSCORE_PERIOD}"].replace(0, np.nan)
df[f"ATR_{ATR_PERIOD}"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)
# ... BB bands, Volume SMA similarly
```

### 10.3 Voting System Uses `VOTING_THRESHOLD`

```python
def voting_system(sig_a, sig_b, sig_c):
    total = sig_a + sig_b + sig_c
    if total >= VOTING_THRESHOLD:
        return ("LONG", total)
    elif total <= -VOTING_THRESHOLD:
        return ("SHORT", total)
    return ("HOLD", total)
```

### 10.4 `evaluate_signal()` in `live_signal.py`

Update column name references to use the dynamic names from `config.py`. Since `compute_indicators()` now produces dynamic column names, `evaluate_signal()` must reference them dynamically:

```python
ema_fast_col = f"EMA_{EMA_FAST}"
ema_slow_col = f"EMA_{EMA_SLOW}"
rsi_col = f"RSI_{RSI_PERIOD}"
atr_col = f"ATR_{ATR_PERIOD}"
# etc.
```

---

## 11. `backtest.py` Modifications

### 11.1 Import from config.py

Replace direct constant imports from `signals.py` with imports from `config.py` for the parameters that `backtest.py` uses directly (COMMISSION, POSITION_SIZE, etc.). Keep signal function imports from `signals.py`.

### 11.2 `CryptoStrategy.__init__()` with Params

```python
class CryptoStrategy(Strategy):
    ema_fast = EMA_FAST       # class defaults from config.py
    ema_slow = EMA_SLOW
    # ... all 13 parameters

    def __init__(self, broker, data, params=None):
        super().__init__(broker, data)
        if params:
            self.ema_fast = params.get("ema_fast", EMA_FAST)
            self.ema_slow = params.get("ema_slow", EMA_SLOW)
            self.rsi_period = params.get("rsi_period", RSI_PERIOD)
            self.zscore_period = params.get("zscore_period", ZSCORE_PERIOD)
            self.zscore_threshold = params.get("zscore_threshold", ZSCORE_THRESHOLD)
            self.bb_period = params.get("bb_period", BB_PERIOD)
            self.bb_std = params.get("bb_std", BB_STD)
            self.volume_period = params.get("volume_period", VOLUME_PERIOD)
            self.volume_multiplier = params.get("volume_multiplier", VOLUME_MULTIPLIER)
            self.atr_period = params.get("atr_period", ATR_PERIOD)
            self.atr_stop_mult = params.get("atr_stop_mult", ATR_STOP_MULT)
            self.atr_tp_mult = params.get("atr_tp_mult", ATR_TP_MULT)
            self.voting_threshold = params.get("voting_threshold", VOTING_THRESHOLD)
```

`init()` and `next()` use `self.ema_fast` etc. instead of the module constants.

### 11.3 Refactored `run_backtest()`

Add `start_date`, `end_date`, `params` parameters. The function:
1. Filters `df` to date range if provided.
2. Passes `params` to `CryptoStrategy`.
3. Returns the stats dict.

### 11.4 Existing CLI Preserved

`main()` calls `run_backtest()` with `start_date=START_DATE`, `end_date=END_DATE`, `params=None`. Behavior unchanged from user perspective.

---

## 12. `live_signal.py` Modifications

Minimal changes — only update column name references in `evaluate_signal()` to use dynamic names from `config.py`.

The existing `compute_indicators()` function (already imported from `signals.py`) will produce dynamic column names, so `evaluate_signal()` must match.

---

## 13. `requirements.txt`

Add:
```
optuna
```

Full list:
```
ccxt
pandas
numpy
pandas_ta
backtesting
requests
python-dotenv
optuna
```

---

## 14. Success Criteria

| Criterion | Target |
|---|---|
| Optuna completes | 200 trials per symbol, no crashes |
| Best training profit factor | No specific target (maximization) |
| Test profit factor | ≥ 1.2 |
| Test profit factor consistency | `pf_test ≥ 0.8 × pf_train` |
| Test win rate | ≥ 40% |
| Test trade count | ≥ 200 |
| `live_signal.py` compatible | Runs without import errors with new params |
| CLI backward compatible | `python backtest.py` still works |

---

## 15. Edge Cases & Error Handling

| Scenario | Handling |
|---|---|
| `ema_slow ≤ ema_fast` in trial | Return `profit_factor = 0.0` (penalty) |
| `atr_tp_mult ≤ atr_stop_mult` in trial | Return `profit_factor = 0.0` (penalty) |
| Data < 500 candles after filtering | Log warning, return empty stats dict |
| Zero trades in a trial | Return `profit_factor = 0.0`, `win_rate = 0.0` |
| Optuna DB locked | Use `storage=None` (in-memory). Save with `joblib.dump()` |
| Validation fails all fallbacks | Log detailed report, suggest manual parameter review |
| Missing `results/` folder | Auto-create on first run |
| `config.py` deleted or corrupted | `signals.py` falls back to hardcoded defaults defined as module-level fallbacks |

---

## 16. Out of Scope

- Optimizing period lengths (RSI_PERIOD, BB_PERIOD, ZSCORE_PERIOD, ATR_PERIOD)
- Multi-timeframe optimization
- Walk-forward with sliding windows (single 70/30 split only)
- Real-time paper trading during optimization
- Genetic algorithms or grid search (Optuna TPE only)
- Portfolio-level optimization (symbols optimized independently)
- Automatic deployment to `live_signal.py` (user manually runs Phase 5)
- Hyperparameter tuning of Optuna itself (TPE defaults)

---

## 17. Key Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Config approach | `config.py` central file | Single source of truth, easy imports, clean history |
| Parameter passing | Strategy constructor `params` dict | Instance attributes, no global mutation, thread-safe |
| Column names | Dynamic f-strings | Correct when EMA/SLOW/RSI periods change |
| Optimized params | 7 only (not periods) | Periods are standard defaults; expanding space risks overfitting |
| Trial count | 200 (bump to 400 if needed) | TPE converges efficiently; 400 as safety margin |
| Per-symbol optimization | Separate for BTC and ETH | Different volatility/trend characteristics |
| Objective | Profit factor (log Sharpe) | Directly targets spec PF ≥ 1.3 criterion |
| File layout | Root-level files, `results/` folder | Clean imports, no deep nesting |
| CLI backward compat | Preserved | `python backtest.py` still works |
| Gap handling | Detect + warn, no forward-fill | Forward-filling biases indicators; skip naturally |
| Fallback sequence | Reduce params → 80/20 split → Sharpe obj | Ordered escalation, no data leakage |
| Train/test ratio | 70/30 (80/20 as fallback) | Standard; 70% provides sufficient training data |
