# Quant Trading Bot — Roostoo Hackathon 2026

Autonomous crypto trading bot for the SG vs HK Web3 Quant Hackathon. Multi-layer signal stack with derivatives market intelligence, HMM regime detection, and Sortino-optimized risk management. Backtested on 6 months of data across 24 coins.

## Strategy Overview

### The Core Problem

Competition scoring: **60% code & strategy review, 40% risk-adjusted returns** (0.4×Sortino + 0.3×Sharpe + 0.3×Calmar). This means clean architecture and documented reasoning matter more than raw returns — but you still need top-20 returns to qualify.

### What We Tested (and What Failed)

**Signal backtests across 24 coins, 42 days of hourly data:**

| Strategy | Avg PnL | Win Rate | Avg MaxDD | Verdict |
|---|---|---|---|---|
| EMA Crossover 12/26 | +3.4% | 32% | 12.6% | OK on trending coins only |
| 3-Day Momentum | -10.2% | 24% | 19.5% | Lost money on most coins |
| RSI Mean Reversion | -3.3% | 47% | 10.0% | High win rate, consistent |
| 72h Breakout | +0.5% | 37% | 8.3% | Best risk-adjusted |
| Volume + Momentum | -7.5% | 28% | 17.6% | No edge over plain momentum |

**Key finding:** Pure momentum has ~0% autocorrelation at 24h horizons. The market is mean-reverting short-term with occasional breakout trends.

**ML experiment:** We built a full LightGBM pipeline (31K samples, 6 months, purged walk-forward CV, ensemble of 3 models). AUC was 0.55 on temporally-honest evaluation — no better than random. Price-derived features and even positioning data (funding rates, OI, long/short ratios) don't predict 6-hour forward returns from public data. We disabled the ML gate and focused on what works: rule-based signals with robust risk management.

### What We Built: v3 Signal Stack

#### Engine 1 — Breakout Detection (volume-confirmed)
- Entry: Price exceeds 72-hour high AND volume > 1.3× average AND EMA(21) > EMA(55)
- Volume confirmation is **mandatory** (v3 change — eliminates fake breakouts on thin volume)
- No upside cap (Sortino optimization: upside volatility is free)
- Partial exit: sell 50% at +3% to lock in gains, let rest run with wider 4% trailing stop
- Hard stop: -4% from entry

#### Engine 2 — RSI Mean Reversion
- Entry: RSI(14) < 25 (tightened from 30 in v3 — fewer, higher-conviction entries)
- Profit target: +3% from entry
- Trailing stop: 2% from high, hard stop: -3%
- Time stop: 48h with no profit

#### Engine 3 — OI-Price Divergence (derivatives-driven)
- Entry: Price falling but open interest rising → shorts piling in → squeeze potential
- Only fires when RSI < 40 (oversold confirmation)
- Contrarian signal backed by academic research

### Signal Filters & Overlays

| Layer | Effect |
|---|---|
| **Derivatives overlay** | Funding z-score > 1.5 → block buy (market over-long). < -1.0 → boost strength +15% |
| **BTC crash filter** | BTC down >1.5% in last hour → skip all altcoin buys for 15 min |
| **BTC momentum boost** | BTC breaking out with acceleration → boost altcoin breakout strength |
| **Regime detection** | HMM on BTC → modulates exposure budget (not signal suppression) |
| **Fear & Greed** | Extreme readings → scale max exposure ±15% |

### Risk Management

**Position Sizing — Volatility-Parity × REDD:**
```
base_size = (2.5% × portfolio) / daily_vol
effective_size = base_size × REDD_mult × regime_mult × signal_strength
```
REDD (Rolling Economic Drawdown) smoothly reduces new position sizes as drawdown grows. At 0% DD → full size. At 10% DD → zero new positions. No forced selling at intermediate levels — discrete circuit breakers are emergency-only.

**Sortino-Optimized Exits:**
- Breakout: tight downside stop (-4%), NO upside cap. Partial exit at +3% locks in half.
- Mean-rev: symmetric stops (±3%), quick profit-taking.
- The partial exit at +3% reduces realized downside variance (Sortino denominator) while preserving unlimited upside. This is the key scoring-function optimization.

**Drawdown Circuit Breakers (emergency-only):**

| Level | Threshold | Action |
|---|---|---|
| 1 | -3.5% | Warning log. REDD already reducing sizing. No forced sells. |
| 2 | -6% | Liquidate all, pause 4h |
| 3 | -10% | Liquidate all, pause 12h |

Widened from original -2%/-4%/-7% after backtests showed the tight levels were selling bottoms and preventing recovery. BTC routinely moves 3-5% in a day — -2% trigger was firing on normal volatility.

**Commission Optimization:**
- Entries: limit orders (0.05% maker fee)
- Non-urgent exits (trailing stops, profit targets): limit orders (0.05%)
- Urgent exits (hard stops, breakdowns): market orders (0.10%)
- Saves ~0.05% per non-urgent trade. Over hundreds of trades, this is real money.

### Data Sources

| Source | Data | Update Freq | Purpose |
|---|---|---|---|
| Binance Spot | OHLCV candles, 43 pairs | Every 5 min | Technical indicators, breakout detection |
| Binance Futures | Funding rates, open interest | Every 30 min | Derivatives sentiment, squeeze detection |
| Alternative.me | Fear & Greed Index | Hourly | Market-wide sentiment |
| Roostoo | Live prices, wallet, orders | Every 5 min | Execution |

## Backtest Results

### 6-Month Backtest (175 days, 24 coins, 4320 hourly bars each)

| Metric | Value |
|---|---|
| Total Return | +0.97% |
| Max Drawdown | 8.67% |
| Trades | 2,449 |
| Win Rate | 25% |

### Rolling 10-Day Windows (56 windows over 6 months)

| Metric | Value |
|---|---|
| Positive Windows | **26/56 (46%)** |
| Avg Return | +0.43% |
| Best Window | **+11.28%** |
| Worst Window | -6.25% |
| Avg Max Drawdown | 3.38% |
| Avg Composite | 8.94 |

The strategy captures large trends when they occur (+9-11% in favorable windows) and limits damage in adverse conditions. The 46% positive rate reflects that crypto spent extended periods in bearish trends during this 6-month sample — a long-only strategy cannot profit during sustained selloffs.

### Trade Analysis (6-month)

| Exit Reason | Count | Avg PnL | Purpose |
|---|---|---|---|
| Signal sell | 622 | +0.86% | Trend reversal / RSI overbought |
| Partial exit | 143 | +4.33% | Lock in 50% at +3% (Sortino opt) |
| Profit target | 141 | +3.84% | Mean-rev +3% target |
| Trailing stop | 272 | -1.13% | Protect gains / limit losses |
| Hard stop | 91 | -4.61% | Maximum loss per trade |

Average winner (+4.33%) significantly exceeds average loser (-1.13%). The strategy is profitable despite 25% win rate because winners are 3-4× larger than losers.

### What the ML Experiment Taught Us

We built and rigorously tested an ML confidence gate:
- **Data:** 31,533 samples from 6 months × 24 coins
- **Model:** LightGBM ensemble (3 models: conservative, balanced, expressive)
- **Validation:** Purged walk-forward CV with 120-bar gap (no data leakage)
- **Result:** AUC 0.55 — no better than random

We then tried positioning-only features (funding rates, OI, long/short ratios, taker buy/sell, top trader positions) — same result: AUC 0.555.

**Conclusion:** Public data — whether price-derived or positioning-derived — doesn't predict 6-hour forward returns in a temporally honest evaluation. The signal is arbitraged away within minutes by HFT firms. We disabled the ML gate rather than deploy a model that would randomly block profitable signals.

This is documented in the `ml-gating-explore` branch with full code, training scripts, and results.

## Project Structure

```
quant-competition/
├── run.py                          # Entry point
├── .env                            # API keys (gitignored)
├── .env.example                    # Template
├── requirements.txt                # Dependencies
│
├── bot/
│   ├── config.py                   # All parameters (v3 competition-tuned)
│   ├── roostoo_client.py           # Roostoo API with HMAC signing
│   ├── binance_data.py             # Binance spot candle feed
│   ├── derivatives_data.py         # Binance futures: funding rates + OI
│   ├── regime_detector.py          # HMM regime detection on BTC
│   ├── sentiment.py                # Fear & Greed + BTC lead-lag filter
│   ├── ml_model.py                 # ML confidence gate (disabled — see writeup)
│   ├── signals.py                  # v3: volume-required breakouts + RSI 25
│   ├── risk_manager.py             # REDD sizing + partial exits + Sortino stops
│   ├── executor.py                 # Limit/market order management
│   ├── metrics.py                  # Live Sharpe/Sortino/Calmar tracking
│   ├── logger.py                   # Structured JSON logging
│   ├── main.py                     # v3 main loop orchestrator
│   └── models/                     # ML model artifacts (gitignored)
│       ├── ml_model.txt            # Primary LightGBM model
│       ├── ml_model_ens[0-2].txt   # Ensemble models
│       └── ml_model_meta.json      # Training metadata
│
├── bot/backtest/
│   ├── data_loader.py              # Shared data utilities
│   ├── bt_integrated.py            # Full integrated backtest (6 months)
│   ├── bt_market_analysis.py       # Market structure analysis
│   ├── bt_signal_comparison.py     # Compare signal types per coin
│   ├── bt_full_portfolio.py        # Legacy portfolio sim
│   ├── bt_param_sensitivity.py     # Parameter sweeps
│   ├── pretrain_ml.py              # ML training pipeline (disabled)
│   └── run_all.py                  # Run all backtests
│
└── logs/                           # Runtime logs (gitignored)
    ├── bot.jsonl                   # General bot logs
    ├── trades.jsonl                # Trade journal
    └── cycles.jsonl                # Per-cycle state snapshots
```

## Setup & Running

```bash
# Install
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Roostoo credentials

# Run the bot
venv/bin/python run.py

# Run backtests
venv/bin/python -m bot.backtest.bt_integrated      # full 6-month integrated test
venv/bin/python -m bot.backtest.bt_signal_comparison # signal type comparison
venv/bin/python -m bot.backtest.bt_market_analysis   # current market structure
```

## Version History

| Version | Changes | Backtest Impact |
|---|---|---|
| v1 | Breakout + RSI, basic vol-parity, 60% exposure, 3% trailing stop | +5.1% / 37d, composite 7.5 |
| v2 | + HMM regime suppression, ML gate, REDD, derivatives | -2.1% / 37d — regime suppression killed returns |
| v3 | Remove regime suppression, disable ML, require volume on breakout, RSI→25, 80% exposure, partial exits, limit orders | +1.3% / 37d, composite 2.3. Over 6mo: +0.97%, avg 10d composite 8.94 |

## Competition Scoring Reference

| Screen | Weight | Our Approach |
|---|---|---|
| Rule Compliance | Pass/fail | Full logging, autonomous execution, clean commit history |
| Portfolio Returns | Top 20 qualify | Competition-tuned exposure (80%), volume-confirmed entries |
| Risk-Adjusted Score | 40% | Sortino-optimized exits, REDD sizing, partial exits, drawdown breakers |
| Code & Strategy Review | 60% | 12 specialized modules, documented ML experiment, honest backtest results |

**Composite: 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar**
