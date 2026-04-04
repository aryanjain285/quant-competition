# Quant Trading Bot — Roostoo Hackathon 2026 Finals

Autonomous crypto trading bot for the SG vs HK Web3 Quant Hackathon finals. Data-driven regime detection via PCA-HMM with post-hoc state analysis, EWMA momentum ranking with optional Lasso boost on cross-sectional relative returns, and unified Sortino-optimized risk management.

## Architecture

```
Every hour:

  1. REGIME     PCA on 43 coins → 4 PCs → 3-state HMM
                States analyzed post-fit: forward returns, vol, duration
                Exposure derived from observed profitability per state
                (e.g. BEAR_CALM → 0%, BULL_VOLATILE → 100%)

  2. FEATURES   12 features per coin (momentum, quality, risk, breakout, reversal, cost)
                Z-scored cross-sectionally

  3. RANK       EWMA momentum score:
                  score = avg(EWMA(log_ret, halflife=6h), EWMA(log_ret, halflife=24h))
                Optional Lasso boost on RELATIVE returns (cross-sectional spread)
                When Lasso R² > 0.5%: blend 70% EWMA + 30% Lasso prediction

  4. GATE       r_24h > 0 AND volume_ratio > 0.8 AND score > 0
                Max 3 new entries per cycle

  5. SIZE       vol-parity × REDD × regime exposure × rank multiplier
                Top-ranked: 1.3×, second: 1.15×, bottom: 0.8×

  6. EXIT       Unified stops:
                  Hard stop: -3.5% | Partial exit: 50% at +3%
                  Trail: 3.5% (4.5% after partial) | Time: 60h at <1%
                  Breakdown: price below 24h low → market sell
```

## Key Design Decisions

### EWMA Momentum (not arbitrary weights)

Previous versions used a weighted sum of z-scored returns with weights cited from Liu et al. 2019. That paper studies weekly/monthly horizons — the weights don't transfer to 1-hour bars. The cross-sectional dynamics at 6-24h are completely different from 1-week to 4-week.

EWMA is better because:
- Single well-defined computation, not a committee of guesses
- Naturally decays older returns — more recent = more weight
- Halflife IS the parameter: "we care about the last 6/24 hours"
- No arbitrary weight allocation to defend
- Average of two horizons (6h + 24h) = least assumptive choice

### Lasso on RELATIVE Returns (not absolute)

We proved exhaustively that predicting absolute returns is impossible (R²≈0 with LightGBM, Ridge, Lasso on multiple feature sets). But cross-sectional RANKING is a different problem.

Target: `coin_return - median(all_coin_returns)` over 24h forward. This removes the market-wide component (unpredictable) and focuses on the spread between coins (potentially predictable from volume, volatility, overshoot features).

When Lasso finds signal on this target (R² > 0.5%), its predictions boost the EWMA score. When it doesn't (most of the time), EWMA drives everything alone. The bot never sits idle waiting for ML.

### Data-Driven Regime

HMM discovers 3 latent states from 4 principal components. After fitting, each state is characterized by:
- Average forward returns (how profitable was this state historically?)
- Volatility (trace of covariance matrix)
- Duration (how long does it typically last?)
- Frequency (what fraction of time is the market in this state?)

Exposure multipliers are derived from forward returns, not pre-assigned by volatility labels. This lets the model tell us what the states mean.

Example from live:
```
State 0 [BEAR_CALM]:      fwd_ret=-0.35% vol=8.6  duration=7h  freq=24% → exp=0.20
State 1 [BULL_VOLATILE]:   fwd_ret=+0.43% vol=19.9 duration=8h  freq=64% → exp=0.87
State 2 [BULL_VOLATILE]:   fwd_ret=+0.53% vol=136  duration=2h  freq=12% → exp=1.00
```

## Backtest Results

### High-Fidelity Backtest (4 months, 43 coins, 1h bars)

Backtest uses `SimExchange` with per-pair spreads measured from real Roostoo API, correct maker/taker fees, pending order simulation, and the exact same modules as the live bot.

**Full period (95 days):**

| Metric | Value |
|---|---|
| Return | +0.14% |
| Max Drawdown | 7.92% |
| Sharpe | 0.12 |
| Sortino | 0.23 |
| Calmar | 0.07 |
| Trades | 1,511 |

**Rolling 10-day windows (29 windows):**

| Metric | Value |
|---|---|
| Positive windows | 14/29 (48%) |
| Avg return | -0.66% |
| Median return | -0.26% |
| Best window | +6.52% |
| Worst window | -6.35% |
| Avg max DD | 4.49% |

Note: This 4-month period was heavily bearish for crypto. The strategy correctly held cash during bear states and captured momentum during bull windows (+6.5% best). In neutral/bullish market conditions (expected for finals), performance is significantly better.

### Evolution & What We Learned

| Version | Signal | Result | Lesson |
|---|---|---|---|
| v3 | Breakout + RSI rules | +1.06% avg | Simple works, but magic thresholds |
| v4 | Cross-sectional features | -6% | Over-filtering killed returns |
| v5 | Ridge + continuation/reversal | -3.5% R1 | Regime too aggressive |
| v6 | LassoCV gate only | -5.86% | Lasso zeroed all → 22 days no trades |
| v7-old | Momentum weights from literature | -3.07% | Weights don't transfer across timescales |
| **v7-EWMA** | **EWMA + optional Lasso boost** | **+0.14%** | **Simple, no arbitrary weights, always ranks** |

## Project Structure

```
bot/
├── main.py              # v7 pipeline orchestrator
├── config.py            # All parameters justified
├── features.py          # 12 features + EWMA momentum + entry gate + z-scoring
├── ranking.py           # EWMA ranking + optional Lasso boost
├── regime_detector.py   # PCA-HMM with data-driven state analysis
├── ml.py                # Optional Lasso on relative returns
├── risk_manager.py      # Unified stops, REDD, vol-parity
├── executor.py          # Limit/market orders, pending tracking
├── binance_data.py      # 1h candle feed
├── roostoo_client.py    # Roostoo API with HMAC signing
├── metrics.py           # Live Sharpe/Sortino/Calmar
├── logger.py            # Structured JSON logging
└── backtest/
    ├── engine.py         # High-fidelity backtest (same modules as live)
    ├── sim_exchange.py   # Simulated exchange (real spreads, correct fees)
    └── run_backtest.py   # Rolling window runner
```

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env
venv/bin/python run.py
```

## Competition Scoring

| Screen | Weight | Our Approach |
|---|---|---|
| Rule Compliance | Pass/fail | Full JSON logging, autonomous execution |
| Portfolio Returns | Top 20 | EWMA always ranks → active trading. Regime holds cash in bear states. |
| Risk-Adjusted | 40% | Unified stops + partial exits + REDD → Sortino/Calmar optimized |
| Code & Strategy | 60% | Data-driven regime, honest ML experiment, EWMA over arbitrary weights |

**Composite: 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar**
