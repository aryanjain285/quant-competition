# Quant Trading Bot — Roostoo Hackathon 2026 Finals

Autonomous crypto trading bot for the SG vs HK Web3 Quant Hackathon finals. Simplified 6-step pipeline: PCA-HMM regime detection, LassoCV cross-sectional ranking, model-derived entry gate, and unified Sortino-optimized risk management. All data on 1-hour candles.

## Architecture

```
Every hour:

  1. REGIME    — PCA on 43 coins → top-3 PCs → 3-state HMM
                 LOW_VOL (1.0×) | MID_VOL (0.7×) | HI_VOL (sit out)

  2. FEATURES  — 12 features per coin, z-scored cross-sectionally
                 Momentum: r_6h, r_24h, r_3d
                 Quality:  persistence, choppiness
                 Risk:     realized_vol, downside_vol, jump_proxy
                 Breakout: breakout_distance, volume_ratio
                 Reversal: overshoot
                 Cost:     spread_pct

  3. RANK      — LassoCV predicts 24h forward return for each coin
                 Retrained every 6h on 400h rolling window
                 Non-overlapping targets (24h sampling)

  4. GATE      — predicted return > 30 bps (covers round-trip commission)
                 Single data-driven threshold replaces 7-condition filter

  5. SIZE      — vol-parity × REDD drawdown scaling × regime mult
                 Top-ranked gets 1.3×, bottom 0.8×

  6. EXIT      — Unified stops (no strategy branching):
                 Hard stop: -3.5% | Partial exit: 50% at +3%
                 Trail: 3.5% (4.5% after partial) | Time: 60h at <1%
                 Breakdown: price below 24h low → immediate market sell
```

## What Changed from Round 1 (v5 → v6)

| Component | v5 (Round 1) | v6 (Finals) | Why |
|---|---|---|---|
| Entry logic | 7-condition conjunction filter | Lasso gate (>30 bps) | Data-driven, no magic thresholds |
| ML model | RidgeCV | LassoCV | L1 sparsity zeroes noise features |
| Training | 6h overlapping samples | 24h non-overlapping | Honest sample count, no inflation |
| HMM input | PC1 only | Top-3 PCs | Richer market structure |
| Exposure | LOW_VOL=0.5× | LOW_VOL=1.0× | Stop double-penalizing low vol |
| Stops | Strategy-specific (mean_rev vs breakout) | Unified | Simpler, no classification dependency |
| Features | 16 including r_1h | 12 (r_1h dropped) | 1h return is pure noise |
| Signal engine | continuation + reversal labels | Lasso score only | Labels were leaking into stop logic |

## Key Design Decisions (with justification)

**Why Lasso over Ridge?** With 12 features and a 10-day evaluation window, overfitting to noise features hurts more than missing weak signal. Lasso's L1 penalty zeros irrelevant features. After training, we log exactly which features survived — directly explainable for code review (30% of score).

**Why 30 bps entry threshold?** Round-trip commission = 10-20 bps (0.05% maker + 0.10% taker). A 30 bps predicted return covers commission + slippage buffer. Below this, expected profit doesn't justify the trade.

**Why 3.5% hard stop?** BTC hourly vol ≈ 0.4 annualized → daily vol ≈ 2.1%. A 3.5% stop = 1.7σ daily move. Gives room for normal noise but caps losses before they compound. Directly reduces the Sortino denominator.

**Why partial exit at +3%?** Locks in profit on half the position, creating asymmetric payoff (capped downside, open upside on remainder). Reduces variance of trade outcomes → better Sharpe and Sortino.

**Why 24h non-overlapping training samples?** With 6h sampling and 24h forward horizon, consecutive samples share 75% of their target window. This inflates effective sample count by 4×, masking overfitting. 24h sampling gives ~690 genuine samples for 12 features (58:1 ratio — healthy for Lasso).

**Why HMM on 3 PCs instead of 1?** PC1 alone captures ~65% of variance but misses sector rotations and dispersion changes. 3 PCs with 3-state HMM = ~35 parameters, stable with 1000+ observations. State mapping uses trace(covariance) to rank states by volatility — no manual label assignment.

## Backtest Infrastructure

High-fidelity engine (`bot/backtest/engine.py`) that mirrors `main.py` exactly:
- Same modules: features, zscore, Lasso, regime, risk manager
- `SimExchange` with per-pair spreads measured from real Roostoo API
- Correct fees: limit = 0.05% maker, market = 0.10% taker
- Pending order simulation with timeout and stale-sell resubmission
- Time monkeypatch so risk_manager holding hours work in simulation

```bash
venv/bin/python -m bot.backtest.run_backtest              # 4 months, 10-day windows
venv/bin/python -m bot.backtest.run_backtest --months 6   # 6 months
```

## Project Structure

```
bot/
├── main.py              # 6-step pipeline orchestrator
├── config.py            # All parameters with justifications
├── features.py          # 12 cross-sectional features + z-scoring
├── ml.py                # LassoCV trainer (24h non-overlapping targets)
├── ranking.py           # Lasso-based ranking + 30 bps gate
├── regime_detector.py   # PCA (3 PCs) → HMM (3 states)
├── risk_manager.py      # Unified stops, REDD, vol-parity sizing
├── executor.py          # Limit/market orders, pending tracking
├── binance_data.py      # 1h candle feed
├── roostoo_client.py    # Roostoo API with HMAC signing
├── metrics.py           # Live Sharpe/Sortino/Calmar
├── logger.py            # Structured JSON logging
├── signals.py           # DEPRECATED (v4 continuation/reversal — replaced by Lasso gate)
└── backtest/
    ├── engine.py         # High-fidelity backtest (same modules as live)
    ├── sim_exchange.py   # Simulated exchange (real spreads, correct fees)
    └── run_backtest.py   # Rolling window runner
```

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env     # add Roostoo API keys
venv/bin/python run.py   # start the bot
```

## Competition Scoring

| Screen | Weight | Our Approach |
|---|---|---|
| Rule Compliance | Pass/fail | Full JSON logging, autonomous execution, clean commits |
| Portfolio Returns | Top 20 | Lasso gate ensures only positive-EV trades. Regime sits out crises. |
| Risk-Adjusted | 40% | Unified stops + partial exits + REDD → optimized Sortino/Calmar |
| Code & Strategy | 60% | Clean 6-step pipeline, documented Lasso experiment, every parameter justified |

**Composite: 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar**
