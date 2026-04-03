# Quant Trading Bot — Roostoo Hackathon 2026 Finals

Autonomous crypto trading bot for the SG vs HK Web3 Quant Hackathon finals. Data-driven regime detection via PCA-HMM with post-hoc state analysis, momentum composite ranking with optional Lasso boost, and unified Sortino-optimized risk management.

## Architecture

```
Every hour:

  1. REGIME     PCA on 43 coins → 4 PCs → 3-state HMM
                States NOT pre-labeled. After fitting:
                  - Analyze each state's forward returns, vol, duration
                  - Derive exposure multiplier from observed profitability
                  - Name states descriptively (e.g. BEAR_CALM, BULL_VOLATILE)
                Example from live: BEAR_CALM (exp=0.00), BULL_VOLATILE (exp=1.00)

  2. FEATURES   12 features per coin, z-scored cross-sectionally
                Momentum: r_6h, r_24h, r_3d (Liu et al. 2019)
                Quality:  persistence, choppiness
                Risk:     realized_vol, downside_vol, jump_proxy
                Breakout: breakout_distance, volume_ratio
                Reversal: overshoot
                Cost:     spread_pct

  3. RANK       Momentum composite score (always available):
                  Score = 0.50·z(r_24h) + 0.35·z(r_3d) + 0.15·z(r_6h)
                        + 0.10·z(persistence) - 0.10·z(choppiness)
                        + 0.05·z(volume_ratio)
                Optional Lasso boost when ML finds signal (R² > 0.5%):
                  blend = 0.70·momentum + 0.30·lasso_pred

  4. GATE       r_24h > 0 AND volume_ratio > 0.8
                Deliberately loose — want active trading.
                Max 3 new entries per cycle.

  5. SIZE       vol-parity × REDD × state exposure multiplier
                Top-ranked: 1.3×, second: 1.15×, last: 0.8×

  6. EXIT       Unified trailing stops (no strategy branching):
                  Hard stop: -3.5% | Partial exit: 50% at +3%
                  Trail: 3.5% (4.5% after partial) | Time: 60h at <1%
                  Breakdown: price below 24h low → market sell
```

## Key Design Decisions

### Data-Driven Regime (v7)

Previous versions pre-labeled HMM states as LOW_VOL/MID_VOL/HI_VOL based on covariance trace. This imposed our assumption that "volatility = regime" onto the model. In v7:

1. Fit HMM on 4 PC score vectors (unsupervised)
2. Get the raw latent state sequence
3. For each state, measure: average forward returns, volatility, duration, frequency
4. Derive exposure multipliers from forward returns (positive → trade, negative → sit out)
5. Name states based on observed properties (for logging only)

This lets the model tell us what the states mean. In practice, it discovers states like:
- BEAR_CALM: negative fwd returns, low vol → exposure 0% (sit out)
- BULL_VOLATILE: positive fwd returns, mid vol → exposure 100% (trade fully)
- BEAR_VOLATILE: negative fwd returns, high vol → exposure ~50% (cautious)

### Why Momentum Composite Instead of ML-Only

v6 used LassoCV as the sole entry gate. Result: Lasso zeroed ALL features (R²=0.000), bot held cash for 22 consecutive days, failed the 8 active trading days rule.

v7 solution: momentum composite always produces rankings (weighted sum of z-scored returns, persistence, choppiness, volume). The bot always has candidates. Lasso runs in background as optional boost — when it finds signal, it blends in. When it doesn't, momentum drives everything.

### Momentum Weights (Liu, Tsyvinski & Wu 2019)

Academic research on crypto momentum shows 1-week (24h-168h) momentum is the strongest cross-sectional predictor. Weights reflect this:
- r_24h: 0.50 (strongest single predictor)
- r_3d: 0.35 (overlaps with 1-week window)
- r_6h: 0.15 (noisier but captures recent shifts)
- persistence: +0.10 (trend quality)
- choppiness: -0.10 (penalize noisy paths)
- volume_ratio: +0.05 (volume confirmation)

### Why Not r_1h?

At 1-hour bar resolution, `r_1h` = return over a single bar. This is pure noise with no cross-sectional predictive power. Dropped to reduce overfitting.

## Risk Management

**Unified stops** (every position, no strategy branching):
- Hard stop: -3.5% (≈1.7σ daily, caps downside → Sortino optimization)
- Partial exit: sell 50% at +3% (locks in gains, asymmetric payoff)
- Trailing: 3.5% from high (4.5% after partial — "house money" effect)
- Time: 60h at <1% gain (opportunity cost recapture)
- Breakdown: price below 24h low → immediate market sell

**REDD**: smooth scaling 1.0× at 0% DD → 0.0× at 10% DD. Primary control.

**Circuit breakers**: -3.5% warning, -6% liquidate (4h pause), -10% emergency (12h pause).

## Project Structure

```
bot/
├── main.py              # v7 pipeline orchestrator
├── config.py            # All parameters with academic citations
├── features.py          # 12 features + z-scoring + momentum score + entry gate
├── ranking.py           # Momentum composite + optional Lasso boost
├── regime_detector.py   # PCA-HMM with data-driven state analysis
├── ml.py                # Optional LassoCV (background boost)
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
| Portfolio Returns | Top 20 | Momentum always ranks → active trading. Regime sits out bear states. |
| Risk-Adjusted | 40% | Unified stops + partial exits + REDD → Sortino/Calmar optimized |
| Code & Strategy | 60% | Data-driven regime, documented evolution v1→v7, every parameter cited |

**Composite: 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar**
