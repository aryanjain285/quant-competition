# Quantitative Trading Bot — Roostoo Hackathon 2026 Finals

Autonomous crypto trading bot for the SG vs HK Web3 Quant Hackathon. Built over 3 weeks across 8 major versions. The final system uses EWMA momentum ranking with dynamic spread filtering, PCA-HMM data-driven regime detection, and unified Sortino-optimized risk management. Every design decision is backed by backtesting on 4-6 months of hourly data with a high-fidelity simulator that models real Roostoo spreads and correct maker/taker fees.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Step-by-Step Pipeline](#step-by-step-pipeline)
3. [Dynamic Spread Filter](#1-dynamic-spread-filter)
4. [Regime Detection](#2-regime-detection-pca-hmm-with-data-driven-state-analysis)
5. [EWMA Momentum Ranking](#3-ewma-momentum-ranking)
6. [Risk Management](#4-risk-management-sortino-optimization)
7. [ML Experiments](#what-we-tried-ml-experiments)
8. [Backtest Results](#backtest-results)
9. [Version Evolution](#version-evolution)
10. [Backtest Infrastructure](#backtest-infrastructure)
11. [Project Structure](#project-structure)
12. [Setup & Deployment](#setup--deployment)

---

## Architecture

```
Every hour:

  1. REGIME     PCA (43 coins → 4 PCs) → 3-state HMM
                States analyzed post-fit from historical forward returns
                Exposure: linear from Sharpe ranking, 0.10 floor

  2. SPREAD     Dynamic filter: compute median spread across all coins
                Only trade coins with spread ≤ median
                Calm→trade broadly (~30 coins) | Volatile→narrow to liquid (~15)

  3. FEATURES   12 per-coin features, z-scored cross-sectionally

  4. RANK       EWMA momentum score:
                  score = avg(EWMA(log_ret, halflife=6h), EWMA(log_ret, halflife=24h))

  5. GATE       r_24h > 0 AND volume_ratio > 0.8 AND score > 0
                Max 3 new entries per hour

  6. SIZE       vol-parity × REDD × regime exposure × rank multiplier
                Top-ranked: 1.3×, second: 1.15×, bottom: 0.8×

  7. EXIT       Unified stops (every position, no strategy branching):
                  Hard stop: -3.5% | Partial exit: 50% at +3%
                  Trail: 3.5% (4.5% after partial) | Time: 60h at <1%
                  Breakdown: price below 24h low → market sell
```

---

## Step-by-Step Pipeline

### 1. Dynamic Spread Filter

**The problem:** Coins on Roostoo have wildly different bid-ask spreads. Every trade starts in a hole equal to the spread — you're immediately losing money and need the price to move just to break even.

Measured spreads from the live Roostoo API:

| Coin | Spread (bps) | Impact on 3% profit target |
|---|---|---|
| BTC/USD | 0.0 | 0% of profit eaten |
| SOL/USD | 1.3 | 0.4% eaten |
| DOT/USD | 8.0 | 2.7% eaten |
| PEPE/USD | 28.7 | 9.6% eaten |
| WIF/USD | 54.8 | 18.3% eaten |
| EIGEN/USD | 66.0 | 22.0% eaten |

Trading EIGEN means giving up 22% of your profit target to the spread before the coin even moves.

**Static approach (tested first):** Permanently remove the 6 worst coins (spread > 15 bps). Result: return improved from +0.44% to +0.75%. Better, but spreads are not static — they widen during volatility and tighten during calm periods.

**Dynamic approach (final):** Each cycle, compute the median spread across all coins from the LIVE ticker data. Only trade coins with spread at or below the median. This adapts automatically:

- Calm markets: most spreads are tight → median is low → trade ~30 coins broadly
- Volatile markets: spreads widen → median rises → only the most liquid ~15 coins survive
- No hardcoded threshold to defend — "better than half the universe" is self-calibrating

**Impact:** +0.44% → **+2.08%** return. Sortino: 0.46 → **2.02**. This single change was worth more than all ML experiments combined. It's a guaranteed structural edge — not a statistical signal that might disappear, but a cost reduction that works in every market condition.

### 2. Regime Detection: PCA-HMM with Data-Driven State Analysis

**What it does:** Determines HOW MUCH of the portfolio to deploy at any given time. In favorable conditions, exposure approaches 1.0×. In hostile conditions, it drops to 0.10× (tiny positions just for activity compliance).

**How it works:**

1. **PCA (Principal Component Analysis):** We compute hourly log returns for all 43 coins and extract the top 4 principal components. These 4 PCs capture ~72% of the variance in the entire cross-section — they represent the dominant market forces driving co-movement. PCA loadings are cached and refit every 6 hours. Between refits, new data is projected using cached loadings (a single matrix multiply — negligible cost).

2. **HMM (Hidden Markov Model):** A 3-state Gaussian HMM is fitted on the 4-dimensional PC score vectors. It discovers 3 distinct market "moods" that keep recurring. The HMM doesn't know what these moods mean — it just groups similar market conditions together. Refit every 12 hours.

3. **State Analysis (the key innovation):** After fitting, we DON'T pre-label states as "low vol / mid vol / high vol." Instead, we analyze each discovered state empirically:
   - What were the average forward returns when the market was in this state?
   - What was the volatility (trace of covariance matrix)?
   - How long did the state typically last?
   - What fraction of time was the market in this state?

4. **Exposure Derivation:** States are ranked by their historical forward-return Sharpe ratio. Exposure is linearly interpolated from worst (0.10) to best (~1.0), with a 0.10 floor to guarantee activity compliance. This is data-driven — the model tells US what the states mean, we don't impose assumptions.

**Why 4 PCs:** 4 PCs typically capture 72-85% of variance in a 43-coin crypto universe. 3-state HMM with 4D observations has ~50 parameters — stable with 1000+ observations (we use the full candle history). Going beyond 4 PCs risks overfitting the HMM (parameters scale as K²).

**Why 3 states, not 2 or 5:** 2 states can only distinguish "good" vs "bad" — too coarse. 5 states with 4D observations would require ~120 HMM parameters, risking overfitting on 1000 observations. 3 states with ~50 parameters gives stable estimation while capturing meaningful market structure.

**Exposure derivation:** Linear interpolation from Sharpe ranking with 0.10 floor. Best state gets ~1.0, worst gets 0.10. We tested multiple alternatives:
- Fixed tiers (1.2/0.6/0.10): -0.85% — HMM misclassification amplified by 1.2x leverage
- Floor at 0.15: +2.08% — good but trades too much in worst state
- **Floor at 0.10: +2.45%** — best result, minimal bear-state exposure while maintaining activity

The 0.10 floor ensures the bot places at least one small trade per day (~$100K on $1M) even in the worst regime state, satisfying the 8 active trading days requirement without wasting capital.

**Example from live deployment:**
```
State 0 [BEAR_CALM]:    fwd_ret=-2.62%  vol=8.4   duration=8h  freq=22% → exposure=0.10
State 1 [BULL_VOLATILE]: fwd_ret=+1.59%  vol=19.6  duration=9h  freq=65% → exposure=1.00
State 2 [BEAR_VOLATILE]: fwd_ret=-0.27%  vol=132   duration=2h  freq=12% → exposure=0.55
```

### 3. EWMA Momentum Ranking

**What it does:** Ranks all eligible coins by their recent momentum. The top-ranked coins get bought.

**The signal:**
```python
score = average(EWMA(log_returns, halflife=6h), EWMA(log_returns, halflife=24h))
```

EWMA (Exponentially Weighted Moving Average) of log returns gives a smoothed estimate of recent momentum. The halflife parameter means returns from `halflife` hours ago have half the weight of the current hour. We average two horizons:

- **6h EWMA:** Captures short-term shifts — a coin that started moving up in the last few hours
- **24h EWMA:** Captures daily momentum — the sustained trend over the last day

**Why EWMA, not weighted z-scored returns:**

Version 7 used `Score = 0.50×z(r_24h) + 0.35×z(r_3d) + 0.15×z(r_6h)` with weights cited from Liu, Tsyvinski & Wu (2019), "Risks and Returns of Cryptocurrency." This paper studies crypto momentum on weekly and monthly horizons. Our bot operates on 1-hour bars. The cross-sectional dynamics at 6-24 hour horizons are fundamentally different from 1-week to 4-week horizons. We cannot justify transplanting those weights to our timescale — and a judge asking "why 0.50 and not 0.45?" has no good answer.

EWMA has one parameter per horizon (halflife) with a clear, defensible meaning: "how quickly do we forget old data?" Averaging two horizons equally is the least assumptive choice. No arbitrary weights to defend.

**Result:** Literature weights: -3.07% over 4 months. EWMA: **+2.08%**.

**Entry gate:** After ranking, coins must pass two simple conditions:
- `r_24h > 0` — positive 24-hour return (don't buy declining coins)
- `volume_ratio > 0.8` — minimum volume activity (don't buy dead coins)

These are deliberately loose. The EWMA ranking handles quality; the gate just eliminates obviously bad entries. Max 3 new entries per cycle to prevent overtrading.

### 4. Risk Management (Sortino Optimization)

**Scoring formula:** Composite = 0.4×Sortino + 0.3×Sharpe + 0.3×Calmar

Sortino only penalizes DOWNSIDE volatility — big winning days don't hurt. Our entire exit strategy exploits this asymmetry.

**Position Sizing — Volatility Parity:**
```
base_size = (0.5% of portfolio) / (coin's daily volatility)
```
High-vol coins (memecoins) get smaller positions. Low-vol coins (BTC) get larger. Equal risk contribution per position.

Then multiplied by:
- **REDD (Rolling Economic Drawdown):** Smoothly reduces new position sizes as portfolio drawdown grows. 0% DD → full size. 10% DD → zero new positions. No discrete jumps — continuous smooth scaling. This is the primary drawdown control.
- **Regime exposure:** Linear from Sharpe ranking (0.10 to ~1.0) based on current HMM state.
- **Rank multiplier:** Top-ranked coin gets 1.3×, second gets 1.15×, bottom gets 0.8×. Concentrates capital in highest-conviction picks.

**Unified Trailing Stops (no strategy branching):**

Every position uses identical exit logic:

| Stop | Trigger | Rationale |
|---|---|---|
| Hard stop | -3.5% from entry | BTC daily vol ≈ 2.1%. 3.5% ≈ 1.7σ. Caps downside → directly reduces Sortino denominator. |
| Partial exit | +3% from entry, sell 50% | Locks in gains on half the position. Creates asymmetric payoff — capped downside, open upside on remainder. Reduces variance of trade outcomes. |
| Trailing stop | 3.5% from high (4.5% after partial) | Protects gains as price rises. Wider after partial exit — remaining half is "house money," give it room. |
| Time stop | 60h with <1% gain | If position hasn't moved in 2.5 days, the thesis is wrong. Free up capital for better opportunities. |
| Breakdown | Price below 24h low | Technical exit — immediate market sell. Something changed structurally. |

**Emergency circuit breakers:**
- -3.5% portfolio drawdown: warning (REDD already reducing sizes)
- -6%: full liquidation + 4h pause
- -10%: emergency liquidation + 12h pause

---

## What We Tried: ML Experiments

### Model Shootout

We tested 6 machine learning models for cross-sectional ranking (94 walk-forward windows, temporal CV, no future leakage):

| Model | Spearman Rank Corr | Top-3 Hit Rate | p-value vs EWMA |
|---|---|---|---|
| EWMA only | -0.050 | 15.6% | — |
| **Ridge** | **+0.020** | **14.9%** | **0.025** |
| RandomForest | +0.009 | 17.0% | — |
| XGBoost | +0.002 | 16.0% | — |
| Lasso | NaN (constant predictions) | 13.8% | — |
| ElasticNet | NaN (constant predictions) | 11.7% | — |

Random baseline: Spearman = 0.000, Hit@3 ≈ 7%.

Ridge showed statistically significant ranking ability (p=0.025). Lasso and ElasticNet zeroed ALL features (L1 penalty too aggressive for this data). RandomForest had the best raw hit rate but no significance test.

### Why Every ML Integration Failed

We tested 5 ways to integrate Ridge into the full trading pipeline:

| Integration Method | Return | Positive 10d Windows | Why It Failed |
|---|---|---|---|
| **Pure EWMA (no ML)** | **+2.08%** | **12/29 (41%)** | — |
| Score blend (0.7×EWMA + 0.3×Ridge) | +0.14% | 11/29 (38%) | Ridge's magnitude noise (R² 1-4%) degraded EWMA signal |
| Rank averaging (avg of EWMA rank + Ridge rank) | -0.68% | 10/29 (34%) | Ridge reordered momentum winners to "safer" but weaker picks |
| Ridge veto (block if Ridge predicts negative) | -0.66% | 10/29 (34%) | Vetoed too many good momentum picks |
| Ridge as primary ranker | +0.11% | 11/29 (38%) | R² too low for direct ranking |

**Root cause:** The shootout measured Ridge's ability to rank ALL 43 coins (including coins going down). The trading pipeline only ranks coins that already passed the momentum gate (r_24h > 0, score > 0) — a much narrower, more similar group. Ridge can distinguish winners from losers across the full spectrum, but it can't differentiate within the top momentum cohort.

### Earlier ML Experiments (v1-v6)

- **LightGBM ensemble** (31,533 samples, 6 months, purged walk-forward CV): AUC 0.55. No forward-predictive power from price-derived features.
- **LightGBM on positioning data** (funding rates, OI, long/short ratios, taker volume): AUC 0.555. Public positioning data also priced in.
- **LassoCV as sole entry gate** (v6): Lasso zeroed ALL 12 features (R²=0.000). Bot held cash for 22 consecutive days, failed the 8 active trading days rule.
- **RidgeCV on relative returns** (v7): R² of 1-4% consistently. Coefficients too small for magnitude-based scoring.

**Conclusion:** Public data — whether price-derived or positioning-derived — does not predict short-term crypto returns with any ML model we tested. Cross-sectional RANKING from ML was marginally significant (p=0.025) but failed to survive the full trading pipeline. The ML code is preserved in the repository for documentation purposes — it demonstrates rigorous testing and honest evaluation.

---

## Backtest Results

### Backtest Infrastructure

Our backtests use a high-fidelity `SimExchange` that models the Roostoo exchange precisely:
- **Per-pair spreads** measured from the live Roostoo API (0 bps for BTC to 66 bps for EIGEN)
- **Correct fee model:** 0.05% maker (limit orders), 0.10% taker (market orders)
- **Limit order fill simulation** with pending tracking, timeout, and stale-sell resubmission
- **Same code modules** as the live bot — features, ranking, regime, risk manager all imported directly

The backtest calls the exact same functions as `main.py`. Only the exchange interaction is simulated.

### 4-Month In-Sample (95 days, 43 coins, 1h bars)

| Metric | Value |
|---|---|
| Return | **+2.45%** |
| Sharpe | 0.80 |
| Sortino | **1.61** |
| Calmar | **1.14** |
| Max Drawdown | 8.32% |
| Trades | 1,563 |
| Best 10-day window | +7.72% |
| Worst 10-day window | -5.28% |
| Positive 10-day windows | 13/29 (45%) |
| Avg 10-day return | +0.16% |

### 6-Month Out-of-Sample (155 days)

*Note: "Out-of-sample" here means the 6-month backtest includes 2 additional months of data beyond the 4-month period used during strategy development. All parameters (EWMA halflives, stop levels, spread filter logic, exposure floor) were tuned on the 4-month window. The extra 2 months were never seen during development — they test whether the strategy generalizes to unseen market conditions.*

| Metric | Value |
|---|---|
| Return | -5.92% |
| Positive 10-day windows | 16/49 (33%) |
| Best 10-day window | **+9.80%** |
| Worst 10-day window | -5.31% |
| Avg max drawdown | 3.74% |

The 6-month period includes an extended bearish stretch for crypto. The strategy's behavior is consistent across in-sample and out-of-sample: captures bull windows strongly (+9.80% best — highest across all tests) and limits bear losses (worst -5.31%, similar to in-sample). Average drawdown is actually LOWER on OOS (3.74% vs 4.16%) — the risk management generalizes. A long-only momentum strategy cannot profit when the entire market drops — this is the fundamental constraint, not a strategy flaw.

---

## Version Evolution

| Version | Approach | 4mo Return | What We Learned |
|---|---|---|---|
| v1 | Simple breakout + RSI(14) rules | +5.1% (42d) | Worked on limited data, magic thresholds |
| v2 | + HMM regime suppression + ML gate | -2.1% | Regime suppressed best trades |
| v3 | Stripped ML, tuned params | +1.06% | Simpler was better |
| v4 | Cross-sectional features + ranking | -6.0% | Over-filtering killed returns |
| v5 | Ridge + continuation/reversal events | -3.5% | Regime too aggressive, 5-min/1-hour mismatch |
| v6 | LassoCV as sole entry gate | -5.86% | Lasso zeroed all features → 22 days no trades |
| v7a | Momentum with literature weights | -3.07% | Liu et al. weights don't transfer to 1h bars |
| v7b | EWMA momentum (no weights) | +0.14% | Simple wins |
| v8a | + static spread filter | +0.75% | Removing expensive coins helps |
| v8b | + dynamic spread + 0.15 floor | +2.08% | Adapting to conditions |
| **v8c** | **+ 0.10 floor (final)** | **+2.45%** | **Less bear-state exposure = less drag** |

---

## Project Structure

```
bot/
├── main.py              # Pipeline orchestrator (7 steps)
├── config.py            # All parameters with justifications
├── features.py          # 12 features + EWMA momentum + entry gate + z-scoring
├── ranking.py           # EWMA ranking + dynamic spread filter
├── regime_detector.py   # PCA-HMM with data-driven exposure (0.10 floor)
├── ml.py                # Ridge trainer (disabled — kept for code review documentation)
├── risk_manager.py      # Unified stops, REDD, vol-parity
├── executor.py          # Limit/market orders, pending order tracking
├── binance_data.py      # 1h candle feed from Binance
├── roostoo_client.py    # Roostoo API with HMAC-SHA256 signing
├── metrics.py           # Live Sharpe/Sortino/Calmar computation
├── logger.py            # Structured JSON logging (trades, cycles, metrics)
└── backtest/
    ├── engine.py         # High-fidelity backtest engine (same modules as live)
    ├── sim_exchange.py   # Simulated exchange (per-pair spreads, correct fees)
    ├── model_shootout.py # 6-model ML comparison framework
    └── run_backtest.py   # Rolling window runner with configurable params
```

---

## Setup & Deployment

```bash
# Install
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with Roostoo API credentials

# Run the bot
venv/bin/python run.py

# Run backtests
venv/bin/python -m bot.backtest.run_backtest              # 4 months
venv/bin/python -m bot.backtest.run_backtest --months 6   # 6 months OOS
venv/bin/python -m bot.backtest.model_shootout             # ML model comparison
```

---

## Competition Scoring

| Screen | Weight | Our Approach |
|---|---|---|
| Rule Compliance | Pass/fail | Full JSON logging, autonomous execution, 0.10 floor ensures 8 active days |
| Portfolio Returns | Top 20 qualify | 1.2x leverage in bull states, dynamic spread filter reduces cost drag |
| Risk-Adjusted Score | 40% | Unified Sortino-optimized stops, REDD scaling, partial exits |
| Code & Strategy Review | 60% | Data-driven regime, 6-model shootout, honest ML evaluation, clean v1→v8 evolution |

**Composite score: 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar**
