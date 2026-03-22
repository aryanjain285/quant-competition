# Quant Trading Bot — Roostoo Hackathon 2026

Autonomous crypto trading bot for the SG vs HK Web3 Quant Hackathon. Five-stage pipeline with PCA-HMM regime detection, cross-sectional feature scoring, Ridge regression ranking, and Sortino-optimized risk management. All data on 1-hour candles.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MAIN LOOP (every 1 hour)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ STEP 0: Data                                                   │ │
│  │   Roostoo ticker + wallet | Binance 1h candles (43 pairs)     │ │
│  └────────────────────┬───────────────────────────────────────────┘ │
│                       │                                             │
│  ┌────────────────────▼───────────────────────────────────────────┐ │
│  │ STEP 1: REGIME FILTER — Should we trade at all?               │ │
│  │   PCA on cross-sectional returns → PC1 market factor          │ │
│  │   3-state HMM on PC1: TREND_SUPPORTIVE / SELECTIVE / HOSTILE  │ │
│  │   HOSTILE → sit in cash (no new entries)                      │ │
│  └────────────────────┬───────────────────────────────────────────┘ │
│                       │                                             │
│  ┌────────────────────▼───────────────────────────────────────────┐ │
│  │ STEP 2: EVENT FILTER — Any valid setups?                      │ │
│  │   Per-coin features → z-score cross-sectionally               │ │
│  │   Continuation: breakout + positive 24h + persistent + volume │ │
│  │   Reversal: deep overshoot + stabilizing + acceptable risk    │ │
│  └────────────────────┬───────────────────────────────────────────┘ │
│                       │                                             │
│  ┌────────────────────▼───────────────────────────────────────────┐ │
│  │ STEP 3: VALID TRADES — Collect survivors                      │ │
│  │   Only setups that passed regime + event filters              │ │
│  │   Empty → hold cash (no forced trades)                        │ │
│  └────────────────────┬───────────────────────────────────────────┘ │
│                       │                                             │
│  ┌────────────────────▼───────────────────────────────────────────┐ │
│  │ STEP 4: RANKING — Which are best?                             │ │
│  │   Ridge regression on z-scored features (retrained daily)     │ │
│  │   Fallback: hand-built weighted score                         │ │
│  │   Score thresholds: continuation > 0.15, reversal > 0.20     │ │
│  └────────────────────┬───────────────────────────────────────────┘ │
│                       │                                             │
│  ┌────────────────────▼───────────────────────────────────────────┐ │
│  │ STEP 5: EXECUTION                                             │ │
│  │   Vol-parity sizing × REDD drawdown scaling × regime mult     │ │
│  │   Breakout: partial exit at +3%, trailing stop, no upside cap │ │
│  │   Reversal: +3% profit target, -3% hard stop, 48h time stop  │ │
│  │   Limit orders (0.05%) for entries + non-urgent exits         │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Strategy

### Regime Detection: PCA-HMM

Instead of using BTC alone to detect market state, we compute the first principal component (PC1) across all 43 tradeable coins. PC1 captures the dominant market factor — the common co-movement that drives ~65% of cross-sectional variance.

A 3-state Gaussian Hidden Markov Model fitted on PC1 returns + rolling volatility identifies:
- **TREND_SUPPORTIVE** (low vol, directional): full exposure budget (1.0×)
- **SELECTIVE** (moderate vol): reduced exposure (0.5×)
- **HOSTILE** (high vol, chaotic): minimal exposure (0.1×), no new entries

PCA loadings are cached and refit every 6 hours. HMM refits every 24 hours. Between refits, new data is projected using cached loadings (a single dot product — negligible cost).

### Signal Generation: Continuation + Reversal

Two independent event detectors, both using z-scored cross-sectional features:

**Continuation** (trend-following): Fires when a coin is breaking out above its 72-hour high with volume confirmation, positive 24h return, high persistence (clean path), low choppiness, and acceptable risk/cost penalties. All 7 conditions must pass — conjunction prevents false entries.

**Reversal** (mean-reversion): Fires when a coin's 6-hour return is more than 1.3 standard deviations below its own historical distribution (overshoot), the 1-hour return shows stabilization (not free-falling), and risk is elevated but not catastrophic.

In bearish markets, continuation signals go dormant (nothing is above its 72h high) and reversal signals become the primary entry path, catching oversold bounces.

### Ranking: Ridge Regression

Valid candidates are ranked by a Ridge regression model trained daily on the trailing 400 hours of data. The model learns which z-scored features predict 24-hour forward returns. Features include multi-horizon returns (1h/6h/24h/3d), persistence, choppiness, breakout distance, volume ratio, risk penalty, cost penalty, and overshoot.

Score thresholds prevent low-conviction entries: continuation requires score > 0.15, reversal requires score > 0.20 (higher bar since mean-reversion is riskier in downtrends).

Fallback: if Ridge training fails, a hand-built weighted score takes over with theory-driven weights.

### Risk Management

**Volatility-parity sizing:** Position size inversely proportional to the coin's realized volatility. High-vol memecoins get smaller positions than low-vol majors. Ensures equal risk contribution per trade.

**REDD (Rolling Economic Drawdown):** Smoothly reduces new position sizes as portfolio drawdown increases. At 0% DD → full size, at 10% DD → zero new positions. This is the primary drawdown control — no discrete jumps.

**Strategy-specific exits:**
- Continuation: -4% hard stop, +3% partial exit (sell 50%), then 3-4% trailing stop on remainder. No upside cap (Sortino optimization: upside vol is free).
- Reversal: -3% hard stop, +3% profit target, 2% trailing stop (only when in profit), 48h time stop.

**Emergency circuit breakers:** -3.5% warning only (REDD handles it), -6% full liquidation + 4h pause, -10% emergency + 12h pause.

### Bearish Market Behavior

In the current very bearish market:
- **HMM detects HOSTILE regime** → 10% max exposure ($80-100K of $1M)
- **Zero continuation signals** — nothing is above its 3-day high
- **Occasional reversal signals** — deeply oversold coins that stabilize
- **Mostly cash** — ~0% return while other teams lose -3% to -10%
- This produces near-zero drawdown → excellent Calmar and Sortino ratios
- Qualifies for top 20 on relative return (losing least), then the 60% code review scores dominate

## What We Tested and What Failed

### Signal Comparison (24 coins, 42 days hourly data)

| Strategy | Avg PnL | Win Rate | Avg MaxDD |
|---|---|---|---|
| EMA Crossover 12/26 | +3.4% | 32% | 12.6% |
| 3-Day Momentum | -10.2% | 24% | 19.5% |
| RSI Mean Reversion | -3.3% | 47% | 10.0% |
| 72h Breakout | +0.5% | 37% | 8.3% |
| Volume + Momentum | -7.5% | 28% | 17.6% |

Pure momentum has near-zero autocorrelation at 24h horizons. The market is mean-reverting short-term with occasional breakout trends.

### ML Experiment

Built a LightGBM ensemble (31K samples, 6 months, purged walk-forward CV). AUC 0.55 on temporally-honest evaluation — no predictive power from price-derived or positioning features. Disabled the model. The Ridge regression in the current system is used only for cross-sectional ranking (which coin is best relative to others), not forward return prediction.

### Rolling 10-Day Backtest (4 months, 24 coins, 1h bars)

| Metric | Value |
|---|---|
| Positive windows | 20/36 (56%) |
| Avg return | +1.06% |
| Best window | +8.28% |
| Worst window | -3.20% |
| Avg max DD | 2.39% |
| Avg composite | 7.27 |

### Key Bugs Found and Fixed

- All lookback constants were calibrated for 5-min bars but data was switched to 1h — every feature was computed over 12× the intended horizon. Volatility was 3.5× overstated, making positions 3.5× too small.
- Regime detector classified neutral markets as HOSTILE due to over-weighted stress penalties.
- Executor returned fill events from limit orders but main.py discarded them — partial fills were lost.
- Mean-reversion trailing stop fired before hard stop check, causing wrong exit reasons.
- Sentiment module computed 12h return but named it `btc_1h_return`, with a 15-minute pause that expired before the next 1h cycle.

## Project Structure

```
bot/
├── main.py              # 5-step pipeline orchestrator
├── config.py            # All parameters (1h bars, competition-tuned)
├── features.py          # Cross-sectional feature engine + z-scoring
├── signals.py           # Continuation + reversal event detection
├── ranking.py           # Hand-built score + Ridge regression
├── regime_detector.py   # PCA-HMM (3-state, PC1 market proxy)
├── ml.py                # Ridge trainer (daily retrain on rolling window)
├── risk_manager.py      # REDD + vol-parity + strategy-specific stops
├── executor.py          # Limit/market orders + pending order tracking
├── binance_data.py      # 1h candle feed from Binance
├── roostoo_client.py    # Roostoo API with HMAC signing
├── metrics.py           # Live Sharpe/Sortino/Calmar
└── logger.py            # Structured JSON logging
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
| Portfolio Returns | Top 20 qualify | In bearish market: lose least. In bull: capture trends. |
| Risk-Adjusted | 40% | REDD + Sortino-optimized exits + regime gating |
| Code & Strategy | 60% | 5-stage pipeline, PCA-HMM, Ridge, documented ML experiment |

**Composite: 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar**
