# Quantitative Trading Bot
## SG vs HK Web3 Quant Hackathon — Finals
### Team: [Your Team Name]

---

## Slide 1: One-Line Thesis

**We built complex ML systems, rigorously proved they don't work in crypto, and discovered that two simple ideas — EWMA momentum and dynamic spread filtering — beat everything else.**

- Tested 6 ML models (Ridge, Lasso, ElasticNet, RF, XGBoost)
- Tested 5 integration methods (blend, rank avg, veto, primary, boost)
- All made things worse
- The winning strategy is the simplest one — but we had to exhaust complexity to find it

---

## Slide 2: Pipeline

```
Every hour:

  1. REGIME     PCA → 4 PCs → 3-state HMM → data-driven exposure (0.10 floor)
  2. SPREAD     Dynamic filter: only trade coins with spread ≤ current median
  3. RANK       EWMA momentum: avg(EWMA(6h), EWMA(24h))
  4. GATE       r_24h > 0, volume > 0.8
  5. SIZE       vol-parity × REDD × regime × rank position
  6. EXIT       Hard -3.5% | Partial 50% at +3% | Trail 3.5%/4.5% | Time 60h
```

12 specialized Python modules. Same code runs in backtest and live.
High-fidelity backtest with per-pair spreads from real Roostoo API.

---

## Slide 3: Dynamic Spread Filter (Key Innovation)

**The problem:** Coins have wildly different spreads on Roostoo.

| Coin | Spread | Impact on 3% profit target |
|---|---|---|
| BTC/USD | 0.0 bps | 0% eaten |
| SOL/USD | 1.3 bps | 0.4% eaten |
| PEPE/USD | 28.7 bps | 9.6% eaten |
| EIGEN/USD | 66.0 bps | 22% eaten |

**Static fix:** Remove worst 6 coins → return +0.44% → +0.75%
**Dynamic fix:** Each hour, compute median spread. Only trade below-median coins.
- Calm markets → tight spreads → trade ~30 coins
- Volatile markets → wide spreads → narrow to ~15 liquid coins

**Result:** +0.44% → **+2.45%**. Sortino: 0.46 → **1.61**. Single biggest improvement.

---

## Slide 4: Data-Driven Regime Detection

**Old approach (v1-v6):** Pre-label HMM states as "low vol / mid vol / high vol." Assign exposure by our assumption about what volatility means. This is circular.

**Our approach:** Fit HMM on 4 principal components. THEN analyze discovered states:

```
State 0 [BEAR_CALM]:    fwd_ret=-2.6%  vol=8.4   → exposure=0.10
State 1 [BULL_VOLATILE]: fwd_ret=+1.6%  vol=19.6  → exposure=1.20
State 2 [BEAR_VOLATILE]: fwd_ret=-0.3%  vol=132   → exposure=0.60
```

The MODEL tells us what the states mean. We don't impose labels.

**Exposure:** Linear interpolation from Sharpe ranking with 0.10 floor:
- Best state → ~1.0 (full deployment in favorable conditions)
- Mid state → ~0.55 (moderate)
- Worst state → 0.10 (minimal — activity compliance)
- Tested 1.2x leverage: worse (-0.85%) because HMM misclassification is amplified

---

## Slide 5: Why EWMA Over Weighted Scores

**v7 used:** `Score = 0.50×z(r_24h) + 0.35×z(r_3d) + 0.15×z(r_6h)` citing Liu et al. 2019.

**Problem:** That paper studies weekly/monthly horizons. We operate on 1-hour bars. We can't justify "why 0.50 and not 0.45?"

**EWMA solution:** One parameter per horizon (halflife). Clear meaning: "how fast do we forget?"

```python
score = avg(EWMA(log_returns, halflife=6h), EWMA(log_returns, halflife=24h))
```

No arbitrary weights. Average of two horizons = least assumptive choice.

**Result:** Literature weights: -3.07%. EWMA: **+2.45%**.

---

## Slide 6: The ML Shootout

We tested 6 models on 94 walk-forward windows:

| Model | Spearman Rank | Top-3 Hit Rate | p-value |
|---|---|---|---|
| EWMA only | -0.050 | 15.6% | — |
| **Ridge** | **+0.020** | **14.9%** | **0.025** |
| RandomForest | +0.009 | 17.0% | — |
| XGBoost | +0.002 | 16.0% | — |
| Lasso | NaN | 13.8% | — |
| ElasticNet | NaN | 11.7% | — |

Ridge was statistically significant. So we tested every way to use it...

---

## Slide 7: Why We Disabled ML (The Honest Slide)

Every Ridge integration made the full pipeline worse:

| Method | Return | Positive Windows |
|---|---|---|
| **Pure EWMA** | **+2.45%** | **13/29 (45%)** |
| Score blend (70/30) | +0.14% | 11/29 (38%) |
| Rank averaging | -0.68% | 10/29 (34%) |
| Ridge veto | -0.66% | 10/29 (34%) |
| Ridge primary | +0.11% | 11/29 (38%) |

**Why?** Shootout measured ranking on ALL 43 coins. Pipeline only ranks coins that already passed the momentum gate — a narrower, more similar group. Ridge's R² of 1-4% means 96%+ noise. Blending noise into a clean signal degrades it.

**Decision:** Disable ML. Pure EWMA. Keep code for documentation.

---

## Slide 8: Risk Management (Sortino Optimization)

**Scoring formula:** 0.4×Sortino + 0.3×Sharpe + 0.3×Calmar

Sortino only penalizes DOWNSIDE volatility. Our exits exploit this asymmetry:

| Stop Type | Trigger | Effect on Sortino |
|---|---|---|
| Hard stop | -3.5% from entry | Caps downside → reduces denominator |
| Partial exit | +3% sells 50% | Locks gains → reduces outcome variance |
| Trailing | 3.5% / 4.5% from high | Protects gains with "house money" width |
| Time stop | 60h at <1% | Frees capital from stale positions |

**REDD scaling:** Smooth drawdown control. 0% DD → full size. 10% DD → zero new positions. No discrete jumps.

---

## Slide 9: Backtest Results

**High-fidelity engine:** Per-pair spreads from real Roostoo API, correct maker/taker fees, same code as live bot.

**4-month (in-sample):**

| Metric | Value |
|---|---|
| Return | **+2.45%** |
| Sortino | **1.61** |
| Sharpe | 0.80 |
| Calmar | **1.14** |
| Max Drawdown | 8.32% |

**6-month (out-of-sample):** -5.23% (heavily bearish period). Best window +7.73%.

The strategy captures bull windows and limits bear losses. Cannot profit when entire market drops — fundamental constraint of long-only.

---

## Slide 10: Evolution — What We Tried, What We Learned

| Version | Approach | Return | Lesson |
|---|---|---|---|
| v3 | Breakout + RSI rules | +1.06% | Magic thresholds aren't defensible |
| v4 | Cross-sectional features | -6.0% | Over-filtering kills returns |
| v5 | Ridge + event filters | -3.5% | Regime too aggressive |
| v6 | LassoCV as sole gate | -5.86% | ML zeroed all → 22 days no trades |
| v7 | Literature weights | -3.07% | Weights don't transfer timescales |
| v8 | EWMA only | +0.44% | Simple wins |
| v8+ | + static spread filter | +0.75% | Remove expensive coins |
| v8++ | + dynamic spread + 0.15 floor | +2.08% | Adapt to conditions |
| **v8 final** | **+ 0.10 floor** | **+2.45%** | **Less bear exposure = less drag** |

Each version taught us something. The final system is simple because we exhausted the complex alternatives.

---

## Slide 11: Live Competition Results

[To be filled with Round 2 results]

- Portfolio return: ___
- Max drawdown: ___
- Sharpe / Sortino / Calmar: ___ / ___ / ___
- Regime states observed: ___
- Dynamic spread filter: avg median spread = ___ bps, avg coins filtered = ___
- Number of trades: ___
- Best / worst individual trade: ___ / ___

---

## Slide 12: Key Takeaways

**1. Simple beats complex in short-horizon crypto trading.**
We tested 8 strategy versions and 6 ML models. The best performer is EWMA momentum — something any quant student could write in 30 minutes.

**2. The edge isn't in the signal — it's in what you DON'T trade.**
Dynamic spread filtering (+0.44% → +2.08%) was worth more than all ML experiments combined. Avoiding expensive coins is a guaranteed, structural edge.

**3. Data-driven > assumption-driven.**
Letting the HMM discover states and deriving exposure from forward returns beats pre-labeling states by volatility.

**4. Honest evaluation > inflated claims.**
We built a LightGBM ensemble, proved it doesn't work (AUC 0.55), and disabled it. We'd rather explain what didn't work than deploy a model we can't defend.

**If we had more time:** Test intraday patterns (hour-of-day effects), add derivatives features (funding rates as regime overlay), explore pairs trading within sectors.
