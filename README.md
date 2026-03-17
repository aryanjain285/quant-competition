# Quant Trading Bot — Roostoo Hackathon 2026

Autonomous crypto trading bot built for the SG vs HK Web3 Quant Hackathon.
Trades on Roostoo's mock exchange via REST API, using a dual-engine signal system backed by historical backtests.

## Strategy Overview

### The Core Problem

In a 10-day crypto competition scored on risk-adjusted metrics (Sortino, Sharpe, Calmar), the winning approach isn't maximum returns — it's **consistent positive returns with minimal drawdowns**. One bad -8% day destroys your Calmar ratio for the entire competition.

### What We Tested (and What Failed)

We backtested 5 signal types across 24 coins over 42 days of hourly data:

| Strategy | Avg PnL | Win Rate | Avg MaxDD | Verdict |
|---|---|---|---|---|
| EMA Crossover 12/26 | +3.4% | 32% | 12.6% | OK on trending coins only |
| 3-Day Momentum | -10.2% | 24% | 19.5% | **Lost money on most coins** |
| RSI Mean Reversion | -3.3% | 47% | 10.0% | High win rate, consistent |
| 72h Breakout | +0.5% | 37% | 8.3% | Best risk-adjusted |
| Volume + Momentum | -7.5% | 28% | 17.6% | No edge over plain momentum |

**Key finding:** Pure momentum strategies have ~0% autocorrelation at 24h horizons in the current market. The market is *mean-reverting* on short timeframes with occasional *breakout trends*. This rules out momentum as a primary signal.

### What We Built: Dual-Engine System

Two independent signal engines that exploit different market behaviors:

#### Engine 1: Breakout Detection
- **Entry:** Price exceeds its 72-hour high while EMA(21) > EMA(55) (trend aligned)
- **Volume confirmation** boosts signal strength
- **Exit:** Trailing stop at -3% from high, or hard stop at -4% from entry
- **Why it works:** Catches large trending moves (FET +43%, ZEC +27%, NEAR +34% in backtests) while the tight stops limit downside to ~4% maxDD
- **Win rate is low (~20%)** but winners are much larger than losers — classic trend-following profile

#### Engine 2: RSI Mean Reversion
- **Entry:** RSI(14) drops below 30 (oversold)
- **Exit:** Profit target at +3% from entry, trailing stop at -2%, or hard stop at -3%
- **Why it works:** Crypto routinely oversells and bounces. Backtest showed 78% win rate on XRP, 53-61% on other large caps
- **Provides consistent small gains** that smooth out the equity curve between breakout trades

### Why Not ML / Regression / Neural Nets?

1. **10 days of live data is not enough** for any statistical model to demonstrate edge over simple rules
2. **Overfitting risk** — ML models tuned on 42 days of history will overfit
3. **Code review is 60% of the score** — judges need to understand the strategy. Rules-based logic is transparent
4. **Time-to-production** — 4 days to build means every hour matters. Simple systems that work > complex systems that break

### Risk Management (Where the Competition is Actually Won)

#### Volatility-Parity Position Sizing
```
position_size = (target_risk × portfolio_value) / daily_volatility
```
- Automatically sizes down in volatile markets
- Target risk: 1.5% of portfolio per position
- This directly optimizes Sharpe and Sortino ratios

#### Strategy-Specific Trailing Stops
| Strategy | Trailing Stop | Hard Stop | Profit Target |
|---|---|---|---|
| Breakout | -3% from high | -4% from entry | None (let it run) |
| Mean Reversion | -2% from high | -3% from entry | +3% from entry |

Tighter stops for mean-reversion (quick in-and-out). Wider for breakouts (let trends develop).

#### Portfolio-Level Drawdown Circuit Breakers
| Drawdown | Action |
|---|---|
| -2% from peak | Cut all positions by 50% |
| -4% from peak | Liquidate everything, pause 12 hours |
| -7% from peak | Emergency: cash only for 48 hours |

This directly protects the Calmar ratio. In backtests, max drawdown never exceeded 4.78% across any 10-day window.

#### Exposure Limits
- Max 60% of portfolio in crypto (always 40%+ cash buffer)
- Max 20% per coin
- Max 8 simultaneous positions

### Data Advantage

Roostoo's API only provides basic ticker data (bid/ask/last/change). We supplement with **Binance public data** (no API key required):
- 1000 historical 5-minute candles per pair on startup
- Live candle updates every 5 minutes
- Volume data for confirmation signals

Since Roostoo mirrors real crypto prices, Binance data gives us 100× richer signals than teams using only the Roostoo ticker.

## Backtest Results

### Full Period (37 days, 24 coins)

| Metric | Value |
|---|---|
| Total Return | **+5.12%** |
| Max Drawdown | **4.72%** |
| Sharpe Ratio | **2.51** |
| Sortino Ratio | **8.87** |
| Calmar Ratio | **10.71** |
| **Composite Score** | **7.52** |
| Trades | 718 |
| Win Rate | 20% (but avg winner >> avg loser) |

### Rolling 10-Day Windows (Competition Format)

| Metric | Value |
|---|---|
| Avg Return | +1.04% |
| Median Return | +0.77% |
| Best Window | +4.69% |
| Worst Window | -2.47% |
| Positive Windows | **7 / 10 (70%)** |
| Avg Max Drawdown | 3.40% |
| Worst Max Drawdown | 4.78% |
| Avg Composite Score | 6.44 |

### Rolling 8-Day Windows (Min Active Days)

| Metric | Value |
|---|---|
| Avg Return | +0.33% |
| Best Window | +4.05% |
| Worst Window | -1.73% |
| Positive Windows | 5 / 10 (50%) |
| Avg Max Drawdown | 2.98% |

## Project Structure

```
quant-competition/
├── run.py                          # Entry point: venv/bin/python run.py
├── .env                            # API keys (gitignored)
├── .env.example                    # Template for .env
├── requirements.txt                # Python dependencies
│
├── bot/
│   ├── config.py                   # All tunable parameters
│   ├── roostoo_client.py           # Roostoo API with HMAC signing
│   ├── binance_data.py             # Binance candle data feed
│   ├── signals.py                  # Dual-engine signal generation
│   ├── risk_manager.py             # Position sizing & drawdown breakers
│   ├── executor.py                 # Order placement & management
│   ├── metrics.py                  # Live Sharpe/Sortino/Calmar tracking
│   ├── logger.py                   # Structured JSON logging
│   └── main.py                     # Main trading loop orchestrator
│
├── bot/backtest/
│   ├── data_loader.py              # Shared data loading utilities
│   ├── bt_market_analysis.py       # Market structure analysis
│   ├── bt_signal_comparison.py     # Compare 5 signal types per coin
│   ├── bt_full_portfolio.py        # Full portfolio sim (8/10-day windows)
│   ├── bt_param_sensitivity.py     # Parameter sweep analysis
│   └── run_all.py                  # Run all backtests
│
└── logs/                           # Runtime logs (gitignored)
    ├── bot.jsonl                   # General bot logs
    ├── trades.jsonl                # Trade journal
    └── cycles.jsonl                # Per-cycle state snapshots
```

## Setup & Running

### Prerequisites
- Python 3.10+
- Internet access (for Roostoo API and Binance data)

### Installation
```bash
cd quant-competition
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

### Configure API Keys
```bash
cp .env.example .env
# Edit .env with your Roostoo API key and secret
```

### Run the Bot
```bash
venv/bin/python run.py
```
The bot will:
1. Connect to Roostoo and verify API access
2. Load 1000 candles of historical data from Binance for all 43 pairs
3. Enter the main loop (every 5 minutes):
   - Poll Roostoo ticker + update Binance candles
   - Compute signals for all pairs
   - Check drawdown breakers and trailing stops
   - Execute buy/sell orders
   - Log everything to `logs/`

### Run Backtests
```bash
# Run all backtests
venv/bin/python -m bot.backtest.run_all

# Run individual backtests
venv/bin/python -m bot.backtest.bt_market_analysis     # Market structure
venv/bin/python -m bot.backtest.bt_signal_comparison    # Signal comparison
venv/bin/python -m bot.backtest.bt_full_portfolio       # Full portfolio sim
venv/bin/python -m bot.backtest.bt_param_sensitivity    # Parameter sweeps
```

## How to Verify the Strategy

1. **Run `bt_signal_comparison`** — confirms that breakout and RSI mean-reversion are the two strongest signals across coins
2. **Run `bt_full_portfolio`** — shows the complete portfolio simulation including 8-day and 10-day rolling windows matching competition format
3. **Run `bt_param_sensitivity`** — shows the strategy is robust across a range of parameters (not fragile/overfit to one setting)
4. **Check `logs/trades.jsonl`** during live trading — every trade is logged with signal type, strength, and P&L

## What to Improve

### Quick Wins (hours of work)
- **Coin-specific RSI thresholds** — the backtest shows RSI works best on XRP (78% WR) but poorly on PEPE. Tune per-coin
- **Adaptive breakout lookback** — use ATR to dynamically set the lookback window (wider in low-vol, tighter in high-vol)
- **Time-of-day filter** — crypto volume/volatility varies by hour. Avoid entries during low-liquidity periods

### Medium Effort (1-2 days)
- **Cross-sectional ranking** — instead of trading every breakout, rank by signal strength and only take the top N
- **Correlation filter** — avoid holding 5 altcoins that all move together. Diversify across sectors
- **Regime overlay** — detect bull/bear/choppy market regime and adjust max exposure accordingly

### Research (if time permits)
- **Order flow signals** — poll Binance order book depth for liquidity signals
- **Funding rate data** — negative funding often precedes short squeezes
- **On-chain metrics** — whale wallet movements as a leading indicator

## Competition Scoring Reference

| Screen | Weight | Our Approach |
|---|---|---|
| Rule Compliance | Pass/fail | Full logging, autonomous trades, clean commits |
| Portfolio Returns | Top 20 qualify | Breakout engine captures uptrends |
| Risk-Adjusted Score | 40% | Volatility sizing (Sharpe), downside focus (Sortino), drawdown breakers (Calmar) |
| Code & Strategy Review | 60% | Modular architecture, clear strategy logic, documented |

Composite score: **0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar**
