#!/usr/bin/env python3
"""
BACKTEST 4: Parameter Sensitivity Analysis
───────────────────────────────────────────
Tests how sensitive the strategy is to key parameters:
- Breakout lookback period (24h, 48h, 72h, 96h)
- Trailing stop width (1%, 2%, 3%, 4%, 5%)
- Max exposure (40%, 50%, 60%, 70%)
- Max positions (4, 6, 8, 10)
- RSI thresholds (20/60, 25/65, 30/70, 35/75)

Shows which parameters matter most and optimal ranges.

Run: venv/bin/python -m bot.backtest.bt_param_sensitivity
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from bot.backtest.data_loader import load_data, compute_metrics, CORE_PAIRS
from bot.signals import compute_signal


def run_backtest(
    closes, highs, lows, volumes, pairs, min_len,
    breakout_lookback=72,
    trail_stop_breakout=0.03,
    trail_stop_meanrev=0.02,
    hard_stop=-0.04,
    profit_target_meanrev=0.03,
    max_exposure=0.60,
    max_positions=8,
    drawdown_limit=0.05,
) -> dict:
    """Parameterized portfolio backtest. Returns metrics dict."""
    INITIAL = 1_000_000
    COMMISSION = 0.001

    cash = INITIAL
    positions = {}
    portfolio_history = []
    peak = INITIAL
    trades = 0
    wins = 0

    for t in range(100, min_len):
        prices = {p: closes[p][t] for p in pairs}
        pv = cash + sum(pos["qty"] * prices.get(p, 0) for p, pos in positions.items())
        portfolio_history.append(pv)
        if pv > peak:
            peak = pv
        dd = (peak - pv) / peak

        if dd > drawdown_limit and positions:
            for pair, pos in positions.items():
                cash += pos["qty"] * prices[pair] * (1 - COMMISSION)
                trades += 1
                if prices[pair] > pos["entry"]:
                    wins += 1
            positions.clear()
            continue

        # Stops
        for pair in list(positions.keys()):
            pos = positions[pair]
            p = prices[pair]
            if p > pos["high"]:
                pos["high"] = p

            trail = trail_stop_breakout if pos["strategy"] == "breakout" else trail_stop_meanrev
            drop_from_high = (pos["high"] - p) / pos["high"]
            pnl_from_entry = (p - pos["entry"]) / pos["entry"]

            exit_it = False
            if drop_from_high >= trail:
                exit_it = True
            elif pnl_from_entry <= hard_stop:
                exit_it = True
            elif pos["strategy"] == "mean_rev" and pnl_from_entry >= profit_target_meanrev:
                exit_it = True

            if exit_it:
                cash += pos["qty"] * p * (1 - COMMISSION)
                trades += 1
                if p > pos["entry"]:
                    wins += 1
                del positions[pair]

        # Signals
        sigs = {}
        for pair in pairs:
            c = closes[pair][:t + 1]
            h = highs[pair][:t + 1]
            l = lows[pair][:t + 1]
            v = volumes[pair][:t + 1]
            if len(c) < 80:
                continue
            sigs[pair] = compute_signal(c, h, l, v,
                                        prices[pair] * 0.9999, prices[pair] * 1.0001,
                                        breakout_lookback)

        for pair in list(positions.keys()):
            sig = sigs.get(pair, {})
            if sig.get("action") == "SELL":
                p = prices[pair]
                pos = positions[pair]
                cash += pos["qty"] * p * (1 - COMMISSION)
                trades += 1
                if p > pos["entry"]:
                    wins += 1
                del positions[pair]

        buys = [(p, s) for p, s in sigs.items() if s["action"] == "BUY" and p not in positions]
        buys.sort(key=lambda x: x[1]["strength"], reverse=True)
        exposure = sum(pos["qty"] * prices.get(p, 0) for p, pos in positions.items())

        for pair, sig in buys:
            if len(positions) >= max_positions:
                break
            remaining = max_exposure * pv - exposure
            if remaining <= 0:
                break
            vol = sig.get("real_vol", 0.5) or 0.5
            daily_vol = vol / np.sqrt(365)
            if daily_vol <= 0:
                daily_vol = 0.03
            size = min(0.015 * pv / daily_vol, 0.20 * pv, remaining) * sig["strength"]
            if size < 100:
                continue
            p = prices[pair]
            qty = size / p
            if qty * p * (1 + COMMISSION) > cash:
                continue
            cash -= qty * p * (1 + COMMISSION)
            positions[pair] = {"qty": qty, "entry": p, "high": p, "strategy": sig["strategy"]}
            exposure += qty * p
            trades += 1

    for pair, pos in positions.items():
        p = closes[pair][min_len - 1]
        cash += pos["qty"] * p * (1 - COMMISSION)
        trades += 1
        if p > pos["entry"]:
            wins += 1

    m = compute_metrics(portfolio_history, INITIAL, daily_sample_rate=24)
    m["trades"] = trades
    m["wins"] = wins
    m["win_rate_pct"] = round(wins / max(trades, 1) * 100, 1)
    return m


def main():
    data = load_data(CORE_PAIRS, interval="1h", limit=1000)
    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    volumes = data["volumes"]
    pairs = data["pairs"]
    min_len = data["min_len"]

    print("\n" + "=" * 90)
    print("  PARAMETER SENSITIVITY ANALYSIS")
    print("  Testing how each parameter affects strategy performance")
    print("=" * 90)

    # ── Test 1: Breakout lookback ──
    print(f"\n{'─' * 90}")
    print("  1. BREAKOUT LOOKBACK PERIOD")
    print(f"{'─' * 90}")
    print(f"  {'Lookback':>10} {'Return':>9} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Calmar':>8} {'Compos':>8} {'Trades':>7}")

    for lookback in [24, 36, 48, 72, 96, 120]:
        m = run_backtest(closes, highs, lows, volumes, pairs, min_len,
                         breakout_lookback=lookback)
        marker = " ◀" if lookback == 72 else ""
        print(f"  {lookback:>7}h  {m['total_return_pct']:>+8.2f}% {m['max_drawdown_pct']:>7.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['calmar']:>7.2f} "
              f"{m['composite']:>7.2f} {m['trades']:>6d}{marker}")

    # ── Test 2: Trailing stop width (breakout) ──
    print(f"\n{'─' * 90}")
    print("  2. BREAKOUT TRAILING STOP WIDTH")
    print(f"{'─' * 90}")
    print(f"  {'Trail %':>10} {'Return':>9} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Calmar':>8} {'Compos':>8} {'WinR':>7}")

    for trail in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        m = run_backtest(closes, highs, lows, volumes, pairs, min_len,
                         trail_stop_breakout=trail)
        marker = " ◀" if trail == 0.03 else ""
        print(f"  {trail:>9.0%}  {m['total_return_pct']:>+8.2f}% {m['max_drawdown_pct']:>7.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['calmar']:>7.2f} "
              f"{m['composite']:>7.2f} {m['win_rate_pct']:>5.0f}%{marker}")

    # ── Test 3: Mean reversion trailing stop ──
    print(f"\n{'─' * 90}")
    print("  3. MEAN REVERSION TRAILING STOP WIDTH")
    print(f"{'─' * 90}")
    print(f"  {'Trail %':>10} {'Return':>9} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Calmar':>8} {'Compos':>8}")

    for trail in [0.01, 0.015, 0.02, 0.025, 0.03]:
        m = run_backtest(closes, highs, lows, volumes, pairs, min_len,
                         trail_stop_meanrev=trail)
        marker = " ◀" if trail == 0.02 else ""
        print(f"  {trail:>9.1%}  {m['total_return_pct']:>+8.2f}% {m['max_drawdown_pct']:>7.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['calmar']:>7.2f} "
              f"{m['composite']:>7.2f}{marker}")

    # ── Test 4: Mean reversion profit target ──
    print(f"\n{'─' * 90}")
    print("  4. MEAN REVERSION PROFIT TARGET")
    print(f"{'─' * 90}")
    print(f"  {'Target %':>10} {'Return':>9} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Calmar':>8} {'Compos':>8}")

    for target in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        m = run_backtest(closes, highs, lows, volumes, pairs, min_len,
                         profit_target_meanrev=target)
        marker = " ◀" if target == 0.03 else ""
        print(f"  {target:>9.0%}  {m['total_return_pct']:>+8.2f}% {m['max_drawdown_pct']:>7.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['calmar']:>7.2f} "
              f"{m['composite']:>7.2f}{marker}")

    # ── Test 5: Max exposure ──
    print(f"\n{'─' * 90}")
    print("  5. MAX PORTFOLIO EXPOSURE")
    print(f"{'─' * 90}")
    print(f"  {'Exposure':>10} {'Return':>9} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Calmar':>8} {'Compos':>8}")

    for exp in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        m = run_backtest(closes, highs, lows, volumes, pairs, min_len,
                         max_exposure=exp)
        marker = " ◀" if exp == 0.60 else ""
        print(f"  {exp:>9.0%}  {m['total_return_pct']:>+8.2f}% {m['max_drawdown_pct']:>7.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['calmar']:>7.2f} "
              f"{m['composite']:>7.2f}{marker}")

    # ── Test 6: Max positions ──
    print(f"\n{'─' * 90}")
    print("  6. MAX SIMULTANEOUS POSITIONS")
    print(f"{'─' * 90}")
    print(f"  {'Positions':>10} {'Return':>9} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Calmar':>8} {'Compos':>8}")

    for pos in [3, 5, 8, 10, 12, 15]:
        m = run_backtest(closes, highs, lows, volumes, pairs, min_len,
                         max_positions=pos)
        marker = " ◀" if pos == 8 else ""
        print(f"  {pos:>10d}  {m['total_return_pct']:>+8.2f}% {m['max_drawdown_pct']:>7.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['calmar']:>7.2f} "
              f"{m['composite']:>7.2f}{marker}")

    # ── Test 7: Drawdown limit ──
    print(f"\n{'─' * 90}")
    print("  7. DRAWDOWN CIRCUIT BREAKER LEVEL")
    print(f"{'─' * 90}")
    print(f"  {'DD Limit':>10} {'Return':>9} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Calmar':>8} {'Compos':>8}")

    for dd_lim in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 1.0]:
        m = run_backtest(closes, highs, lows, volumes, pairs, min_len,
                         drawdown_limit=dd_lim)
        label = "OFF" if dd_lim >= 1.0 else f"{dd_lim:.0%}"
        marker = " ◀" if dd_lim == 0.05 else ""
        print(f"  {label:>10s}  {m['total_return_pct']:>+8.2f}% {m['max_drawdown_pct']:>7.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['calmar']:>7.2f} "
              f"{m['composite']:>7.2f}{marker}")

    print(f"\n  ◀ = current setting")
    print(f"  Recommendation: pick parameters that maximize COMPOSITE score")
    print(f"  (competition metric = 0.4*Sortino + 0.3*Sharpe + 0.3*Calmar)")


if __name__ == "__main__":
    main()
