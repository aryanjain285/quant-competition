#!/usr/bin/env python3
"""
BACKTEST 2: Signal Comparison
──────────────────────────────
Tests 5 different signal types per coin to find what works best:
1. EMA crossover (12/26)
2. 3-day momentum
3. RSI mean reversion (buy <30, sell >60)
4. 72h breakout
5. Volume-confirmed momentum

For each: reports PnL, trades, win rate, max drawdown.

Run: venv/bin/python -m bot.backtest.bt_signal_comparison
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from bot.backtest.data_loader import load_data, CORE_PAIRS
from bot.signals import ema, rsi


COMMISSION = 0.002  # 0.1% each way


def backtest_single(closes, volumes, entry_fn, exit_fn) -> dict:
    """Run a simple long-only backtest on one coin.

    Returns dict with pnl, trades, win_rate, max_drawdown.
    """
    position = False
    entry_price = 0.0
    total_pnl = 0.0
    trades = 0
    wins = 0
    max_dd = 0.0
    peak_equity = 1.0
    equity = 1.0

    for i in range(100, len(closes)):
        if not position and entry_fn(closes, volumes, i):
            position = True
            entry_price = closes[i]
            trades += 1
        elif position and exit_fn(closes, volumes, i, entry_price):
            pnl = (closes[i] / entry_price) - 1 - COMMISSION
            total_pnl += pnl
            equity *= (1 + pnl)
            if pnl > 0:
                wins += 1
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity
            max_dd = max(max_dd, dd)
            position = False

    # Close any open position
    if position:
        pnl = (closes[-1] / entry_price) - 1 - COMMISSION
        total_pnl += pnl
        equity *= (1 + pnl)
        if pnl > 0:
            wins += 1
        trades += 1

    return {
        "pnl": total_pnl,
        "trades": trades,
        "win_rate": wins / trades if trades > 0 else 0,
        "max_dd": max_dd,
    }


def main():
    data = load_data(CORE_PAIRS, interval="1h", limit=1000)
    closes = data["closes"]
    volumes = data["volumes"]
    pairs = data["pairs"]

    # ── Define strategies ──
    strategies = {
        "EMA 12/26": {
            "entry": lambda c, v, i: (
                len(c[:i+1]) > 26 and
                ema(c[:i+1], 12)[-1] > ema(c[:i+1], 26)[-1] and
                ema(c[:i], 12)[-1] <= ema(c[:i], 26)[-1]
            ),
            "exit": lambda c, v, i, ep: (
                ema(c[:i+1], 12)[-1] < ema(c[:i+1], 26)[-1] or
                (c[i] / ep - 1) < -0.03
            ),
        },
        "3d Momentum": {
            "entry": lambda c, v, i: i >= 72 and (c[i] / c[i - 72] - 1) > 0.03,
            "exit": lambda c, v, i, ep: (
                (i >= 72 and (c[i] / c[i - 72] - 1) < -0.01) or
                (c[i] / ep - 1) < -0.03
            ),
        },
        "RSI Mean-Rev": {
            "entry": lambda c, v, i: rsi(c[:i+1], 14) < 30,
            "exit": lambda c, v, i, ep: (
                rsi(c[:i+1], 14) > 60 or
                (c[i] / ep - 1) < -0.03
            ),
        },
        "72h Breakout": {
            "entry": lambda c, v, i: (
                i >= 72 and c[i] >= np.max(c[i - 72:i])
            ),
            "exit": lambda c, v, i, ep: (
                (i >= 24 and c[i] <= np.min(c[i - 24:i])) or
                (c[i] / ep - 1) < -0.03
            ),
        },
        "Vol+Momentum": {
            "entry": lambda c, v, i: (
                i >= 72 and
                (c[i] / c[i - 72] - 1) > 0.02 and
                len(v) > i and i >= 48 and
                v[i] > 1.3 * np.mean(v[i - 48:i])
            ),
            "exit": lambda c, v, i, ep: (
                (i >= 72 and (c[i] / c[i - 72] - 1) < -0.01) or
                (c[i] / ep - 1) < -0.03
            ),
        },
    }

    # ── Run backtests ──
    print("\n" + "=" * 100)
    print("  SIGNAL COMPARISON BACKTEST (hourly data)")
    print("=" * 100)

    # Per-coin results
    all_results = {}  # strategy -> list of (pair, result)
    for strat_name in strategies:
        all_results[strat_name] = []

    for pair in sorted(pairs):
        c = closes[pair]
        v = volumes[pair]

        print(f"\n  {pair}:")
        print(f"  {'Strategy':<18} {'PnL':>8} {'Trades':>7} {'WinR':>6} {'MaxDD':>7}")
        print(f"  {'─' * 50}")

        for strat_name, strat in strategies.items():
            result = backtest_single(c, v, strat["entry"], strat["exit"])
            all_results[strat_name].append((pair, result))

            # Color coding via symbols
            pnl_sym = "+" if result["pnl"] > 0.01 else ("-" if result["pnl"] < -0.01 else "~")
            print(f"  {strat_name:<18} {result['pnl']:>+7.1%} {result['trades']:>6d} "
                  f"{result['win_rate']:>5.0%} {result['max_dd']:>6.1%}  {pnl_sym}")

    # ── Aggregate summary ──
    print(f"\n\n{'=' * 100}")
    print("  AGGREGATE SUMMARY ACROSS ALL COINS")
    print("=" * 100)
    print(f"\n  {'Strategy':<18} {'Avg PnL':>9} {'Med PnL':>9} {'Win Coins':>10} "
          f"{'Avg WinR':>9} {'Avg MaxDD':>10} {'Best Coin':>16} {'Worst Coin':>16}")
    print(f"  {'─' * 100}")

    for strat_name, results in all_results.items():
        pnls = [r["pnl"] for _, r in results]
        wrs = [r["win_rate"] for _, r in results]
        dds = [r["max_dd"] for _, r in results]
        win_coins = sum(1 for p in pnls if p > 0)

        best = max(results, key=lambda x: x[1]["pnl"])
        worst = min(results, key=lambda x: x[1]["pnl"])

        print(f"  {strat_name:<18} {np.mean(pnls):>+8.1%} {np.median(pnls):>+8.1%} "
              f"{win_coins:>5}/{len(results):<4} {np.mean(wrs):>8.0%} {np.mean(dds):>9.1%} "
              f"{best[0]:>12}({best[1]['pnl']:>+.0%}) "
              f"{worst[0]:>12}({worst[1]['pnl']:>+.0%})")

    # ── Best strategy per coin ──
    print(f"\n\n{'=' * 100}")
    print("  BEST STRATEGY PER COIN")
    print("=" * 100)
    for pair in sorted(pairs):
        best_strat = None
        best_pnl = -999
        for strat_name, results in all_results.items():
            for p, r in results:
                if p == pair and r["pnl"] > best_pnl:
                    best_pnl = r["pnl"]
                    best_strat = strat_name
        print(f"  {pair:<14} → {best_strat:<18} ({best_pnl:>+.1%})")


if __name__ == "__main__":
    main()
