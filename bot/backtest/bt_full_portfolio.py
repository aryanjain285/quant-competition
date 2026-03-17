#!/usr/bin/env python3
"""
BACKTEST 3: Full Portfolio Simulation
─────────────────────────────────────
Simulates the complete dual-engine strategy with:
- Breakout + RSI mean reversion signals
- Volatility-parity position sizing
- Strategy-specific trailing stops
- Drawdown circuit breakers
- Portfolio-level risk management

Runs over the full available history AND rolling 8/10-day windows
to match the actual competition format.

Run: venv/bin/python -m bot.backtest.bt_full_portfolio
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from bot.backtest.data_loader import load_data, compute_metrics, print_metrics, CORE_PAIRS
from bot.signals import compute_signal


# ─── Configuration (mirrors live bot) ───
INITIAL = 1_000_000
COMMISSION_MARKET = 0.001    # 0.1% taker
COMMISSION_LIMIT = 0.0005    # 0.05% maker
MAX_POSITIONS = 8
MAX_EXPOSURE_PCT = 0.60
MAX_POSITION_PCT = 0.20
BREAKOUT_LOOKBACK = 72       # hours

# Strategy-specific stops
STOPS = {
    "breakout": {"trail": 0.03, "hard": -0.04},
    "mean_rev": {"trail": 0.02, "hard": -0.03, "profit_target": 0.03},
}

# Drawdown breaker
DRAWDOWN_LIQUIDATE = 0.05


def run_portfolio_backtest(
    closes: dict, highs: dict, lows: dict, volumes: dict,
    pairs: list, start: int, end: int,
    verbose: bool = False,
) -> dict:
    """Run a full portfolio backtest over a range of bars.

    Returns dict with:
        portfolio_history, trades, wins, final_value,
        strategy_counts, trade_log
    """
    cash = INITIAL
    positions = {}   # pair -> {qty, entry, high, strategy}
    portfolio_history = []
    peak = INITIAL
    trades = 0
    wins = 0
    strategy_counts = {"breakout": 0, "mean_rev": 0}
    trade_log = []

    for t in range(start, end):
        prices = {p: closes[p][t] for p in pairs if t < len(closes[p])}
        pv = cash + sum(pos["qty"] * prices.get(p, 0) for p, pos in positions.items())
        portfolio_history.append(pv)

        if pv > peak:
            peak = pv
        dd = (peak - pv) / peak

        # ── Drawdown breaker ──
        if dd > DRAWDOWN_LIQUIDATE and positions:
            for pair, pos in positions.items():
                p = prices.get(pair, pos["entry"])
                cash += pos["qty"] * p * (1 - COMMISSION_MARKET)
                trades += 1
                if p > pos["entry"]:
                    wins += 1
                trade_log.append({"pair": pair, "side": "SELL", "reason": "drawdown_breaker",
                                  "pnl": (p / pos["entry"] - 1)})
            positions.clear()
            if verbose:
                print(f"  [bar {t}] DRAWDOWN BREAKER fired at {dd:.1%}, liquidated all")
            continue

        # ── Strategy-specific trailing stops ──
        for pair in list(positions.keys()):
            pos = positions[pair]
            p = prices.get(pair, 0)
            if p <= 0:
                continue
            if p > pos["high"]:
                pos["high"] = p

            stop = STOPS.get(pos["strategy"], STOPS["breakout"])
            drop_from_high = (pos["high"] - p) / pos["high"]
            pnl_from_entry = (p - pos["entry"]) / pos["entry"]

            should_exit = False
            reason = ""
            if drop_from_high >= stop["trail"]:
                should_exit = True
                reason = "trailing_stop"
            elif pnl_from_entry <= stop["hard"]:
                should_exit = True
                reason = "hard_stop"
            elif pos["strategy"] == "mean_rev" and pnl_from_entry >= stop.get("profit_target", 999):
                should_exit = True
                reason = "profit_target"

            if should_exit:
                cash += pos["qty"] * p * (1 - COMMISSION_MARKET)
                trades += 1
                if p > pos["entry"]:
                    wins += 1
                trade_log.append({"pair": pair, "side": "SELL", "reason": reason,
                                  "pnl": pnl_from_entry, "strategy": pos["strategy"]})
                if verbose and reason != "trailing_stop":
                    print(f"  [bar {t}] EXIT {pair} via {reason} pnl={pnl_from_entry:+.2%}")
                del positions[pair]

        # ── Compute signals ──
        sigs = {}
        for pair in pairs:
            if t >= len(closes[pair]):
                continue
            c = closes[pair][:t + 1]
            h = highs[pair][:t + 1]
            l = lows[pair][:t + 1]
            v = volumes[pair][:t + 1]
            if len(c) < 80:
                continue
            sigs[pair] = compute_signal(
                c, h, l, v,
                prices[pair] * 0.9999, prices[pair] * 1.0001,
                BREAKOUT_LOOKBACK,
            )

        # ── Signal-based sells ──
        for pair in list(positions.keys()):
            sig = sigs.get(pair, {})
            if sig.get("action") == "SELL" or sig.get("breakdown"):
                p = prices.get(pair, 0)
                pos = positions[pair]
                pnl = (p / pos["entry"]) - 1
                cash += pos["qty"] * p * (1 - COMMISSION_MARKET)
                trades += 1
                if p > pos["entry"]:
                    wins += 1
                trade_log.append({"pair": pair, "side": "SELL", "reason": "signal",
                                  "pnl": pnl, "strategy": pos["strategy"]})
                del positions[pair]

        # ── Signal-based buys (strongest first) ──
        buy_sigs = [
            (p, s) for p, s in sigs.items()
            if s["action"] == "BUY" and p not in positions
        ]
        buy_sigs.sort(key=lambda x: x[1]["strength"], reverse=True)

        exposure = sum(pos["qty"] * prices.get(p, 0) for p, pos in positions.items())

        for pair, sig in buy_sigs:
            if len(positions) >= MAX_POSITIONS:
                break
            remaining = MAX_EXPOSURE_PCT * pv - exposure
            if remaining <= 0:
                break

            # Volatility-parity sizing
            vol = sig.get("real_vol", 0.5) or 0.5
            daily_vol = vol / np.sqrt(365)
            if daily_vol <= 0:
                daily_vol = 0.03
            size = min(0.015 * pv / daily_vol, MAX_POSITION_PCT * pv, remaining)
            size *= sig["strength"]

            if size < 100:
                continue

            p = prices[pair]
            qty = size / p
            cost = qty * p * (1 + COMMISSION_LIMIT)
            if cost > cash:
                continue

            cash -= cost
            positions[pair] = {
                "qty": qty, "entry": p, "high": p, "strategy": sig["strategy"],
            }
            exposure += qty * p
            trades += 1
            strategy_counts[sig["strategy"]] = strategy_counts.get(sig["strategy"], 0) + 1
            trade_log.append({"pair": pair, "side": "BUY", "reason": "signal",
                              "strategy": sig["strategy"], "size_usd": size})

            if verbose:
                print(f"  [bar {t}] BUY {pair} [{sig['strategy']}] "
                      f"str={sig['strength']:.2f} size=${size:,.0f}")

    # ── Close remaining positions ──
    for pair, pos in positions.items():
        p = closes[pair][min(end - 1, len(closes[pair]) - 1)]
        cash += pos["qty"] * p * (1 - COMMISSION_MARKET)
        pnl = (p / pos["entry"]) - 1
        trades += 1
        if p > pos["entry"]:
            wins += 1
        trade_log.append({"pair": pair, "side": "SELL", "reason": "end_of_backtest",
                          "pnl": pnl, "strategy": pos["strategy"]})

    return {
        "portfolio_history": portfolio_history,
        "final_value": cash,
        "trades": trades,
        "wins": wins,
        "strategy_counts": strategy_counts,
        "trade_log": trade_log,
    }


def main():
    data = load_data(CORE_PAIRS, interval="1h", limit=1000)
    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    volumes = data["volumes"]
    pairs = data["pairs"]
    min_len = data["min_len"]

    # ═══════════════════════════════════════════════════
    # TEST 1: Full period backtest
    # ═══════════════════════════════════════════════════
    print("\n\n" + "▓" * 70)
    print("  TEST 1: FULL PERIOD BACKTEST")
    print("▓" * 70)

    result = run_portfolio_backtest(closes, highs, lows, volumes, pairs, 100, min_len)
    metrics = compute_metrics(result["portfolio_history"], INITIAL, daily_sample_rate=24)
    metrics["trades"] = result["trades"]
    metrics["win_rate"] = f"{result['wins']}/{result['trades']} ({result['wins']/max(result['trades'],1):.0%})"
    metrics["strategies"] = result["strategy_counts"]

    print_metrics(metrics, f"FULL PERIOD ({metrics['days']} days, {len(pairs)} pairs)")
    print(f"  Trades:           {metrics['trades']:>13d}")
    print(f"  Win Rate:         {metrics['win_rate']:>13s}")
    print(f"  Strategies:       breakout={result['strategy_counts'].get('breakout',0)}, "
          f"mean_rev={result['strategy_counts'].get('mean_rev',0)}")

    # Trade analysis
    trade_log = result["trade_log"]
    sells = [t for t in trade_log if t["side"] == "SELL" and "pnl" in t]
    if sells:
        pnls = [t["pnl"] for t in sells]
        print(f"\n  Trade PnL Distribution:")
        print(f"    Avg:    {np.mean(pnls):>+7.2%}")
        print(f"    Median: {np.median(pnls):>+7.2%}")
        print(f"    Best:   {np.max(pnls):>+7.2%}")
        print(f"    Worst:  {np.min(pnls):>+7.2%}")

        # By exit reason
        reasons = {}
        for t in sells:
            r = t.get("reason", "unknown")
            if r not in reasons:
                reasons[r] = []
            reasons[r].append(t["pnl"])
        print(f"\n  Exit Reasons:")
        for reason, rpnls in sorted(reasons.items()):
            print(f"    {reason:<20} count={len(rpnls):>4} avg_pnl={np.mean(rpnls):>+6.2%}")

    # ═══════════════════════════════════════════════════
    # TEST 2: Rolling 10-day windows (competition format)
    # ═══════════════════════════════════════════════════
    print("\n\n" + "▓" * 70)
    print("  TEST 2: ROLLING 10-DAY WINDOWS (competition format)")
    print("▓" * 70)

    window_10d = 240  # 10 days * 24 hours
    step = 72         # slide by 3 days

    results_10d = []
    print(f"\n  {'Window':<10} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Calmar':>8} {'Compos':>8} {'Trades':>7}")
    print(f"  {'─' * 70}")

    for start in range(100, min_len - window_10d, step):
        end = start + window_10d
        r = run_portfolio_backtest(closes, highs, lows, volumes, pairs, start, end)
        m = compute_metrics(r["portfolio_history"], INITIAL, daily_sample_rate=24)
        if not m:
            continue
        m["trades"] = r["trades"]
        m["wins"] = r["wins"]
        results_10d.append(m)

        day_start = (start - 100) // 24
        print(f"  d{day_start:>3}-{day_start+10:<3} {m['total_return_pct']:>+7.2f}% "
              f"{m['max_drawdown_pct']:>7.2f}% {m['sharpe']:>7.2f} "
              f"{m['sortino']:>7.2f} {m['calmar']:>7.2f} {m['composite']:>7.2f} "
              f"{m['trades']:>6d}")

    if results_10d:
        rets = [m["total_return_pct"] for m in results_10d]
        dds = [m["max_drawdown_pct"] for m in results_10d]
        comps = [m["composite"] for m in results_10d]
        print(f"\n  SUMMARY ({len(results_10d)} windows):")
        print(f"    Avg return:      {np.mean(rets):>+7.2f}%")
        print(f"    Median return:   {np.median(rets):>+7.2f}%")
        print(f"    Best return:     {np.max(rets):>+7.2f}%")
        print(f"    Worst return:    {np.min(rets):>+7.2f}%")
        print(f"    Positive windows: {sum(1 for r in rets if r > 0)}/{len(rets)} "
              f"({sum(1 for r in rets if r > 0)/len(rets):.0%})")
        print(f"    Avg max DD:      {np.mean(dds):>7.2f}%")
        print(f"    Worst max DD:    {np.max(dds):>7.2f}%")
        print(f"    Avg composite:   {np.mean(comps):>7.2f}")

    # ═══════════════════════════════════════════════════
    # TEST 3: Rolling 8-day windows (minimum active days)
    # ═══════════════════════════════════════════════════
    print("\n\n" + "▓" * 70)
    print("  TEST 3: ROLLING 8-DAY WINDOWS (min active trading requirement)")
    print("▓" * 70)

    window_8d = 192  # 8 days * 24 hours

    results_8d = []
    print(f"\n  {'Window':<10} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Calmar':>8} {'Compos':>8} {'Trades':>7}")
    print(f"  {'─' * 70}")

    for start in range(100, min_len - window_8d, step):
        end = start + window_8d
        r = run_portfolio_backtest(closes, highs, lows, volumes, pairs, start, end)
        m = compute_metrics(r["portfolio_history"], INITIAL, daily_sample_rate=24)
        if not m:
            continue
        m["trades"] = r["trades"]
        results_8d.append(m)

        day_start = (start - 100) // 24
        print(f"  d{day_start:>3}-{day_start+8:<3} {m['total_return_pct']:>+7.2f}% "
              f"{m['max_drawdown_pct']:>7.2f}% {m['sharpe']:>7.2f} "
              f"{m['sortino']:>7.2f} {m['calmar']:>7.2f} {m['composite']:>7.2f} "
              f"{m['trades']:>6d}")

    if results_8d:
        rets = [m["total_return_pct"] for m in results_8d]
        dds = [m["max_drawdown_pct"] for m in results_8d]
        comps = [m["composite"] for m in results_8d]
        print(f"\n  SUMMARY ({len(results_8d)} windows):")
        print(f"    Avg return:      {np.mean(rets):>+7.2f}%")
        print(f"    Median return:   {np.median(rets):>+7.2f}%")
        print(f"    Best return:     {np.max(rets):>+7.2f}%")
        print(f"    Worst return:    {np.min(rets):>+7.2f}%")
        print(f"    Positive windows: {sum(1 for r in rets if r > 0)}/{len(rets)} "
              f"({sum(1 for r in rets if r > 0)/len(rets):.0%})")
        print(f"    Avg max DD:      {np.mean(dds):>7.2f}%")
        print(f"    Avg composite:   {np.mean(comps):>7.2f}")

    # ═══════════════════════════════════════════════════
    # TEST 4: Worst-case stress test
    # ═══════════════════════════════════════════════════
    print("\n\n" + "▓" * 70)
    print("  TEST 4: WORST-CASE ANALYSIS")
    print("▓" * 70)

    if results_10d:
        worst_idx = np.argmin([m["total_return_pct"] for m in results_10d])
        worst = results_10d[worst_idx]
        print(f"\n  Worst 10-day window:")
        print(f"    Return:    {worst['total_return_pct']:>+7.2f}%")
        print(f"    Max DD:    {worst['max_drawdown_pct']:>7.2f}%")
        print(f"    Sharpe:    {worst['sharpe']:>7.2f}")
        print(f"    Sortino:   {worst['sortino']:>7.2f}")
        print(f"    Calmar:    {worst['calmar']:>7.2f}")
        print(f"    Composite: {worst['composite']:>7.2f}")
        print(f"    Trades:    {worst['trades']:>7d}")

        # How bad could it get?
        all_dds = [m["max_drawdown_pct"] for m in results_10d]
        print(f"\n  Drawdown distribution across 10-day windows:")
        print(f"    Min:    {np.min(all_dds):>6.2f}%")
        print(f"    25th:   {np.percentile(all_dds, 25):>6.2f}%")
        print(f"    Median: {np.median(all_dds):>6.2f}%")
        print(f"    75th:   {np.percentile(all_dds, 75):>6.2f}%")
        print(f"    Max:    {np.max(all_dds):>6.2f}%")


if __name__ == "__main__":
    main()
