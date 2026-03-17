#!/usr/bin/env python3
"""
BACKTEST 1: Market Structure Analysis
──────────────────────────────────────
Analyzes the current market to understand:
- Which coins are trending vs choppy
- Volatility regimes
- Momentum persistence (does past predict future?)
- Cross-sectional rankings
- Volume patterns

Run: venv/bin/python -m bot.backtest.bt_market_analysis
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from bot.backtest.data_loader import load_data, CORE_PAIRS
from bot.signals import ema, rsi, realized_volatility, downside_volatility


def main():
    data = load_data(CORE_PAIRS, interval="1h", limit=1000)
    closes = data["closes"]
    volumes = data["volumes"]
    pairs = data["pairs"]
    min_len = data["min_len"]

    print("\n" + "=" * 90)
    print("  MARKET STRUCTURE ANALYSIS")
    print("=" * 90)

    # ── Per-coin analysis ──
    print(f"\n{'Pair':<16} {'1d':>7} {'3d':>7} {'7d':>7} {'14d':>7} {'30d':>7} "
          f"{'Vol7d':>7} {'MaxDD':>7} {'AutoC':>7} {'RSI':>5}")
    print("─" * 90)

    coin_data = []
    for pair in sorted(pairs):
        c = closes[pair]
        v = volumes[pair]

        # Returns
        r1d = (c[-1] / c[-25] - 1) if len(c) > 25 else 0
        r3d = (c[-1] / c[-73] - 1) if len(c) > 73 else 0
        r7d = (c[-1] / c[-169] - 1) if len(c) > 169 else 0
        r14d = (c[-1] / c[-337] - 1) if len(c) > 337 else 0
        r30d = (c[-1] / c[-721] - 1) if len(c) > 721 else 0

        # 7-day vol (annualized)
        log_rets = np.diff(np.log(c[-169:])) if len(c) > 169 else np.array([])
        vol_7d = float(np.std(log_rets) * np.sqrt(24 * 365)) if len(log_rets) > 0 else 0

        # Max drawdown (30d)
        window = c[-720:] if len(c) >= 720 else c
        peak = np.maximum.accumulate(window)
        dd = (peak - window) / peak
        max_dd = float(np.max(dd))

        # Autocorrelation of hourly returns (momentum persistence)
        rets = np.diff(c[-169:]) / c[-169:-1] if len(c) > 169 else np.array([])
        autocorr = float(np.corrcoef(rets[:-1], rets[1:])[0, 1]) if len(rets) > 2 else 0

        # RSI
        current_rsi = rsi(c, 14)

        # Trend classification
        all_up = r1d > 0 and r3d > 0 and r7d > 0
        all_down = r1d < 0 and r3d < 0 and r7d < 0

        print(f"{pair:<16} {r1d:>+6.1%} {r3d:>+6.1%} {r7d:>+6.1%} {r14d:>+6.1%} {r30d:>+6.1%} "
              f"{vol_7d:>6.0%} {max_dd:>6.1%} {autocorr:>+6.3f} {current_rsi:>5.0f}")

        coin_data.append({
            "pair": pair, "r1d": r1d, "r3d": r3d, "r7d": r7d,
            "vol_7d": vol_7d, "max_dd": max_dd, "autocorr": autocorr, "rsi": current_rsi,
        })

    # ── Cross-sectional ranking ──
    print(f"\n{'=' * 90}")
    print("  CROSS-SECTIONAL RANKINGS (7-day returns)")
    print("─" * 90)
    ranked = sorted(coin_data, key=lambda x: x["r7d"], reverse=True)
    for i, d in enumerate(ranked):
        trend = "UP  " if d["r1d"] > 0 and d["r3d"] > 0 else ("DOWN" if d["r1d"] < 0 and d["r3d"] < 0 else "MIX ")
        bar = "█" * max(0, int(d["r7d"] * 100))
        print(f"  {i+1:>2}. {d['pair']:<14} 7d={d['r7d']:>+6.1%}  [{trend}]  {bar}")

    # ── Momentum persistence test ──
    print(f"\n{'=' * 90}")
    print("  MOMENTUM PERSISTENCE TEST")
    print("  Does past 24h return predict next 24h direction?")
    print("─" * 90)
    for pair in ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "LINK/USD", "FET/USD", "ZEC/USD"]:
        if pair not in closes:
            continue
        c = closes[pair]
        correct, total = 0, 0
        for i in range(48, len(c) - 24, 24):
            past = (c[i] / c[i - 24]) - 1
            future = (c[i + 24] / c[i]) - 1
            if past != 0 and future != 0:
                if np.sign(past) == np.sign(future):
                    correct += 1
                total += 1
        hit = correct / total if total > 0 else 0
        verdict = "TRENDING" if hit > 0.55 else ("REVERTING" if hit < 0.45 else "RANDOM")
        print(f"  {pair:<14} {hit:>5.0%} hit rate ({total} samples) → {verdict}")

    # ── Volatility regime ──
    print(f"\n{'=' * 90}")
    print("  VOLATILITY REGIME")
    print("─" * 90)
    vols = [d["vol_7d"] for d in coin_data if d["vol_7d"] > 0]
    median_vol = np.median(vols)
    print(f"  Median annualized vol: {median_vol:.0%}")
    print(f"  Low vol coins (<{median_vol:.0%}):")
    for d in sorted(coin_data, key=lambda x: x["vol_7d"]):
        if d["vol_7d"] < median_vol and d["vol_7d"] > 0:
            print(f"    {d['pair']:<14} vol={d['vol_7d']:.0%}")
    print(f"  High vol coins (>{median_vol:.0%}):")
    for d in sorted(coin_data, key=lambda x: x["vol_7d"], reverse=True):
        if d["vol_7d"] >= median_vol:
            print(f"    {d['pair']:<14} vol={d['vol_7d']:.0%}")

    # ── RSI distribution ──
    print(f"\n{'=' * 90}")
    print("  RSI DISTRIBUTION (potential mean-reversion entries)")
    print("─" * 90)
    oversold = [d for d in coin_data if d["rsi"] < 30]
    overbought = [d for d in coin_data if d["rsi"] > 70]
    neutral = [d for d in coin_data if 30 <= d["rsi"] <= 70]
    print(f"  Oversold  (RSI < 30): {len(oversold)} coins")
    for d in sorted(oversold, key=lambda x: x["rsi"]):
        print(f"    {d['pair']:<14} RSI={d['rsi']:.0f}")
    print(f"  Overbought (RSI > 70): {len(overbought)} coins")
    for d in sorted(overbought, key=lambda x: x["rsi"], reverse=True):
        print(f"    {d['pair']:<14} RSI={d['rsi']:.0f}")
    print(f"  Neutral (30-70): {len(neutral)} coins")


if __name__ == "__main__":
    main()
