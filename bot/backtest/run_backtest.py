#!/usr/bin/env python3
"""
Backtest runner using the high-fidelity engine.

Downloads 4 months of 1h data, fetches real Roostoo exchange_info,
and runs rolling 10-day windows + a full continuous test.

Usage:
    venv/bin/python -m bot.backtest.run_backtest
    venv/bin/python -m bot.backtest.run_backtest --months 6
    venv/bin/python -m bot.backtest.run_backtest --window 240 --step 72
"""
import sys
import os
import argparse
import time as pytime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import requests
from bot.config import BINANCE_BASE_URL, BINANCE_SYMBOL_MAP, TRADEABLE_COINS
from bot.backtest.engine import BacktestEngine
from bot.backtest.sim_exchange import MEASURED_SPREAD_BPS
from bot.roostoo_client import RoostooClient


def download_candles(pair: str, session: requests.Session, months: int = 4) -> list[dict]:
    symbol = BINANCE_SYMBOL_MAP.get(pair)
    if not symbol:
        return []
    end_ms = int(pytime.time() * 1000)
    start_ms = end_ms - (months * 30 * 24 * 3_600_000)
    candles = []
    cur = start_ms
    while cur < end_ms:
        try:
            r = session.get(
                f"{BINANCE_BASE_URL}/api/v3/klines",
                params={"symbol": symbol, "interval": "1h",
                        "startTime": cur, "endTime": end_ms, "limit": 1000},
                timeout=15,
            )
            r.raise_for_status()
            raw = r.json()
        except Exception:
            break
        if not raw or not isinstance(raw, list):
            break
        for k in raw:
            candles.append({
                "open_time": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": k[6],
            })
        cur = raw[-1][0] + 3_600_000
        if len(raw) < 1000:
            break
        pytime.sleep(0.05)
    return candles


def fetch_exchange_info() -> dict:
    """Fetch real exchange info from Roostoo API."""
    try:
        client = RoostooClient()
        info = client.exchange_info()
        if info and "TradePairs" in info:
            return info["TradePairs"]
    except Exception:
        pass
    # Fallback: generate from known pairs
    return {p: {"PricePrecision": 4, "AmountPrecision": 2, "MiniOrder": 1.0}
            for p in TRADEABLE_COINS}


def compute_metrics(history: list[float], initial: float) -> dict:
    """Compute competition metrics from portfolio history."""
    if len(history) < 2:
        return {}
    arr = np.array(history)
    final = arr[-1]
    total_ret = (final - initial) / initial

    peak = np.maximum.accumulate(arr)
    dd = (peak - arr) / peak
    max_dd = float(np.max(dd))

    daily = arr[::24]
    dr = np.diff(daily) / daily[:-1] if len(daily) > 1 else np.array([])
    days = len(dr)

    sharpe = float(np.mean(dr) / np.std(dr) * np.sqrt(365)) if len(dr) > 1 and np.std(dr) > 0 else 0
    neg = dr[dr < 0]
    sortino = float(np.mean(dr) / np.std(neg) * np.sqrt(365)) if len(neg) > 0 and np.std(neg) > 0 else (10 if np.mean(dr) > 0 else 0)
    calmar = float((total_ret * 365 / max(days, 1)) / max_dd) if max_dd > 0 else (10 if total_ret > 0 else 0)
    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    return {
        "total_return_pct": round(total_ret * 100, 3),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "composite": round(composite, 3),
        "days": days,
    }


def main():
    parser = argparse.ArgumentParser(description="High-fidelity backtest")
    parser.add_argument("--months", type=int, default=4, help="Months of data to download")
    parser.add_argument("--window", type=int, default=240, help="Rolling window size (bars)")
    parser.add_argument("--step", type=int, default=72, help="Rolling window step (bars)")
    parser.add_argument("--warmup", type=int, default=600, help="Warmup bars (must be > ML_LOOKBACK + ML_FORWARD + 100 = 524)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  HIGH-FIDELITY BACKTEST ({args.months} months, {args.window//24}-day windows)")
    print(f"  Real exchange info, per-pair spreads, correct fees")
    print("=" * 70)

    # Download data
    session = requests.Session()
    session.timeout = 15
    all_candles = {}
    min_len = 999999

    print(f"\n  Downloading {args.months} months of 1h data...")
    pairs = [p for p in TRADEABLE_COINS if p in BINANCE_SYMBOL_MAP]
    for i, pair in enumerate(pairs):
        candles = download_candles(pair, session, args.months)
        if len(candles) < args.warmup + args.window:
            print(f"    [{i+1}/{len(pairs)}] {pair}: {len(candles)} bars — skip")
            continue
        all_candles[pair] = candles
        min_len = min(min_len, len(candles))
        print(f"    [{i+1}/{len(pairs)}] {pair}: {len(candles)} bars ({len(candles)//24}d)")

    # Trim to common length
    all_candles = {p: v[:min_len] for p, v in all_candles.items()}
    print(f"  Loaded {len(all_candles)} pairs, {min_len} bars (~{min_len//24}d)")

    # Fetch real exchange info
    print("  Fetching exchange info from Roostoo...")
    exchange_info = fetch_exchange_info()
    # Filter to pairs we have data for
    exchange_info = {p: exchange_info.get(p, {"PricePrecision": 4, "AmountPrecision": 2, "MiniOrder": 1.0})
                     for p in all_candles}
    print(f"  Exchange info for {len(exchange_info)} pairs")

    # Print spread calibration
    spread_pairs = [(p, MEASURED_SPREAD_BPS.get(p, 8.0)) for p in all_candles]
    spread_pairs.sort(key=lambda x: x[1])
    print(f"  Spread range: {spread_pairs[0][1]:.1f} bps ({spread_pairs[0][0]}) to "
          f"{spread_pairs[-1][1]:.1f} bps ({spread_pairs[-1][0]})")

    engine = BacktestEngine(all_candles, exchange_info)

    # ═══════════════════════════════════════════════════════════
    # TEST 1: Full continuous run
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  TEST 1: FULL CONTINUOUS ({(min_len - args.warmup) // 24} days)")
    print(f"{'=' * 70}")

    full_result = engine.run(start=args.warmup, end=min_len)
    full_m = compute_metrics(full_result.portfolio_history, 1_000_000)

    print(f"\n  Return:     {full_m.get('total_return_pct', 0):>+8.2f}%")
    print(f"  Max DD:     {full_m.get('max_drawdown_pct', 0):>8.2f}%")
    print(f"  Sharpe:     {full_m.get('sharpe', 0):>8.2f}")
    print(f"  Sortino:    {full_m.get('sortino', 0):>8.2f}")
    print(f"  Calmar:     {full_m.get('calmar', 0):>8.2f}")
    print(f"  Composite:  {full_m.get('composite', 0):>8.2f}")
    print(f"  Trades:     {full_result.total_trades:>8}")

    # ═══════════════════════════════════════════════════════════
    # TEST 2: Rolling windows
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  TEST 2: ROLLING {args.window//24}-DAY WINDOWS")
    print(f"{'=' * 70}")

    results = []
    print(f"\n  {'Window':<10} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Calmar':>8} {'Compos':>8} {'Trades':>7}")
    print(f"  {'─' * 70}")

    for s in range(args.warmup, min_len - args.window, args.step):
        e = s + args.window
        r = engine.run(start=s, end=e)
        m = compute_metrics(r.portfolio_history, 1_000_000)
        if not m:
            continue
        m["trades"] = r.total_trades
        results.append(m)
        d = (s - args.warmup) // 24
        print(f"  d{d:>3}-{d + args.window//24:<3} "
              f"{m['total_return_pct']:>+7.2f}% {m['max_drawdown_pct']:>7.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['calmar']:>7.2f} "
              f"{m['composite']:>7.2f} {m['trades']:>6}")

    if results:
        rets = [m["total_return_pct"] for m in results]
        comps = [m["composite"] for m in results]
        dds = [m["max_drawdown_pct"] for m in results]
        print(f"\n  {'=' * 70}")
        print(f"  SUMMARY ({len(results)} windows)")
        print(f"  {'=' * 70}")
        print(f"    Avg return:       {np.mean(rets):>+7.2f}%")
        print(f"    Median return:    {np.median(rets):>+7.2f}%")
        print(f"    Best / Worst:     {np.max(rets):>+7.2f}% / {np.min(rets):>+7.2f}%")
        print(f"    Positive:         {sum(1 for r in rets if r > 0)}/{len(rets)} "
              f"({sum(1 for r in rets if r > 0)/len(rets):.0%})")
        print(f"    Avg max DD:       {np.mean(dds):>7.2f}%")
        print(f"    Avg composite:    {np.mean(comps):>7.2f}")
        print(f"    Median composite: {np.median(comps):>7.2f}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
