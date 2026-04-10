#!/usr/bin/env python3
"""
Split backtest runner for sizing calibration.

Downloads a contiguous data sample, then evaluates:
  - in-sample segment: first `is_months`
  - out-of-sample segment: remaining months

Each segment reports:
  - full continuous metrics
  - rolling-window summary within that segment

Typical usage:
    python -m bot.backtest.run_split_backtest --months 6 --is-months 4
"""
import argparse
import json
import time as pytime
from pathlib import Path

import numpy as np
import requests

from bot.backtest.engine import BacktestEngine
from bot.backtest.run_backtest import download_candles, fetch_exchange_info, compute_metrics
from bot.backtest.sim_exchange import MEASURED_SPREAD_BPS
from bot.config import BINANCE_SYMBOL_MAP, TRADEABLE_COINS


def rolling_summary(engine: BacktestEngine, start: int, end: int, window: int, step: int) -> dict:
    results = []
    for s in range(start, end - window + 1, step):
        r = engine.run(start=s, end=s + window)
        m = compute_metrics(r.portfolio_history, 1_000_000)
        if not m:
            continue
        m["trades"] = r.total_trades
        m["window_start_bar"] = s
        m["window_end_bar"] = s + window
        results.append(m)

    if not results:
        return {"window_count": 0, "windows": []}

    rets = [m["total_return_pct"] for m in results]
    dds = [m["max_drawdown_pct"] for m in results]
    comps = [m["composite"] for m in results]
    trades = [m["trades"] for m in results]

    return {
        "window_count": len(results),
        "avg_return_pct": round(float(np.mean(rets)), 3),
        "median_return_pct": round(float(np.median(rets)), 3),
        "best_return_pct": round(float(np.max(rets)), 3),
        "worst_return_pct": round(float(np.min(rets)), 3),
        "positive_windows": int(sum(1 for r in rets if r > 0)),
        "avg_max_dd_pct": round(float(np.mean(dds)), 3),
        "avg_composite": round(float(np.mean(comps)), 3),
        "median_composite": round(float(np.median(comps)), 3),
        "avg_trades": round(float(np.mean(trades)), 3),
        "windows": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Split backtest for in-sample / out-of-sample evaluation")
    parser.add_argument("--months", type=int, default=6, help="Total months of data to download")
    parser.add_argument("--is-months", type=int, default=4, help="Months allocated to in-sample")
    parser.add_argument("--warmup", type=int, default=600, help="Warmup bars before in-sample starts")
    parser.add_argument("--window", type=int, default=240, help="Rolling window size in bars")
    parser.add_argument("--step", type=int, default=72, help="Rolling window step in bars")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    if args.is_months >= args.months:
        raise SystemExit("--is-months must be smaller than --months")

    session = requests.Session()
    all_candles = {}
    min_len = 999999

    print(f"Downloading {args.months} months of 1h data...")
    pairs = [p for p in TRADEABLE_COINS if p in BINANCE_SYMBOL_MAP]
    for pair in pairs:
        candles = download_candles(pair, session, args.months)
        if len(candles) < args.warmup + args.window:
            continue
        all_candles[pair] = candles
        min_len = min(min_len, len(candles))
        pytime.sleep(0.05)

    all_candles = {p: v[:min_len] for p, v in all_candles.items()}
    exchange_info = fetch_exchange_info()
    exchange_info = {
        p: exchange_info.get(p, {"PricePrecision": 4, "AmountPrecision": 2, "MiniOrder": 1.0})
        for p in all_candles
    }

    spread_pairs = sorted((p, MEASURED_SPREAD_BPS.get(p, 8.0)) for p in all_candles)
    engine = BacktestEngine(all_candles, exchange_info)

    split_bar = args.is_months * 30 * 24
    if min_len < split_bar + args.window:
        raise SystemExit(f"Not enough bars ({min_len}) for requested split")

    is_start = args.warmup
    is_end = split_bar
    oos_start = split_bar
    oos_end = min_len

    is_result = engine.run(start=is_start, end=is_end)
    oos_result = engine.run(start=oos_start, end=oos_end)

    report = {
        "months": args.months,
        "is_months": args.is_months,
        "loaded_pairs": len(all_candles),
        "bars": min_len,
        "split_bar": split_bar,
        "spread_min_bps": {"pair": spread_pairs[0][0], "bps": spread_pairs[0][1]} if spread_pairs else None,
        "spread_max_bps": {"pair": spread_pairs[-1][0], "bps": spread_pairs[-1][1]} if spread_pairs else None,
        "in_sample": {
            "full": {
                **compute_metrics(is_result.portfolio_history, 1_000_000),
                "trades": is_result.total_trades,
                "final_value": is_result.final_value,
            },
            "rolling": rolling_summary(engine, is_start, is_end, args.window, args.step),
        },
        "out_of_sample": {
            "full": {
                **compute_metrics(oos_result.portfolio_history, 1_000_000),
                "trades": oos_result.total_trades,
                "final_value": oos_result.final_value,
            },
            "rolling": rolling_summary(engine, oos_start, oos_end, args.window, args.step),
        },
    }

    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Wrote {args.output}")
    print(output)


if __name__ == "__main__":
    main()
