#!/usr/bin/env python3
"""
BACKTEST v3: Matches live bot exactly.
Changes from v2:
  - Breakout requires volume confirmation (mandatory)
  - RSI oversold tightened to 25
  - Regime filter REMOVED for breakout suppression (kept for exposure sizing)
  - Non-urgent exits use limit order commission (0.05% vs 0.1%)
  - Widened drawdown breakers

Run: venv/bin/python -m bot.backtest.bt_integrated
"""
import sys
import os
import time as _time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import requests
from bot.backtest.data_loader import compute_metrics, print_metrics, CORE_PAIRS
from bot.signals import compute_signal
from bot.config import (
    BINANCE_FUTURES_URL, BINANCE_SYMBOL_MAP,
    MAX_POSITIONS, MAX_TOTAL_EXPOSURE_PCT, MAX_POSITION_PCT,
    TARGET_RISK_PER_TRADE,
    DRAWDOWN_LEVEL_1, DRAWDOWN_LEVEL_2, DRAWDOWN_LEVEL_3,
)
from bot.binance_data import BinanceData
from bot.logger import get_logger

log = get_logger("bt_integrated")

INITIAL = 1_000_000
COMMISSION_TAKER = 0.001   # 0.1% market orders
COMMISSION_MAKER = 0.0005  # 0.05% limit orders
BREAKOUT_LOOKBACK = 72


def detect_regime(btc_closes, idx, vol_window=24, baseline_window=72):
    if idx < baseline_window + 1:
        return 1.0, "TRENDING"
    lr = np.diff(np.log(btc_closes[idx - baseline_window:idx + 1]))
    if len(lr) < vol_window:
        return 1.0, "TRENDING"
    recent_vol = np.std(lr[-vol_window:])
    baseline_vol = np.std(lr)
    if baseline_vol <= 0:
        return 1.0, "TRENDING"
    if recent_vol > 1.5 * baseline_vol:
        confidence = min(1.0, recent_vol / (2 * baseline_vol))
        return max(0.3, 0.6 - 0.3 * confidence), "VOLATILE"
    return 1.0, "TRENDING"


def download_funding_rates(pairs):
    session = requests.Session()
    session.timeout = 10
    all_funding = {}
    for pair in pairs:
        sym = BINANCE_SYMBOL_MAP.get(pair)
        if not sym:
            continue
        try:
            r = session.get(f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate",
                            params={"symbol": sym, "limit": 1000})
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                all_funding[pair] = [
                    {"t": d["fundingTime"], "rate": float(d["fundingRate"])}
                    for d in data
                ]
        except Exception:
            pass
        _time.sleep(0.05)
    return all_funding


def compute_funding_zscore(funding_list, timestamp, window=30):
    rates = []
    for f in funding_list:
        if f["t"] <= timestamp:
            rates.append(f["rate"])
    if len(rates) < 5:
        return 0.0
    recent = rates[-window:] if len(rates) >= window else rates
    mean_r = np.mean(recent)
    std_r = np.std(recent)
    if std_r <= 0:
        return 0.0
    return (rates[-1] - mean_r) / std_r


def redd_multiplier(dd):
    if dd <= 0:
        return 1.0
    return max(0.0, 1.0 - (dd / 0.10))  # matches risk_manager.py


def run_simulation(
    closes, highs, lows, volumes, timestamps,
    pairs, btc_closes, btc_timestamps,
    funding_data,
    start, end,
    verbose=False,
):
    cash = INITIAL
    positions = {}
    portfolio_history = []
    peak = INITIAL
    trades = 0
    wins = 0
    trade_log = []
    dd_level = 0
    pause_until_bar = 0

    for t in range(start, end):
        prices = {}
        for p in pairs:
            if t < len(closes[p]):
                prices[p] = closes[p][t]

        pv = cash + sum(pos["qty"] * prices.get(p, 0) for p, pos in positions.items())
        portfolio_history.append(pv)
        if pv > peak:
            peak = pv
        dd = (peak - pv) / peak if peak > 0 else 0

        # ── Drawdown breakers (widened) ──
        if dd >= DRAWDOWN_LEVEL_3 and dd_level < 3:
            dd_level = 3
            pause_until_bar = t + 12
            for pair, pos in positions.items():
                p = prices.get(pair, pos["entry"])
                cash += pos["qty"] * p * (1 - COMMISSION_TAKER)
                trades += 1
                if p > pos["entry"]: wins += 1
                trade_log.append({"pair": pair, "reason": "dd_level3", "pnl": (p / pos["entry"] - 1)})
            positions.clear()
            continue
        elif dd >= DRAWDOWN_LEVEL_2 and dd_level < 2:
            dd_level = 2
            pause_until_bar = t + 4
            for pair, pos in positions.items():
                p = prices.get(pair, pos["entry"])
                cash += pos["qty"] * p * (1 - COMMISSION_TAKER)
                trades += 1
                if p > pos["entry"]: wins += 1
                trade_log.append({"pair": pair, "reason": "dd_level2", "pnl": (p / pos["entry"] - 1)})
            positions.clear()
            continue
        elif dd >= DRAWDOWN_LEVEL_1 and dd_level < 1:
            dd_level = 1
        if dd < DRAWDOWN_LEVEL_1 * 0.5:
            dd_level = 0

        if t < pause_until_bar:
            continue

        regime_mult, regime_name = detect_regime(btc_closes, t)

        # BTC crash filter
        btc_crash_skip = False
        if t >= 1 and btc_closes[t - 1] > 0:
            btc_1h_ret = (btc_closes[t] / btc_closes[t - 1] - 1)
            if btc_1h_ret < -0.015:
                btc_crash_skip = True

        # ── Trailing stops with partial exits ──
        for pair in list(positions.keys()):
            pos = positions[pair]
            p = prices.get(pair, 0)
            if p <= 0:
                continue
            pos["high"] = max(pos["high"], p)

            high = pos["high"]
            entry = pos["entry"]
            strategy = pos["strategy"]
            partial_taken = pos.get("partial_taken", False)
            holding_bars = t - pos.get("entry_bar", t)

            drop_from_high = (high - p) / high if high > 0 else 0
            pnl = (p - entry) / entry if entry > 0 else 0

            should_exit = False
            exit_fraction = 1.0
            reason = ""
            urgent = False  # determines taker vs maker commission

            if strategy == "mean_rev":
                if drop_from_high >= 0.02:
                    should_exit, reason = True, "trailing_stop"
                elif pnl <= -0.03:
                    should_exit, reason, urgent = True, "hard_stop", True
                elif pnl >= 0.03:
                    should_exit, reason = True, "profit_target"
                elif holding_bars > 48 and pnl < 0.005:
                    should_exit, reason = True, "time_stop"
            else:
                if pnl <= -0.04:
                    should_exit, reason, urgent = True, "hard_stop", True
                elif pnl >= 0.03 and not partial_taken:
                    should_exit, reason, exit_fraction = True, "partial_exit", 0.5
                    pos["partial_taken"] = True
                elif (partial_taken or pnl > 0.02):
                    trail = 0.04 if partial_taken else 0.03
                    if drop_from_high >= trail:
                        should_exit, reason = True, "trailing_stop"
                if not should_exit and holding_bars > 72 and pnl < 0.01:
                    should_exit, reason = True, "time_stop"

            if should_exit:
                sell_qty = pos["qty"] * exit_fraction
                comm = COMMISSION_TAKER if urgent else COMMISSION_MAKER
                cash += sell_qty * p * (1 - comm)
                trades += 1
                if pnl > 0: wins += 1
                trade_log.append({"pair": pair, "reason": reason, "pnl": pnl,
                                  "strategy": strategy, "fraction": exit_fraction,
                                  "commission": "taker" if urgent else "maker"})
                if exit_fraction >= 1.0:
                    del positions[pair]
                else:
                    pos["qty"] -= sell_qty

        # ── Compute signals (v3: volume required, RSI 25) ──
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
            sig = compute_signal(c, h, l, v,
                                 prices[pair] * 0.9999, prices[pair] * 1.0001,
                                 BREAKOUT_LOOKBACK)

            # Derivatives overlay
            fr = funding_data.get(pair, [])
            if fr and t < len(timestamps.get(pair, [])):
                ts = timestamps[pair][t]
                fz = compute_funding_zscore(fr, ts)
                if sig["action"] == "BUY" and fz > 1.5:
                    sig["action"] = "HOLD"
                elif sig["action"] == "BUY" and fz < -1.0:
                    sig["strength"] = min(1.0, sig["strength"] + 0.15)

            # NO regime breakout suppression (removed in v3)
            # Regime only affects exposure via regime_mult in position sizing

            # BTC crash filter (kept)
            if sig["action"] == "BUY" and pair != "BTC/USD" and btc_crash_skip:
                sig["action"] = "HOLD"

            sigs[pair] = sig

        # ── Signal-based sells (non-urgent use limit commission) ──
        for pair in list(positions.keys()):
            sig = sigs.get(pair, {})
            if sig.get("action") == "SELL" or sig.get("breakdown"):
                p = prices.get(pair, 0)
                pos = positions[pair]
                pnl = (p / pos["entry"]) - 1
                urgent = sig.get("breakdown", False)
                comm = COMMISSION_TAKER if urgent else COMMISSION_MAKER
                cash += pos["qty"] * p * (1 - comm)
                trades += 1
                if pnl > 0: wins += 1
                trade_log.append({"pair": pair, "reason": "signal_sell", "pnl": pnl,
                                  "strategy": pos["strategy"],
                                  "commission": "taker" if urgent else "maker"})
                del positions[pair]

        # ── Buy signals (limit commission) ──
        buys = [(p, s) for p, s in sigs.items()
                if s["action"] == "BUY" and p not in positions]
        buys.sort(key=lambda x: x[1]["strength"], reverse=True)

        exposure = sum(pos["qty"] * prices.get(p, 0) for p, pos in positions.items())
        effective_max = MAX_TOTAL_EXPOSURE_PCT * regime_mult * 0.85

        for pair, sig in buys:
            if len(positions) >= MAX_POSITIONS:
                break
            remaining = effective_max * pv - exposure
            if remaining <= 0:
                break

            vol = sig.get("real_vol", 0.5) or 0.5
            daily_vol = vol / np.sqrt(365)
            if daily_vol <= 0:
                daily_vol = 0.03

            size = (TARGET_RISK_PER_TRADE * pv) / daily_vol
            size = min(size, MAX_POSITION_PCT * pv, remaining)
            size *= redd_multiplier(dd)
            size *= max(0.3, sig["strength"])

            if size < 100:
                continue

            p = prices[pair]
            qty = size / p
            cost = qty * p * (1 + COMMISSION_MAKER)  # limit order for entry
            if cost > cash:
                continue

            cash -= cost
            positions[pair] = {
                "qty": qty, "entry": p, "high": p,
                "strategy": sig["strategy"],
                "partial_taken": False,
                "entry_bar": t,
            }
            exposure += qty * p
            trades += 1
            trade_log.append({"pair": pair, "reason": "buy", "strategy": sig["strategy"],
                              "size": size})

    # Close remaining
    final_prices = {p: closes[p][min(end - 1, len(closes[p]) - 1)] for p in pairs}
    for pair, pos in positions.items():
        p = final_prices.get(pair, pos["entry"])
        cash += pos["qty"] * p * (1 - COMMISSION_MAKER)
        pnl = (p / pos["entry"]) - 1
        trades += 1
        if pnl > 0: wins += 1
        trade_log.append({"pair": pair, "reason": "end", "pnl": pnl})

    return {
        "portfolio_history": portfolio_history,
        "final": cash,
        "trades": trades,
        "wins": wins,
        "trade_log": trade_log,
    }


def download_spot_paginated(symbol, session, months=6):
    """Download months of 1h candles via pagination (1000 bars per request)."""
    end_ms = int(_time.time() * 1000)
    start_ms = end_ms - (months * 30 * 24 * 3_600_000)
    candles = []
    cur = start_ms
    while cur < end_ms:
        try:
            from bot.config import BINANCE_BASE_URL
            r = session.get(f"{BINANCE_BASE_URL}/api/v3/klines",
                            params={"symbol": symbol, "interval": "1h",
                                    "startTime": cur, "endTime": end_ms, "limit": 1000})
            r.raise_for_status()
            raw = r.json()
        except Exception as e:
            break
        if not raw:
            break
        for k in raw:
            candles.append({"t": k[0], "c": float(k[4]), "h": float(k[2]),
                            "l": float(k[3]), "v": float(k[5])})
        cur = raw[-1][0] + 3_600_000
        if len(raw) < 1000:
            break
        _time.sleep(0.05)
    return candles


def main():
    MONTHS = 6
    print("=" * 70)
    print(f"  INTEGRATED BACKTEST v3 ({MONTHS} MONTHS)")
    print("  Volume-required breakouts + RSI 25 + No regime suppression")
    print("  + Limit order commissions + Widened breakers + Competition params")
    print("=" * 70)

    session = requests.Session()
    session.timeout = 15

    closes, highs, lows, volumes, timestamps_dict = {}, {}, {}, {}, {}
    min_len = 999999

    print(f"\n  Downloading {MONTHS} months of hourly data...")
    for i, pair in enumerate(CORE_PAIRS):
        sym = BINANCE_SYMBOL_MAP.get(pair)
        if not sym:
            continue
        candles = download_spot_paginated(sym, session, months=MONTHS)
        if len(candles) < 500:
            print(f"    [{i+1}/{len(CORE_PAIRS)}] {pair}: {len(candles)} bars — skip")
            continue
        closes[pair] = np.array([c["c"] for c in candles])
        highs[pair] = np.array([c["h"] for c in candles])
        lows[pair] = np.array([c["l"] for c in candles])
        volumes[pair] = np.array([c["v"] for c in candles])
        timestamps_dict[pair] = np.array([c["t"] for c in candles])
        min_len = min(min_len, len(candles))
        print(f"    [{i+1}/{len(CORE_PAIRS)}] {pair}: {len(candles)} bars ({len(candles)//24}d)")

    pairs = list(closes.keys())
    btc_closes = closes.get("BTC/USD", np.array([]))
    btc_ts = timestamps_dict.get("BTC/USD", np.array([]))
    print(f"  Loaded {len(pairs)} pairs, {min_len} bars (~{min_len // 24}d)")

    print("  Loading funding rates...")
    funding_data = download_funding_rates(pairs)
    print(f"  Funding data for {len(funding_data)} pairs")

    # ═══════════════════════════════════════════════════════════
    # TEST 1: Full period
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 1: FULL PERIOD")
    print("=" * 70)

    result = run_simulation(closes, highs, lows, volumes, timestamps_dict,
                            pairs, btc_closes, btc_ts, funding_data,
                            100, min_len)
    metrics = compute_metrics(result["portfolio_history"], INITIAL, daily_sample_rate=24)
    print_metrics(metrics, f"FULL PERIOD ({metrics.get('days', 0)} days, {len(pairs)} pairs)")
    print(f"  Trades:    {result['trades']}")
    wr = result['wins'] / max(result['trades'], 1)
    print(f"  Win Rate:  {result['wins']}/{result['trades']} ({wr:.0%})")

    sells = [t for t in result["trade_log"] if "pnl" in t and t["reason"] != "buy"]
    if sells:
        pnls = [t["pnl"] for t in sells]
        print(f"\n  Trade PnL: avg={np.mean(pnls):+.2%}  median={np.median(pnls):+.2%}  "
              f"best={np.max(pnls):+.2%}  worst={np.min(pnls):+.2%}")

        # Commission savings
        maker_exits = sum(1 for t in sells if t.get("commission") == "maker")
        taker_exits = sum(1 for t in sells if t.get("commission") == "taker")
        print(f"  Exit orders: {maker_exits} limit (0.05%) + {taker_exits} market (0.1%)")

        reasons = {}
        for t in sells:
            r = t["reason"]
            reasons.setdefault(r, []).append(t["pnl"])
        print(f"\n  Exit reasons:")
        for reason, rpnls in sorted(reasons.items()):
            print(f"    {reason:<20} n={len(rpnls):>4}  avg_pnl={np.mean(rpnls):>+6.2%}")

    # ═══════════════════════════════════════════════════════════
    # TEST 2: Rolling 10-day windows
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 2: ROLLING 10-DAY WINDOWS (competition format)")
    print("=" * 70)

    window = 240
    step = 72
    results_10d = []

    print(f"\n  {'Window':<10} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Calmar':>8} {'Compos':>8} {'Trades':>7}")
    print(f"  {'─' * 70}")

    for s in range(100, min_len - window, step):
        e = s + window
        r = run_simulation(closes, highs, lows, volumes, timestamps_dict,
                           pairs, btc_closes, btc_ts, funding_data, s, e)
        m = compute_metrics(r["portfolio_history"], INITIAL, daily_sample_rate=24)
        if not m:
            continue
        m["trades"] = r["trades"]
        m["wins"] = r["wins"]
        results_10d.append(m)
        d = (s - 100) // 24
        print(f"  d{d:>3}-{d+10:<3} {m['total_return_pct']:>+7.2f}% {m['max_drawdown_pct']:>7.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['calmar']:>7.2f} "
              f"{m['composite']:>7.2f} {m['trades']:>6}")

    if results_10d:
        rets = [m["total_return_pct"] for m in results_10d]
        dds = [m["max_drawdown_pct"] for m in results_10d]
        comps = [m["composite"] for m in results_10d]
        print(f"\n  SUMMARY ({len(results_10d)} windows):")
        print(f"    Avg return:       {np.mean(rets):>+7.2f}%")
        print(f"    Median return:    {np.median(rets):>+7.2f}%")
        print(f"    Best / Worst:     {np.max(rets):>+7.2f}% / {np.min(rets):>+7.2f}%")
        print(f"    Positive windows: {sum(1 for r in rets if r > 0)}/{len(rets)} "
              f"({sum(1 for r in rets if r > 0) / len(rets):.0%})")
        print(f"    Avg max DD:       {np.mean(dds):>7.2f}%")
        print(f"    Avg composite:    {np.mean(comps):>7.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()