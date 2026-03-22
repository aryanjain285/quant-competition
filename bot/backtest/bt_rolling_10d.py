#!/usr/bin/env python3
"""
Rolling 10-day backtest over 4 months of 1h data.

Downloads 4 months of hourly candles (2880 bars per coin), then runs
the full v4 pipeline in rolling 10-day (240-bar) windows stepped by
3 days (72 bars).

This matches the competition format: 10-day live trading period,
scored on Sortino/Sharpe/Calmar composite.

Run: venv/bin/python -m bot.backtest.bt_rolling_10d
"""
import sys
import os
import time as _time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import requests
from bot.config import (
    BINANCE_BASE_URL, BINANCE_SYMBOL_MAP, ANNUALIZATION_FACTOR,
    MAX_POSITIONS, MAX_TOTAL_EXPOSURE_PCT, MAX_POSITION_PCT,
    TARGET_RISK_PER_TRADE,
    DRAWDOWN_LEVEL_1, DRAWDOWN_LEVEL_2, DRAWDOWN_LEVEL_3,
    BREAKOUT_LOOKBACK,
)
from bot.features import (
    compute_returns, compute_persistence, compute_choppiness,
    compute_realized_vol, compute_downside_vol, compute_jump_proxy,
    compute_breakout_distance, compute_volume_ratio, compute_overshoot,
    zscore_array, compute_risk_penalty, compute_cost_penalty,
)
from bot.ranking import Ranker
from bot.backtest.data_loader import compute_metrics, print_metrics, CORE_PAIRS

INITIAL = 1_000_000
COMMISSION_TAKER = 0.001
COMMISSION_MAKER = 0.0005


# ─── Data Download ───────────────────────────────────────────

def download_klines(symbol, session, months=4):
    end_ms = int(_time.time() * 1000)
    start_ms = end_ms - (months * 30 * 24 * 3_600_000)
    candles = []
    cur = start_ms
    while cur < end_ms:
        try:
            r = session.get(f"{BINANCE_BASE_URL}/api/v3/klines",
                            params={"symbol": symbol, "interval": "1h",
                                    "startTime": cur, "endTime": end_ms, "limit": 1000})
            r.raise_for_status()
            raw = r.json()
        except Exception:
            break
        if not raw: break
        for k in raw:
            candles.append({"c": float(k[4]), "h": float(k[2]),
                            "l": float(k[3]), "v": float(k[5])})
        cur = raw[-1][0] + 3_600_000
        if len(raw) < 1000: break
        _time.sleep(0.05)
    return candles


# ─── Feature computation at bar index (1h bars) ──────────────

def compute_features_at(c, h, l, v, idx):
    if idx < 80: return None
    price = c[idx]
    if price <= 0: return None

    cs, hs, ls, vs = c[:idx+1], h[:idx+1], l[:idx+1], v[:idx+1]
    lookbacks = {"1h": 1, "6h": 6, "24h": 24, "3d": 72}
    rets = compute_returns(cs, lookbacks)
    return {
        **rets,
        "persistence": compute_persistence(cs, 24),
        "choppiness": compute_choppiness(cs, 24),
        "realized_vol": compute_realized_vol(cs, 24),
        "downside_vol": compute_downside_vol(cs, 24),
        "jump_proxy": compute_jump_proxy(cs, 24),
        "breakout_distance": compute_breakout_distance(cs, hs, BREAKOUT_LOOKBACK),
        "volume_ratio": compute_volume_ratio(vs, BREAKOUT_LOOKBACK),
        "overshoot": compute_overshoot(cs, 168),
        "spread_pct": 0.0,
        "short_vol": float(np.std(np.diff(np.log(cs[-7:]))) * ANNUALIZATION_FACTOR) if len(cs) > 7 else 0,
        "price": price,
    }


# ─── Regime detection ────────────────────────────────────────

def detect_regime(all_raw):
    if not all_raw: return 1, 0.75
    r24h = [all_raw[p].get("r_24h", 0) for p in all_raw]
    breadth = sum(1 for r in r24h if r > 0) / len(r24h)
    trend = float(np.median(r24h))
    dvs = [all_raw[p].get("downside_vol", 0) for p in all_raw]
    dn = min(1.0, float(np.median(dvs)) / 0.5) if np.median(dvs) > 0 else 0
    score = (breadth - 0.5) * 2 + np.clip(trend * 20, -1, 1) - dn * 0.5
    if score > 0.3: return 0, 1.0
    elif score < -0.3: return 2, 0.25
    else: return 1, 0.75


# ─── Event filters ───────────────────────────────────────────

def check_continuation(raw, z):
    bo = raw.get("breakout_distance", 0)
    if bo <= 0: return None
    if raw.get("r_24h", 0) <= 0: return None
    if raw.get("persistence", 0.5) < 0.52: return None
    if raw.get("choppiness", 0.5) > 0.85: return None
    if raw.get("volume_ratio", 1.0) < 1.3: return None
    if z.get("risk_penalty", 0) > 1.5: return None
    if z.get("cost_penalty", 0) > 1.5: return None
    s = 0.5 + min(0.15, bo * 5) + min(0.1, raw.get("r_24h", 0) * 2)
    return min(s, 1.0)


def check_reversal(raw, z):
    ov = raw.get("overshoot", 0)
    if ov > -1.5: return None
    if z.get("risk_penalty", 0) > 2.5: return None
    if z.get("cost_penalty", 0) > 1.5: return None
    if raw.get("r_1h", 0) < -0.03: return None
    s = 0.4 + min(0.25, abs(ov) * 0.1) + (0.1 if raw.get("r_1h", 0) > 0 else 0)
    return min(s, 1.0)


def check_breakdown(c, l, idx, lookback=72):
    if idx < lookback + 1: return False
    bl = max(lookback // 3, 12)
    prior_low = np.min(l[max(0, idx - bl):idx])
    return c[idx] < prior_low and prior_low > 0


def redd_mult(dd):
    return max(0.0, 1.0 - dd / 0.10) if dd > 0 else 1.0


# ─── Simulation ──────────────────────────────────────────────

def run_simulation(data, pairs, start, end):
    cash = INITIAL
    positions = {}
    history = []
    peak = INITIAL
    trades, wins = 0, 0
    trade_log = []
    dd_level = 0
    pause_until = 0
    ranker = Ranker()

    for t in range(start, end):
        prices = {p: data[p]["c"][t] for p in pairs if t < len(data[p]["c"])}
        pv = cash + sum(pos["qty"] * prices.get(p, 0) for p, pos in positions.items())
        history.append(pv)
        if pv > peak: peak = pv
        dd = (peak - pv) / peak if peak > 0 else 0

        # Drawdown breakers
        if dd >= DRAWDOWN_LEVEL_3 and dd_level < 3:
            dd_level = 3; pause_until = t + 12
            for p, pos in positions.items():
                cash += pos["qty"] * prices.get(p, pos["entry"]) * (1 - COMMISSION_TAKER)
                trades += 1
                if prices.get(p, 0) > pos["entry"]: wins += 1
                trade_log.append({"reason": "dd_breaker", "pnl": prices.get(p, pos["entry"])/pos["entry"]-1})
            positions.clear(); continue
        elif dd >= DRAWDOWN_LEVEL_2 and dd_level < 2:
            dd_level = 2; pause_until = t + 4
            for p, pos in positions.items():
                cash += pos["qty"] * prices.get(p, pos["entry"]) * (1 - COMMISSION_TAKER)
                trades += 1
                if prices.get(p, 0) > pos["entry"]: wins += 1
            positions.clear(); continue
        elif dd >= DRAWDOWN_LEVEL_1 and dd_level < 1:
            dd_level = 1
        if dd < DRAWDOWN_LEVEL_1 * 0.5: dd_level = 0
        if t < pause_until: continue

        # Features + regime
        all_raw = {}
        for p in pairs:
            if t >= len(data[p]["c"]): continue
            f = compute_features_at(data[p]["c"], data[p]["h"], data[p]["l"], data[p]["v"], t)
            if f: all_raw[p] = f
        if not all_raw: continue

        regime_state, regime_mult = detect_regime(all_raw)

        # Trailing stops
        for p in list(positions.keys()):
            pos = positions[p]
            price = prices.get(p, 0)
            if price <= 0: continue
            pos["high"] = max(pos["high"], price)
            pnl = (price - pos["entry"]) / pos["entry"]
            dfh = (pos["high"] - price) / pos["high"] if pos["high"] > 0 else 0
            strat = pos["strategy"]
            partial = pos.get("partial_taken", False)
            bars = t - pos.get("entry_bar", t)

            should_exit, fraction, reason = False, 1.0, ""
            urgent = False

            if strat == "reversal":
                if dfh >= 0.02: should_exit, reason = True, "trailing_stop"
                elif pnl <= -0.03: should_exit, reason, urgent = True, "hard_stop", True
                elif pnl >= 0.03: should_exit, reason = True, "profit_target"
                elif bars > 48 and pnl < 0.005: should_exit, reason = True, "time_stop"
            else:
                if pnl <= -0.04: should_exit, reason, urgent = True, "hard_stop", True
                elif pnl >= 0.03 and not partial:
                    should_exit, reason, fraction = True, "partial_exit", 0.5
                    pos["partial_taken"] = True
                elif partial or pnl > 0.02:
                    trail = 0.04 if partial else 0.03
                    if dfh >= trail: should_exit, reason = True, "trailing_stop"
                if not should_exit and bars > 72 and pnl < 0.01:
                    should_exit, reason = True, "time_stop"

            if should_exit:
                sq = pos["qty"] * fraction
                comm = COMMISSION_TAKER if urgent else COMMISSION_MAKER
                cash += sq * price * (1 - comm)
                trades += 1
                if pnl > 0: wins += 1
                trade_log.append({"reason": reason, "pnl": pnl, "strategy": strat})
                if fraction >= 1.0: del positions[p]
                else: pos["qty"] -= sq

        # Z-score + signals
        zscored = {}
        feature_keys = [k for k in next(iter(all_raw.values())) if isinstance(all_raw[next(iter(all_raw))].get(k), (int, float)) and k != "price"]
        for key in feature_keys:
            vals = np.array([all_raw[p].get(key, 0) for p in all_raw])
            z = zscore_array(vals)
            for i, p in enumerate(all_raw):
                if p not in zscored: zscored[p] = {}
                zscored[p][key] = float(z[i])
        for p in zscored:
            zscored[p]["risk_penalty"] = compute_risk_penalty(zscored[p].get("realized_vol",0), zscored[p].get("downside_vol",0), zscored[p].get("jump_proxy",0))
            zscored[p]["cost_penalty"] = compute_cost_penalty(zscored[p].get("spread_pct",0), zscored[p].get("short_vol",0))

        signals = {}
        for p in all_raw:
            cs_str = check_continuation(all_raw[p], zscored[p])
            rv_str = check_reversal(all_raw[p], zscored[p])
            bd = check_breakdown(data[p]["c"], data[p]["l"], t, BREAKOUT_LOOKBACK)
            if bd: signals[p] = ("SELL", "breakdown", 0.8)
            elif cs_str is not None: signals[p] = ("BUY", "continuation", cs_str)
            elif rv_str is not None: signals[p] = ("BUY", "reversal", rv_str)

        # Signal sells
        for p in list(positions.keys()):
            sig = signals.get(p)
            if sig and sig[0] == "SELL":
                price = prices.get(p, 0)
                pos = positions[p]
                pnl = (price / pos["entry"]) - 1
                cash += pos["qty"] * price * (1 - COMMISSION_MAKER)
                trades += 1
                if pnl > 0: wins += 1
                trade_log.append({"reason": "signal_sell", "pnl": pnl})
                del positions[p]

        # Valid candidates + ranking
        if regime_state == 2: continue  # HOSTILE
        valid = {}
        for p, sig in signals.items():
            if sig[0] == "BUY" and p not in positions:
                valid[p] = zscored.get(p, {}).copy()
                valid[p]["_strength"] = sig[2]; valid[p]["_strategy"] = sig[1]
        if not valid: continue

        ranked = ranker.rank(valid, max_results=MAX_POSITIONS)
        exposure = sum(pos["qty"] * prices.get(p, 0) for p, pos in positions.items())
        eff_max = MAX_TOTAL_EXPOSURE_PCT * regime_mult

        for p, score, feats in ranked:
            if len(positions) >= MAX_POSITIONS: break
            remaining = eff_max * pv - exposure
            if remaining <= 0: break
            vol = all_raw[p].get("realized_vol", 0.5) or 0.5
            dvol = vol / np.sqrt(365)
            if dvol <= 0: dvol = 0.03
            size = (TARGET_RISK_PER_TRADE * pv) / dvol
            size = min(size, MAX_POSITION_PCT * pv, remaining)
            size *= redd_mult(dd) * max(0.3, feats.get("_strength", 0.5))
            if size < 100: continue
            price = prices[p]
            qty = size / price
            cost = qty * price * (1 + COMMISSION_MAKER)
            if cost > cash: continue
            cash -= cost
            positions[p] = {"qty": qty, "entry": price, "high": price,
                            "strategy": feats.get("_strategy", "continuation"),
                            "partial_taken": False, "entry_bar": t}
            exposure += qty * price; trades += 1

    # Close remaining
    for p, pos in positions.items():
        price = data[p]["c"][min(end-1, len(data[p]["c"])-1)]
        cash += pos["qty"] * price * (1 - COMMISSION_MAKER)
        pnl = (price / pos["entry"]) - 1
        trades += 1
        if pnl > 0: wins += 1
    return {"history": history, "final": cash, "trades": trades, "wins": wins, "trade_log": trade_log}


# ─── Main ────────────────────────────────────────────────────

def main():
    MONTHS = 4
    print("=" * 70)
    print(f"  ROLLING 10-DAY BACKTEST ({MONTHS} MONTHS, 1H BARS)")
    print(f"  All lookbacks corrected for hourly data")
    print("=" * 70)

    session = requests.Session(); session.timeout = 15
    data = {}; min_len = 999999

    print(f"\n  Downloading {MONTHS} months of 1h data...")
    for i, pair in enumerate(CORE_PAIRS):
        sym = BINANCE_SYMBOL_MAP.get(pair)
        if not sym: continue
        candles = download_klines(sym, session, months=MONTHS)
        if len(candles) < 300:
            print(f"    [{i+1}/{len(CORE_PAIRS)}] {pair}: {len(candles)} — skip"); continue
        data[pair] = {
            "c": np.array([x["c"] for x in candles]),
            "h": np.array([x["h"] for x in candles]),
            "l": np.array([x["l"] for x in candles]),
            "v": np.array([x["v"] for x in candles]),
        }
        min_len = min(min_len, len(candles))
        print(f"    [{i+1}/{len(CORE_PAIRS)}] {pair}: {len(candles)} bars ({len(candles)//24}d)")

    pairs = list(data.keys())
    print(f"  Loaded {len(pairs)} pairs, {min_len} bars (~{min_len//24}d)")


    # Rolling 10-day windows
    window = 240   # 10 days × 24h
    step = 72      # 3-day step

    results = []
    print(f"\n  {'Window':<10} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Calmar':>8} {'Compos':>8} {'Trades':>7}")
    print(f"  {'─' * 70}")

    for s in range(80, min_len - window, step):
        e = s + window
        r = run_simulation(data, pairs, s, e)
        m = compute_metrics(r["history"], INITIAL, daily_sample_rate=24)
        if not m: continue
        m["trades"] = r["trades"]
        m["wins"] = r["wins"]
        results.append(m)
        d = (s - 80) // 24
        print(f"  d{d:>3}-{d+10:<3} {m['total_return_pct']:>+7.2f}% {m['max_drawdown_pct']:>7.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['calmar']:>7.2f} "
              f"{m['composite']:>7.2f} {m['trades']:>6}")

    if results:
        rets = [m["total_return_pct"] for m in results]
        dds = [m["max_drawdown_pct"] for m in results]
        comps = [m["composite"] for m in results]
        sharpes = [m["sharpe"] for m in results]
        sortinos = [m["sortino"] for m in results]
        calmars = [m["calmar"] for m in results]

        print(f"\n  {'=' * 70}")
        print(f"  SUMMARY ({len(results)} windows)")
        print(f"  {'=' * 70}")
        print(f"    Avg return:       {np.mean(rets):>+7.2f}%")
        print(f"    Median return:    {np.median(rets):>+7.2f}%")
        print(f"    Best / Worst:     {np.max(rets):>+7.2f}% / {np.min(rets):>+7.2f}%")
        print(f"    Positive:         {sum(1 for r in rets if r > 0)}/{len(rets)} "
              f"({sum(1 for r in rets if r > 0)/len(rets):.0%})")
        print(f"    Avg max DD:       {np.mean(dds):>7.2f}%")
        print(f"    Avg Sharpe:       {np.mean(sharpes):>7.2f}")
        print(f"    Avg Sortino:      {np.mean(sortinos):>7.2f}")
        print(f"    Avg Calmar:       {np.mean(calmars):>7.2f}")
        print(f"    Avg composite:    {np.mean(comps):>7.2f}")
        print(f"    Median composite: {np.median(comps):>7.2f}")


if __name__ == "__main__":
    main()
