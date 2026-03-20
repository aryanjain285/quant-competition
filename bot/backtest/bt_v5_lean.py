#!/usr/bin/env python3
"""
v4 Lean Pipeline Backtest: Regime (PCA-HMM) -> Events -> Ranking -> Execution.
Matches the 10-day competition format with a continuous long-term view.

Run: python -m bot.backtest.bt_v4_lean
"""
import sys
import os
import time as _time
import numpy as np
import requests

# Ensure path resolution for the bot module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bot.config import (
    BINANCE_BASE_URL, BINANCE_SYMBOL_MAP, ANNUALIZATION_FACTOR,
    MAX_POSITIONS, MAX_TOTAL_EXPOSURE_PCT, MAX_POSITION_PCT,
    TARGET_RISK_PER_TRADE, BREAKOUT_LOOKBACK,
    DRAWDOWN_LEVEL_1, DRAWDOWN_LEVEL_2, DRAWDOWN_LEVEL_3
)
from bot.regime_detector import RegimeDetector
from bot.signals import compute_signal
from bot.ranking import Ranker
from bot.features import compute_coin_features, zscore_universe, compute_submodel_scores
from bot.backtest.data_loader import compute_metrics, print_metrics, CORE_PAIRS

INITIAL_CASH = 1_000_000
COMMISSION_MAKER = 0.0005 # Using maker fee for backtest as bot uses limit orders

# ─── Data Downloader ──────────────────────────────────────────

def download_klines(symbol, session, months=4):
    """Downloads historical 1h klines from Binance."""
    end_ms = int(_time.time() * 1000)
    start_ms = end_ms - (months * 30 * 24 * 3_600_000)
    candles = []
    cur = start_ms
    while cur < end_ms:
        try:
            r = session.get(
                f"{BINANCE_BASE_URL}/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": "1h",
                    "startTime": cur,
                    "endTime": end_ms,
                    "limit": 1000,
                },
                timeout=15,
            )
            r.raise_for_status()
            raw = r.json()
        except Exception: break
        if not raw: break
        for k in raw:
            candles.append({"c": float(k[4]), "h": float(k[2]), "l": float(k[3]), "v": float(k[5])})
        cur = raw[-1][0] + 3_600_000
        if len(raw) < 1000: break
        _time.sleep(0.05)
    return candles

# ─── Helper: Price Matrix for PCA ─────────────────────────────

def build_price_matrix(data, pairs, t, lookback=1000):
    matrix = []
    for p in pairs:
        if t < len(data[p]["c"]):
            matrix.append(data[p]["c"][max(0, t-lookback+1):t+1])
    if not matrix: return None
    min_len = min(len(arr) for arr in matrix)
    return np.array([arr[-min_len:] for arr in matrix])

# ─── Simulation Engine ────────────────────────────────────────

def run_simulation(data, pairs, start, end):
    cash = INITIAL_CASH
    positions = {}
    history = []
    peak = INITIAL_CASH
    regime = RegimeDetector()
    ranker = Ranker()
    last_pca_fit = -100
    trades = 0
    wins = 0
    trade_log = []

    for t in range(start, end):
        prices = {p: data[p]["c"][t] for p in pairs}
        pv = cash + sum(pos["qty"] * prices[p] for p, pos in positions.items())
        history.append(pv)
        if pv > peak:
            peak = pv
        dd = (peak - pv) / peak if peak > 0 else 0

        pm = build_price_matrix(data, pairs, t)
        if pm is not None:
            if t - last_pca_fit >= 6:
                regime._pca_last_fit = 0
                last_pca_fit = t
            pc1_series, _, _, _ = regime.compute_pc1_market_proxy(pm.T)
            regime.update(pc1_series=pc1_series)
            if t % 24 == 0 and pc1_series is not None:
                regime.fit_hmm(pc1_series)

        # exits
        for p in list(positions.keys()):
            pos = positions[p]
            price = prices[p]
            pos["high"] = max(pos["high"], price)
            pnl = (price - pos["entry"]) / pos["entry"]
            dfh = (pos["high"] - price) / pos["high"] if pos["high"] > 0 else 0

            should_exit = False
            if pos["strategy"] == "reversal":
                if pnl <= -0.03 or pnl >= 0.03 or dfh >= 0.02:
                    should_exit = True
            else:
                if pnl <= -0.04:
                    should_exit = True
                elif pnl >= 0.03 and not pos["partial_taken"]:
                    cash += (pos["qty"] * 0.5) * price * (1 - COMMISSION_MAKER)
                    pos["qty"] *= 0.5
                    pos["partial_taken"] = True
                    continue
                elif pos["partial_taken"] or pnl > 0.02:
                    if dfh >= (0.04 if pos["partial_taken"] else 0.03):
                        should_exit = True

            if should_exit:
                cash += pos["qty"] * price * (1 - COMMISSION_MAKER)
                trades += 1
                if pnl > 0:
                    wins += 1
                trade_log.append({"pair": p, "reason": "exit", "pnl": pnl})
                del positions[p]

        raw_feats = {}
        for p in pairs:
            f = compute_coin_features(
                data[p]["c"][:t+1],
                data[p]["h"][:t+1],
                data[p]["l"][:t+1],
                data[p]["v"][:t+1],
                prices[p],
                prices[p],
            )
            if f:
                raw_feats[p] = f

        zscored = zscore_universe(raw_feats)
        for p in zscored:
            compute_submodel_scores(zscored[p])

        candidates = {}
        for p in raw_feats:
            sig = compute_signal(raw_feats[p], zscored[p], data[p]["c"][:t+1], data[p]["l"][:t+1])
            if sig["action"] == "BUY" and p not in positions and regime.should_trade():
                candidates[p] = zscored[p].copy()
                candidates[p]["_signal"] = sig

        ranked = ranker.rank(candidates, max_results=MAX_POSITIONS)

        eff_max = MAX_TOTAL_EXPOSURE_PCT * regime.get_exposure_multiplier()
        current_exp = sum(pos["qty"] * prices[p] for p, pos in positions.items())

        for p, score, feats in ranked:
            if len(positions) >= MAX_POSITIONS:
                break
            room = (eff_max * pv) - current_exp
            if room <= 0:
                break

            dvol = raw_feats[p]["realized_vol"] / np.sqrt(365)
            dvol = max(dvol, 0.03)
            size = (TARGET_RISK_PER_TRADE * pv) / dvol
            size = min(size, MAX_POSITION_PCT * pv, room)
            size *= max(0.0, 1.0 - (dd / 0.10))
            size *= max(0.3, feats["_signal"]["strength"])

            if size < 100:
                continue

            qty = size / prices[p]
            cost = qty * prices[p] * (1 + COMMISSION_MAKER)
            if cost > cash:
                continue

            cash -= cost
            positions[p] = {
                "qty": qty,
                "entry": prices[p],
                "high": prices[p],
                "strategy": feats["_signal"]["strategy"],
                "partial_taken": False,
            }
            current_exp += qty * prices[p]
            trades += 1

    # force close at window end
    for p, pos in positions.items():
        price = data[p]["c"][min(end - 1, len(data[p]["c"]) - 1)]
        pnl = (price - pos["entry"]) / pos["entry"]
        cash += pos["qty"] * price * (1 - COMMISSION_MAKER)
        trades += 1
        if pnl > 0:
            wins += 1
        trade_log.append({"pair": p, "reason": "window_end", "pnl": pnl})

    return {"history": history, "final": cash, "trades": trades, "wins": wins, "trade_log": trade_log}

# ─── Main Execution ───────────────────────────────────────────

def main():
    print("=" * 70)
    print("  v4 LEAN PIPELINE BACKTEST: Regime -> Event -> Rank -> Execute")
    print("=" * 70)

    # Data Fetching
    session = requests.Session(); data = {}; min_len = 9999
    for pair in CORE_PAIRS:
        sym = BINANCE_SYMBOL_MAP.get(pair)
        if not sym: continue
        candles = download_klines(sym, session, months=4)
        if len(candles) < 500: continue
        data[pair] = {"c": np.array([x["c"] for x in candles]), "h": np.array([x["h"] for x in candles]),
                      "l": np.array([x["l"] for x in candles]), "v": np.array([x["v"] for x in candles])}
        min_len = min(min_len, len(candles))
        print(f"  Loaded {pair}: {len(candles)} bars")

    pairs = list(data.keys())

    # 1. LONG TERM CONTINUOUS TEST
    print("\n[TEST 1] 4-MONTH CONTINUOUS RUN")
    res = run_simulation(data, pairs, 80, min_len)
    metrics = compute_metrics(res["history"], INITIAL_CASH, daily_sample_rate=24)
    print_metrics(metrics, "FULL CONTINUOUS PERIOD")

    # 2. ROLLING 10-DAY TEST (Competition Format)
    print("\n[TEST 2] ROLLING 10-DAY WINDOWS")
    window, step = 240, 72 # 10 days, step 3 days
    results = []
    for s in range(80, min_len - window, step):
        r = run_simulation(data, pairs, s, s + window)
        m = compute_metrics(r["history"], INITIAL_CASH, daily_sample_rate=24)
        if not m:
            continue
        m["trades"] = r["trades"]
        m["wins"] = r["wins"]
        results.append(m)
        print(
            f"  Window {s//24}: "
            f"Ret={m['total_return_pct']:>+6.2f}% | "
            f"Comp={m['composite']:>5.2f} | "
            f"Trades={m['trades']:>3}"
        )

    if results:
        print(f"\nROLLING SUMMARY: Avg Return={np.mean([m['total_return_pct'] for m in results]):.2f}% | "
              f"Avg Composite={np.mean([m['composite'] for m in results]):.2f}")

if __name__ == "__main__":
    main()