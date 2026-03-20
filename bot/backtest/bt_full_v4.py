#!/usr/bin/env python3
"""
FULL FIDELITY v4 PIPELINE BACKTEST
Matches the exact logic of the full main.py:
- Rule-based + PCA-HMM Regime Detection (market breadth & trend)
- Derivatives Overlay (Trade vetoes, OI-divergence buys)
- Derivative-adjusted Position Sizing
- Event Filters (Continuation/Reversal/Breakdown)
- Sortino-Optimized Exits & REDD Scaling
"""
import sys
import os
import time as _time
import numpy as np
import requests
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bot.config import (
    BINANCE_BASE_URL, BINANCE_SYMBOL_MAP, MAX_POSITIONS, 
    MAX_TOTAL_EXPOSURE_PCT, MAX_POSITION_PCT, TARGET_RISK_PER_TRADE, 
    BREAKOUT_LOOKBACK
)
from bot.regime_detector import RegimeDetector
from bot.signals import compute_signal
from bot.ranking import Ranker
from bot.features import (
    compute_coin_features, zscore_universe, 
    compute_submodel_scores, compute_market_features
)
from bot.backtest.data_loader import compute_metrics, print_metrics, CORE_PAIRS

INITIAL_CASH = 1_000_000
COMMISSION = 0.0005 

# ─── Data Downloader ──────────────────────────────────────────

def download_klines(symbol, session, months=4):
    end_ms = int(_time.time() * 1000)
    start_ms = end_ms - (months * 30 * 24 * 3_600_000)
    candles = []
    cur = start_ms
    while cur < end_ms:
        try:
            r = session.get(f"{BINANCE_BASE_URL}/api/v3/klines",
                            params={"symbol": symbol, "interval": "1h",
                                    "startTime": cur, "endTime": end_ms, "limit": 1000}, timeout=15)
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

def build_price_matrix(data, pairs, t, lookback=1000):
    matrix = []
    for p in pairs:
        if t < len(data[p]["c"]):
            matrix.append(data[p]["c"][max(0, t-lookback+1):t+1])
    if not matrix: return None
    min_len = min(len(arr) for arr in matrix)
    return np.array([arr[-min_len:] for arr in matrix])

# ─── Historical Derivatives Simulator ─────────────────────────

class HistoricalDerivativesMock:
    """
    Simulates historical derivatives data. Calling the live API 2880 times 
    would result in an IP ban and inaccurate data (using today's funding for past trades).
    This injects realistic statistical noise to trigger the logic in main.py.
    """
    def compute_signals(self, active_pairs, spot_prices):
        signals = {}
        for pair in active_pairs:
            # Generate a realistic composite score (Normal dist bounded [-1.0, 1.0])
            score = np.clip(random.gauss(0, 0.4), -1.0, 1.0)
            
            # Simulate occasional OI divergence (happens ~5% of the time)
            has_divergence = random.random() < 0.05
            
            signals[pair] = {
                "composite_deriv_score": score,
                "oi_price_divergence": has_divergence,
                "oi_signal": 1 if has_divergence else 0,
                "funding_zscore": score * 2 # Proxy
            }
        return signals

# ─── Simulation Engine ────────────────────────────────────────

def run_v4_full_simulation(data, pairs, start, end):
    cash = INITIAL_CASH
    positions = {}
    history = []
    peak = INITIAL_CASH
    regime = RegimeDetector()
    ranker = Ranker()
    derivatives = HistoricalDerivativesMock()
    last_pca_fit = -100
    trades = 0
    trade_log = []

    for t in range(start, end):
        prices = {p: data[p]["c"][t] for p in pairs}
        pv = cash + sum(pos["qty"] * prices[p] for p, pos in positions.items())
        history.append(pv)
        if pv > peak: peak = pv
        dd = (peak - pv) / peak if peak > 0 else 0

        # ══════════════════════════════════════════════════════
        # STEP 1: REGIME FILTER (Full rules + PCA-HMM)
        # ══════════════════════════════════════════════════════
        all_raw_feats = {}
        for p in pairs:
            # Match main.py: strictly require 100 bars for regime stability
            if t < 100: continue 
            f = compute_coin_features(data[p]["c"][:t+1], data[p]["h"][:t+1], 
                                      data[p]["l"][:t+1], data[p]["v"][:t+1], 
                                      prices[p], prices[p])
            if f: all_raw_feats[p] = f
            
        market_feats = compute_market_features(all_raw_feats)

        pm = build_price_matrix(data, pairs, t)
        pc1_series = None
        if pm is not None:
            if t - last_pca_fit >= 6:
                regime._pca_last_fit = 0
                last_pca_fit = t
            pc1_series, _, pc1_var, _ = regime.compute_pc1_market_proxy(pm.T)
            if pc1_var > 0: market_feats["pc1_explained_var"] = pc1_var

        regime.update(market_feats, pc1_series)
        if t % 24 == 0 and pc1_series is not None:
            regime.fit_hmm(pc1_series)

        # ══════════════════════════════════════════════════════
        # EXITS (Trailing Stops, Partial Exits & Time Stops)
        # ══════════════════════════════════════════════════════
        for p in list(positions.keys()):
            pos = positions[p]
            price = prices[p]
            pos["high"] = max(pos["high"], price)
            pnl = (price - pos["entry"]) / pos["entry"]
            dfh = (pos["high"] - price) / pos["high"] if pos["high"] > 0 else 0
            bars_held = t - pos["entry_bar"]

            should_exit = False
            if pos["strategy"] == "reversal":
                if pnl <= -0.03 or pnl >= 0.03 or dfh >= 0.02: should_exit = True
                elif bars_held > 48 and pnl < 0.005: should_exit = True # Time stop
            else:
                if pnl <= -0.04: should_exit = True
                elif pnl >= 0.03 and not pos["partial_taken"]:
                    cash += (pos["qty"] * 0.5) * price * (1 - COMMISSION)
                    pos["qty"] *= 0.5; pos["partial_taken"] = True; continue
                elif pos["partial_taken"] or pnl > 0.02:
                    if dfh >= (0.04 if pos["partial_taken"] else 0.03): should_exit = True
                if not should_exit and bars_held > 72 and pnl < 0.01: should_exit = True # Time stop

            if should_exit:
                cash += pos["qty"] * price * (1 - COMMISSION)
                trades += 1
                trade_log.append({"pair": p, "reason": "trailing/time_stop", "pnl": pnl})
                del positions[p]

        if not regime.should_trade():
            continue

        # ══════════════════════════════════════════════════════
        # STEP 2: EVENT FILTER & DERIVATIVES OVERLAY
        # ══════════════════════════════════════════════════════
        zscored = zscore_universe(all_raw_feats)
        for p in zscored: compute_submodel_scores(zscored[p])

        deriv_signals = derivatives.compute_signals(pairs, prices)
        
        signals = {}
        candidates = {}

        for p in all_raw_feats:
            sig = compute_signal(all_raw_feats[p], zscored[p], data[p]["c"][:t+1], data[p]["l"][:t+1], BREAKOUT_LOOKBACK)
            
            # --- EXACT DERIVATIVES LOGIC FROM main.py ---
            dsig = deriv_signals.get(p, {})
            deriv_score = dsig.get("composite_deriv_score", 0)

            if sig["action"] == "BUY":
                if deriv_score < -0.5:
                    sig["action"] = "HOLD"
                    sig["strategy"] = "deriv_suppress"
                elif deriv_score > 0.3:
                    sig["strength"] = min(1.0, sig["strength"] + 0.15)

                if dsig.get("oi_price_divergence") and dsig.get("oi_signal", 0) > 0:
                    if sig["action"] == "HOLD" and all_raw_feats[p].get("overshoot", 0) < -1.0:
                        sig["action"] = "BUY"
                        sig["strategy"] = "oi_divergence"
                        sig["strength"] = 0.7

            sig["deriv_score"] = round(deriv_score, 3)
            signals[p] = sig

            if sig["action"] == "BUY" and p not in positions:
                candidates[p] = zscored[p].copy()
                candidates[p]["_signal"] = sig

        # ══════════════════════════════════════════════════════
        # STEP 4: RANKING AND EXECUTION (BUYS)
        # ══════════════════════════════════════════════════════
        ranked = ranker.rank(candidates, max_results=MAX_POSITIONS)

        eff_max = MAX_TOTAL_EXPOSURE_PCT * regime.get_exposure_multiplier()
        current_exp = sum(pos["qty"] * prices[p] for p, pos in positions.items())

        for p, score, feats in ranked:
            if len(positions) >= MAX_POSITIONS: break
            room = (eff_max * pv) - current_exp
            if room <= 0: break

            sig = feats["_signal"]
            deriv_score = sig.get("deriv_score", 0)

            dvol = max(all_raw_feats[p]["realized_vol"] / np.sqrt(365), 0.03)
            size = (TARGET_RISK_PER_TRADE * pv) / dvol
            size = min(size, MAX_POSITION_PCT * pv, room)
            
            # Risk Management Adjustments
            size *= max(0.0, 1.0 - (dd / 0.10)) # REDD
            size *= max(0.3, sig["strength"])   # Signal Strength
            
            # EXACT SIZING FROM main.py
            deriv_adj = 1.0 + np.clip(deriv_score * 0.2, -0.2, 0.2)
            size *= deriv_adj

            if size > 100:
                qty = size / prices[p]
                cash -= qty * prices[p] * (1 + COMMISSION)
                positions[p] = {
                    "qty": qty, "entry": prices[p], "high": prices[p],
                    "strategy": sig["strategy"], "partial_taken": False,
                    "entry_bar": t
                }
                current_exp += size
                trades += 1

        # ══════════════════════════════════════════════════════
        # SIGNAL SELLS (Breakdowns) - Matching main.py
        # ══════════════════════════════════════════════════════
        for p, sig in signals.items():
            if sig["action"] == "SELL" and p in positions:
                cash += positions[p]["qty"] * prices[p] * (1 - COMMISSION)
                trades += 1
                trade_log.append({"pair": p, "reason": "breakdown_signal", "pnl": (prices[p]/positions[p]["entry"])-1})
                del positions[p]

    for p, pos in positions.items():
        cash += pos["qty"] * data[p]["c"][end-1] * (1 - COMMISSION)
        trades += 1

    return {"history": history, "final": cash, "trades": trades}

# ─── Main Execution ───────────────────────────────────────────

def main():
    print("=" * 70)
    print("  FULL FIDELITY v4 PIPELINE BACKTEST (with Derivatives)")
    print("=" * 70)

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
    res = run_v4_full_simulation(data, pairs, 80, min_len)
    full = compute_metrics(res["history"], INITIAL_CASH, daily_sample_rate=24)
    print_metrics(full, "FULL CONTINUOUS PERIOD")

    # 2. ROLLING 10-DAY TEST (Competition Format)
    print("\n[TEST 2] ROLLING 10-DAY WINDOWS")
    window, step = 240, 72 # 10 days, step 3 days
    results = []
    for s in range(80, min_len - window, step):
        r = run_v4_full_simulation(data, pairs, s, s + window)
        m = compute_metrics(r["history"], INITIAL_CASH, daily_sample_rate=24)
        if m:
            m["trades"] = r["trades"]
            results.append(m)
            print(f"  Window {s//24:>3}: Ret={m['total_return_pct']:>+6.2f}% | "
                  f"Comp={m['composite']:>5.2f} | Trades={m['trades']:>3}")

    # ════════════════════════════════════════════════════════════
    # FINAL UNIFIED SCORECARD
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL BACKTEST SCORECARD (FULL v4)")
    print("=" * 70)

    print("\n  [1] LONG-TERM CONTINUOUS (4 MONTHS)")
    print(f"      Total Return:  {full['total_return_pct']:>8.2f}%")
    print(f"      Max Drawdown:  {full['max_drawdown_pct']:>8.2f}%")
    print(f"      Sharpe:        {full['sharpe']:>8.2f}")
    print(f"      Sortino:       {full['sortino']:>8.2f}")
    print(f"      Calmar:        {full['calmar']:>8.2f}")
    print(f"      Composite:     {full['composite']:>8.2f}")

    if results:
        print("\n  [2] ROLLING 10-DAY WINDOWS (AVERAGES)")
        print(f"      Avg Return:    {np.mean([m['total_return_pct'] for m in results]):>8.2f}%")
        print(f"      Avg Max DD:    {np.mean([m['max_drawdown_pct'] for m in results]):>8.2f}%")
        print(f"      Avg Sharpe:    {np.mean([m['sharpe'] for m in results]):>8.2f}")
        print(f"      Avg Sortino:   {np.mean([m['sortino'] for m in results]):>8.2f}")
        print(f"      Avg Calmar:    {np.mean([m['calmar'] for m in results]):>8.2f}")
        print(f"      Avg Compos:    {np.mean([m['composite'] for m in results]):>8.2f}")
        print(f"      Avg Trades:    {np.mean([m['trades'] for m in results]):>8.1f}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()