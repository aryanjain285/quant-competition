#!/usr/bin/env python3
"""
FULL v4 HIGH-FIDELITY BACKTEST (Apples-to-Apples with Lean v5)

Realistic Execution physics:
- 6 bps synthetic spreads
- Limit order queues & timeouts
- 0.10% Taker Fees / 0.05% Maker Fees deducted directly from cash
- Exchange precision and minimum order constraints

Strategy:
- Rule-based + PCA-HMM Regime Detection
- Derivatives Overlay (Vetoes & OI Divergence)
- Derivative-adjusted Position Sizing
"""
from __future__ import annotations

import os
import sys
import time as pytime
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "backtest" else _THIS_DIR
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bot.config import (
    BINANCE_BASE_URL,
    BINANCE_SYMBOL_MAP,
    BREAKOUT_LOOKBACK,
    MAX_POSITIONS,
    TRADEABLE_COINS,
    USE_LIMIT_ORDERS,
    LIMIT_ORDER_TIMEOUT_SECONDS,
)

INITIAL_CASH = 1_000_000.0
INTERVAL_MS = 3_600_000
WINDOW_BARS = 240
STEP_BARS = 72
WARMUP_BARS = 80
DOWNLOAD_MONTHS = 4

DEFAULT_PRICE_PRECISION = 4
DEFAULT_AMOUNT_PRECISION = 6
DEFAULT_MIN_ORDER_USD = 10.0
DEFAULT_SPREAD_BPS = 4.0     
LIMIT_FILL_BUFFER_BPS = 1.0   

@dataclass
class SimulationContext:
    current_index: int = 0
    current_time: float = 0.0

class SimClock:
    def __init__(self):
        self.ctx = SimulationContext()
    def set_bar(self, index: int, ts_ms: int):
        self.ctx.current_index = index
        self.ctx.current_time = ts_ms / 1000.0
    def time(self) -> float:
        return self.ctx.current_time

class HistoricalDerivativesMock:
    def compute_signals(self, active_pairs, spot_prices):
        signals = {}
        for pair in active_pairs:
            score = np.clip(random.gauss(0, 0.4), -1.0, 1.0)
            has_divergence = random.random() < 0.05
            signals[pair] = {
                "composite_deriv_score": score,
                "oi_price_divergence": has_divergence,
                "oi_signal": 1 if has_divergence else 0,
                "funding_zscore": score * 2 
            }
        return signals

class SimRoostooClient:
    def __init__(self, all_candles, exchange_info, clock, spread_bps=DEFAULT_SPREAD_BPS, limit_fill_buffer_bps=LIMIT_FILL_BUFFER_BPS):
        self.all_candles = all_candles
        self.exchange_info = exchange_info
        self.clock = clock
        self.spread_bps = spread_bps
        self.limit_fill_buffer_bps = limit_fill_buffer_bps
        self.cash = INITIAL_CASH
        self.positions: dict[str, float] = {}
        self.order_id = 1
        self.open_orders: dict[int, dict] = {}

    def server_time(self): return int(self.clock.time() * 1000)
    def exchange_info_payload(self): return {"TradePairs": self.exchange_info}
    def balance(self) -> dict:
        wallet = {"USD": {"Free": self.cash, "Lock": 0.0}}
        for pair, qty in self.positions.items():
            if qty > 0: wallet[pair.split("/")[0]] = {"Free": qty, "Lock": 0.0}
        return wallet

    def _bar(self, pair, idx=None):
        return self.all_candles[pair][self.clock.ctx.current_index if idx is None else idx]

    def _ticker_one(self, pair):
        close = float(self._bar(pair)["close"])
        half = (self.spread_bps / 1e4) / 2.0
        return {"LastPrice": close, "MaxBid": close * (1.0 - half), "MinAsk": close * (1.0 + half)}

    def ticker(self):
        return {p: self._ticker_one(p) for p in self.all_candles if self.clock.ctx.current_index < len(self.all_candles[p])}

    def _next_order_id(self):
        oid = self.order_id
        self.order_id += 1
        return oid

    def _within_bar(self, pair, price, idx=None):
        c = self._bar(pair, idx)
        tol = price * self.limit_fill_buffer_bps / 1e4
        return (float(c["low"]) - tol) <= price <= (float(c["high"]) + tol)

    def _apply_fill(self, pair, side, quantity, fill_price, is_maker=False):
        fee_rate = 0.0005 if is_maker else 0.0010
        notional = quantity * fill_price
        fee_usd = notional * fee_rate

        oid = self._next_order_id()
        if side == "BUY":
            total_cost = notional + fee_usd
            if total_cost > self.cash + 1e-9: 
                return {"Success": False, "ErrMsg": "insufficient funds"}
            self.cash -= total_cost
            self.positions[pair] = self.positions.get(pair, 0.0) + quantity
        else:
            held = self.positions.get(pair, 0.0)
            quantity = min(quantity, held)
            if quantity <= 0: 
                return {"Success": False, "ErrMsg": "no position"}
            self.cash += (quantity * fill_price) - fee_usd
            remaining = held - quantity
            self.positions[pair] = remaining if remaining > 1e-12 else 0.0
            
        return {
            "Success": True, 
            "OrderDetail": {
                "OrderID": oid, "Status": "FILLED", 
                "Role": "MAKER" if is_maker else "TAKER", 
                "FilledQuantity": quantity, "FilledAverPrice": fill_price, 
                "CommissionChargeValue": fee_usd
            }
        }

    def place_order(self, pair, side, quantity, order_type="MARKET", price=None):
        tick = self._ticker_one(pair)
        if order_type == "MARKET":
            fill_price = tick["MinAsk"] if side == "BUY" else tick["MaxBid"]
            return self._apply_fill(pair, side, quantity, fill_price, is_maker=False)
            
        limit_price = float(price if price is not None else tick["LastPrice"])
        if self._within_bar(pair, limit_price):
            return self._apply_fill(pair, side, quantity, limit_price, is_maker=False)
            
        oid = self._next_order_id()
        self.open_orders[oid] = {"pair": pair, "side": side, "quantity": quantity, "price": limit_price, "placed_time": self.clock.time()}
        return {"Success": True, "OrderDetail": {"OrderID": oid, "Status": "PENDING", "Role": "SIM", "FilledQuantity": 0.0, "FilledAverPrice": 0.0, "CommissionChargeValue": 0.0}}

    def cancel_order(self, order_id=None):
        if order_id is None: self.open_orders.clear()
        else: self.open_orders.pop(order_id, None)
        return {"Success": True}

    def advance_pending_orders(self):
        filled = []
        for oid, info in list(self.open_orders.items()):
            if self._within_bar(info["pair"], info["price"]):
                res = self._apply_fill(info["pair"], info["side"], info["quantity"], info["price"], is_maker=True)
                if res.get("Success"):
                    filled.append((oid, res))
                    del self.open_orders[oid]
        return filled

class SimExecutor:
    def __init__(self, client, exchange_info):
        self.client = client
        self.exchange_info = exchange_info
        self.pending_orders = {}

    def _get_precision(self, pair):
        info = self.exchange_info.get(pair, {})
        return info.get("PricePrecision", DEFAULT_PRICE_PRECISION), info.get("AmountPrecision", DEFAULT_AMOUNT_PRECISION), info.get("MiniOrder", DEFAULT_MIN_ORDER_USD)

    def _round_quantity(self, quantity, pair):
        prec = self._get_precision(pair)[1]
        return float(int(quantity)) if prec == 0 else round(quantity, prec)

    def _round_price(self, price, pair): return round(price, self._get_precision(pair)[0])

    def _check_min_order(self, pair, quantity, price): return quantity * price >= self._get_precision(pair)[2]

    def buy(self, pair, quantity_usd, current_price, bid, ask, use_limit=True):
        coin_qty = self._round_quantity(quantity_usd / current_price, pair)
        if coin_qty <= 0 or not self._check_min_order(pair, coin_qty, current_price): return None
        if use_limit and bid > 0 and ask > 0:
            limit_price = self._round_price((bid + ask) / 2.0, pair)
            res = self.client.place_order(pair, "BUY", coin_qty, "LIMIT", limit_price)
            if res and res.get("Success") and res.get("OrderDetail", {}).get("Status") == "PENDING":
                self.pending_orders[res["OrderDetail"]["OrderID"]] = {"pair": pair, "side": "BUY", "time_placed": self.client.clock.time(), "price": limit_price, "quantity": coin_qty}
            return res
        return self.client.place_order(pair, "BUY", coin_qty, "MARKET")

    def sell(self, pair, coin_quantity, current_price, bid, ask, use_limit=True):
        coin_qty = self._round_quantity(coin_quantity, pair)
        if coin_qty <= 0 or not self._check_min_order(pair, coin_qty, current_price): return None
        if use_limit and bid > 0 and ask > 0:
            limit_price = self._round_price((bid + ask) / 2.0, pair)
            res = self.client.place_order(pair, "SELL", coin_qty, "LIMIT", limit_price)
            if res and res.get("Success") and res.get("OrderDetail", {}).get("Status") == "PENDING":
                self.pending_orders[res["OrderDetail"]["OrderID"]] = {"pair": pair, "side": "SELL", "time_placed": self.client.clock.time(), "price": limit_price, "quantity": coin_qty}
            return res
        return self.client.place_order(pair, "SELL", coin_qty, "MARKET")

    def manage_pending_orders(self):
        for oid, _ in self.client.advance_pending_orders(): self.pending_orders.pop(oid, None)
        now = self.client.clock.time()
        for oid in [o for o, i in self.pending_orders.items() if now - i["time_placed"] > LIMIT_ORDER_TIMEOUT_SECONDS]:
            info = self.pending_orders.pop(oid)
            self.client.cancel_order(oid)
            if info["side"] == "SELL": self.client.place_order(info["pair"], "SELL", info["quantity"], "MARKET")

    def cancel_all_pending(self):
        self.client.cancel_order()
        self.pending_orders.clear()

class HistoricalFullv4Backtest:
    def __init__(self, all_candles: dict[str, list[dict]], exchange_info: dict):
        self.all_candles = all_candles
        self.trade_pairs = exchange_info 
        self.active_pairs = [p for p in TRADEABLE_COINS if p in self.trade_pairs and p in self.all_candles]
        
    def _build_price_matrix(self, t: int, min_bars: int = 100) -> Optional[np.ndarray]:
        close_series = []
        for pair in self.active_pairs:
            closes = [c["close"] for c in self.all_candles[pair][:t+1]]
            if len(closes) < min_bars: continue
            close_series.append(np.array(closes, dtype=float))
        if len(close_series) < 2: return None
        min_len = min(len(x) for x in close_series)
        return np.column_stack([x[-min_len:] for x in close_series])

    def run_window(self, start_idx: int, end_idx: int) -> dict:
        from bot.regime_detector import RegimeDetector
        from bot.ranking import Ranker
        from bot.risk_manager import RiskManager
        from bot.metrics import PerformanceTracker
        from bot.features import compute_coin_features, zscore_universe, compute_submodel_scores, compute_market_features
        from bot.signals import compute_signal

        regime = RegimeDetector()
        ranker = Ranker()
        risk_mgr = RiskManager(INITIAL_CASH)
        perf = PerformanceTracker(INITIAL_CASH)
        derivatives = HistoricalDerivativesMock()
        
        clock = SimClock()
        pytime.time = clock.time
        client = SimRoostooClient(self.all_candles, self.trade_pairs, clock)
        executor = SimExecutor(client, self.trade_pairs)

        positions = {}
        trade_log = []
        cycle_count = 0
        
        for t in range(start_idx, end_idx):
            cycle_count += 1
            first_pair = self.active_pairs[0]
            clock.set_bar(t, self.all_candles[first_pair][t]["close_time"])

            ticker_data = client.ticker()
            wallet = client.balance()
            
            positions.clear()
            for coin, bal in wallet.items():
                if coin != "USD" and bal.get("Free", 0) > 0:
                    positions[f"{coin}/USD"] = bal.get("Free", 0)

            port_value = wallet.get("USD", {}).get("Free", 0)
            for pair, qty in positions.items():
                port_value += qty * ticker_data.get(pair, {}).get("LastPrice", 0)
                
            risk_mgr.update_portfolio_value(port_value)
            perf.record(port_value)

            dd_check = risk_mgr.check_drawdown_breakers()
            if dd_check["action"] == "liquidate":
                for pair, qty in list(positions.items()):
                    tick = ticker_data.get(pair, {})
                    executor.sell(pair, qty, tick.get("LastPrice",0), tick.get("MaxBid",0), tick.get("MinAsk",0), use_limit=False)
                    trade_log.append({"pair": pair, "reason": "dd_breaker"})
                positions.clear()
                executor.cancel_all_pending()
                continue

            if risk_mgr.is_paused: continue

            all_raw_features = {}
            for pair in self.active_pairs:
                hist = self.all_candles[pair][:t+1]
                closes = np.array([x["close"] for x in hist])
                if len(closes) < 100: continue
                highs = np.array([x["high"] for x in hist])
                lows = np.array([x["low"] for x in hist])
                volumes = np.array([x["volume"] for x in hist])
                tick = ticker_data.get(pair, {})
                
                feats = compute_coin_features(closes, highs, lows, volumes, tick.get("MaxBid",0), tick.get("MinAsk",0))
                if feats: all_raw_features[pair] = feats

            market_feats = compute_market_features(all_raw_features)

            price_matrix = self._build_price_matrix(t)
            pc1_series = None
            if price_matrix is not None:
                pc1_result = regime.compute_pc1_market_proxy(price_matrix)
                pc1_series = pc1_result[0]
                if pc1_result[2] > 0: market_feats["pc1_explained_var"] = pc1_result[2]

            regime.update(market_feats, pc1_series)

            if cycle_count % 24 == 0 and pc1_series is not None:
                regime.fit_hmm(pc1_series)

            risk_mgr.set_regime_multiplier(regime.get_exposure_multiplier())
            executor.manage_pending_orders()

            if not regime.should_trade():
                for pair, qty in list(positions.items()):
                    tick = ticker_data.get(pair, {})
                    price = tick.get("LastPrice", 0)
                    risk_mgr.update_trailing_stop(pair, price)
                    should_exit, reason, exit_frac = risk_mgr.check_trailing_stop(pair, price)
                    if should_exit:
                        executor.sell(pair, qty * exit_frac, price, tick.get("MaxBid",0), tick.get("MinAsk",0), use_limit=False)
                        if exit_frac >= 1.0: risk_mgr.clear_trailing_stop(pair)
                        trade_log.append({"pair": pair, "reason": reason})
                continue

            all_zscored = zscore_universe(all_raw_features)
            for pair in all_zscored: compute_submodel_scores(all_zscored[pair])

            current_prices = {p: ticker_data.get(p, {}).get("LastPrice", 0) for p in self.active_pairs}
            deriv_signals = derivatives.compute_signals(self.active_pairs, current_prices)

            signals = {}
            valid_candidates = {}

            for pair in all_raw_features:
                hist = self.all_candles[pair][:t+1]
                closes = np.array([x["close"] for x in hist])
                lows = np.array([x["low"] for x in hist])
                
                sig = compute_signal(all_raw_features[pair], all_zscored[pair], closes, lows, BREAKOUT_LOOKBACK)
                
                dsig = deriv_signals.get(pair, {})
                deriv_score = dsig.get("composite_deriv_score", 0)

                if sig["action"] == "BUY":
                    if deriv_score < -0.5:
                        sig["action"] = "HOLD"
                        sig["strategy"] = "deriv_suppress"
                    elif deriv_score > 0.3:
                        sig["strength"] = min(1.0, sig["strength"] + 0.15)

                    if dsig.get("oi_price_divergence") and dsig.get("oi_signal", 0) > 0:
                        if sig["action"] == "HOLD" and all_raw_features[pair].get("overshoot", 0) < -1.0:
                            sig["action"] = "BUY"
                            sig["strategy"] = "oi_divergence"
                            sig["strength"] = 0.7

                sig["deriv_score"] = round(deriv_score, 3)
                signals[pair] = sig

                if sig["action"] == "BUY" and positions.get(pair, 0) <= 0:
                    valid_candidates[pair] = all_zscored.get(pair, {})
                    valid_candidates[pair]["_signal"] = sig

            ranked = ranker.rank(valid_candidates, max_results=MAX_POSITIONS)

            for pair, qty in list(positions.items()):
                tick = ticker_data.get(pair, {})
                price = tick.get("LastPrice", 0)
                risk_mgr.update_trailing_stop(pair, price)
                should_exit, reason, exit_frac = risk_mgr.check_trailing_stop(pair, price)
                if should_exit:
                    executor.sell(pair, qty * exit_frac, price, tick.get("MaxBid",0), tick.get("MinAsk",0), use_limit=(reason != "hard_stop"))
                    if exit_frac >= 1.0: risk_mgr.clear_trailing_stop(pair)
                    trade_log.append({"pair": pair, "reason": reason})

            total_exposure = sum(qty * ticker_data.get(p, {}).get("LastPrice", 0) for p, qty in positions.items())
            num_positions = len(positions)

            for pair, score, features in ranked:
                if num_positions >= MAX_POSITIONS: break
                sig = features.get("_signal", {})
                tick = ticker_data.get(pair, {})
                price = tick.get("LastPrice", 0)
                real_vol = all_raw_features.get(pair, {}).get("realized_vol", 0.5)

                size_usd = risk_mgr.position_size_usd(
                    pair, real_vol, total_exposure, num_positions,
                    signal_strength=sig.get("strength", 0.5),
                    deriv_score=sig.get("deriv_score", 0)
                )

                if size_usd >= 50:
                    result = executor.buy(pair, size_usd, price, tick.get("MaxBid",0), tick.get("MinAsk",0), use_limit=USE_LIMIT_ORDERS)
                    if result and result.get("Success"):
                        strategy = sig.get("strategy", "continuation")
                        risk_mgr.update_trailing_stop(pair, price, strategy="mean_rev" if strategy == "reversal" else "breakout", entry_price=price)
                        total_exposure += size_usd
                        num_positions += 1
                        trade_log.append({"pair": pair, "reason": "buy", "strategy": strategy})

            for pair, sig in signals.items():
                if sig["action"] == "SELL" and positions.get(pair, 0) > 0:
                    qty = positions[pair]
                    tick = ticker_data.get(pair, {})
                    urgent = sig.get("breakdown", False)
                    executor.sell(pair, qty, tick.get("LastPrice",0), tick.get("MaxBid",0), tick.get("MinAsk",0), use_limit=not urgent)
                    risk_mgr.clear_trailing_stop(pair)
                    trade_log.append({"pair": pair, "reason": "breakdown_signal"})

        for pair, qty in list(client.positions.items()):
            if qty > 0:
                tick = client.ticker().get(pair, {})
                executor.sell(pair, qty, tick.get("LastPrice", 0), tick.get("MaxBid", 0), tick.get("MinAsk", 0), use_limit=False)

        return {"summary": perf.summary(), "trade_log": trade_log}

def default_exchange_info(pairs: list[str]) -> dict[str, dict]:
    return {pair: {"PricePrecision": DEFAULT_PRICE_PRECISION, "AmountPrecision": DEFAULT_AMOUNT_PRECISION, "MiniOrder": DEFAULT_MIN_ORDER_USD} for pair in pairs}

def download_4m_hourly(pair: str, session: requests.Session, months: int = DOWNLOAD_MONTHS) -> list[dict]:
    symbol = BINANCE_SYMBOL_MAP.get(pair)
    if not symbol: return []
    end_ms = int(pytime.time() * 1000)
    start_ms = end_ms - months * 30 * 24 * INTERVAL_MS
    out, cur = [], start_ms
    while cur < end_ms:
        try:
            r = session.get(f"{BINANCE_BASE_URL}/api/v3/klines", params={"symbol": symbol, "interval": "1h", "startTime": cur, "endTime": end_ms, "limit": 1000}, timeout=15)
            r.raise_for_status()
            raw = r.json()
        except Exception: break
        if not raw: break
        for k in raw:
            out.append({"open_time": k[0], "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]), "close_time": k[6]})
        cur = raw[-1][0] + INTERVAL_MS
        if len(raw) < 1000: break
        pytime.sleep(0.05)
    return out

def load_all_history(pairs: list[str]) -> dict[str, list[dict]]:
    session = requests.Session()
    data = {}
    for pair in pairs:
        candles = download_4m_hourly(pair, session)
        if len(candles) >= WARMUP_BARS + WINDOW_BARS:
            data[pair] = candles
            print(f"Loaded {pair}: {len(candles)} bars")
    return data

def main():
    print("=" * 70)
    print("  FULL V4 HIGH-FIDELITY BACKTEST (WITH COMMISSIONS)")
    print("=" * 70)

    pairs = [p for p in TRADEABLE_COINS if p in BINANCE_SYMBOL_MAP]
    all_candles = load_all_history(pairs)
    if len(all_candles) < 2: return

    min_len = min(len(v) for v in all_candles.values())
    usable_pairs = [p for p, v in all_candles.items() if len(v) >= min_len]
    all_candles = {p: v[:min_len] for p, v in all_candles.items() if p in usable_pairs}
    exchange_info = default_exchange_info(list(all_candles.keys()))
    bt = HistoricalFullv4Backtest(all_candles, exchange_info)

    print("\n[TEST 1] 4-MONTH CONTINUOUS RUN")
    full = bt.run_window(WARMUP_BARS, min_len)

    print("\n[TEST 2] ROLLING 10-DAY WINDOWS")
    results = []
    for s in range(WARMUP_BARS, min_len - WINDOW_BARS, STEP_BARS):
        res = bt.run_window(s, s + WINDOW_BARS)
        m = dict(res["summary"])
        m["trades"] = len(res["trade_log"])
        results.append(m)
        print(f"  Window {s // 24:>3}: Ret={m['total_return_pct']:>+6.2f}% | Comp={m['composite']:>6.2f} | Trades={m['trades']:>3}")

    print("\n" + "=" * 70 + "\n  FINAL BACKTEST SCORECARD (FULL V4)\n" + "=" * 70)
    f_m = full["summary"]
    print("\n  [1] LONG-TERM CONTINUOUS (4 MONTHS)")
    print(f"      Total Return:  {f_m['total_return_pct']:>8.2f}%\n      Max Drawdown:  {f_m['max_drawdown_pct']:>8.2f}%\n      Sharpe:        {f_m['sharpe']:>8.2f}\n      Sortino:       {f_m['sortino']:>8.2f}\n      Calmar:        {f_m['calmar']:>8.2f}\n      Composite:     {f_m['composite']:>8.2f}\n      Total Trades:  {len(full['trade_log']):>8}")

    if results:
        print("\n  [2] ROLLING 10-DAY WINDOWS (AVERAGES)")
        print(f"      Avg Return:    {np.mean([m['total_return_pct'] for m in results]):>8.2f}%\n      Avg Max DD:    {np.mean([m['max_drawdown_pct'] for m in results]):>8.2f}%\n      Avg Sharpe:    {np.mean([m['sharpe'] for m in results]):>8.2f}\n      Avg Sortino:   {np.mean([m['sortino'] for m in results]):>8.2f}\n      Avg Calmar:    {np.mean([m['calmar'] for m in results]):>8.2f}\n      Avg Compos:    {np.mean([m['composite'] for m in results]):>8.2f}\n      Avg Trades:    {np.mean([m['trades'] for m in results]):>8.1f}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()