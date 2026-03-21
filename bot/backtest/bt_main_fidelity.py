#!/usr/bin/env python3
"""
Historical backtest that mimics main.py as closely as practical on hourly OHLCV.
- Incorporates dynamic Ridge Regression for ranking.
- Accurately models 0.1% Taker and 0.05% Maker fees.
"""
from __future__ import annotations

import os
import sys
import math
import time as pytime
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests

from sklearn.linear_model import Ridge

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "backtest" else _THIS_DIR
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bot.config import (
    BINANCE_BASE_URL, BINANCE_SYMBOL_MAP, BREAKOUT_LOOKBACK,
    MAX_POSITIONS, TRADEABLE_COINS, USE_LIMIT_ORDERS, LIMIT_ORDER_TIMEOUT_SECONDS,
)
from bot.features import compute_coin_features, zscore_universe, compute_submodel_scores
from bot.signals import compute_signal
from bot.ranking import Ranker
from bot.regime_detector import RegimeDetector
from bot.risk_manager import RiskManager
from bot.metrics import PerformanceTracker

INITIAL_CASH = 1_000_000.0
INTERVAL_MS = 3_600_000
WINDOW_BARS = 240
STEP_BARS = 72
WARMUP_BARS = 500  # Increased for ML lookback
DOWNLOAD_MONTHS = 4

DEFAULT_PRICE_PRECISION = 4
DEFAULT_AMOUNT_PRECISION = 6
DEFAULT_MIN_ORDER_USD = 10.0
DEFAULT_SPREAD_BPS = 6.0    
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

class SimRoostooClient:
    def __init__(self, all_candles: dict[str, list[dict]], exchange_info: dict[str, dict], clock: SimClock):
        self.all_candles = all_candles
        self.exchange_info = exchange_info
        self.clock = clock
        self.spread_bps = DEFAULT_SPREAD_BPS
        self.limit_fill_buffer_bps = LIMIT_FILL_BUFFER_BPS
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

    def _bar(self, pair: str, idx: Optional[int] = None) -> dict:
        return self.all_candles[pair][self.clock.ctx.current_index if idx is None else idx]

    def _ticker_one(self, pair: str) -> dict:
        close = float(self._bar(pair)["close"])
        half = (self.spread_bps / 1e4) / 2.0
        return {"LastPrice": close, "MaxBid": close * (1.0 - half), "MinAsk": close * (1.0 + half)}

    def ticker(self) -> dict:
        return {p: self._ticker_one(p) for p in self.all_candles.keys() if self.clock.ctx.current_index < len(self.all_candles[p])}

    def _next_order_id(self) -> int:
        oid = self.order_id
        self.order_id += 1
        return oid

    def _within_bar(self, pair: str, price: float, idx: Optional[int] = None) -> bool:
        c = self._bar(pair, idx=idx)
        tol = price * self.limit_fill_buffer_bps / 1e4
        return (float(c["low"]) - tol) <= price <= (float(c["high"]) + tol)

    # STRICT FEES ENFORCED HERE
    def _apply_fill(self, pair: str, side: str, quantity: float, fill_price: float, is_maker: bool = False) -> dict:
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
            if quantity <= 0: return {"Success": False, "ErrMsg": "no position"}
            self.cash += (quantity * fill_price) - fee_usd
            remaining = held - quantity
            self.positions[pair] = remaining if remaining > 1e-12 else 0.0

        return {"Success": True, "OrderDetail": {"OrderID": oid, "Status": "FILLED", "FilledQuantity": quantity, "FilledAverPrice": fill_price}}

    def place_order(self, pair: str, side: str, quantity: float, order_type: str = "MARKET", price: Optional[float] = None):
        tick = self._ticker_one(pair)
        if order_type == "MARKET":
            fill_price = tick["MinAsk"] if side == "BUY" else tick["MaxBid"]
            return self._apply_fill(pair, side, quantity, fill_price, is_maker=False)

        limit_price = float(price if price is not None else tick["LastPrice"])
        if self._within_bar(pair, limit_price):
            return self._apply_fill(pair, side, quantity, limit_price, is_maker=False)

        oid = self._next_order_id()
        self.open_orders[oid] = {"pair": pair, "side": side, "quantity": quantity, "price": limit_price, "placed_time": self.clock.time()}
        return {"Success": True, "OrderDetail": {"OrderID": oid, "Status": "PENDING"}}

    def cancel_order(self, order_id: Optional[int] = None):
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
    def __init__(self, client: SimRoostooClient, exchange_info: dict):
        self.client = client
        self.exchange_info = exchange_info
        self.pending_orders: dict[int, dict] = {}

    def _get_precision(self, pair: str):
        info = self.exchange_info.get(pair, {})
        return info.get("PricePrecision", DEFAULT_PRICE_PRECISION), info.get("AmountPrecision", DEFAULT_AMOUNT_PRECISION), info.get("MiniOrder", DEFAULT_MIN_ORDER_USD)

    def _round_quantity(self, quantity: float, pair: str) -> float:
        prec = self._get_precision(pair)[1]
        return float(int(quantity)) if prec == 0 else round(quantity, prec)

    def _round_price(self, price: float, pair: str) -> float:
        return round(price, self._get_precision(pair)[0])

    def buy(self, pair: str, quantity_usd: float, current_price: float, bid: float, ask: float, use_limit: bool = True):
        coin_qty = self._round_quantity(quantity_usd / current_price, pair)
        if coin_qty <= 0 or (coin_qty * current_price < self._get_precision(pair)[2]): return None
        if use_limit and bid > 0 and ask > 0:
            limit_price = self._round_price((bid + ask) / 2.0, pair)
            res = self.client.place_order(pair, "BUY", coin_qty, "LIMIT", limit_price)
            if res and res.get("Success") and res.get("OrderDetail", {}).get("Status") == "PENDING":
                self.pending_orders[res["OrderDetail"]["OrderID"]] = {"pair": pair, "side": "BUY", "time_placed": self.client.clock.time(), "price": limit_price, "quantity": coin_qty}
            return res
        return self.client.place_order(pair, "BUY", coin_qty, "MARKET")

    def sell(self, pair: str, coin_quantity: float, current_price: float, bid: float, ask: float, use_limit: bool = True):
        coin_qty = self._round_quantity(coin_quantity, pair)
        if coin_qty <= 0 or (coin_qty * current_price < self._get_precision(pair)[2]): return None
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

class HistoricalMainLikeBacktest:
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

    def train_ridge_model(self, current_t: int, ranker_features: list, lookback: int = 400, forward: int = 24):
        start_idx = max(100, current_t - lookback - forward)
        end_idx = current_t - forward
        if end_idx - start_idx < 50: return None
        
        X_train, y_train = [], []
        for t in range(start_idx, end_idx, 6):
            raw_features = {}
            for pair in self.active_pairs:
                closes = [x["close"] for x in self.all_candles[pair][:t+1]]
                if len(closes) < 100: continue
                highs = [x["high"] for x in self.all_candles[pair][:t+1]]
                lows = [x["low"] for x in self.all_candles[pair][:t+1]]
                volumes = [x["volume"] for x in self.all_candles[pair][:t+1]]
                feats = compute_coin_features(closes, highs, lows, volumes, closes[-1], closes[-1])
                if feats: raw_features[pair] = feats
                
            zscored = zscore_universe(raw_features)
            for pair, f_z in zscored.items():
                c0 = self.all_candles[pair][t]["close"]
                cf = self.all_candles[pair][t + forward]["close"]
                X_train.append([f_z.get(k, 0.0) for k in ranker_features])
                y_train.append((cf - c0) / c0)
                
        if len(X_train) > 50:
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            return model
        return None

    def run_window(self, start_idx: int, end_idx: int) -> dict:
        regime = RegimeDetector()
        ranker = Ranker()
        risk_mgr = RiskManager(INITIAL_CASH)
        perf = PerformanceTracker(INITIAL_CASH)
        
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

            # RIDGE TRAINING INJECTION
            if cycle_count % 24 == 1:
                try:
                    new_model = self.train_ridge_model(t, ranker.ridge_features)
                    if new_model: ranker.set_ridge_model(new_model)
                except Exception: pass

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

            price_matrix = self._build_price_matrix(t)
            pc1_series = None
            if price_matrix is not None:
                pc1_series = regime.compute_pc1_market_proxy(price_matrix)[0]

            regime.update(pc1_series=pc1_series)
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

            signals = {}
            valid_candidates = {}
            for pair in all_raw_features:
                hist = self.all_candles[pair][:t+1]
                closes = np.array([x["close"] for x in hist])
                lows = np.array([x["low"] for x in hist])
                sig = compute_signal(all_raw_features[pair], all_zscored[pair], closes, lows, BREAKOUT_LOOKBACK)
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
                size_usd = risk_mgr.position_size_usd(pair, all_raw_features.get(pair, {}).get("realized_vol", 0.5), total_exposure, num_positions, signal_strength=sig.get("strength", 0.5))
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
                    tick = ticker_data.get(pair, {})
                    executor.sell(pair, positions[pair], tick.get("LastPrice",0), tick.get("MaxBid",0), tick.get("MinAsk",0), use_limit=not sig.get("breakdown", False))
                    risk_mgr.clear_trailing_stop(pair)
                    trade_log.append({"pair": pair, "reason": "breakdown_signal"})

        for pair, qty in list(client.positions.items()):
            if qty > 0:
                tick = client.ticker().get(pair, {})
                executor.sell(pair, qty, tick.get("LastPrice", 0), tick.get("MaxBid", 0), tick.get("MinAsk", 0), use_limit=False)

        return {"summary": perf.summary(), "trade_log": trade_log}

def default_exchange_info(pairs: list[str]) -> dict[str, dict]:
    return {p: {"PricePrecision": DEFAULT_PRICE_PRECISION, "AmountPrecision": DEFAULT_AMOUNT_PRECISION, "MiniOrder": DEFAULT_MIN_ORDER_USD} for p in pairs}

def download_4m_hourly(pair: str, session: requests.Session, months: int = DOWNLOAD_MONTHS) -> list[dict]:
    symbol = BINANCE_SYMBOL_MAP.get(pair)
    if not symbol: return []
    end_ms = int(pytime.time() * 1000)
    start_ms = end_ms - months * 30 * 24 * INTERVAL_MS
    out, cur = [], start_ms
    while cur < end_ms:
        try:
            r = session.get(f"{BINANCE_BASE_URL}/api/v3/klines", params={"symbol": symbol, "interval": "1h", "startTime": cur, "endTime": end_ms, "limit": 1000}, timeout=15)
            raw = r.json()
        except Exception: break
        if not raw: break
        for k in raw: out.append({"open_time": k[0], "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]), "close_time": k[6]})
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
    print("  LEAN V5 BACKTEST (RIDGE REGRESSION & STRICT FEES)")
    print("=" * 70)

    pairs = [p for p in TRADEABLE_COINS if p in BINANCE_SYMBOL_MAP]
    all_candles = load_all_history(pairs)
    if len(all_candles) < 2: return

    min_len = min(len(v) for v in all_candles.values())
    usable_pairs = [p for p, v in all_candles.items() if len(v) >= min_len]
    all_candles = {p: v[:min_len] for p, v in all_candles.items() if p in usable_pairs}
    bt = HistoricalMainLikeBacktest(all_candles, default_exchange_info(list(all_candles.keys())))

    print(f"\n[TEST 1] {DOWNLOAD_MONTHS}-MONTH CONTINUOUS RUN (Using rolling ML model)")
    full = bt.run_window(WARMUP_BARS, min_len)

    print("\n[TEST 2] ROLLING 10-DAY WINDOWS")
    results = []
    # Steps forward by STEP_BARS (72 hours), testing WINDOW_BARS (240 hours / 10 days)
    for s in range(WARMUP_BARS, min_len - WINDOW_BARS, STEP_BARS):
        res = bt.run_window(s, s + WINDOW_BARS)
        m = dict(res["summary"])
        m["trades"] = len(res["trade_log"])
        results.append(m)
        print(f"  Window {s // 24:>3}: Ret={m['total_return_pct']:>+6.2f}% | Comp={m['composite']:>6.2f} | Trades={m['trades']:>3}")

    print("\n" + "=" * 70 + "\n  FINAL BACKTEST SCORECARD (RIDGE ML)\n" + "=" * 70)
    f_m = full["summary"]
    print("\n  [1] LONG-TERM CONTINUOUS (4 MONTHS)")
    print(f"      Total Return:  {f_m['total_return_pct']:>8.2f}%")
    print(f"      Max Drawdown:  {f_m['max_drawdown_pct']:>8.2f}%")
    print(f"      Sharpe:        {f_m['sharpe']:>8.2f}")
    print(f"      Sortino:       {f_m['sortino']:>8.2f}")
    print(f"      Calmar:        {f_m['calmar']:>8.2f}")
    print(f"      Composite:     {f_m['composite']:>8.2f}")
    print(f"      Total Trades:  {len(full['trade_log']):>8}")

    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
