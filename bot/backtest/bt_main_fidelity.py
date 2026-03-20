#!/usr/bin/env python3
"""
Historical backtest that mimics main.py as closely as practical on hourly OHLCV.

What it reuses from the live bot:
- bot.features / bot.signals / bot.ranking / bot.regime_detector
- bot.risk_manager / bot.metrics
- bot.config parameters and universe
- the same orchestration order as TradingBot.run_cycle()

What it approximates:
- ticker bid/ask from hourly close using a configurable synthetic spread
- Roostoo execution using a simulated client over historical bars
- exchange precision / min-order constraints with configurable defaults

Outputs:
- 4-month continuous backtest
- rolling 10-day windows (240 hourly bars, step 72 bars)
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

# Allow imports from bot package when this file lives in bot/backtest or standalone.
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
from bot.binance_data import BinanceData
from bot.features import compute_coin_features, zscore_universe, compute_submodel_scores
from bot.signals import compute_signal
from bot.ranking import Ranker
from bot.regime_detector import RegimeDetector
from bot.risk_manager import RiskManager
from bot.metrics import PerformanceTracker
import bot.risk_manager as risk_manager_module
import bot.executor as executor_module


INITIAL_CASH = 1_000_000.0
INTERVAL_MS = 3_600_000
WINDOW_BARS = 240
STEP_BARS = 72
WARMUP_BARS = 80
DOWNLOAD_MONTHS = 4

# Backtest assumptions for live-mimicking execution.
DEFAULT_PRICE_PRECISION = 4
DEFAULT_AMOUNT_PRECISION = 6
DEFAULT_MIN_ORDER_USD = 10.0
DEFAULT_SPREAD_BPS = 6.0      # synthetic spread around close when no real bid/ask history exists
LIMIT_FILL_BUFFER_BPS = 1.0   # small tolerance around bar range for immediate fills


@dataclass
class SimulationContext:
    current_index: int = 0
    current_time: float = 0.0


class SimClock:
    def __init__(self):
        self.ctx = SimulationContext()

    def set_bar(self, index: int, ts_ms: int):
        # Use candle close time in seconds.
        self.ctx.current_index = index
        self.ctx.current_time = ts_ms / 1000.0

    def time(self) -> float:
        return self.ctx.current_time


class SimRoostooClient:
    """Historical simulator that exposes a small subset of RoostooClient methods.

    This lets us preserve Executor-style semantics without touching live APIs.
    """

    def __init__(
        self,
        all_candles: dict[str, list[dict]],
        exchange_info: dict[str, dict],
        clock: SimClock,
        spread_bps: float = DEFAULT_SPREAD_BPS,
        limit_fill_buffer_bps: float = LIMIT_FILL_BUFFER_BPS,
    ):
        self.all_candles = all_candles
        self.exchange_info = exchange_info
        self.clock = clock
        self.spread_bps = spread_bps
        self.limit_fill_buffer_bps = limit_fill_buffer_bps
        self.cash = INITIAL_CASH
        self.positions: dict[str, float] = {}
        self.order_id = 1
        self.open_orders: dict[int, dict] = {}

    def server_time(self):
        return int(self.clock.time() * 1000)

    def exchange_info_payload(self):
        return {"TradePairs": self.exchange_info}

    def balance(self) -> dict:
        wallet = {"USD": {"Free": self.cash, "Lock": 0.0}}
        for pair, qty in self.positions.items():
            if qty <= 0:
                continue
            coin = pair.split("/")[0]
            wallet[coin] = {"Free": qty, "Lock": 0.0}
        return wallet

    def _bar(self, pair: str, idx: Optional[int] = None) -> dict:
        idx = self.clock.ctx.current_index if idx is None else idx
        return self.all_candles[pair][idx]

    def _ticker_one(self, pair: str) -> dict:
        c = self._bar(pair)
        close = float(c["close"])
        half = (self.spread_bps / 1e4) / 2.0
        bid = close * (1.0 - half)
        ask = close * (1.0 + half)
        return {
            "LastPrice": close,
            "MaxBid": bid,
            "MinAsk": ask,
        }

    def ticker(self) -> dict:
        return {p: self._ticker_one(p) for p in self.all_candles.keys() if self.clock.ctx.current_index < len(self.all_candles[p])}

    def _next_order_id(self) -> int:
        oid = self.order_id
        self.order_id += 1
        return oid

    def _within_bar(self, pair: str, price: float, idx: Optional[int] = None) -> bool:
        c = self._bar(pair, idx=idx)
        lo = float(c["low"])
        hi = float(c["high"])
        tol = price * self.limit_fill_buffer_bps / 1e4
        return (lo - tol) <= price <= (hi + tol)

    def _apply_fill(self, pair: str, side: str, quantity: float, fill_price: float) -> dict:
        oid = self._next_order_id()
        if side == "BUY":
            notional = quantity * fill_price
            if notional > self.cash + 1e-9:
                return {"Success": False, "ErrMsg": "insufficient funds"}
            self.cash -= notional
            self.positions[pair] = self.positions.get(pair, 0.0) + quantity
        else:
            held = self.positions.get(pair, 0.0)
            quantity = min(quantity, held)
            if quantity <= 0:
                return {"Success": False, "ErrMsg": "no position"}
            self.cash += quantity * fill_price
            remaining = held - quantity
            self.positions[pair] = remaining if remaining > 1e-12 else 0.0

        return {
            "Success": True,
            "OrderDetail": {
                "OrderID": oid,
                "Status": "FILLED",
                "Role": "SIM",
                "FilledQuantity": quantity,
                "FilledAverPrice": fill_price,
                "CommissionChargeValue": 0.0,
            },
        }

    def place_order(self, pair: str, side: str, quantity: float, order_type: str = "MARKET", price: Optional[float] = None):
        tick = self._ticker_one(pair)
        bid = tick["MaxBid"]
        ask = tick["MinAsk"]
        last = tick["LastPrice"]

        if order_type == "MARKET":
            fill_price = ask if side == "BUY" else bid
            return self._apply_fill(pair, side, quantity, fill_price)

        # LIMIT behavior: immediate fill if limit is inside the current bar range.
        limit_price = float(price if price is not None else last)
        if self._within_bar(pair, limit_price):
            return self._apply_fill(pair, side, quantity, limit_price)

        oid = self._next_order_id()
        self.open_orders[oid] = {
            "pair": pair,
            "side": side,
            "quantity": quantity,
            "price": limit_price,
            "placed_time": self.clock.time(),
        }
        return {
            "Success": True,
            "OrderDetail": {
                "OrderID": oid,
                "Status": "PENDING",
                "Role": "SIM",
                "FilledQuantity": 0.0,
                "FilledAverPrice": 0.0,
                "CommissionChargeValue": 0.0,
            },
        }

    def cancel_order(self, order_id: Optional[int] = None):
        if order_id is None:
            self.open_orders.clear()
        else:
            self.open_orders.pop(order_id, None)
        return {"Success": True}

    def advance_pending_orders(self):
        """Check whether older pending limit orders got touched by the current bar."""
        filled = []
        for oid, info in list(self.open_orders.items()):
            if self._within_bar(info["pair"], info["price"]):
                result = self._apply_fill(info["pair"], info["side"], info["quantity"], info["price"])
                if result.get("Success"):
                    filled.append((oid, result))
                    del self.open_orders[oid]
        return filled


class SimExecutor:
    """Historical analogue of Executor.

    Keeps the same public methods used by main.py, but routes to SimRoostooClient.
    """

    def __init__(self, client: SimRoostooClient, exchange_info: dict):
        self.client = client
        self.exchange_info = exchange_info
        self.pending_orders: dict[int, dict] = {}

    def _get_precision(self, pair: str) -> tuple[int, int, float]:
        info = self.exchange_info.get(pair, {})
        return (
            info.get("PricePrecision", DEFAULT_PRICE_PRECISION),
            info.get("AmountPrecision", DEFAULT_AMOUNT_PRECISION),
            info.get("MiniOrder", DEFAULT_MIN_ORDER_USD),
        )

    def _round_quantity(self, quantity: float, pair: str) -> float:
        _, amt_prec, _ = self._get_precision(pair)
        if amt_prec == 0:
            return float(int(quantity))
        return round(quantity, amt_prec)

    def _round_price(self, price: float, pair: str) -> float:
        price_prec, _, _ = self._get_precision(pair)
        return round(price, price_prec)

    def _check_min_order(self, pair: str, quantity: float, price: float) -> bool:
        _, _, mini = self._get_precision(pair)
        return quantity * price >= mini

    def buy(self, pair: str, quantity_usd: float, current_price: float, bid: float, ask: float, use_limit: bool = True):
        coin_qty = self._round_quantity(quantity_usd / current_price, pair)
        if coin_qty <= 0 or not self._check_min_order(pair, coin_qty, current_price):
            return None

        if use_limit and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            limit_price = self._round_price(mid, pair)
            result = self.client.place_order(pair=pair, side="BUY", quantity=coin_qty, order_type="LIMIT", price=limit_price)
            if result and result.get("Success"):
                detail = result.get("OrderDetail", {})
                oid = detail.get("OrderID")
                if detail.get("Status") == "PENDING" and oid is not None:
                    self.pending_orders[oid] = {
                        "pair": pair,
                        "side": "BUY",
                        "time_placed": self.client.clock.time(),
                        "price": limit_price,
                        "quantity": coin_qty,
                    }
                return result

        return self.client.place_order(pair=pair, side="BUY", quantity=coin_qty, order_type="MARKET")

    def sell(self, pair: str, coin_quantity: float, current_price: float, bid: float, ask: float, use_limit: bool = True):
        coin_quantity = self._round_quantity(coin_quantity, pair)
        if coin_quantity <= 0 or not self._check_min_order(pair, coin_quantity, current_price):
            return None

        if use_limit and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            limit_price = self._round_price(mid, pair)
            result = self.client.place_order(pair=pair, side="SELL", quantity=coin_quantity, order_type="LIMIT", price=limit_price)
            if result and result.get("Success"):
                detail = result.get("OrderDetail", {})
                oid = detail.get("OrderID")
                if detail.get("Status") == "PENDING" and oid is not None:
                    self.pending_orders[oid] = {
                        "pair": pair,
                        "side": "SELL",
                        "time_placed": self.client.clock.time(),
                        "price": limit_price,
                        "quantity": coin_quantity,
                    }
                return result

        return self.client.place_order(pair=pair, side="SELL", quantity=coin_quantity, order_type="MARKET")

    def manage_pending_orders(self):
        # First, fill any pending orders touched by the current bar.
        filled = self.client.advance_pending_orders()
        for oid, _ in filled:
            self.pending_orders.pop(oid, None)

        if not self.pending_orders:
            return

        now = self.client.clock.time()
        stale_ids = []
        for oid, info in self.pending_orders.items():
            if now - info["time_placed"] > LIMIT_ORDER_TIMEOUT_SECONDS:
                stale_ids.append(oid)

        for oid in stale_ids:
            info = self.pending_orders.pop(oid)
            self.client.cancel_order(order_id=oid)
            if info["side"] == "SELL":
                self.client.place_order(pair=info["pair"], side="SELL", quantity=info["quantity"], order_type="MARKET")

    def cancel_all_pending(self):
        self.client.cancel_order()
        self.pending_orders.clear()


class HistoricalMainLikeBacktest:
    def __init__(self, all_candles: dict[str, list[dict]], exchange_info: dict):
        self.all_candles = all_candles
        # FIX: Directly use exchange_info (no "TradePairs" wrapper in this script)
        self.trade_pairs = exchange_info 
        self.active_pairs = [p for p in TRADEABLE_COINS if p in self.trade_pairs and p in self.all_candles]
        
    def _build_price_matrix(self, t: int, min_bars: int = 100) -> Optional[np.ndarray]:
        close_series = []
        for pair in self.active_pairs:
            # FIX: Use "close" instead of "c"
            closes = [c["close"] for c in self.all_candles[pair][:t+1]]
            if len(closes) < min_bars:
                continue
            close_series.append(np.array(closes, dtype=float))
        if len(close_series) < 2:
            return None
        min_len = min(len(x) for x in close_series)
        return np.column_stack([x[-min_len:] for x in close_series])

    def run_window(self, start_idx: int, end_idx: int) -> dict:
        """Runs the exact 5-step Lean v4 pipeline over a historical window."""
        
        # 1. Initialize live modules
        from bot.regime_detector import RegimeDetector
        from bot.ranking import Ranker
        from bot.risk_manager import RiskManager
        from bot.metrics import PerformanceTracker
        from bot.features import compute_coin_features, zscore_universe, compute_submodel_scores
        from bot.signals import compute_signal

        regime = RegimeDetector()
        ranker = Ranker()
        risk_mgr = RiskManager(INITIAL_CASH)
        perf = PerformanceTracker(INITIAL_CASH)
        
        # Initialize Simulated Clock and sync global time
        clock = SimClock()
        pytime.time = clock.time
        
        # Client takes: all_candles, exchange_info, clock
        client = SimRoostooClient(self.all_candles, self.trade_pairs, clock)
        executor = SimExecutor(client, self.trade_pairs)

        positions = {}
        trade_log = []
        cycle_count = 0
        
        for t in range(start_idx, end_idx):
            cycle_count += 1
            
            # Sync the simulated clock to the actual Binance candle timestamp
            first_pair = self.active_pairs[0]
            current_ts_ms = self.all_candles[first_pair][t]["close_time"]
            clock.set_bar(t, current_ts_ms)

            # ══════════════════════════════════════════════════════
            # STEP 0: Fetch Data & Check Breakers
            # ══════════════════════════════════════════════════════
            ticker_data = client.ticker()
            wallet = client.balance()
            
            # Sync positions
            positions.clear()
            for coin, bal in wallet.items():
                if coin == "USD": continue
                free = bal.get("Free", 0)
                if free > 0:
                    pair = f"{coin}/USD"
                    positions[pair] = free

            # Portfolio Value & DD Breakers
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

            if risk_mgr.is_paused:
                continue

            # ══════════════════════════════════════════════════════
            # STEP 1: REGIME FILTER (PCA-HMM Only)
            # ══════════════════════════════════════════════════════
            all_raw_features = {}
            for pair in self.active_pairs:
                hist = self.all_candles[pair][:t+1]
                # FIX: Use explicit key names "close", "high", "low", "volume"
                closes = np.array([x["close"] for x in hist])
                if len(closes) < 100:
                    continue
                highs = np.array([x["high"] for x in hist])
                lows = np.array([x["low"] for x in hist])
                volumes = np.array([x["volume"] for x in hist])
                tick = ticker_data.get(pair, {})
                
                feats = compute_coin_features(closes, highs, lows, volumes, tick.get("MaxBid",0), tick.get("MinAsk",0))
                if feats:
                    all_raw_features[pair] = feats

            # PCA market proxy
            price_matrix = self._build_price_matrix(t)
            pc1_series = None
            if price_matrix is not None:
                pc1_result = regime.compute_pc1_market_proxy(price_matrix)
                pc1_series = pc1_result[0]

            # Update regime (Only accepts pc1_series now!)
            regime.update(pc1_series=pc1_series)

            if cycle_count % 24 == 0 and pc1_series is not None:
                regime.fit_hmm(pc1_series)

            risk_mgr.set_regime_multiplier(regime.get_exposure_multiplier())
            executor.manage_pending_orders()

            if not regime.should_trade():
                # Just check exits and skip
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

            # ══════════════════════════════════════════════════════
            # STEP 2: EVENT FILTER (No Derivatives)
            # ══════════════════════════════════════════════════════
            all_zscored = zscore_universe(all_raw_features)
            for pair in all_zscored:
                compute_submodel_scores(all_zscored[pair])

            signals = {}
            for pair in all_raw_features:
                hist = self.all_candles[pair][:t+1]
                closes = np.array([x["close"] for x in hist])
                lows = np.array([x["low"] for x in hist])
                signals[pair] = compute_signal(all_raw_features[pair], all_zscored[pair], closes, lows, BREAKOUT_LOOKBACK)

            # ══════════════════════════════════════════════════════
            # STEP 3 & 4: VALID TRADES & RANKING
            # ══════════════════════════════════════════════════════
            valid_candidates = {}
            for pair, sig in signals.items():
                if sig["action"] == "BUY" and positions.get(pair, 0) <= 0:
                    valid_candidates[pair] = all_zscored.get(pair, {})
                    valid_candidates[pair]["_signal"] = sig

            ranked = ranker.rank(valid_candidates, max_results=MAX_POSITIONS)

            # ══════════════════════════════════════════════════════
            # STEP 5: EXECUTION
            # ══════════════════════════════════════════════════════
            
            # Check Exits First
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

            # Buys
            for pair, score, features in ranked:
                if num_positions >= MAX_POSITIONS: break
                sig = features.get("_signal", {})
                tick = ticker_data.get(pair, {})
                price = tick.get("LastPrice", 0)
                real_vol = all_raw_features.get(pair, {}).get("realized_vol", 0.5)

                # Lean position sizer (no deriv_score)
                size_usd = risk_mgr.position_size_usd(
                    pair, real_vol, total_exposure, num_positions,
                    signal_strength=sig.get("strength", 0.5)
                )

                if size_usd >= 50:
                    result = executor.buy(pair, size_usd, price, tick.get("MaxBid",0), tick.get("MinAsk",0), use_limit=USE_LIMIT_ORDERS)
                    if result and result.get("Success"):
                        strategy = sig.get("strategy", "continuation")
                        risk_mgr.update_trailing_stop(
                            pair, price, 
                            strategy="mean_rev" if strategy == "reversal" else "breakout",
                            entry_price=price
                        )
                        total_exposure += size_usd
                        num_positions += 1
                        trade_log.append({"pair": pair, "reason": "buy", "strategy": strategy})

            # Signal Sells (Breakdowns)
            for pair, sig in signals.items():
                if sig["action"] == "SELL" and positions.get(pair, 0) > 0:
                    qty = positions[pair]
                    tick = ticker_data.get(pair, {})
                    urgent = sig.get("breakdown", False)
                    executor.sell(pair, qty, tick.get("LastPrice",0), tick.get("MaxBid",0), tick.get("MinAsk",0), use_limit=not urgent)
                    risk_mgr.clear_trailing_stop(pair)
                    trade_log.append({"pair": pair, "reason": "breakdown_signal"})

        # End of window teardown (liquidate to get final cash value)
        for pair, qty in list(client.positions.items()):
            if qty > 0:
                tick = client.ticker().get(pair, {})
                executor.sell(pair, qty, tick.get("LastPrice", 0), tick.get("MaxBid", 0), tick.get("MinAsk", 0), use_limit=False)

        summary = perf.summary()
        return {"summary": summary, "trade_log": trade_log}

def default_exchange_info(pairs: list[str]) -> dict[str, dict]:
    info = {}
    for pair in pairs:
        info[pair] = {
            "PricePrecision": DEFAULT_PRICE_PRECISION,
            "AmountPrecision": DEFAULT_AMOUNT_PRECISION,
            "MiniOrder": DEFAULT_MIN_ORDER_USD,
        }
    return info


def download_4m_hourly(pair: str, session: requests.Session, months: int = DOWNLOAD_MONTHS) -> list[dict]:
    symbol = BINANCE_SYMBOL_MAP.get(pair)
    if not symbol:
        return []
    end_ms = int(pytime.time() * 1000)
    start_ms = end_ms - months * 30 * 24 * INTERVAL_MS
    out = []
    cur = start_ms
    while cur < end_ms:
        try:
            r = session.get(
                f"{BINANCE_BASE_URL}/api/v3/klines",
                params={"symbol": symbol, "interval": "1h", "startTime": cur, "endTime": end_ms, "limit": 1000},
                timeout=15,
            )
            r.raise_for_status()
            raw = r.json()
        except Exception:
            break
        if not raw:
            break
        for k in raw:
            out.append({
                "open_time": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": k[6],
            })
        cur = raw[-1][0] + INTERVAL_MS
        if len(raw) < 1000:
            break
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


def print_summary(label: str, metrics: dict):
    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"  Total Return: {metrics['total_return_pct']:>10.3f}%")
    print(f"  Max Drawdown:{metrics['max_drawdown_pct']:>10.3f}%")
    print(f"  Sharpe:      {metrics['sharpe']:>10.3f}")
    print(f"  Sortino:     {metrics['sortino']:>10.3f}")
    print(f"  Calmar:      {metrics['calmar']:>10.3f}")
    print(f"  Composite:   {metrics['composite']:>10.3f}")
    print(f"  Snapshots:   {metrics['num_snapshots']:>10}")


def main():
    print("=" * 70)
    print("  MAIN-LIKE HISTORICAL BACKTEST")
    print("=" * 70)

    pairs = [p for p in TRADEABLE_COINS if p in BINANCE_SYMBOL_MAP]
    all_candles = load_all_history(pairs)
    if len(all_candles) < 2:
        print("Not enough historical data loaded.")
        return

    min_len = min(len(v) for v in all_candles.values())
    usable_pairs = [p for p, v in all_candles.items() if len(v) >= min_len]
    all_candles = {p: v[:min_len] for p, v in all_candles.items() if p in usable_pairs}
    exchange_info = default_exchange_info(list(all_candles.keys()))
    bt = HistoricalMainLikeBacktest(all_candles, exchange_info)

    print("\n[TEST 1] 4-MONTH CONTINUOUS RUN")
    full = bt.run_window(WARMUP_BARS, min_len)
    print_summary("FULL CONTINUOUS PERIOD", full["summary"])

    print("\n[TEST 2] ROLLING 10-DAY WINDOWS")
    results = []
    for s in range(WARMUP_BARS, min_len - WINDOW_BARS, STEP_BARS):
        res = bt.run_window(s, s + WINDOW_BARS)
        m = dict(res["summary"])
        m["trades"] = len(res["trade_log"])
        results.append(m)
        print(
            f"  Window {s // 24:>3}: Ret={m['total_return_pct']:>+6.2f}% | "
            f"Comp={m['composite']:>6.2f} | Trades={m['trades']:>3}"
        )

    # ════════════════════════════════════════════════════════════
    # FINAL UNIFIED SCORECARD
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL BACKTEST SCORECARD")
    print("=" * 70)

    # 1. Long Term Continuous
    f_m = full["summary"]
    print("\n  [1] LONG-TERM CONTINUOUS (4 MONTHS)")
    print(f"      Total Return:  {f_m['total_return_pct']:>8.2f}%")
    print(f"      Max Drawdown:  {f_m['max_drawdown_pct']:>8.2f}%")
    print(f"      Sharpe:        {f_m['sharpe']:>8.2f}")
    print(f"      Sortino:       {f_m['sortino']:>8.2f}")
    print(f"      Calmar:        {f_m['calmar']:>8.2f}")
    print(f"      Composite:     {f_m['composite']:>8.2f}")
    print(f"      Total Trades:  {len(full['trade_log']):>8}")

    # 2. Rolling Averages
    if results:
        avg_ret = np.mean([m['total_return_pct'] for m in results])
        avg_dd = np.mean([m['max_drawdown_pct'] for m in results])
        avg_sharpe = np.mean([m['sharpe'] for m in results])
        avg_sortino = np.mean([m['sortino'] for m in results])
        avg_calmar = np.mean([m['calmar'] for m in results])
        avg_comp = np.mean([m['composite'] for m in results])
        avg_trades = np.mean([m['trades'] for m in results])

        print("\n  [2] ROLLING 10-DAY WINDOWS (AVERAGES)")
        print(f"      Avg Return:    {avg_ret:>8.2f}%")
        print(f"      Avg Max DD:    {avg_dd:>8.2f}%")
        print(f"      Avg Sharpe:    {avg_sharpe:>8.2f}")
        print(f"      Avg Sortino:   {avg_sortino:>8.2f}")
        print(f"      Avg Calmar:    {avg_calmar:>8.2f}")
        print(f"      Avg Compos:    {avg_comp:>8.2f}")
        print(f"      Avg Trades:    {avg_trades:>8.1f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
