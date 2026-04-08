"""
Simulated Roostoo exchange for backtesting.

Design principle: the backtest should call the SAME bot modules (signals,
features, ranking, regime, risk_manager) as main.py. Only the exchange
interaction is simulated. This eliminates divergence between backtest
and live.

Key fidelity features:
  - Per-pair spreads calibrated from real Roostoo API measurements
  - Real exchange_info (precision, min order) fetched once and cached
  - Correct fee model: limit = 0.05% maker, market = 0.10% taker
  - Limit order fills: pending -> eligible from the next bar if price is touched
  - Pending orders are managed through the same Executor timeout path as live
  - Simulated clock that risk_manager.check_trailing_stop uses for holding hours
"""
import time as _real_time
import numpy as np
from typing import Optional
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════
# MEASURED REAL ROOSTOO SPREADS (from API, April 2026)
# Used when historical Binance H/L data isn't available for spread estimation
# ═══════════════════════════════════════════════════════════════

# Spread in basis points per pair (measured from live Roostoo ticker)
MEASURED_SPREAD_BPS = {
    "BTC/USD": 0.0, "ETH/USD": 0.0, "BNB/USD": 0.2, "PAXG/USD": 0.0,
    "XRP/USD": 0.8, "SOL/USD": 1.3, "DOGE/USD": 1.1, "SUI/USD": 1.1,
    "LTC/USD": 1.9, "ZEC/USD": 0.4, "HBAR/USD": 1.1, "PENGU/USD": 1.6,
    "VIRTUAL/USD": 1.5, "S/USD": 2.4, "TRX/USD": 3.2, "UNI/USD": 3.1,
    "TRUMP/USD": 3.5, "TAO/USD": 3.3, "FLOKI/USD": 3.7, "CFX/USD": 3.8,
    "ONDO/USD": 3.8, "WLD/USD": 3.7, "ZEN/USD": 3.9,
    "FET/USD": 4.2, "ADA/USD": 4.1, "ICP/USD": 4.3, "CRV/USD": 4.7,
    "FORM/USD": 4.9, "CAKE/USD": 7.4, "PUMP/USD": 6.0, "SOMI/USD": 6.4,
    "OPEN/USD": 6.4, "XLM/USD": 6.1, "BMT/USD": 7.0, "DOT/USD": 8.0,
    "TON/USD": 8.0, "XPL/USD": 8.2, "NEAR/USD": 8.2, "STO/USD": 8.5,
    "PLUME/USD": 10.3, "WLFI/USD": 10.1, "POL/USD": 10.8, "ARB/USD": 10.7,
    "LINK/USD": 11.4, "AVAX/USD": 11.1, "LISTA/USD": 11.1, "APT/USD": 11.6,
    "ENA/USD": 12.2, "FIL/USD": 11.8, "MIRA/USD": 13.3, "PENDLE/USD": 9.2,
    "SHIB/USD": 16.5, "BONK/USD": 17.3, "SEI/USD": 18.7, "TUT/USD": 21.2,
    "1000CHEEMS/USD": 24.3, "HEMI/USD": 25.6, "PEPE/USD": 28.7,
    "EDEN/USD": 30.6, "LINEA/USD": 33.1, "WIF/USD": 54.8, "BIO/USD": 57.3,
    "EIGEN/USD": 66.0,
}
DEFAULT_SPREAD_BPS = 8.0  # fallback for unmeasured pairs


@dataclass
class SimClock:
    """Simulated clock that time.time() can reference."""
    _current_time: float = 0.0

    def set_time(self, ts_seconds: float):
        self._current_time = ts_seconds

    def time(self) -> float:
        return self._current_time


class SimExchange:
    """Simulated Roostoo exchange with realistic spreads, fees, and fills.

    Usage:
        exchange = SimExchange(candles, exchange_info, clock)
        exchange.set_bar(t)  # advance to bar t
        ticker = exchange.ticker()
        result = exchange.place_order(pair, "BUY", qty, "MARKET")
    """

    MAKER_FEE = 0.0005   # 0.05% for limit orders
    TAKER_FEE = 0.0010   # 0.10% for market orders

    def __init__(
        self,
        all_candles: dict[str, list[dict]],
        exchange_info: dict[str, dict],
        clock: SimClock,
    ):
        self.all_candles = all_candles
        self.exchange_info = exchange_info
        self.clock = clock

        self.cash: float = 0.0
        self.holdings: dict[str, float] = {}  # pair -> quantity

        self._current_bar: int = 0
        self._order_id: int = 0
        self.pending_orders: dict[int, dict] = {}

    def reset(self, initial_cash: float = 1_000_000.0):
        self.cash = initial_cash
        self.holdings.clear()
        self.pending_orders.clear()
        self._order_id = 0

    def set_bar(self, idx: int):
        self._current_bar = idx
        # Set clock from first available pair's close_time
        for pair in self.all_candles:
            candles = self.all_candles[pair]
            if idx < len(candles):
                self.clock.set_time(candles[idx]["close_time"] / 1000.0)
                break

    # ─── Ticker (with realistic per-pair spreads) ──────────────

    def _get_spread_bps(self, pair: str) -> float:
        """Get realistic spread for this pair."""
        return MEASURED_SPREAD_BPS.get(pair, DEFAULT_SPREAD_BPS)

    def _ticker_one(self, pair: str) -> dict:
        candles = self.all_candles[pair]
        if self._current_bar >= len(candles):
            return {}
        bar = candles[self._current_bar]
        close = float(bar["close"])
        half_spread = (self._get_spread_bps(pair) / 10000) / 2.0
        return {
            "LastPrice": close,
            "MaxBid": close * (1 - half_spread),
            "MinAsk": close * (1 + half_spread),
            "Change": 0.0,
            "CoinTradeValue": float(bar.get("volume", 0)),
            "UnitTradeValue": float(bar.get("volume", 0)) * close,
        }

    def ticker(self) -> dict:
        return {
            pair: self._ticker_one(pair)
            for pair in self.all_candles
            if self._current_bar < len(self.all_candles[pair])
        }

    def server_time(self) -> int:
        return int(self.clock.time() * 1000)

    def exchange_info_payload(self) -> dict:
        return {"IsRunning": True, "TradePairs": self.exchange_info}

    # ─── Balance ───────────────────────────────────────────────

    def balance(self) -> dict:
        wallet = {"USD": {"Free": self.cash, "Lock": 0.0}}
        for pair, qty in self.holdings.items():
            if qty > 0:
                coin = pair.split("/")[0]
                wallet[coin] = {"Free": qty, "Lock": 0.0}
        return wallet

    # ─── Order execution ──────────────────────────────────────

    def _next_oid(self) -> int:
        self._order_id += 1
        return self._order_id

    def _fill(self, pair: str, side: str, qty: float, price: float, is_maker: bool) -> dict:
        """Execute a fill with correct fee model."""
        fee_rate = self.MAKER_FEE if is_maker else self.TAKER_FEE
        notional = qty * price
        fee = notional * fee_rate
        oid = self._next_oid()

        if side == "BUY":
            total_cost = notional + fee
            if total_cost > self.cash + 0.01:
                return {"Success": False, "ErrMsg": "insufficient funds"}
            self.cash -= total_cost
            self.holdings[pair] = self.holdings.get(pair, 0.0) + qty
        else:
            held = self.holdings.get(pair, 0.0)
            qty = min(qty, held)
            if qty <= 0:
                return {"Success": False, "ErrMsg": "no position"}
            self.cash += notional - fee
            remaining = held - qty
            self.holdings[pair] = remaining if remaining > 1e-12 else 0.0

        return {
            "Success": True,
            "OrderDetail": {
                "OrderID": oid,
                "Status": "FILLED",
                "FilledQuantity": qty,
                "FilledAverPrice": price,
                "Role": "MAKER" if is_maker else "TAKER",
                "CommissionChargeValue": fee,
            },
        }

    def _bar_contains_price(self, pair: str, price: float, bar_idx: int = None) -> bool:
        """Check if price is within the bar's high-low range."""
        idx = bar_idx if bar_idx is not None else self._current_bar
        candles = self.all_candles[pair]
        if idx >= len(candles):
            return False
        bar = candles[idx]
        return float(bar["low"]) <= price <= float(bar["high"])

    def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> dict:
        tick = self._ticker_one(pair)
        if not tick:
            return {"Success": False, "ErrMsg": "pair not available"}

        if order_type == "MARKET":
            # Market orders fill at ask (buy) or bid (sell)
            fill_price = tick["MinAsk"] if side == "BUY" else tick["MaxBid"]
            return self._fill(pair, side, quantity, fill_price, is_maker=False)

        # Limit order
        limit_price = float(price) if price is not None else tick["LastPrice"]

        # Live orders are placed after the decision for the current hour has already
        # been made. With 1h bars, the earliest fill we can justify is the NEXT bar.
        oid = self._next_oid()
        self.pending_orders[oid] = {
            "pair": pair,
            "side": side,
            "quantity": quantity,
            "price": limit_price,
            "placed_time": self.clock.time(),
            "eligible_bar": self._current_bar + 1,
        }
        return {
            "Success": True,
            "OrderDetail": {
                "OrderID": oid,
                "Status": "PENDING",
                "FilledQuantity": 0,
                "FilledAverPrice": 0,
            },
        }

    def advance_pending_orders(self) -> list[dict]:
        """Check if any pending orders can fill on current bar. Returns fill events."""
        fill_events = []
        for oid in list(self.pending_orders.keys()):
            info = self.pending_orders[oid]
            eligible_bar = info.get("eligible_bar", 0)
            if self._current_bar >= eligible_bar and self._bar_contains_price(info["pair"], info["price"]):
                result = self._fill(
                    info["pair"], info["side"], info["quantity"],
                    info["price"], is_maker=True,
                )
                if result.get("Success"):
                    detail = result["OrderDetail"]
                    fill_events.append({
                        "pair": info["pair"],
                        "side": info["side"],
                        "filled_qty": detail["FilledQuantity"],
                        "filled_avg_price": detail["FilledAverPrice"],
                    })
                del self.pending_orders[oid]
        return fill_events

    def cancel_order(self, order_id: Optional[int] = None) -> dict:
        if order_id is None:
            self.pending_orders.clear()
        else:
            self.pending_orders.pop(order_id, None)
        return {"Success": True, "CanceledList": [order_id] if order_id else []}

    def query_order(self, order_id: int = None, **kwargs) -> list:
        """Simulate query_order for pending order checks."""
        if order_id is not None and order_id in self.pending_orders:
            info = self.pending_orders[order_id]
            eligible_bar = info.get("eligible_bar", 0)
            if self._current_bar >= eligible_bar and self._bar_contains_price(info["pair"], info["price"]):
                result = self._fill(
                    info["pair"], info["side"], info["quantity"],
                    info["price"], is_maker=True,
                )
                self.pending_orders.pop(order_id, None)
                if result.get("Success"):
                    detail = result["OrderDetail"]
                    return [{
                        "OrderID": detail.get("OrderID", order_id),
                        "Status": "FILLED",
                        "FilledQuantity": detail.get("FilledQuantity", 0),
                        "FilledAverPrice": detail.get("FilledAverPrice", 0),
                        "Pair": info["pair"],
                        "Side": info["side"],
                    }]
            return [{"OrderID": order_id, "Status": "PENDING",
                     "FilledQuantity": 0, "FilledAverPrice": 0,
                     "Pair": info["pair"], "Side": info["side"]}]
        return []

    def portfolio_value(self, ticker_data: dict = None) -> float:
        """Compute current portfolio value."""
        if ticker_data is None:
            ticker_data = self.ticker()
        value = self.cash
        for pair, qty in self.holdings.items():
            if qty > 0:
                tick = ticker_data.get(pair, {})
                value += qty * tick.get("LastPrice", 0)
        return value
