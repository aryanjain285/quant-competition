"""
Order executor: translates portfolio decisions into Roostoo API calls.
Handles limit-to-market fallback and pending order management.
"""
import time
from typing import Optional
from bot.roostoo_client import RoostooClient
from bot.config import LIMIT_ORDER_TIMEOUT_SECONDS, LIMIT_ORDER_OFFSET_BPS
from bot.logger import get_logger, log_trade

log = get_logger("executor")


class Executor:
    """Executes orders on Roostoo with limit-to-market fallback."""

    def __init__(self, client: RoostooClient, exchange_info: dict):
        self.client = client
        self.exchange_info = exchange_info  # pair -> {PricePrecision, AmountPrecision, MiniOrder}
        # Track pending limit orders: order_id -> {pair, side, time_placed, ...}
        self.pending_orders: dict[int, dict] = {}

    def _get_precision(self, pair: str) -> tuple[int, int, float]:
        """Get price precision, amount precision, and min order for a pair."""
        info = self.exchange_info.get(pair, {})
        return (
            info.get("PricePrecision", 4),
            info.get("AmountPrecision", 2),
            info.get("MiniOrder", 1.0),
        )

    def _round_quantity(self, quantity: float, pair: str) -> float:
        """Round quantity to the pair's amount precision."""
        _, amt_prec, _ = self._get_precision(pair)
        if amt_prec == 0:
            return int(quantity)
        return round(quantity, amt_prec)

    def _round_price(self, price: float, pair: str) -> float:
        """Round price to the pair's price precision."""
        price_prec, _, _ = self._get_precision(pair)
        return round(price, price_prec)

    def _check_min_order(self, pair: str, quantity: float, price: float) -> bool:
        """Check if order meets minimum order value."""
        _, _, mini = self._get_precision(pair)
        return (quantity * price) >= mini

    def buy(
        self,
        pair: str,
        quantity_usd: float,
        current_price: float,
        bid: float,
        ask: float,
        use_limit: bool = True,
    ) -> Optional[dict]:
        """Place a buy order.

        Args:
            pair: e.g. "BTC/USD"
            quantity_usd: how much USD to spend
            current_price: latest price
            bid: current best bid
            ask: current best ask
            use_limit: whether to try limit order first
        """
        # Calculate coin quantity from USD amount
        coin_qty = quantity_usd / current_price
        coin_qty = self._round_quantity(coin_qty, pair)

        if coin_qty <= 0:
            return None
        if not self._check_min_order(pair, coin_qty, current_price):
            log.debug(f"Order too small: {pair} qty={coin_qty} price={current_price}")
            return None

        if use_limit and bid > 0 and ask > 0:
            # Place limit at mid-price or slightly above bid
            mid = (bid + ask) / 2
            limit_price = self._round_price(mid, pair)

            result = self.client.place_order(
                pair=pair, side="BUY", quantity=coin_qty,
                order_type="LIMIT", price=limit_price,
            )
            if result and result.get("Success"):
                detail = result.get("OrderDetail", {})
                order_id = detail.get("OrderID")
                if detail.get("Status") == "PENDING" and order_id is not None:
                    self.pending_orders[order_id] = {
                        "pair": pair, "side": "BUY",
                        "time_placed": time.time(),
                        "price": limit_price, "quantity": coin_qty,
                    }
                self._log_trade_record(detail, pair, "BUY", coin_qty, limit_price)
                return result

            # Only fall through to market if the error was fill-related, not auth/permission
            err = (result or {}).get("ErrMsg", "")
            if "permission" in err.lower() or "unauthorized" in err.lower():
                return result
            log.debug(f"Limit buy failed for {pair}, trying market")

        # Market order fallback
        result = self.client.place_order(
            pair=pair, side="BUY", quantity=coin_qty, order_type="MARKET",
        )
        if result and result.get("Success"):
            self._log_trade_record(result.get("OrderDetail", {}), pair, "BUY", coin_qty, current_price)
        return result

    def sell(
        self,
        pair: str,
        coin_quantity: float,
        current_price: float,
        bid: float,
        ask: float,
        use_limit: bool = True,
    ) -> Optional[dict]:
        """Place a sell order.

        Args:
            pair: e.g. "BTC/USD"
            coin_quantity: amount of coin to sell
            current_price: latest price
            bid/ask: current best bid/ask
            use_limit: whether to try limit order first
        """
        coin_quantity = self._round_quantity(coin_quantity, pair)
        if coin_quantity <= 0:
            return None
        if not self._check_min_order(pair, coin_quantity, current_price):
            log.debug(f"Sell too small: {pair} qty={coin_quantity}")
            return None

        if use_limit and bid > 0 and ask > 0:
            mid = (bid + ask) / 2
            limit_price = self._round_price(mid, pair)

            result = self.client.place_order(
                pair=pair, side="SELL", quantity=coin_quantity,
                order_type="LIMIT", price=limit_price,
            )
            if result and result.get("Success"):
                detail = result.get("OrderDetail", {})
                order_id = detail.get("OrderID")
                if detail.get("Status") == "PENDING" and order_id is not None:
                    self.pending_orders[order_id] = {
                        "pair": pair, "side": "SELL",
                        "time_placed": time.time(),
                        "price": limit_price, "quantity": coin_quantity,
                    }
                self._log_trade_record(detail, pair, "SELL", coin_quantity, limit_price)
                return result

            err = (result or {}).get("ErrMsg", "")
            if "permission" in err.lower() or "unauthorized" in err.lower():
                return result
            log.debug(f"Limit sell failed for {pair}, trying market")

        result = self.client.place_order(
            pair=pair, side="SELL", quantity=coin_quantity, order_type="MARKET",
        )
        if result and result.get("Success"):
            self._log_trade_record(result.get("OrderDetail", {}), pair, "SELL", coin_quantity, current_price)
        return result

    def manage_pending_orders(self) -> list[dict]:
        """Check tracked pending orders, clean up filled ones, cancel stale ones."""
        if not self.pending_orders:
            return []

        now = time.time()
        fill_events = []

        for order_id in list(self.pending_orders.keys()):
            info = self.pending_orders.get(order_id)
            if not info:
                continue

            matches = self.client.query_order(order_id=order_id)
            if matches:
                od = matches[0]
                status = str(od.get("Status", "")).upper()
                filled_qty = float(od.get("FilledQuantity", 0) or 0)

                if status in {"FILLED", "CANCELED", "CANCELLED", "REJECTED"}:
                    if filled_qty > 0:
                        fill_events.append({
                            "pair": info["pair"],
                            "side": info["side"],
                            "filled_qty": filled_qty,
                            "filled_avg_price": float(od.get("FilledAverPrice", 0) or 0),
                        })
                    self.pending_orders.pop(order_id, None)
                    continue

            age = now - info["time_placed"]
            if age > LIMIT_ORDER_TIMEOUT_SECONDS:
                log.info(f"Cancelling stale order {order_id} ({info['pair']} {info['side']})")
                self.client.cancel_order(order_id=order_id)
                self.pending_orders.pop(order_id, None)

                result = self.client.place_order(
                    pair=info["pair"],
                    side=info["side"],
                    quantity=info["quantity"],
                    order_type="MARKET",
                )
                if result and result.get("Success"):
                    detail = result.get("OrderDetail", {})
                    filled_qty = float(detail.get("FilledQuantity", 0) or 0)
                    if filled_qty > 0:
                        fill_events.append({
                            "pair": info["pair"],
                            "side": info["side"],
                            "filled_qty": filled_qty,
                            "filled_avg_price": float(detail.get("FilledAverPrice", 0) or 0),
                        })
        return fill_events
    

    def cancel_all_pending(self):
        """Cancel all pending orders (used during drawdown breakers)."""
        if not self.pending_orders:
            return
        log.info(f"Cancelling all {len(self.pending_orders)} pending orders")
        self.client.cancel_order()  # no args = cancel all
        self.pending_orders.clear()

    def _log_trade_record(self, detail: dict, pair: str, side: str, qty: float, price: float):
        """Log a trade to the trade journal."""
        log_trade({
            "pair": pair,
            "side": side,
            "quantity": qty,
            "price": price,
            "order_id": detail.get("OrderID"),
            "status": detail.get("Status"),
            "role": detail.get("Role"),
            "filled_qty": detail.get("FilledQuantity"),
            "filled_avg_price": detail.get("FilledAverPrice"),
            "commission": detail.get("CommissionChargeValue"),
        })
