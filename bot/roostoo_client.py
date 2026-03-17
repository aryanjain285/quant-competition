"""
Roostoo API client with HMAC-SHA256 signing.
Handles all interactions with the Roostoo mock exchange.
"""
import time
import hmac
import hashlib
import requests
from typing import Optional
from bot.config import API_KEY, API_SECRET, BASE_URL
from bot.logger import get_logger

log = get_logger("roostoo_client")


def _timestamp() -> str:
    return str(int(time.time() * 1000))


def _sign(params: dict) -> tuple[dict, str]:
    """Generate signed headers and encoded body/query string."""
    params["timestamp"] = _timestamp()
    sorted_keys = sorted(params.keys())
    total_params = "&".join(f"{k}={params[k]}" for k in sorted_keys)

    signature = hmac.new(
        API_SECRET.encode("utf-8"),
        total_params.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "RST-API-KEY": API_KEY,
        "MSG-SIGNATURE": signature,
    }
    return headers, total_params


class RoostooClient:
    """Thin wrapper around the Roostoo REST API."""

    def __init__(self):
        self.base = BASE_URL
        self.session = requests.Session()
        self.session.timeout = 10

    # ---- Public (no auth) ----

    def server_time(self) -> Optional[int]:
        try:
            r = self.session.get(f"{self.base}/v3/serverTime")
            r.raise_for_status()
            return r.json().get("ServerTime")
        except Exception as e:
            log.error(f"server_time failed: {e}")
            return None

    def exchange_info(self) -> Optional[dict]:
        try:
            r = self.session.get(f"{self.base}/v3/exchangeInfo")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error(f"exchange_info failed: {e}")
            return None

    def ticker(self, pair: Optional[str] = None) -> Optional[dict]:
        params = {"timestamp": _timestamp()}
        if pair:
            params["pair"] = pair
        try:
            r = self.session.get(f"{self.base}/v3/ticker", params=params)
            r.raise_for_status()
            data = r.json()
            if not data.get("Success"):
                log.warning(f"ticker error: {data.get('ErrMsg')}")
                return None
            return data.get("Data", {})
        except Exception as e:
            log.error(f"ticker failed: {e}")
            return None

    # ---- Signed (auth required) ----

    def balance(self) -> Optional[dict]:
        params = {"timestamp": _timestamp()}
        headers, _ = _sign(params)
        try:
            r = self.session.get(f"{self.base}/v3/balance", headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            if not data.get("Success"):
                log.warning(f"balance error: {data.get('ErrMsg')}")
                return None
            # API may return "SpotWallet" or "Wallet" depending on account type
            return data.get("SpotWallet", data.get("Wallet", {}))
        except Exception as e:
            log.error(f"balance failed: {e}")
            return None

    def pending_count(self) -> Optional[dict]:
        params = {"timestamp": _timestamp()}
        headers, _ = _sign(params)
        try:
            r = self.session.get(f"{self.base}/v3/pending_count", headers=headers, params=params)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error(f"pending_count failed: {e}")
            return None

    def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> Optional[dict]:
        payload = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        if order_type.upper() == "LIMIT" and price is not None:
            payload["price"] = str(price)

        headers, total_params = _sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            r = self.session.post(f"{self.base}/v3/place_order", headers=headers, data=total_params)
            r.raise_for_status()
            data = r.json()
            if not data.get("Success"):
                log.warning(f"place_order error [{pair} {side}]: {data.get('ErrMsg')}")
            else:
                detail = data.get("OrderDetail", {})
                log.info(
                    f"ORDER {detail.get('Status')}: {side} {quantity} {pair} "
                    f"@ {detail.get('FilledAverPrice', price)} "
                    f"[ID={detail.get('OrderID')}]"
                )
            return data
        except Exception as e:
            log.error(f"place_order failed [{pair} {side}]: {e}")
            return None

    def query_order(
        self,
        order_id: Optional[int] = None,
        pair: Optional[str] = None,
        pending_only: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> Optional[list]:
        payload = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        else:
            if pair:
                payload["pair"] = pair
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
            if limit is not None:
                payload["limit"] = str(limit)

        headers, total_params = _sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            r = self.session.post(f"{self.base}/v3/query_order", headers=headers, data=total_params)
            r.raise_for_status()
            data = r.json()
            if not data.get("Success"):
                return []
            return data.get("OrderMatched", [])
        except Exception as e:
            log.error(f"query_order failed: {e}")
            return None

    def cancel_order(
        self, order_id: Optional[int] = None, pair: Optional[str] = None
    ) -> Optional[list]:
        payload = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair

        headers, total_params = _sign(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            r = self.session.post(f"{self.base}/v3/cancel_order", headers=headers, data=total_params)
            r.raise_for_status()
            data = r.json()
            if data.get("Success"):
                log.info(f"Cancelled orders: {data.get('CanceledList', [])}")
            return data.get("CanceledList", [])
        except Exception as e:
            log.error(f"cancel_order failed: {e}")
            return None
