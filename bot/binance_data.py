"""
Binance public data feed.
Fetches historical klines and live ticker data — no API key required.
Used as the primary signal source (richer than Roostoo's ticker).
"""
import time
import requests
import numpy as np
from collections import defaultdict
from typing import Optional
from bot.config import BINANCE_BASE_URL, BINANCE_SYMBOL_MAP, TRADEABLE_COINS
from bot.logger import get_logger

log = get_logger("binance_data")


class BinanceData:
    """Fetches and stores candle data from Binance public API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 10
        # pair -> list of candle dicts, newest last
        # each candle: {open_time, open, high, low, close, volume, close_time}
        self.candles: dict[str, list[dict]] = defaultdict(list)

    def _binance_symbol(self, pair: str) -> Optional[str]:
        return BINANCE_SYMBOL_MAP.get(pair)

    def fetch_klines(
        self, pair: str, interval: str = "5m", limit: int = 1000
    ) -> Optional[list[dict]]:
        """Fetch klines from Binance REST API.

        Args:
            pair: Roostoo pair like "BTC/USD"
            interval: Binance interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: number of candles (max 1000)
        """
        symbol = self._binance_symbol(pair)
        if not symbol:
            return None

        try:
            r = self.session.get(
                f"{BINANCE_BASE_URL}/api/v3/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
            )
            r.raise_for_status()
            raw = r.json()
        except Exception as e:
            log.error(f"klines failed for {pair} ({symbol}): {e}")
            return None

        candles = []
        for k in raw:
            candles.append({
                "open_time": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": k[6],
            })
        return candles

    def load_history(self, pairs: Optional[list[str]] = None, interval: str = "5m", limit: int = 1000):
        """Load historical candles for all tradeable pairs.

        Called once on startup to seed the candle store.
        """
        pairs = pairs or TRADEABLE_COINS
        loaded = 0
        for pair in pairs:
            candles = self.fetch_klines(pair, interval=interval, limit=limit)
            if candles:
                self.candles[pair] = candles
                loaded += 1
            time.sleep(0.1)  # respect rate limits
        log.info(f"Loaded history for {loaded}/{len(pairs)} pairs ({limit} x {interval} candles)")

    def update_latest(self, pairs: Optional[list[str]] = None):
        """Fetch the latest few candles and append/update the store.

        Called every cycle to keep data fresh.
        """
        pairs = pairs or TRADEABLE_COINS
        for pair in pairs:
            candles = self.fetch_klines(pair, interval="1h", limit=3)
            if not candles:
                continue

            existing = self.candles[pair]
            for c in candles:
                # replace if same open_time, else append
                replaced = False
                for i in range(len(existing) - 1, max(len(existing) - 5, -1), -1):
                    if existing[i]["open_time"] == c["open_time"]:
                        existing[i] = c
                        replaced = True
                        break
                if not replaced:
                    existing.append(c)

            # keep only last 2000 candles to bound memory
            if len(existing) > 2000:
                self.candles[pair] = existing[-2000:]

        log.debug(f"Updated candles for {len(pairs)} pairs")

    def get_closes(self, pair: str, n: Optional[int] = None) -> np.ndarray:
        """Get array of close prices, newest last."""
        candles = self.candles.get(pair, [])
        if not candles:
            return np.array([])
        closes = np.array([c["close"] for c in candles])
        if n is not None:
            closes = closes[-n:]
        return closes

    def get_volumes(self, pair: str, n: Optional[int] = None) -> np.ndarray:
        """Get array of volumes, newest last."""
        candles = self.candles.get(pair, [])
        if not candles:
            return np.array([])
        vols = np.array([c["volume"] for c in candles])
        if n is not None:
            vols = vols[-n:]
        return vols

    def get_highs(self, pair: str, n: Optional[int] = None) -> np.ndarray:
        candles = self.candles.get(pair, [])
        if not candles:
            return np.array([])
        highs = np.array([c["high"] for c in candles])
        if n is not None:
            highs = highs[-n:]
        return highs

    def get_lows(self, pair: str, n: Optional[int] = None) -> np.ndarray:
        candles = self.candles.get(pair, [])
        if not candles:
            return np.array([])
        lows = np.array([c["low"] for c in candles])
        if n is not None:
            lows = lows[-n:]
        return lows
