"""
Derivatives data feed: funding rates, open interest, liquidations.
Fetches from Binance Futures API (public, no auth needed).

These are the signals most teams won't have. Academic research shows:
- Extreme funding rates predict mean-reversion within 1-3 days
- OI divergence from price predicts squeezes and reversals
- Liquidation cascades create exploitable price dislocations
"""
import time
import requests
import numpy as np
from collections import defaultdict
from typing import Optional
from bot.config import BINANCE_FUTURES_URL, BINANCE_SYMBOL_MAP
from bot.logger import get_logger

log = get_logger("derivatives")

# Binance futures uses same symbols as spot (BTCUSDT etc.)
# but some altcoins may not have futures. We handle that gracefully.


class DerivativesData:
    """Fetches and interprets derivatives market data from Binance Futures."""

    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 10

        # Cached data: symbol -> latest values
        self.funding_rates: dict[str, list[dict]] = defaultdict(list)  # history
        self.open_interest: dict[str, list[dict]] = defaultdict(list)  # history
        self.available_symbols: set[str] = set()

        # Aggregated signals per pair
        self.signals: dict[str, dict] = {}

    def _futures_symbol(self, pair: str) -> Optional[str]:
        """Convert Roostoo pair to Binance futures symbol."""
        return BINANCE_SYMBOL_MAP.get(pair)

    def _fetch_json(self, url: str, params: dict = None) -> Optional[any]:
        try:
            r = self.session.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.debug(f"Derivatives fetch failed: {url} — {e}")
            return None

    # ─── Data Fetching ──────────────────────────────────────────

    def fetch_funding_rate(self, pair: str, limit: int = 30) -> Optional[list[dict]]:
        """Fetch recent funding rate history for a pair.

        Returns list of {fundingTime, fundingRate, markPrice}.
        Funding settles every 8h on Binance. Rates are per-period (not annualized).
        """
        symbol = self._futures_symbol(pair)
        if not symbol:
            return None

        data = self._fetch_json(
            f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": limit},
        )
        if data and isinstance(data, list):
            parsed = []
            for entry in data:
                parsed.append({
                    "time": entry.get("fundingTime", 0),
                    "rate": float(entry.get("fundingRate", 0)),
                    "mark_price": float(entry.get("markPrice", 0)),
                })
            self.funding_rates[pair] = parsed
            self.available_symbols.add(pair)
            return parsed
        return None

    def fetch_open_interest(self, pair: str) -> Optional[dict]:
        """Fetch current open interest for a pair.

        Returns {openInterest (in coins), time}.
        """
        symbol = self._futures_symbol(pair)
        if not symbol:
            return None

        data = self._fetch_json(
            f"{BINANCE_FUTURES_URL}/fapi/v1/openInterest",
            {"symbol": symbol},
        )
        if data and isinstance(data, dict):
            entry = {
                "time": data.get("time", 0),
                "oi": float(data.get("openInterest", 0)),
            }
            history = self.open_interest[pair]
            # Append if new timestamp, else update last
            if not history or history[-1]["time"] != entry["time"]:
                history.append(entry)
            else:
                history[-1] = entry
            # Keep last 500 snapshots
            if len(history) > 500:
                self.open_interest[pair] = history[-500:]
            return entry
        return None

    def fetch_oi_history(self, pair: str, period: str = "5m", limit: int = 500) -> Optional[list]:
        """Fetch OI history (kline-style). Only available for some endpoints."""
        symbol = self._futures_symbol(pair)
        if not symbol:
            return None

        data = self._fetch_json(
            f"{BINANCE_FUTURES_URL}/futures/data/openInterestHist",
            {"symbol": symbol, "period": period, "limit": limit},
        )
        if data and isinstance(data, list):
            parsed = []
            for entry in data:
                parsed.append({
                    "time": entry.get("timestamp", 0),
                    "oi": float(entry.get("sumOpenInterest", 0)),
                    "oi_value": float(entry.get("sumOpenInterestValue", 0)),
                })
            self.open_interest[pair] = parsed
            return parsed
        return None

    def load_all(self, pairs: list[str]):
        """Fetch funding rates + OI for all pairs. Called periodically."""
        loaded = 0
        for pair in pairs:
            fr = self.fetch_funding_rate(pair, limit=30)
            oi = self.fetch_open_interest(pair)
            if fr or oi:
                loaded += 1
            time.sleep(0.05)  # rate limit respect
        log.info(f"Derivatives data loaded for {loaded}/{len(pairs)} pairs")

    # ─── Signal Computation ─────────────────────────────────────

    def compute_signals(self, pairs: list[str], spot_prices: dict[str, float] = None) -> dict[str, dict]:
        """Compute derivatives-based signals for all pairs.

        Returns dict[pair -> signal_dict] with:
            - funding_signal: -1 (extreme positive = bearish), 0 (neutral), +1 (extreme negative = bullish)
            - funding_rate: latest funding rate
            - funding_zscore: z-score of current funding vs recent history
            - oi_signal: -1 (OI declining = exhaustion), 0 (neutral), +1 (OI rising = conviction)
            - oi_change_pct: recent OI change as percentage
            - oi_price_divergence: True if OI and price moving opposite directions
            - composite_deriv_score: combined score [-1, +1]
        """
        self.signals.clear()

        for pair in pairs:
            sig = {
                "funding_signal": 0,
                "funding_rate": 0.0,
                "funding_zscore": 0.0,
                "oi_signal": 0,
                "oi_change_pct": 0.0,
                "oi_price_divergence": False,
                "composite_deriv_score": 0.0,
            }

            # ── Funding rate analysis ──
            rates = self.funding_rates.get(pair, [])
            if len(rates) >= 5:
                current_rate = rates[-1]["rate"]
                recent_rates = np.array([r["rate"] for r in rates])
                mean_rate = np.mean(recent_rates)
                std_rate = np.std(recent_rates)

                sig["funding_rate"] = round(current_rate, 8)

                if std_rate > 0:
                    zscore = (current_rate - mean_rate) / std_rate
                    sig["funding_zscore"] = round(float(zscore), 3)

                    # Extreme positive funding → market too long → bearish signal
                    # Extreme negative funding → market too short → bullish signal (squeeze)
                    if zscore > 1.5:
                        sig["funding_signal"] = -1  # bearish
                    elif zscore < -1.5:
                        sig["funding_signal"] = 1   # bullish (short squeeze potential)

                # Also check absolute level
                # Funding > 0.05% per 8h is elevated; > 0.1% is extreme
                if current_rate > 0.001:  # 0.1%
                    sig["funding_signal"] = min(sig["funding_signal"], -1)
                elif current_rate < -0.0005:  # -0.05%
                    sig["funding_signal"] = max(sig["funding_signal"], 1)

            # ── Open interest analysis ──
            oi_history = self.open_interest.get(pair, [])
            if len(oi_history) >= 3:
                current_oi = oi_history[-1]["oi"]
                # Compare to OI from ~24h ago if available, else earliest
                lookback_idx = max(0, len(oi_history) - 24)
                past_oi = oi_history[lookback_idx]["oi"]

                if past_oi > 0:
                    oi_change = (current_oi - past_oi) / past_oi
                    sig["oi_change_pct"] = round(oi_change * 100, 2)

                    # Rising OI = new money entering = conviction
                    # Falling OI = positions closing = exhaustion
                    if oi_change > 0.05:  # +5% OI increase
                        sig["oi_signal"] = 1
                    elif oi_change < -0.05:  # -5% OI decrease
                        sig["oi_signal"] = -1

                    # OI-price divergence: the key signal
                    # Price down + OI up = shorts piling in = squeeze potential (bullish)
                    # Price up + OI down = longs exiting = exhaustion (bearish)
                    if spot_prices and pair in spot_prices:
                        price_now = spot_prices[pair]
                        # We need historical price too; use mark_price from funding if available
                        if rates and len(rates) > lookback_idx:
                            price_past = rates[lookback_idx]["mark_price"] if rates[lookback_idx]["mark_price"] > 0 else price_now
                            price_change = (price_now - price_past) / price_past if price_past > 0 else 0

                            if price_change < -0.02 and oi_change > 0.03:
                                # Price down, OI up → shorts stacking → squeeze incoming
                                sig["oi_price_divergence"] = True
                                sig["oi_signal"] = 1  # bullish override
                            elif price_change > 0.02 and oi_change < -0.03:
                                # Price up, OI down → longs taking profit → exhaustion
                                sig["oi_price_divergence"] = True
                                sig["oi_signal"] = -1  # bearish override

            # ── Composite derivatives score ──
            # Weighted: funding 0.4, OI 0.6 (OI divergence is stronger signal)
            composite = 0.4 * sig["funding_signal"] + 0.6 * sig["oi_signal"]
            sig["composite_deriv_score"] = round(composite, 3)

            self.signals[pair] = sig

        return self.signals

    def get_signal(self, pair: str) -> dict:
        """Get derivatives signal for a specific pair."""
        return self.signals.get(pair, {
            "funding_signal": 0, "funding_rate": 0.0, "funding_zscore": 0.0,
            "oi_signal": 0, "oi_change_pct": 0.0, "oi_price_divergence": False,
            "composite_deriv_score": 0.0,
        })
