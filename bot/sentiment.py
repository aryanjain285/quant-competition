"""
External sentiment data: Fear & Greed Index + BTC lead-lag filter.

Fear & Greed Index (alternative.me):
- Free API, updates daily
- Values 0-100: 0=Extreme Fear, 100=Extreme Greed
- Used as a contrarian regime filter:
  - Extreme Fear (<25): historically a buying opportunity, but reduce exposure defensively
  - Extreme Greed (>75): reduce exposure, market likely to correct
  - Neutral (25-75): no adjustment

BTC Lead-Lag Filter:
- BTC leads altcoins by 1-3 candles during large moves
- If BTC drops >1.5% in the last hour, skip altcoin buys for a few cycles
- If BTC is breaking out strongly, boost altcoin signal confidence
"""
import time
import requests
from typing import Optional
from bot.logger import get_logger

log = get_logger("sentiment")

FEAR_GREED_URL = "https://api.alternative.me/fng/"


class SentimentAnalyzer:
    """Fetches and interprets market sentiment data."""

    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 10

        # Fear & Greed
        self.fear_greed_value: int = 50  # default neutral
        self.fear_greed_label: str = "Neutral"
        self.fear_greed_updated: float = 0

        # BTC lead-lag
        self.btc_1h_return: float = 0.0
        self.btc_recent_returns: list[float] = []  # last N 5-min returns
        self.skip_buys_until: float = 0  # timestamp until when to skip buys
        self.btc_momentum_boost: float = 0.0  # positive = BTC breaking out

    # ─── Fear & Greed Index ─────────────────────────────────────

    def update_fear_greed(self):
        """Fetch latest Fear & Greed Index. Call once per hour max."""
        # Don't fetch more than once per hour
        if time.time() - self.fear_greed_updated < 3600:
            return

        try:
            r = self.session.get(FEAR_GREED_URL, params={"limit": 1})
            r.raise_for_status()
            data = r.json()
            if data.get("data") and len(data["data"]) > 0:
                entry = data["data"][0]
                self.fear_greed_value = int(entry.get("value", 50))
                self.fear_greed_label = entry.get("value_classification", "Neutral")
                self.fear_greed_updated = time.time()
                log.info(f"Fear & Greed: {self.fear_greed_value} ({self.fear_greed_label})")
        except Exception as e:
            log.warning(f"Fear & Greed fetch failed: {e}")

    def get_fg_exposure_multiplier(self) -> float:
        """Get exposure adjustment based on Fear & Greed.

        Returns multiplier (0.5 to 1.0):
        - Extreme Fear (<20): 0.7 (cautious — could get worse before better)
        - Fear (20-40): 0.85 (mild caution)
        - Neutral (40-60): 1.0 (no adjustment)
        - Greed (60-80): 0.85 (market hot, reduce risk)
        - Extreme Greed (>80): 0.6 (correction likely, pull back hard)
        """
        v = self.fear_greed_value
        if v < 20:
            return 0.7
        elif v < 40:
            return 0.85
        elif v <= 60:
            return 1.0
        elif v <= 80:
            return 0.85
        else:
            return 0.6

    # ─── BTC Lead-Lag Filter ────────────────────────────────────

    def update_btc_context(self, btc_closes: list[float]):
        """Update BTC lead-lag analysis from recent BTC prices.

        Args:
            btc_closes: recent BTC close prices (1h candles), newest last.
                        Need at least 3 for 1h return + recent bars.
        """
        if len(btc_closes) < 3:
            return

        # 1-hour return (1 bar back on 1h data)
        self.btc_1h_return = (btc_closes[-1] / btc_closes[-2]) - 1

        # Recent per-bar returns (last 6 bars = 6 hours)
        self.btc_recent_returns = []
        for i in range(-min(6, len(btc_closes) - 1), 0):
            if abs(i) < len(btc_closes):
                r = (btc_closes[i] / btc_closes[i - 1]) - 1
                self.btc_recent_returns.append(r)

        # BTC crash filter: if BTC dropped >1.5% in last 1 hour, pause altcoin buys
        if self.btc_1h_return < -0.015:
            # Skip buys for 2 hours (2 x 1h cycles)
            self.skip_buys_until = time.time() + 7200
            log.warning(f"BTC CRASH FILTER: BTC 1h return={self.btc_1h_return:.2%}, pausing altcoin buys 2h")

        # BTC momentum boost: if BTC up >2% in last hour with acceleration
        self.btc_momentum_boost = 0.0
        if self.btc_1h_return > 0.02:
            # Check if the move is accelerating (recent bars stronger than earlier)
            if len(self.btc_recent_returns) >= 4:
                recent_avg = sum(self.btc_recent_returns[-2:]) / 2
                earlier_avg = sum(self.btc_recent_returns[:2]) / 2
                if recent_avg > earlier_avg > 0:
                    self.btc_momentum_boost = min(0.3, self.btc_1h_return * 5)
                    log.info(f"BTC MOMENTUM BOOST: +{self.btc_momentum_boost:.2f} (BTC 1h={self.btc_1h_return:.2%})")

    def should_skip_altcoin_buys(self) -> bool:
        """Whether to suppress altcoin buy signals due to BTC weakness."""
        return time.time() < self.skip_buys_until

    def get_btc_signal_boost(self) -> float:
        """Extra signal strength to add when BTC is strongly bullish.

        Returns 0.0 to 0.3. Add to altcoin signal strength for breakout entries.
        """
        return self.btc_momentum_boost

    def get_status(self) -> dict:
        return {
            "fear_greed": self.fear_greed_value,
            "fear_greed_label": self.fear_greed_label,
            "fg_exposure_mult": round(self.get_fg_exposure_multiplier(), 2),
            "btc_1h_return": round(self.btc_1h_return * 100, 2),
            "btc_skip_buys": self.should_skip_altcoin_buys(),
            "btc_momentum_boost": round(self.btc_momentum_boost, 3),
        }
