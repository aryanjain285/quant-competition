"""
Risk manager: position sizing, drawdown circuit breakers, trailing stops.
This is where the competition is won — protects Sortino, Sharpe, and Calmar.
"""
import time
from typing import Optional
from bot.config import (
    MAX_POSITION_PCT, MAX_TOTAL_EXPOSURE_PCT, TARGET_RISK_PER_TRADE,
    MAX_POSITIONS, TRAILING_STOP_PCT,
    DRAWDOWN_LEVEL_1, DRAWDOWN_LEVEL_2, DRAWDOWN_LEVEL_3,
    DRAWDOWN_PAUSE_HOURS,
)
from bot.logger import get_logger

log = get_logger("risk_manager")


class RiskManager:
    """Manages portfolio risk: sizing, stops, and drawdown breakers."""

    def __init__(self, initial_portfolio_value: float):
        self.initial_value = initial_portfolio_value
        self.peak_value = initial_portfolio_value
        self.current_value = initial_portfolio_value

        # Trailing stops: pair -> {high, strategy, entry_price}
        self.trailing_stops: dict[str, dict] = {}

        # Drawdown pause: timestamp when trading can resume
        self.pause_until: float = 0.0

        # Current drawdown level active (0 = none)
        self.drawdown_level: int = 0

    def update_portfolio_value(self, value: float):
        """Call every cycle with current portfolio value."""
        self.current_value = value
        if value > self.peak_value:
            self.peak_value = value

    @property
    def drawdown_from_peak(self) -> float:
        """Current drawdown as a positive fraction (0.05 = 5% down from peak)."""
        if self.peak_value <= 0:
            return 0.0
        return max(0.0, (self.peak_value - self.current_value) / self.peak_value)

    @property
    def is_paused(self) -> bool:
        """Whether trading is paused due to drawdown breaker."""
        return time.time() < self.pause_until

    def check_drawdown_breakers(self) -> dict:
        """Check drawdown levels and return action to take.

        Returns:
            dict with keys:
                - level: 0 (ok), 1 (reduce), 2 (liquidate), 3 (emergency)
                - action: "none", "reduce", "liquidate"
                - reduce_pct: fraction to reduce positions by (for level 1)
        """
        dd = self.drawdown_from_peak

        if dd >= DRAWDOWN_LEVEL_3 and self.drawdown_level < 3:
            self.drawdown_level = 3
            self.pause_until = time.time() + DRAWDOWN_PAUSE_HOURS[3] * 3600
            log.warning(f"DRAWDOWN LEVEL 3: {dd:.2%} — EMERGENCY LIQUIDATION, pausing {DRAWDOWN_PAUSE_HOURS[3]}h")
            return {"level": 3, "action": "liquidate", "reduce_pct": 1.0}

        elif dd >= DRAWDOWN_LEVEL_2 and self.drawdown_level < 2:
            self.drawdown_level = 2
            self.pause_until = time.time() + DRAWDOWN_PAUSE_HOURS[2] * 3600
            log.warning(f"DRAWDOWN LEVEL 2: {dd:.2%} — FULL LIQUIDATION, pausing {DRAWDOWN_PAUSE_HOURS[2]}h")
            return {"level": 2, "action": "liquidate", "reduce_pct": 1.0}

        elif dd >= DRAWDOWN_LEVEL_1 and self.drawdown_level < 1:
            self.drawdown_level = 1
            log.warning(f"DRAWDOWN LEVEL 1: {dd:.2%} — reducing positions 50%")
            return {"level": 1, "action": "reduce", "reduce_pct": 0.5}

        # Reset level if recovered
        if dd < DRAWDOWN_LEVEL_1 * 0.5:
            self.drawdown_level = 0

        return {"level": 0, "action": "none", "reduce_pct": 0.0}

    def position_size_usd(
        self,
        pair: str,
        annualized_vol: float,
        current_exposure_usd: float,
        num_current_positions: int,
    ) -> float:
        """Calculate position size in USD using volatility-parity.

        Targets a fixed risk per position, automatically sizing down
        in volatile markets.

        Args:
            pair: trading pair
            annualized_vol: annualized volatility of the coin
            current_exposure_usd: total USD currently in crypto positions
            num_current_positions: number of positions currently held
        """
        if self.is_paused:
            return 0.0

        if num_current_positions >= MAX_POSITIONS:
            return 0.0

        portfolio = self.current_value

        # Max exposure check
        remaining_exposure = (MAX_TOTAL_EXPOSURE_PCT * portfolio) - current_exposure_usd
        if remaining_exposure <= 0:
            return 0.0

        # Volatility-based sizing
        # Convert annualized vol to per-period (5-min) vol for holding period estimate
        if annualized_vol <= 0:
            annualized_vol = 0.5  # default 50% annual vol if unknown

        # Target: risk TARGET_RISK_PER_TRADE of portfolio
        # position_size = (target_risk * portfolio) / vol_per_holding_period
        # Assume ~24h holding period = 288 5-min bars
        daily_vol = annualized_vol / np.sqrt(365)
        if daily_vol <= 0:
            daily_vol = 0.03  # 3% default

        size = (TARGET_RISK_PER_TRADE * portfolio) / daily_vol

        # Cap at max position size
        max_size = MAX_POSITION_PCT * portfolio
        size = min(size, max_size)

        # Cap at remaining exposure room
        size = min(size, remaining_exposure)

        # Reduce size if drawdown level 1
        if self.drawdown_level >= 1:
            size *= 0.5

        return max(0.0, size)

    def update_trailing_stop(self, pair: str, current_price: float,
                            strategy: str = "breakout", entry_price: float = 0):
        """Update or create trailing stop for a position.

        Different strategies get different stop widths:
        - breakout: wider stop (4%) to let trends run
        - mean_rev: tighter stop (2%) for quick profit-taking,
                    plus hard stop at -3% from entry
        """
        if pair in self.trailing_stops:
            info = self.trailing_stops[pair]
            info["high"] = max(info["high"], current_price)
        else:
            self.trailing_stops[pair] = {
                "high": current_price,
                "strategy": strategy,
                "entry_price": entry_price or current_price,
            }

    def check_trailing_stop(self, pair: str, current_price: float) -> bool:
        """Check if trailing stop has been hit. Returns True if should exit."""
        if pair not in self.trailing_stops:
            return False

        info = self.trailing_stops[pair]
        high = info["high"]
        strategy = info["strategy"]
        entry = info["entry_price"]

        # Strategy-specific trailing stop width
        if strategy == "mean_rev":
            # Tight trailing: 2% from high, or hard stop at -3% from entry
            trail_pct = 0.02
            hard_stop = -0.03
        else:
            # Breakout: moderate trailing (3%) — data shows tighter stops work better
            trail_pct = 0.03
            hard_stop = -0.04

        drop_from_high = (high - current_price) / high
        drop_from_entry = (current_price - entry) / entry

        if drop_from_high >= trail_pct:
            log.info(f"TRAILING STOP [{strategy}]: {pair} dropped {drop_from_high:.2%} from high {high:.4f}")
            return True

        if drop_from_entry <= hard_stop:
            log.info(f"HARD STOP [{strategy}]: {pair} down {drop_from_entry:.2%} from entry {entry:.4f}")
            return True

        # Mean reversion profit target: take profit at +3% from entry
        if strategy == "mean_rev" and (current_price - entry) / entry >= 0.03:
            log.info(f"PROFIT TARGET [mean_rev]: {pair} up {(current_price - entry) / entry:.2%}")
            return True

        return False

    def clear_trailing_stop(self, pair: str):
        """Remove trailing stop when position is closed."""
        self.trailing_stops.pop(pair, None)

    def get_status(self) -> dict:
        """Return current risk state for logging."""
        return {
            "portfolio_value": round(self.current_value, 2),
            "peak_value": round(self.peak_value, 2),
            "drawdown_pct": round(self.drawdown_from_peak * 100, 2),
            "drawdown_level": self.drawdown_level,
            "is_paused": self.is_paused,
            "pause_remaining_min": max(0, round((self.pause_until - time.time()) / 60, 1)),
        }


# Need numpy for sqrt
import numpy as np
