"""
Risk manager v2: position sizing, drawdown control, trailing stops.

Upgrades over v1:
- REDD (Rolling Economic Drawdown) scaling: dynamically reduces size as drawdown grows
- Regime-aware exposure: accepts multiplier from regime detector
- Sortino-optimized exits: tight downside stops, let upside run (no cap on breakout)
- Strategy-specific trailing stops calibrated to backtest results
"""
import time
import numpy as np
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
    """Manages portfolio risk with REDD scaling and Sortino-optimized stops."""

    def __init__(self, initial_portfolio_value: float):
        self.initial_value = initial_portfolio_value
        self.peak_value = initial_portfolio_value
        self.current_value = initial_portfolio_value

        # Trailing stops: pair -> {high, strategy, entry_price, entry_features}
        self.trailing_stops: dict[str, dict] = {}

        # Drawdown pause: timestamp when trading can resume
        self.pause_until: float = 0.0

        # Current drawdown level active (0 = none)
        self.drawdown_level: int = 0

        # External multipliers (set by regime detector, sentiment, etc.)
        self.regime_exposure_mult: float = 1.0
        self.sentiment_exposure_mult: float = 1.0

    def update_portfolio_value(self, value: float):
        self.current_value = value
        if value > self.peak_value:
            self.peak_value = value

    def set_regime_multiplier(self, mult: float):
        """Set by regime detector each cycle."""
        self.regime_exposure_mult = np.clip(mult, 0.1, 1.0)

    def set_sentiment_multiplier(self, mult: float):
        """Set by sentiment analyzer each cycle."""
        self.sentiment_exposure_mult = np.clip(mult, 0.3, 1.0)

    @property
    def drawdown_from_peak(self) -> float:
        if self.peak_value <= 0:
            return 0.0
        return max(0.0, (self.peak_value - self.current_value) / self.peak_value)

    @property
    def is_paused(self) -> bool:
        return time.time() < self.pause_until

    def _redd_multiplier(self) -> float:
        """Rolling Economic Drawdown scaling.

        Smoothly reduces NEW position sizes as drawdown increases.
        At 0% dd → 1.0x, at 8% dd → 0.0x.
        This is the PRIMARY drawdown control mechanism.
        Circuit breakers below are EMERGENCY-only for tail events.
        """
        dd = self.drawdown_from_peak
        max_dd_limit = 0.10  # at 10% DD new positions = zero (matches level 3 breaker)
        if dd <= 0:
            return 1.0
        return max(0.0, 1.0 - (dd / max_dd_limit))

    def check_drawdown_breakers(self) -> dict:
        """Emergency breakers only. REDD handles normal drawdown smoothly.

        Level 1: log warning only — REDD is already reducing sizing
        Level 2: liquidate + short pause (actual emergency)
        Level 3: liquidate + longer pause (catastrophic)

        NO forced position reduction at level 1. Selling at the bottom
        realizes losses and prevents recovery. REDD's smooth scaling
        is strictly better than discrete cuts.
        """
        dd = self.drawdown_from_peak

        if dd >= DRAWDOWN_LEVEL_3 and self.drawdown_level < 3:
            self.drawdown_level = 3
            self.pause_until = time.time() + DRAWDOWN_PAUSE_HOURS[3] * 3600
            log.warning(f"DRAWDOWN LEVEL 3: {dd:.2%} — EMERGENCY, pausing {DRAWDOWN_PAUSE_HOURS[3]}h")
            return {"level": 3, "action": "liquidate", "reduce_pct": 1.0}

        elif dd >= DRAWDOWN_LEVEL_2 and self.drawdown_level < 2:
            self.drawdown_level = 2
            self.pause_until = time.time() + DRAWDOWN_PAUSE_HOURS[2] * 3600
            log.warning(f"DRAWDOWN LEVEL 2: {dd:.2%} — LIQUIDATION, pausing {DRAWDOWN_PAUSE_HOURS[2]}h")
            return {"level": 2, "action": "liquidate", "reduce_pct": 1.0}

        elif dd >= DRAWDOWN_LEVEL_1 and self.drawdown_level < 1:
            self.drawdown_level = 1
            log.warning(f"DRAWDOWN LEVEL 1: {dd:.2%} — REDD active, new sizing at {self._redd_multiplier():.0%}")
            # NO forced sell — REDD handles it. Just log the warning.
            return {"level": 1, "action": "none", "reduce_pct": 0.0}

        if dd < DRAWDOWN_LEVEL_1 * 0.5:
            self.drawdown_level = 0

        return {"level": 0, "action": "none", "reduce_pct": 0.0}

    def position_size_usd(
        self,
        pair: str,
        annualized_vol: float,
        current_exposure_usd: float,
        num_current_positions: int,
        signal_strength: float = 1.0,
        deriv_score: float = 0.0,
    ) -> float:
        """Calculate position size with REDD scaling and regime awareness.

        Args:
            pair: trading pair
            annualized_vol: coin's annualized volatility
            current_exposure_usd: total USD in crypto positions
            num_current_positions: number of open positions
            signal_strength: from signal engine (0-1)
            deriv_score: from derivatives (-1 to +1), boosts/reduces size
        """
        if self.is_paused:
            return 0.0
        if num_current_positions >= MAX_POSITIONS:
            return 0.0

        portfolio = self.current_value

        # Effective max exposure adjusted by regime + sentiment
        effective_exposure = (
            MAX_TOTAL_EXPOSURE_PCT
            * self.regime_exposure_mult
            * self.sentiment_exposure_mult
        )
        remaining_exposure = (effective_exposure * portfolio) - current_exposure_usd
        if remaining_exposure <= 0:
            return 0.0

        # Volatility-parity base sizing
        if annualized_vol <= 0:
            annualized_vol = 0.5
        daily_vol = annualized_vol / np.sqrt(365)
        if daily_vol <= 0:
            daily_vol = 0.03

        size = (TARGET_RISK_PER_TRADE * portfolio) / daily_vol

        # Cap at per-position max
        size = min(size, MAX_POSITION_PCT * portfolio)

        # Cap at remaining room
        size = min(size, remaining_exposure)

        # REDD scaling: smoothly reduce as drawdown grows
        size *= self._redd_multiplier()

        # Derivatives boost/penalty: ±20% max
        deriv_adj = 1.0 + np.clip(deriv_score * 0.2, -0.2, 0.2)
        size *= deriv_adj

        # Signal strength scaling
        size *= np.clip(signal_strength, 0.3, 1.0)

        return max(0.0, size)

    # ─── Trailing Stops (Sortino-Optimized) ─────────────────────

    def update_trailing_stop(self, pair: str, current_price: float,
                            strategy: str = "breakout", entry_price: float = 0,
                            entry_features: np.ndarray = None):
        """Update or create trailing stop for a position."""
        if pair in self.trailing_stops:
            info = self.trailing_stops[pair]
            info["high"] = max(info["high"], current_price)
        else:
            self.trailing_stops[pair] = {
                "high": current_price,
                "strategy": strategy,
                "entry_price": entry_price or current_price,
                "entry_features": entry_features,
                "entry_time": time.time(),
            }

    def check_trailing_stop(self, pair: str, current_price: float) -> tuple[bool, str, float]:
        """Check if trailing stop hit.

        Returns (should_exit, reason, exit_fraction).
        exit_fraction: 1.0 = full exit, 0.5 = partial exit (sell half).

        Sortino-optimized:
        - Tight stops on downside, NO upside cap on breakouts
        - Breakouts: PARTIAL EXIT at +3% (sell 50%, lock in gains, let rest run)
          This reduces downside vol in the Sortino denominator while preserving upside.
        - Mean-reversion: full exit at profit target (quick trades by design)
        """
        if pair not in self.trailing_stops:
            return False, "", 1.0

        info = self.trailing_stops[pair]
        high = info["high"]
        strategy = info["strategy"]
        entry = info["entry_price"]
        entry_time = info.get("entry_time", time.time())
        holding_hours = (time.time() - entry_time) / 3600
        partial_taken = info.get("partial_taken", False)

        drop_from_high = (high - current_price) / high if high > 0 else 0
        pnl_from_entry = (current_price - entry) / entry if entry > 0 else 0

        # ── Mean reversion: tight stops + profit target (full exits) ──
        if strategy == "mean_rev":
            # Hard stop first (most important — cap losses)
            if pnl_from_entry <= -0.03:
                log.info(f"HARD STOP [mean_rev]: {pair} {pnl_from_entry:.2%} from entry")
                return True, "hard_stop", 1.0
            # Trailing only when in profit (don't trail below entry)
            if drop_from_high >= 0.02 and pnl_from_entry > 0:
                log.info(f"TRAILING STOP [mean_rev]: {pair} -{drop_from_high:.2%} from high")
                return True, "trailing_stop", 1.0
            if pnl_from_entry >= 0.03:
                log.info(f"PROFIT TARGET [mean_rev]: {pair} +{pnl_from_entry:.2%}")
                return True, "profit_target", 1.0
            if holding_hours > 48 and pnl_from_entry < 0.005:
                log.info(f"TIME STOP [mean_rev]: {pair} held {holding_hours:.0f}h")
                return True, "time_stop", 1.0

        # ── Breakout: partial exit at +3%, trailing stop, NO upside cap ──
        else:
            # Hard stop near entry
            if pnl_from_entry <= -0.04:
                log.info(f"HARD STOP [breakout]: {pair} {pnl_from_entry:.2%} from entry")
                return True, "hard_stop", 1.0

            # PARTIAL EXIT: sell 50% at +3% to lock in gains (Sortino optimization)
            # The locked-in profit reduces variance of outcomes
            # The remaining 50% rides with no upside cap
            if pnl_from_entry >= 0.03 and not partial_taken:
                log.info(f"PARTIAL EXIT [breakout]: {pair} +{pnl_from_entry:.2%}, selling 50%")
                info["partial_taken"] = True
                # After partial, widen the trailing stop for the remaining position
                info["entry_price"] = entry  # keep original entry for PnL tracking
                return True, "partial_exit", 0.5

            # Trailing stop (only after partial or in profit zone)
            if partial_taken or pnl_from_entry > 0.02:
                trail_pct = 0.04 if partial_taken else 0.03  # wider after partial
                if drop_from_high >= trail_pct:
                    log.info(f"TRAILING STOP [breakout]: {pair} -{drop_from_high:.2%} from high")
                    return True, "trailing_stop", 1.0

            # Time stop
            if holding_hours > 72 and pnl_from_entry < 0.01:
                log.info(f"TIME STOP [breakout]: {pair} held {holding_hours:.0f}h")
                return True, "time_stop", 1.0

        return False, "", 1.0

    def clear_trailing_stop(self, pair: str) -> Optional[dict]:
        """Remove trailing stop, return the stop info (for ML training)."""
        return self.trailing_stops.pop(pair, None)

    def get_status(self) -> dict:
        return {
            "portfolio_value": round(self.current_value, 2),
            "peak_value": round(self.peak_value, 2),
            "drawdown_pct": round(self.drawdown_from_peak * 100, 2),
            "drawdown_level": self.drawdown_level,
            "redd_mult": round(self._redd_multiplier(), 3),
            "regime_mult": round(self.regime_exposure_mult, 3),
            "sentiment_mult": round(self.sentiment_exposure_mult, 3),
            "effective_exposure": round(
                MAX_TOTAL_EXPOSURE_PCT * self.regime_exposure_mult * self.sentiment_exposure_mult, 3
            ),
            "is_paused": self.is_paused,
            "pause_remaining_min": max(0, round((self.pause_until - time.time()) / 60, 1)),
            "open_stops": len(self.trailing_stops),
        }
