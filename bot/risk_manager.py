"""
Risk manager v6: unified trailing stops, REDD scaling, regime-aware sizing.

Key changes from v2:
- UNIFIED trailing stop policy (no strategy branching)
- All stop parameters from config.py (single source of truth)
- Exposure multipliers: LOW_VOL=1.0, MID_VOL=0.7, HI_VOL=0.0

Stop design rationale (Sortino optimization):
- Hard stop at -3.5%: caps downside, reduces Sortino denominator
- Partial exit (50%) at +3%: locks in gains, creates asymmetric payoff
- Trailing stop: 3.5% from high (4.5% after partial for house-money effect)
- Time stop: 60h with <1% gain (opportunity cost recapture)
"""
import time
import numpy as np
from typing import Optional
from bot.config import (
    MAX_POSITION_PCT, MAX_TOTAL_EXPOSURE_PCT, TARGET_RISK_PER_TRADE,
    MAX_POSITIONS,
    HARD_STOP_PCT, PARTIAL_EXIT_PCT, PARTIAL_EXIT_FRACTION,
    TRAILING_STOP_PCT, TRAILING_STOP_WIDE_PCT,
    TIME_STOP_HOURS, TIME_STOP_MIN_GAIN,
    DRAWDOWN_LEVEL_1, DRAWDOWN_LEVEL_2, DRAWDOWN_LEVEL_3,
    DRAWDOWN_PAUSE_HOURS, REDD_MAX_DD,
)
from bot.logger import get_logger

log = get_logger("risk_manager")


class RiskManager:
    """Manages portfolio risk with REDD scaling and unified trailing stops."""

    def __init__(self, initial_portfolio_value: float):
        self.initial_value = initial_portfolio_value
        self.peak_value = initial_portfolio_value
        self.current_value = initial_portfolio_value

        # Trailing stops: pair -> {high, entry_price, entry_time, partial_taken}
        self.trailing_stops: dict[str, dict] = {}

        # Drawdown pause
        self.pause_until: float = 0.0
        self.drawdown_level: int = 0

        # External multiplier (set by regime detector)
        self.regime_exposure_mult: float = 1.0

    def update_portfolio_value(self, value: float):
        self.current_value = value
        if value > self.peak_value:
            self.peak_value = value

    def set_regime_multiplier(self, mult: float):
        self.regime_exposure_mult = np.clip(mult, 0.1, 1.0)

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

        Smoothly reduces NEW position sizes as drawdown grows.
        At 0% DD → 1.0x. At REDD_MAX_DD → 0.0x. Linear interpolation.
        This is the PRIMARY risk control — circuit breakers are emergency-only.
        """
        dd = self.drawdown_from_peak
        if dd <= 0:
            return 1.0
        return max(0.0, 1.0 - (dd / REDD_MAX_DD))

    def check_drawdown_breakers(self) -> dict:
        """Emergency breakers. REDD handles normal drawdown smoothly.

        Level 1: warning only (REDD is already reducing)
        Level 2: liquidate + pause
        Level 3: liquidate + longer pause
        """
        dd = self.drawdown_from_peak

        if dd >= DRAWDOWN_LEVEL_3 and self.drawdown_level < 3:
            self.drawdown_level = 3
            self.pause_until = time.time() + DRAWDOWN_PAUSE_HOURS[3] * 3600
            log.warning(f"DRAWDOWN LEVEL 3: {dd:.2%} — EMERGENCY LIQUIDATION")
            return {"level": 3, "action": "liquidate", "reduce_pct": 1.0}

        if dd >= DRAWDOWN_LEVEL_2 and self.drawdown_level < 2:
            self.drawdown_level = 2
            self.pause_until = time.time() + DRAWDOWN_PAUSE_HOURS[2] * 3600
            log.warning(f"DRAWDOWN LEVEL 2: {dd:.2%} — LIQUIDATION + PAUSE")
            return {"level": 2, "action": "liquidate", "reduce_pct": 1.0}

        if dd >= DRAWDOWN_LEVEL_1 and self.drawdown_level < 1:
            self.drawdown_level = 1
            redd = self._redd_multiplier()
            log.warning(f"DRAWDOWN LEVEL 1: {dd:.2%} — REDD active, sizing at {redd:.0%}")
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
        rank_multiplier: float = 1.0,
    ) -> float:
        """Calculate position size.

        Sizing stack (multiplicative):
        1. Vol-parity base: TARGET_RISK / daily_vol → risk-budgeted sizing
        2. Cap: MAX_POSITION_PCT, remaining exposure room
        3. REDD: smooth drawdown scaling
        4. Signal strength: model confidence
        5. Rank multiplier: top-ranked positions get more capital
        """
        if self.is_paused:
            return 0.0
        if num_current_positions >= MAX_POSITIONS:
            return 0.0

        portfolio = self.current_value

        # Effective max exposure = base × regime multiplier
        effective_exposure = MAX_TOTAL_EXPOSURE_PCT * self.regime_exposure_mult
        remaining = (effective_exposure * portfolio) - current_exposure_usd
        if remaining <= 0:
            return 0.0

        # Vol-parity base
        if annualized_vol <= 0:
            annualized_vol = 0.5
        daily_vol = annualized_vol / np.sqrt(365)
        if daily_vol <= 0:
            daily_vol = 0.03

        size = (TARGET_RISK_PER_TRADE * portfolio) / daily_vol
        size = min(size, MAX_POSITION_PCT * portfolio)
        size = min(size, remaining)

        # REDD scaling
        size *= self._redd_multiplier()

        # Signal & rank scaling
        size *= np.clip(signal_strength, 0.2, 1.3)
        size *= rank_multiplier

        return max(0.0, size)

    # ─── Unified Trailing Stops ─────────────────────────────────

    def update_trailing_stop(
        self,
        pair: str,
        current_price: float,
        entry_price: float = 0,
    ):
        """Update or create trailing stop for a position."""
        if pair in self.trailing_stops:
            info = self.trailing_stops[pair]
            info["high"] = max(info["high"], current_price)
        else:
            self.trailing_stops[pair] = {
                "high": current_price,
                "entry_price": entry_price or current_price,
                "entry_time": time.time(),
                "partial_taken": False,
            }

    def check_trailing_stop(
        self, pair: str, current_price: float
    ) -> tuple[bool, str, float]:
        """Check if trailing stop is hit.

        Returns (should_exit, reason, exit_fraction).

        Unified policy (no strategy branching):
        1. Hard stop: -HARD_STOP_PCT from entry → full exit
        2. Partial exit: +PARTIAL_EXIT_PCT from entry → sell PARTIAL_EXIT_FRACTION
        3. Trailing stop: TRAILING_STOP_PCT from high (TRAILING_STOP_WIDE_PCT after partial)
        4. Time stop: TIME_STOP_HOURS with < TIME_STOP_MIN_GAIN → full exit
        """
        if pair not in self.trailing_stops:
            return False, "", 1.0

        info = self.trailing_stops[pair]
        high = info["high"]
        entry = info["entry_price"]
        entry_time = info.get("entry_time", time.time())
        partial_taken = info.get("partial_taken", False)

        drop_from_high = (high - current_price) / high if high > 0 else 0
        pnl = (current_price - entry) / entry if entry > 0 else 0
        holding_hours = (time.time() - entry_time) / 3600

        # 1. Hard stop (most important — cap losses)
        if pnl <= -HARD_STOP_PCT:
            log.info(f"HARD STOP: {pair} {pnl:+.2%} from entry")
            return True, "hard_stop", 1.0

        # 2. Partial exit at profit target (Sortino optimization)
        if pnl >= PARTIAL_EXIT_PCT and not partial_taken:
            log.info(
                f"PARTIAL EXIT: {pair} +{pnl:.2%}, "
                f"selling {PARTIAL_EXIT_FRACTION:.0%}"
            )
            info["partial_taken"] = True
            return True, "partial_exit", PARTIAL_EXIT_FRACTION

        # 3. Trailing stop
        trail_pct = TRAILING_STOP_WIDE_PCT if partial_taken else TRAILING_STOP_PCT
        # Only trail when we're in profit or after partial
        if (partial_taken or pnl > 0.02) and drop_from_high >= trail_pct:
            log.info(
                f"TRAILING STOP: {pair} -{drop_from_high:.2%} from high "
                f"(threshold: {trail_pct:.1%})"
            )
            return True, "trailing_stop", 1.0

        # 4. Time stop
        if holding_hours > TIME_STOP_HOURS and pnl < TIME_STOP_MIN_GAIN:
            log.info(
                f"TIME STOP: {pair} held {holding_hours:.0f}h, "
                f"PnL={pnl:+.2%} < {TIME_STOP_MIN_GAIN:.0%}"
            )
            return True, "time_stop", 1.0

        return False, "", 1.0

    def clear_trailing_stop(self, pair: str) -> Optional[dict]:
        return self.trailing_stops.pop(pair, None)

    def get_status(self) -> dict:
        return {
            "portfolio_value": round(self.current_value, 2),
            "peak_value": round(self.peak_value, 2),
            "drawdown_pct": round(self.drawdown_from_peak * 100, 2),
            "drawdown_level": self.drawdown_level,
            "redd_mult": round(self._redd_multiplier(), 3),
            "regime_mult": round(self.regime_exposure_mult, 3),
            "effective_exposure": round(
                MAX_TOTAL_EXPOSURE_PCT * self.regime_exposure_mult, 3
            ),
            "is_paused": self.is_paused,
            "pause_remaining_min": max(
                0, round((self.pause_until - time.time()) / 60, 1)
            ),
            "open_stops": len(self.trailing_stops),
        }