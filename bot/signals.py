"""
Signal engine v4: Event filter using continuation/reversal framework.

Replaces v3's breakout/RSI with grounded event detection:

  1. CONTINUATION setup: coin is trending up cleanly with volume.
     Fires when: breakout_distance > 0, r_24h > 0, persistence high,
     choppiness low, risk/cost acceptable, volume confirmed.
     "It is moving up, the move is clean, it is not too dangerous,
      and it is not too expensive to trade."

  2. REVERSAL setup: coin has dropped unusually hard, bounce potential.
     Fires when: overshoot extreme (standardized drop vs own history),
     risk elevated but not catastrophic, cost acceptable.
     "The coin fell much more than normal, but conditions still allow
      a bounce trade."

Both use features computed in features.py (persistence, choppiness,
risk/cost penalties, overshoot). Volume confirmation is MANDATORY
for continuation (proven in v3 backtest: single biggest improvement).

Pure functions — same code runs in backtest and live.
"""
import numpy as np
from bot.features import (
    compute_coin_features, zscore_universe, compute_submodel_scores,
)


# ═══════════════════════════════════════════════════════════════
# EVENT DETECTION (replaces breakout_signal / mean_reversion_signal)
# ═══════════════════════════════════════════════════════════════

def check_continuation(features: dict, zscored: dict) -> dict:
    """Check if a coin shows a valid continuation (trend-following) setup.

    Uses raw features for thresholds, z-scored for scoring.
    Returns {"valid": bool, "strength": float, "reasons": list}
    """
    result = {"valid": False, "strength": 0.0, "reasons": []}

    # All conditions must pass (conjunction — avoids false entries)
    # 1. Price above recent high (breakout distance > 0)
    if features.get("breakout_distance", 0) <= 0:
        return result
    result["reasons"].append("breakout")

    # 2. Medium-horizon return positive (24h)
    if features.get("r_24h", 0) <= 0:
        return result
    result["reasons"].append("r24h_positive")

    # 3. Persistence high (move is internally consistent)
    if features.get("persistence", 0.5) < 0.52:
        return result
    result["reasons"].append("persistent")

    # 4. Choppiness low (path is clean)
    if features.get("choppiness", 0.5) > 0.85:
        return result
    result["reasons"].append("clean_path")

    # 5. Volume confirmed (mandatory — v3 proven)
    if features.get("volume_ratio", 1.0) < 1.3:
        return result
    result["reasons"].append("volume")

    # 6. Risk penalty not extreme (z-scored)
    if zscored.get("risk_penalty", 0) > 1.5:
        return result
    result["reasons"].append("risk_ok")

    # 7. Cost penalty not extreme (z-scored)
    if zscored.get("cost_penalty", 0) > 1.5:
        return result
    result["reasons"].append("cost_ok")

    # All passed — valid continuation
    result["valid"] = True

    # Strength: based on how strong the features are
    strength = 0.5
    strength += min(0.15, features["breakout_distance"] * 5)  # stronger breakout
    strength += min(0.10, features["r_24h"] * 2)               # stronger momentum
    strength += 0.05 if features["persistence"] > 0.58 else 0  # extra clean
    strength += 0.05 if features["volume_ratio"] > 2.0 else 0  # strong volume
    strength -= 0.05 if zscored.get("risk_penalty", 0) > 0.5 else 0  # risk discount
    result["strength"] = round(min(strength, 1.0), 3)

    return result


def check_reversal(features: dict, zscored: dict) -> dict:
    """Check if a coin shows a valid reversal (mean-reversion) setup.

    Uses overshoot score: standardized recent drop vs coin's own history.
    Returns {"valid": bool, "strength": float, "reasons": list}
    """
    result = {"valid": False, "strength": 0.0, "reasons": []}

    # Overshoot must be extreme (large negative z-score = unusually large drop)
    overshoot = features.get("overshoot", 0)
    if overshoot > -1.5:
        # Not oversold enough — overshoot is negative for drops
        return result
    result["reasons"].append(f"oversold(z={overshoot:.1f})")

    # Risk must be elevated but not catastrophic
    # High risk = the coin moved a lot = there's something to revert
    # But if risk is extreme, it might be a real crash not a bounce candidate
    risk_z = zscored.get("risk_penalty", 0)
    if risk_z > 2.5:
        return result  # too dangerous
    result["reasons"].append("risk_manageable")

    # Cost must be acceptable
    if zscored.get("cost_penalty", 0) > 1.5:
        return result
    result["reasons"].append("cost_ok")

    # Recent price action should show some stabilization
    # (not free-falling — r_1h should not be deeply negative)
    r_1h = features.get("r_1h", 0)
    if r_1h < -0.03:
        return result  # still crashing, don't catch falling knife
    result["reasons"].append("stabilizing")

    result["valid"] = True

    # Strength: deeper overshoot = stronger signal
    strength = 0.4
    strength += min(0.25, abs(overshoot) * 0.1)  # deeper = stronger
    strength += 0.1 if r_1h > 0 else 0            # already bouncing = bonus
    strength -= 0.05 if risk_z > 1.0 else 0        # risk discount
    result["strength"] = round(min(strength, 1.0), 3)

    return result


def check_breakdown(closes: np.ndarray, lows: np.ndarray, lookback: int = 72) -> bool:
    """Check if price has broken below recent low (exit signal)."""
    if len(closes) < lookback + 1:
        return False
    breakdown_lb = max(lookback // 3, 12)
    if len(lows) >= breakdown_lb + 1:
        prior_low = np.min(lows[-breakdown_lb - 1:-1])
    else:
        prior_low = np.min(closes[-breakdown_lb - 1:-1])
    return closes[-1] < prior_low and prior_low > 0


# ═══════════════════════════════════════════════════════════════
# MAIN SIGNAL COMPUTATION (called by main.py for each coin)
# ═══════════════════════════════════════════════════════════════

def compute_signal(
    raw_features: dict,
    zscored_features: dict,
    closes: np.ndarray,
    lows: np.ndarray,
    breakout_lookback: int = 72,
) -> dict:
    """Compute signal for one coin using continuation/reversal framework.

    Args:
        raw_features: raw per-coin features from features.py
        zscored_features: z-scored version (with submodel scores)
        closes: raw close prices (for breakdown detection)
        lows: raw low prices (for breakdown detection)

    Returns dict compatible with main.py:
        action: "BUY" / "SELL" / "HOLD"
        strategy: "continuation" / "reversal" / "breakdown" / "none"
        strength: 0-1 signal confidence
        + all feature values for logging
    """
    result = {
        "action": "HOLD",
        "strategy": "none",
        "strength": 0.0,
        "continuation": {},
        "reversal": {},
        "breakdown": False,
        # Pass through features for logging and derivatives overlay
        "rsi": 50.0,  # not used for decisions anymore, kept for logging
        "real_vol": raw_features.get("realized_vol", 0),
        "down_vol": raw_features.get("downside_vol", 0),
        "spread": raw_features.get("spread_pct", 0),
        "breakout_strength": raw_features.get("breakout_distance", 0),
        "volume_confirm": raw_features.get("volume_ratio", 1.0) > 1.3,
    }

    # Check continuation (trend-following)
    cont = check_continuation(raw_features, zscored_features)
    result["continuation"] = cont

    # Check reversal (mean-reversion)
    rev = check_reversal(raw_features, zscored_features)
    result["reversal"] = rev

    # Check breakdown (exit signal)
    is_breakdown = check_breakdown(closes, lows, breakout_lookback)
    result["breakdown"] = is_breakdown

    # Decision priority: breakdown > continuation > reversal
    if is_breakdown:
        result["action"] = "SELL"
        result["strategy"] = "breakdown"
        result["strength"] = 0.8

    elif cont["valid"]:
        result["action"] = "BUY"
        result["strategy"] = "continuation"
        result["strength"] = cont["strength"]

    elif rev["valid"]:
        result["action"] = "BUY"
        result["strategy"] = "reversal"
        result["strength"] = rev["strength"]

    return result
