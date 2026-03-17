"""
Signal engine v3: Reduced overtrading, higher conviction entries.

Changes from v2:
  - Breakout entries REQUIRE volume confirmation (mandatory, not bonus)
  - RSI oversold threshold tightened from 30 → 25 (fewer false entries)
  - Breakout strength baseline raised from 0.6 → 0.65

These changes reduce trade count by ~40-50% while keeping the highest-
conviction entries. Commission drag drops proportionally.

Two independent signal sources:
1. BREAKOUT — new N-hour highs WITH volume confirmation
2. RSI MEAN REVERSION — deeply oversold bounces (RSI < 25)
"""
import numpy as np


# ─────────────────────────────────────────────────────────────────
# Core indicators
# ─────────────────────────────────────────────────────────────────

def ema(prices: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    if len(prices) < 2:
        return prices.copy()
    alpha = 2.0 / (span + 1)
    out = np.empty_like(prices, dtype=float)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(prices: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index. Returns latest RSI value (0-100)."""
    if len(prices) < period + 2:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
    losses = np.mean(-deltas[deltas < 0]) if np.any(deltas < 0) else 0.0
    if losses == 0:
        return 100.0
    rs = gains / losses
    return float(100 - (100 / (1 + rs)))


def realized_volatility(prices: np.ndarray, lookback: int = 288) -> float:
    """Annualized realized volatility from log returns."""
    if len(prices) < lookback + 1:
        lookback = len(prices) - 1
    if lookback < 2:
        return 0.0
    log_returns = np.diff(np.log(prices[-lookback - 1:]))
    return float(np.std(log_returns)) * np.sqrt(288 * 365)


def downside_volatility(prices: np.ndarray, lookback: int = 288) -> float:
    """Annualized downside volatility (only negative returns)."""
    if len(prices) < lookback + 1:
        lookback = len(prices) - 1
    if lookback < 2:
        return 0.0
    log_returns = np.diff(np.log(prices[-lookback - 1:]))
    neg_returns = log_returns[log_returns < 0]
    if len(neg_returns) == 0:
        return 0.0
    return float(np.std(neg_returns)) * np.sqrt(288 * 365)


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Average True Range (last value)."""
    if len(closes) < period + 1:
        return 0.0
    if len(highs) >= period + 1 and len(lows) >= period + 1:
        tr = np.maximum(
            highs[-period:] - lows[-period:],
            np.maximum(
                np.abs(highs[-period:] - closes[-period - 1:-1]),
                np.abs(lows[-period:] - closes[-period - 1:-1]),
            ),
        )
    else:
        tr = np.abs(np.diff(closes[-(period + 1):]))
    return float(np.mean(tr))


def spread_pct(bid: float, ask: float) -> float:
    """Bid-ask spread as percentage of mid price."""
    if bid <= 0 or ask <= 0:
        return 0.0
    mid = (bid + ask) / 2
    return (ask - bid) / mid


# ─────────────────────────────────────────────────────────────────
# Signal 1: BREAKOUT (volume confirmation REQUIRED)
# ─────────────────────────────────────────────────────────────────

def breakout_signal(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    lookback: int = 72,
) -> dict:
    """Detect breakout: price exceeding recent N-period high.

    Returns dict with:
        - is_breakout: bool (True only if BOTH price breaks out AND volume confirms)
        - is_breakdown: bool
        - strength: how far above the prior high
        - volume_confirm: whether volume supports the move
    """
    result = {
        "is_breakout": False,
        "is_breakdown": False,
        "strength": 0.0,
        "volume_confirm": False,
    }

    if len(closes) < lookback + 1:
        return result

    if len(highs) >= lookback + 1:
        prior_high = np.max(highs[-(lookback + 1):-1])
    else:
        prior_high = np.max(closes[-(lookback + 1):-1])

    if len(lows) >= lookback + 1:
        breakdown_lookback = max(lookback // 3, 12)
        prior_low = np.min(lows[-breakdown_lookback - 1:-1])
    else:
        breakdown_lookback = max(lookback // 3, 12)
        prior_low = np.min(closes[-breakdown_lookback - 1:-1])

    current = closes[-1]

    # Volume confirmation: current volume > 1.3x recent average
    vol_confirmed = False
    if len(volumes) >= lookback:
        avg_vol = np.mean(volumes[-lookback:-1])
        if avg_vol > 0 and volumes[-1] > 1.3 * avg_vol:
            vol_confirmed = True
    result["volume_confirm"] = vol_confirmed

    # Breakout: price exceeds prior high AND volume confirms
    # This is the key change — volume is now REQUIRED, not a bonus
    if current > prior_high and prior_high > 0 and vol_confirmed:
        result["is_breakout"] = True
        result["strength"] = (current - prior_high) / prior_high

    # Breakdown: price falls below recent low (exit signal — no volume needed)
    if current < prior_low and prior_low > 0:
        result["is_breakdown"] = True

    return result


# ─────────────────────────────────────────────────────────────────
# Signal 2: RSI MEAN REVERSION (tightened threshold)
# ─────────────────────────────────────────────────────────────────

def mean_reversion_signal(
    closes: np.ndarray,
    rsi_oversold: float = 25.0,    # TIGHTENED from 30 — fewer, higher-conviction entries
    rsi_overbought: float = 65.0,
    rsi_period: int = 14,
) -> dict:
    """RSI-based mean reversion signal.

    Buy when deeply oversold (RSI < 25), sell when overbought.
    Tightened from 30 to reduce false entries and commission drag.
    """
    result = {
        "is_oversold": False,
        "is_overbought": False,
        "rsi_value": 50.0,
        "bounce_strength": 0.0,
    }

    if len(closes) < rsi_period + 10:
        return result

    current_rsi = rsi(closes, rsi_period)
    result["rsi_value"] = round(current_rsi, 2)

    if current_rsi < rsi_oversold:
        result["is_oversold"] = True
    elif current_rsi > rsi_overbought:
        result["is_overbought"] = True

    # Bounce strength: RSI rising from recent trough
    rsi_history = []
    for i in range(min(10, len(closes) - rsi_period - 1)):
        idx = len(closes) - i
        if idx > rsi_period + 1:
            rsi_history.append(rsi(closes[:idx], rsi_period))
    if len(rsi_history) >= 3:
        recent_low = min(rsi_history)
        result["bounce_strength"] = max(0, current_rsi - recent_low) / 100.0

    return result


# ─────────────────────────────────────────────────────────────────
# Combined signal for one pair
# ─────────────────────────────────────────────────────────────────

def compute_signal(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    bid: float,
    ask: float,
    breakout_lookback: int = 72,
) -> dict:
    """Compute all signals for one pair.

    v3 changes:
      - Breakout requires volume confirmation (baked into breakout_signal)
      - RSI oversold at 25 (baked into mean_reversion_signal)
      - Higher base strength for breakout (0.65 vs 0.6)
    """
    result = {
        "action": "HOLD",
        "strategy": "none",
        "strength": 0.0,

        "breakout": False,
        "breakdown": False,
        "breakout_strength": 0.0,
        "volume_confirm": False,

        "rsi": 50.0,
        "oversold": False,
        "overbought": False,
        "bounce_strength": 0.0,

        "real_vol": 0.0,
        "down_vol": 0.0,
        "spread": 0.0,
        "atr": 0.0,

        "ema_fast": 0.0,
        "ema_slow": 0.0,
        "trend_up": False,
    }

    if len(closes) < 80:
        return result

    bo = breakout_signal(closes, highs, lows, volumes, breakout_lookback)
    mr = mean_reversion_signal(closes)

    ema_f = ema(closes, 21)[-1]
    ema_s = ema(closes, 55)[-1]
    trend_up = closes[-1] > ema_f > ema_s

    real_vol = realized_volatility(closes)
    down_vol = downside_volatility(closes)
    current_atr = atr(highs, lows, closes) if len(highs) > 14 else 0.0
    sprd = spread_pct(bid, ask)

    result.update({
        "breakout": bo["is_breakout"],
        "breakdown": bo["is_breakdown"],
        "breakout_strength": round(bo["strength"], 6),
        "volume_confirm": bo["volume_confirm"],

        "rsi": mr["rsi_value"],
        "oversold": mr["is_oversold"],
        "overbought": mr["is_overbought"],
        "bounce_strength": round(mr["bounce_strength"], 4),

        "real_vol": round(real_vol, 4),
        "down_vol": round(down_vol, 4),
        "spread": round(sprd, 6),
        "atr": round(current_atr, 6),

        "ema_fast": round(ema_f, 6),
        "ema_slow": round(ema_s, 6),
        "trend_up": trend_up,
    })

    # ── Decision logic ──
    # Breakout: volume confirmation is already required inside breakout_signal
    if bo["is_breakout"] and trend_up:
        result["action"] = "BUY"
        result["strategy"] = "breakout"
        # Higher base (0.65) since volume is already confirmed
        strength = 0.65 + min(0.35, bo["strength"] * 8)
        result["strength"] = round(min(strength, 1.0), 3)

    # RSI mean reversion: RSI < 25 required (tighter than before)
    elif mr["is_oversold"] and not bo["is_breakdown"]:
        result["action"] = "BUY"
        result["strategy"] = "mean_rev"
        strength = 0.5 + (0.2 if mr["rsi_value"] < 20 else 0) + (0.1 if trend_up else 0)
        result["strength"] = round(strength, 3)

    # Exit signals
    elif bo["is_breakdown"]:
        result["action"] = "SELL"
        result["strategy"] = "breakout"
        result["strength"] = 0.8

    elif mr["is_overbought"] and mr["rsi_value"] > 70:
        result["action"] = "SELL"
        result["strategy"] = "mean_rev"
        result["strength"] = 0.5

    return result