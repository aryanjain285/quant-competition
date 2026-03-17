"""
Signal engine v2: Dual-strategy system backed by backtest results.

Two independent signal sources:
1. BREAKOUT — buy coins making new N-hour highs. Catches large trending moves.
   Backtest: +27-43% on trending coins, maxDD ~4%, 57-67% win rate.
2. RSI MEAN REVERSION — buy oversold dips for quick rebounds.
   Backtest: +9.4% on XRP with 78% WR, consistent across stable coins.

Each signal fires independently. The portfolio manager decides allocation.
Pure functions — same code runs in backtest and live.
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
    """Average True Range (last value). Uses close-to-close if no high/low."""
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
# Signal 1: BREAKOUT
# ─────────────────────────────────────────────────────────────────

def breakout_signal(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    lookback: int = 72,      # 72 periods = 72h if hourly, 6h if 5-min
) -> dict:
    """Detect breakout: price exceeding recent N-period high.

    Returns dict with:
        - is_breakout: bool
        - is_breakdown: bool (below recent low — exit signal)
        - strength: how far above the prior high (as fraction)
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

    # Use highs for breakout detection if available, else closes
    if len(highs) >= lookback + 1:
        prior_high = np.max(highs[-(lookback + 1):-1])
    else:
        prior_high = np.max(closes[-(lookback + 1):-1])

    if len(lows) >= lookback + 1:
        # Use shorter lookback for breakdown (exit faster)
        breakdown_lookback = max(lookback // 3, 12)
        prior_low = np.min(lows[-breakdown_lookback - 1:-1])
    else:
        breakdown_lookback = max(lookback // 3, 12)
        prior_low = np.min(closes[-breakdown_lookback - 1:-1])

    current = closes[-1]

    # Breakout: price exceeds prior high
    if current > prior_high and prior_high > 0:
        result["is_breakout"] = True
        result["strength"] = (current - prior_high) / prior_high

    # Breakdown: price falls below recent low (exit signal)
    if current < prior_low and prior_low > 0:
        result["is_breakdown"] = True

    # Volume confirmation: current volume > 1.3x recent average
    if len(volumes) >= lookback:
        avg_vol = np.mean(volumes[-lookback:-1])
        if avg_vol > 0 and volumes[-1] > 1.3 * avg_vol:
            result["volume_confirm"] = True

    return result


# ─────────────────────────────────────────────────────────────────
# Signal 2: RSI MEAN REVERSION
# ─────────────────────────────────────────────────────────────────

def mean_reversion_signal(
    closes: np.ndarray,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 65.0,
    rsi_period: int = 14,
) -> dict:
    """RSI-based mean reversion signal.

    Buy when oversold, sell/exit when overbought or recovered.
    Backtest showed 78% win rate on XRP, 53-61% on other coins.

    Returns dict with:
        - is_oversold: bool (entry)
        - is_overbought: bool (exit)
        - rsi_value: current RSI
        - bounce_strength: how much RSI has recovered from trough
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

    Returns a unified signal dict that the portfolio manager uses.
    Two independent signal sources can both be active simultaneously.
    """
    result = {
        # Overall
        "action": "HOLD",
        "strategy": "none",       # which strategy triggered: "breakout", "mean_rev", "none"
        "strength": 0.0,          # signal confidence for position sizing

        # Breakout signals
        "breakout": False,
        "breakdown": False,
        "breakout_strength": 0.0,
        "volume_confirm": False,

        # Mean reversion signals
        "rsi": 50.0,
        "oversold": False,
        "overbought": False,
        "bounce_strength": 0.0,

        # Risk metrics
        "real_vol": 0.0,
        "down_vol": 0.0,
        "spread": 0.0,
        "atr": 0.0,

        # Trend context (for filtering, not primary signal)
        "ema_fast": 0.0,
        "ema_slow": 0.0,
        "trend_up": False,
    }

    if len(closes) < 80:
        return result

    # ── Compute all sub-signals ──
    bo = breakout_signal(closes, highs, lows, volumes, breakout_lookback)
    mr = mean_reversion_signal(closes)

    # Trend context (EMA)
    ema_f = ema(closes, 21)[-1]
    ema_s = ema(closes, 55)[-1]
    trend_up = closes[-1] > ema_f > ema_s

    # Volatility
    real_vol = realized_volatility(closes)
    down_vol = downside_volatility(closes)
    current_atr = atr(highs, lows, closes) if len(highs) > 14 else 0.0
    sprd = spread_pct(bid, ask)

    # ── Populate result ──
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
    # Priority 1: Breakout entry (strongest signal, captures big moves)
    if bo["is_breakout"] and trend_up:
        result["action"] = "BUY"
        result["strategy"] = "breakout"
        strength = 0.6 + (0.2 if bo["volume_confirm"] else 0) + min(0.2, bo["strength"] * 5)
        result["strength"] = round(min(strength, 1.0), 3)

    # Priority 2: RSI mean reversion entry (consistent, high win rate)
    elif mr["is_oversold"] and not bo["is_breakdown"]:
        result["action"] = "BUY"
        result["strategy"] = "mean_rev"
        # Stronger if deeply oversold
        strength = 0.4 + (0.2 if mr["rsi_value"] < 25 else 0) + (0.1 if trend_up else 0)
        result["strength"] = round(strength, 3)

    # Exit signals
    elif bo["is_breakdown"]:
        result["action"] = "SELL"
        result["strategy"] = "breakout"
        result["strength"] = 0.8  # exit breakdowns urgently

    elif mr["is_overbought"] and mr["rsi_value"] > 70:
        # Only exit mean-rev at RSI > 70 (was 65, too early)
        result["action"] = "SELL"
        result["strategy"] = "mean_rev"
        result["strength"] = 0.5

    return result
