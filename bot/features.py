"""
Cross-sectional feature engine — Finals v6.

ALL LOOKBACKS ARE IN 1-HOUR BARS. One bar = one hour.
Annualization: sqrt(24 * 365) = sqrt(8760) ≈ 93.6 for hourly data.

Features are designed for cross-sectional ranking via Lasso:
- Multi-horizon momentum (r_6h, r_24h, r_3d)
- Trend quality (persistence, choppiness)
- Risk (realized_vol, downside_vol, jump_proxy)
- Breakout (breakout_distance, volume_ratio)
- Mean-reversion (overshoot)
- Cost (spread_pct)

r_1h is intentionally excluded — at 1-hour resolution it's essentially noise
and adds no cross-sectional predictive power.
"""
import numpy as np
from typing import Optional
from bot.config import ANNUALIZATION_FACTOR, MOMENTUM_LOOKBACKS


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def zscore_array(arr: np.ndarray) -> np.ndarray:
    std = np.std(arr)
    if std == 0 or np.isnan(std):
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / std


# ═══════════════════════════════════════════════════════════════
# PER-COIN FEATURES (all lookbacks in 1h bars)
# ═══════════════════════════════════════════════════════════════

def compute_returns(closes: np.ndarray, lookbacks: dict[str, int]) -> dict[str, float]:
    """Multi-horizon returns. Each lookback is in 1h bars."""
    result = {}
    for label, lb in lookbacks.items():
        if len(closes) > lb + 1 and closes[-(lb + 1)] > 0:
            result[f"r_{label}"] = (closes[-1] / closes[-(lb + 1)]) - 1
        else:
            result[f"r_{label}"] = 0.0
    return result


def compute_persistence(closes: np.ndarray, lookback: int = 24) -> float:
    """Fraction of sub-period returns aligned with net direction. 24 bars = 24h.

    High persistence (> 0.6) → consistent trend, low false signals.
    Low persistence (< 0.4) → choppy, mean-reverting microstructure.
    """
    if len(closes) < lookback + 1:
        lookback = len(closes) - 1
    if lookback < 2:
        return 0.5
    rets = np.diff(closes[-lookback - 1:]) / closes[-lookback - 1:-1]
    total_ret = np.sum(rets)
    if total_ret == 0:
        return 0.5
    aligned = np.sum(np.abs(rets[np.sign(rets) == np.sign(total_ret)]))
    total_abs = np.sum(np.abs(rets))
    return float(aligned / total_abs) if total_abs > 0 else 0.5


def compute_choppiness(closes: np.ndarray, lookback: int = 24) -> float:
    """Path noisiness. 0 = pure trend, 1 = pure chop. 24 bars = 24h.

    Measures net displacement / total path length.
    """
    if len(closes) < lookback + 1:
        lookback = len(closes) - 1
    if lookback < 2:
        return 0.5
    p = closes[-lookback - 1:]
    net_move = abs(p[-1] - p[0])
    total_path = np.sum(np.abs(np.diff(p)))
    return 1.0 - (net_move / total_path) if total_path > 0 else 1.0


def compute_realized_vol(closes: np.ndarray, lookback: int = 24) -> float:
    """Annualized realized vol. 24 bars = 24h. Annualize with sqrt(8760)."""
    if len(closes) < lookback + 1:
        lookback = len(closes) - 1
    if lookback < 2:
        return 0.0
    log_rets = np.diff(np.log(closes[-lookback - 1:]))
    return float(np.std(log_rets)) * ANNUALIZATION_FACTOR


def compute_downside_vol(closes: np.ndarray, lookback: int = 24) -> float:
    """Annualized downside vol. 24 bars = 24h.

    Only negative returns contribute — directly relevant to Sortino denominator.
    """
    if len(closes) < lookback + 1:
        lookback = len(closes) - 1
    if lookback < 2:
        return 0.0
    log_rets = np.diff(np.log(closes[-lookback - 1:]))
    neg = log_rets[log_rets < 0]
    return float(np.std(neg)) * ANNUALIZATION_FACTOR if len(neg) > 0 else 0.0


def compute_jump_proxy(closes: np.ndarray, lookback: int = 24) -> float:
    """Max |return| / realized vol. Measures tail risk.

    High jump_proxy → fat tails, gap risk. Lasso can learn to penalize this.
    """
    if len(closes) < lookback + 1:
        lookback = len(closes) - 1
    if lookback < 10:
        return 0.0
    log_rets = np.diff(np.log(closes[-lookback - 1:]))
    vol = np.std(log_rets)
    return float(np.max(np.abs(log_rets)) / vol) if vol > 0 else 0.0


def compute_breakout_distance(closes: np.ndarray, highs: np.ndarray, lookback: int = 72) -> float:
    """Distance above prior rolling high. 72 bars = 72h = 3 days.

    Positive = trading above recent high (breakout).
    Negative = below recent high (consolidation/decline).
    """
    if len(closes) < lookback + 1:
        return 0.0
    if len(highs) >= lookback + 1:
        prior_high = np.max(highs[-(lookback + 1):-1])
    else:
        prior_high = np.max(closes[-(lookback + 1):-1])
    return (closes[-1] - prior_high) / prior_high if prior_high > 0 else 0.0


def compute_volume_ratio(volumes: np.ndarray, lookback: int = 72) -> float:
    """Recent volume (last 3 bars = 3h) vs lookback average. 72 bars = 72h.

    >1.0 = elevated volume (confirms move).
    Proven in v3 backtest as single biggest signal improvement.
    """
    if len(volumes) < lookback + 3:
        return 1.0
    recent = np.mean(volumes[-3:])
    avg = np.mean(volumes[-(lookback + 3):-3])
    return float(recent / avg) if avg > 0 else 1.0


def compute_overshoot(closes: np.ndarray, lookback: int = 168) -> float:
    """Z-score of 6h return vs distribution of 6h returns over lookback.

    horizon = 6 bars (6h on 1h data).
    lookback = 168 bars (7 days) for reference distribution.

    Strongly negative → coin dropped much more than its own recent history.
    This is the mean-reversion signal — Lasso learns the coefficient.
    """
    horizon = 6
    if len(closes) < lookback + horizon:
        lookback = len(closes) - horizon - 1
    if lookback < horizon * 2:
        return 0.0

    current_ret = (closes[-1] / closes[-horizon] - 1)

    rets = []
    for offset in range(horizon, lookback):
        end_idx = len(closes) - offset
        start_idx = end_idx - horizon
        if start_idx >= 0 and closes[start_idx] > 0:
            rets.append((closes[end_idx] / closes[start_idx]) - 1)

    if len(rets) < 10:
        return 0.0
    mean_r, std_r = np.mean(rets), np.std(rets)
    return float((current_ret - mean_r) / std_r) if std_r > 0 else 0.0


def compute_spread_pct(bid: float, ask: float) -> float:
    """Bid-ask spread as fraction of mid price. Measures liquidity/cost."""
    if bid <= 0 or ask <= 0:
        return 0.0
    return (ask - bid) / ((bid + ask) / 2)


def check_breakdown(closes: np.ndarray, lows: np.ndarray, lookback: int = 72) -> bool:
    """Check if price has broken below recent low (exit signal).

    Uses lookback/3 (min 6h) as the breakdown window.
    This is the only hard-coded exit signal — everything else is trailing stops.
    """
    if len(closes) < lookback + 1:
        return False
    breakdown_lb = max(lookback // 3, 6)
    if len(lows) >= breakdown_lb + 1:
        prior_low = np.min(lows[-breakdown_lb - 1:-1])
    else:
        prior_low = np.min(closes[-breakdown_lb - 1:-1])
    return closes[-1] < prior_low and prior_low > 0


# ═══════════════════════════════════════════════════════════════
# FULL PER-COIN FEATURE VECTOR
# ═══════════════════════════════════════════════════════════════

def compute_coin_features(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volumes: np.ndarray, bid: float, ask: float,
    breakout_lookback: int = 72,
) -> Optional[dict]:
    """Compute ALL raw features for one coin. All lookbacks in 1h bars.

    Returns dict with keys matching LASSO_FEATURES in config.py,
    plus 'price' for reference.
    """
    if len(closes) < 80:
        return None
    if closes[-1] <= 0:
        return None

    rets = compute_returns(closes, MOMENTUM_LOOKBACKS)
    persist = compute_persistence(closes, 24)
    choppy = compute_choppiness(closes, 24)
    rv = compute_realized_vol(closes, 24)
    dv = compute_downside_vol(closes, 24)
    jump = compute_jump_proxy(closes, 24)
    bo_dist = compute_breakout_distance(closes, highs, breakout_lookback)
    vol_ratio = compute_volume_ratio(volumes, breakout_lookback)
    overshoot = compute_overshoot(closes, 168)
    spread = compute_spread_pct(bid, ask)

    return {
        **rets,
        "persistence": persist,
        "choppiness": choppy,
        "realized_vol": rv,
        "downside_vol": dv,
        "jump_proxy": jump,
        "breakout_distance": bo_dist,
        "volume_ratio": vol_ratio,
        "overshoot": overshoot,
        "spread_pct": spread,
        "price": closes[-1],
    }


# ═══════════════════════════════════════════════════════════════
# CROSS-SECTIONAL Z-SCORING
# ═══════════════════════════════════════════════════════════════

def zscore_universe(
    all_features: dict[str, dict],
    feature_keys: list[str] = None,
) -> dict[str, dict]:
    """Z-score features cross-sectionally across all coins.

    Each feature is standardized to mean=0, std=1 across the universe.
    This makes Lasso coefficients comparable and prevents high-magnitude
    features from dominating.
    """
    if not all_features:
        return {}
    pairs = list(all_features.keys())
    if feature_keys is None:
        first = next(iter(all_features.values()))
        feature_keys = [
            k for k, v in first.items()
            if isinstance(v, (int, float)) and k != "price"
        ]

    zscored = {p: {} for p in pairs}
    for key in feature_keys:
        vals = np.array([all_features[p].get(key, 0.0) for p in pairs])
        z = zscore_array(vals)
        for i, p in enumerate(pairs):
            zscored[p][key] = float(z[i])

    # Pass through non-numeric / excluded fields
    for p in pairs:
        for k, v in all_features[p].items():
            if k not in feature_keys:
                zscored[p][k] = v
    return zscored