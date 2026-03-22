"""
Cross-sectional feature engine.

ALL LOOKBACKS ARE IN 1-HOUR BARS. One bar = one hour.
Annualization: sqrt(24 * 365) = sqrt(8760) ≈ 93.6 for hourly data.
"""
import numpy as np
from typing import Optional
from bot.config import ANNUALIZATION_FACTOR


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
    result = {}
    for label, lb in lookbacks.items():
        if len(closes) > lb + 1 and closes[-(lb + 1)] > 0:
            result[f"r_{label}"] = (closes[-1] / closes[-(lb + 1)]) - 1
        else:
            result[f"r_{label}"] = 0.0
    return result


def compute_persistence(closes: np.ndarray, lookback: int = 24) -> float:
    """Fraction of sub-period returns aligned with net direction. 24 bars = 24h."""
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
    """Path noisiness. 0 = trending, 1 = choppy. 24 bars = 24h."""
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
    """Annualized downside vol. 24 bars = 24h."""
    if len(closes) < lookback + 1:
        lookback = len(closes) - 1
    if lookback < 2:
        return 0.0
    log_rets = np.diff(np.log(closes[-lookback - 1:]))
    neg = log_rets[log_rets < 0]
    return float(np.std(neg)) * ANNUALIZATION_FACTOR if len(neg) > 0 else 0.0


def compute_jump_proxy(closes: np.ndarray, lookback: int = 24) -> float:
    """Max |return| / realized vol. Measures tail risk."""
    if len(closes) < lookback + 1:
        lookback = len(closes) - 1
    if lookback < 10:
        return 0.0
    log_rets = np.diff(np.log(closes[-lookback - 1:]))
    vol = np.std(log_rets)
    return float(np.max(np.abs(log_rets)) / vol) if vol > 0 else 0.0


def compute_risk_penalty(rv: float, dv: float, jp: float,
                         a=0.25, b=0.45, c=0.30) -> float:
    return a * rv + b * dv + c * jp


def compute_cost_penalty(spread: float, short_vol: float,
                         delta=0.60, gamma=0.40) -> float:
    return delta * spread + gamma * short_vol


def compute_breakout_distance(closes: np.ndarray, highs: np.ndarray, lookback: int = 72) -> float:
    """Distance above prior rolling high. 72 bars = 72h = 3 days."""
    if len(closes) < lookback + 1:
        return 0.0
    prior_high = np.max(highs[-(lookback + 1):-1]) if len(highs) >= lookback + 1 else np.max(closes[-(lookback + 1):-1])
    return (closes[-1] - prior_high) / prior_high if prior_high > 0 else 0.0


def compute_volume_ratio(volumes: np.ndarray, lookback: int = 72) -> float:
    """Recent volume (last 3 bars = 3h) vs lookback average. 72 bars = 72h."""
    if len(volumes) < lookback + 3:
        return 1.0
    recent = np.mean(volumes[-3:])
    avg = np.mean(volumes[-(lookback + 3):-3])
    return float(recent / avg) if avg > 0 else 1.0


def compute_overshoot(closes: np.ndarray, lookback: int = 168) -> float:
    """Z-score of 6h return vs distribution of 6h returns over lookback.

    horizon = 6 bars (6h on 1h data).
    lookback = 168 bars (7 days) for reference distribution.
    """
    horizon = 6  # 6 hours
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
    if bid <= 0 or ask <= 0:
        return 0.0
    return (ask - bid) / ((bid + ask) / 2)


# ═══════════════════════════════════════════════════════════════
# FULL PER-COIN FEATURE VECTOR
# ═══════════════════════════════════════════════════════════════

def compute_coin_features(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volumes: np.ndarray, bid: float, ask: float,
    lookbacks: dict[str, int] = None, breakout_lookback: int = 72,
) -> Optional[dict]:
    """Compute ALL raw features for one coin. All lookbacks in 1h bars."""
    if lookbacks is None:
        lookbacks = {"1h": 1, "6h": 6, "24h": 24, "3d": 72}

    if len(closes) < 80:
        return None
    if closes[-1] <= 0:
        return None

    rets = compute_returns(closes, lookbacks)
    persist = compute_persistence(closes, 24)
    choppy = compute_choppiness(closes, 24)
    rv = compute_realized_vol(closes, 24)
    dv = compute_downside_vol(closes, 24)
    jump = compute_jump_proxy(closes, 24)
    bo_dist = compute_breakout_distance(closes, highs, breakout_lookback)
    vol_ratio = compute_volume_ratio(volumes, breakout_lookback)
    overshoot = compute_overshoot(closes, 168)
    spread = compute_spread_pct(bid, ask)

    # Short-term vol (last 6h for cost penalty)
    if len(closes) > 7:
        lr = np.diff(np.log(closes[-7:]))
        short_vol = float(np.std(lr)) * ANNUALIZATION_FACTOR if len(lr) > 0 else 0
    else:
        short_vol = 0

    return {
        **rets,
        "persistence": persist, "choppiness": choppy,
        "realized_vol": rv, "downside_vol": dv, "jump_proxy": jump,
        "breakout_distance": bo_dist, "volume_ratio": vol_ratio,
        "overshoot": overshoot, "spread_pct": spread, "short_vol": short_vol,
        "price": closes[-1],
    }


# ═══════════════════════════════════════════════════════════════
# CROSS-SECTIONAL Z-SCORING + SUBMODELS + MARKET FEATURES
# ═══════════════════════════════════════════════════════════════

def zscore_universe(all_features: dict[str, dict], feature_keys: list[str] = None) -> dict[str, dict]:
    if not all_features:
        return {}
    pairs = list(all_features.keys())
    if feature_keys is None:
        first = next(iter(all_features.values()))
        feature_keys = [k for k, v in first.items() if isinstance(v, (int, float)) and k != "price"]

    zscored = {p: {} for p in pairs}
    for key in feature_keys:
        vals = np.array([all_features[p].get(key, 0.0) for p in pairs])
        z = zscore_array(vals)
        for i, p in enumerate(pairs):
            zscored[p][key] = float(z[i])
    for p in pairs:
        for k, v in all_features[p].items():
            if k not in feature_keys:
                zscored[p][k] = v
    return zscored


def compute_submodel_scores(zf: dict) -> dict:
    zf["risk_penalty"] = round(compute_risk_penalty(
        zf.get("realized_vol", 0), zf.get("downside_vol", 0), zf.get("jump_proxy", 0)), 4)
    zf["cost_penalty"] = round(compute_cost_penalty(
        zf.get("spread_pct", 0), zf.get("short_vol", 0)), 4)
    return zf


def compute_market_features(all_raw: dict[str, dict]) -> dict:
    if not all_raw:
        return {"breadth": 0.5, "trend_strength": 0.0, "mkt_downside_vol": 0.0, "cost_stress": 0.0}
    pairs = list(all_raw.keys())
    r24h = [all_raw[p].get("r_24h", 0) for p in pairs]
    dvs = [all_raw[p].get("downside_vol", 0) for p in pairs]
    spreads = [all_raw[p].get("spread_pct", 0) for p in pairs]
    return {
        "breadth": round(sum(1 for r in r24h if r > 0) / len(pairs), 4),
        "trend_strength": round(float(np.median(r24h)), 6),
        "mkt_downside_vol": round(float(np.median(dvs)), 4),
        "cost_stress": round(float(np.median(spreads)), 6),
    }
