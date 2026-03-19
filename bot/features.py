"""
Cross-sectional feature engine.

Computes all features for the scoring/ranking pipeline, both per-coin
and market-level. All per-coin features are z-scored cross-sectionally
(relative to the universe at the same timestamp) so they're comparable.

Architecture:
  Raw OHLCV + derivatives → per-coin features → z-score across universe
                          → market-level features (breadth, stress, etc.)

Feature groups:
  A. Momentum:     multi-horizon returns (1h, 6h, 24h, 3d)
  B. Path quality: persistence, choppiness
  C. Risk:         realized vol, downside vol, jump proxy → RiskPenalty
  D. Cost:         spread %, short-horizon vol → CostPenalty
  E. Breakout:     distance above rolling high, volume ratio
  F. Reversal:     overshoot score (standardized drop vs own history)
  G. Market-level: breadth, trend strength, downside stress, cost stress,
                   leadership stability

Weights for submodels (risk, cost) are set by theory from the brief:
  Risk:  a=0.25 (vol), b=0.45 (downside vol), c=0.30 (jump proxy)
  Cost:  δ=0.50 (spread), λ=0.30 (size proxy), γ=0.20 (short vol)
"""
import numpy as np
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# CORE COMPUTATIONS (pure functions, no state)
# ═══════════════════════════════════════════════════════════════

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def zscore_array(arr: np.ndarray) -> np.ndarray:
    """Z-score an array. Returns 0s if std is 0."""
    std = np.std(arr)
    if std == 0 or np.isnan(std):
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / std


# ═══════════════════════════════════════════════════════════════
# PER-COIN FEATURES
# ═══════════════════════════════════════════════════════════════

def compute_returns(closes: np.ndarray, lookbacks: dict[str, int]) -> dict[str, float]:
    """Multi-horizon returns.

    Args:
        closes: price array, newest last
        lookbacks: {"1h": 12, "6h": 72, "24h": 288, "3d": 864} (bar counts)

    Returns dict like {"r_1h": 0.02, "r_6h": -0.01, ...}
    """
    result = {}
    for label, lb in lookbacks.items():
        if len(closes) > lb + 1 and closes[-(lb + 1)] > 0:
            result[f"r_{label}"] = (closes[-1] / closes[-(lb + 1)]) - 1
        else:
            result[f"r_{label}"] = 0.0
    return result


def compute_persistence(closes: np.ndarray, lookback: int = 288) -> float:
    """Fraction of sub-period returns aligned with net direction.

    High persistence = clean directional move.
    Low persistence = choppy, indecisive.
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
    if total_abs == 0:
        return 0.5
    return float(aligned / total_abs)


def compute_choppiness(closes: np.ndarray, lookback: int = 288) -> float:
    """Path noisiness. 0 = perfectly trending, 1 = perfectly choppy."""
    if len(closes) < lookback + 1:
        lookback = len(closes) - 1
    if lookback < 2:
        return 0.5

    p = closes[-lookback - 1:]
    net_move = abs(p[-1] - p[0])
    total_path = np.sum(np.abs(np.diff(p)))
    if total_path == 0:
        return 1.0
    return 1.0 - (net_move / total_path)


def compute_realized_vol(closes: np.ndarray, lookback: int = 288) -> float:
    """Annualized realized volatility from log returns."""
    if len(closes) < lookback + 1:
        lookback = len(closes) - 1
    if lookback < 2:
        return 0.0
    log_rets = np.diff(np.log(closes[-lookback - 1:]))
    return float(np.std(log_rets)) * np.sqrt(lookback * (365 * 24 / (lookback / 12)))


def compute_downside_vol(closes: np.ndarray, lookback: int = 288) -> float:
    """Annualized downside volatility (negative returns only)."""
    if len(closes) < lookback + 1:
        lookback = len(closes) - 1
    if lookback < 2:
        return 0.0
    log_rets = np.diff(np.log(closes[-lookback - 1:]))
    neg = log_rets[log_rets < 0]
    if len(neg) == 0:
        return 0.0
    return float(np.std(neg)) * np.sqrt(lookback * (365 * 24 / (lookback / 12)))


def compute_jump_proxy(closes: np.ndarray, lookback: int = 288) -> float:
    """Jump proxy: max absolute return / realized vol.

    High = large discrete moves relative to normal vol.
    Captures tail risk beyond what vol alone measures.
    """
    if len(closes) < lookback + 1:
        lookback = len(closes) - 1
    if lookback < 10:
        return 0.0
    log_rets = np.diff(np.log(closes[-lookback - 1:]))
    vol = np.std(log_rets)
    if vol == 0:
        return 0.0
    return float(np.max(np.abs(log_rets)) / vol)


def compute_risk_penalty(
    realized_vol: float, downside_vol: float, jump_proxy: float,
    a: float = 0.25, b: float = 0.45, c: float = 0.30,
) -> float:
    """Risk penalty submodel (from brief).

    Weights: a=vol importance, b=downside vol importance, c=jump importance.
    Inputs should be z-scored before calling this.
    """
    return a * realized_vol + b * downside_vol + c * jump_proxy


def compute_cost_penalty(
    spread_pct: float, short_vol: float,
    delta: float = 0.60, gamma: float = 0.40,
) -> float:
    """Cost penalty submodel (simplified from brief).

    We drop SizeRatio (λ) because Roostoo doesn't provide order book depth.
    Reallocate: δ=0.60 (spread), γ=0.40 (short-horizon vol).
    Inputs should be z-scored.
    """
    return delta * spread_pct + gamma * short_vol


def compute_breakout_distance(closes: np.ndarray, highs: np.ndarray, lookback: int = 72) -> float:
    """Distance above prior rolling high. Positive = breakout territory."""
    if len(closes) < lookback + 1:
        return 0.0
    if len(highs) >= lookback + 1:
        prior_high = np.max(highs[-(lookback + 1):-1])
    else:
        prior_high = np.max(closes[-(lookback + 1):-1])
    if prior_high <= 0:
        return 0.0
    return (closes[-1] - prior_high) / prior_high


def compute_volume_ratio(volumes: np.ndarray, lookback: int = 72) -> float:
    """Current volume / average volume over lookback."""
    if len(volumes) < lookback + 1:
        return 1.0
    avg = np.mean(volumes[-lookback - 1:-1])
    if avg <= 0:
        return 1.0
    return float(volumes[-1] / avg)


def compute_overshoot(closes: np.ndarray, lookback: int = 72) -> float:
    """Standardized recent drop relative to coin's own history.

    Overshoot = (recent_drop - mean_drop) / std_drop
    High overshoot = unusually large drop = mean reversion candidate.
    """
    if len(closes) < lookback + 10:
        return 0.0
    # Current short-term return (6h)
    short_lb = min(72, lookback)  # 6h in 5-min bars
    if len(closes) > short_lb:
        current_ret = (closes[-1] / closes[-short_lb] - 1)
    else:
        return 0.0

    # Historical distribution of similar-horizon returns
    rets = []
    step = max(short_lb // 2, 1)
    for i in range(short_lb, min(len(closes) - short_lb, lookback * 5), step):
        idx = len(closes) - i
        if idx > short_lb:
            r = (closes[idx] / closes[idx - short_lb] - 1)
            rets.append(r)
    if len(rets) < 10:
        return 0.0

    mean_ret = np.mean(rets)
    std_ret = np.std(rets)
    if std_ret == 0:
        return 0.0

    # Negative z-score means unusually bad drop
    return float((current_ret - mean_ret) / std_ret)


def compute_spread_pct(bid: float, ask: float) -> float:
    """Bid-ask spread as fraction of mid price."""
    if bid <= 0 or ask <= 0:
        return 0.0
    mid = (bid + ask) / 2
    return (ask - bid) / mid


# ═══════════════════════════════════════════════════════════════
# FULL PER-COIN FEATURE VECTOR
# ═══════════════════════════════════════════════════════════════

def compute_coin_features(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    bid: float,
    ask: float,
    lookbacks: dict[str, int] = None,
    breakout_lookback: int = 72,
) -> Optional[dict]:
    """Compute ALL raw features for one coin.

    Returns dict of raw (non-z-scored) feature values, or None if insufficient data.
    """
    if lookbacks is None:
        lookbacks = {"1h": 12, "6h": 72, "24h": 288, "3d": 864}

    min_needed = max(lookbacks.values()) if lookbacks else 288
    if len(closes) < min(min_needed + 1, 100):
        return None
    if closes[-1] <= 0:
        return None

    rets = compute_returns(closes, lookbacks)
    persist = compute_persistence(closes, min(288, len(closes) - 1))
    choppy = compute_choppiness(closes, min(288, len(closes) - 1))
    real_vol = compute_realized_vol(closes)
    down_vol = compute_downside_vol(closes)
    jump = compute_jump_proxy(closes)
    bo_dist = compute_breakout_distance(closes, highs, breakout_lookback)
    vol_ratio = compute_volume_ratio(volumes, breakout_lookback)
    overshoot = compute_overshoot(closes)
    spread = compute_spread_pct(bid, ask)

    # Short-horizon vol (for cost penalty): 1h of 5-min bars
    if len(closes) > 13:
        short_lr = np.diff(np.log(closes[-13:]))
        short_vol = float(np.std(short_lr)) * np.sqrt(12 * 24 * 365) if len(short_lr) > 0 else 0
    else:
        short_vol = 0

    features = {
        **rets,                        # r_1h, r_6h, r_24h, r_3d
        "persistence": persist,
        "choppiness": choppy,
        "realized_vol": real_vol,
        "downside_vol": down_vol,
        "jump_proxy": jump,
        "breakout_distance": bo_dist,
        "volume_ratio": vol_ratio,
        "overshoot": overshoot,
        "spread_pct": spread,
        "short_vol": short_vol,
        "price": closes[-1],
    }
    return features


# ═══════════════════════════════════════════════════════════════
# CROSS-SECTIONAL Z-SCORING
# ═══════════════════════════════════════════════════════════════

def zscore_universe(
    all_features: dict[str, dict],
    feature_keys: list[str] = None,
) -> dict[str, dict]:
    """Z-score features across the coin universe.

    Args:
        all_features: {pair: {feature_name: raw_value, ...}, ...}
        feature_keys: which features to z-score (defaults to all numeric)

    Returns same structure but with z-scored values.
    """
    if not all_features:
        return {}

    pairs = list(all_features.keys())
    if feature_keys is None:
        # Auto-detect numeric features from first entry
        first = next(iter(all_features.values()))
        feature_keys = [k for k, v in first.items() if isinstance(v, (int, float)) and k != "price"]

    zscored = {pair: {} for pair in pairs}

    for key in feature_keys:
        values = np.array([all_features[p].get(key, 0.0) for p in pairs])
        z_values = zscore_array(values)
        for i, pair in enumerate(pairs):
            zscored[pair][key] = float(z_values[i])

    # Copy non-zscored fields (price, etc.)
    for pair in pairs:
        for k, v in all_features[pair].items():
            if k not in feature_keys:
                zscored[pair][k] = v

    return zscored


# ═══════════════════════════════════════════════════════════════
# SUBMODEL SCORES (computed on z-scored features)
# ═══════════════════════════════════════════════════════════════

def compute_submodel_scores(zscored_features: dict) -> dict:
    """Compute risk and cost penalty submodels from z-scored features.

    Args:
        zscored_features: z-scored feature dict for ONE coin

    Returns dict with risk_penalty, cost_penalty added.
    """
    f = zscored_features

    risk_penalty = compute_risk_penalty(
        f.get("realized_vol", 0),
        f.get("downside_vol", 0),
        f.get("jump_proxy", 0),
    )
    cost_penalty = compute_cost_penalty(
        f.get("spread_pct", 0),
        f.get("short_vol", 0),
    )

    f["risk_penalty"] = round(risk_penalty, 4)
    f["cost_penalty"] = round(cost_penalty, 4)
    return f


# ═══════════════════════════════════════════════════════════════
# MARKET-LEVEL FEATURES (regime filter inputs)
# ═══════════════════════════════════════════════════════════════

def compute_market_features(all_raw_features: dict[str, dict]) -> dict:
    """Compute market-wide regime features from the raw (non-z-scored) universe.

    These are used by the regime filter to decide: should we trade at all?
    """
    if not all_raw_features:
        return {"breadth": 0.5, "trend_strength": 0.0, "mkt_downside_vol": 0.0,
                "cost_stress": 0.0, "leadership_stability": 0.5}

    pairs = list(all_raw_features.keys())

    # Breadth: fraction of coins with positive 24h return
    r24h_values = [all_raw_features[p].get("r_24h", 0) for p in pairs]
    breadth = sum(1 for r in r24h_values if r > 0) / len(pairs) if pairs else 0.5

    # Trend strength: median 24h return
    trend_strength = float(np.median(r24h_values)) if r24h_values else 0.0

    # Market downside stress: median downside vol
    dv_values = [all_raw_features[p].get("downside_vol", 0) for p in pairs]
    mkt_downside_vol = float(np.median(dv_values)) if dv_values else 0.0

    # Cost stress: median spread
    spread_values = [all_raw_features[p].get("spread_pct", 0) for p in pairs]
    cost_stress = float(np.median(spread_values)) if spread_values else 0.0

    return {
        "breadth": round(breadth, 4),
        "trend_strength": round(trend_strength, 6),
        "mkt_downside_vol": round(mkt_downside_vol, 4),
        "cost_stress": round(cost_stress, 6),
    }
