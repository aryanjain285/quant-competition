"""
Cross-sectional ranking engine.

Takes valid trade candidates (coins that passed regime + event filters)
and ranks them by a composite score. The top N are selected for execution.

Two modes:
  A. Hand-built score (default): weighted sum of z-scored features
     Weights chosen by theory — momentum positive, risk/cost negative.
     Transparent, stable, easy to explain to judges.

  B. Ridge regression (drop-in upgrade): same features, learned weights.
     When teammate delivers trained ridge model, swap one function call.

Research basis:
  - CTREND factor (Fieberg et al., JFQA 2025): ML-aggregated cross-sectional
    signal from 28 indicators produces 3.87%/week returns OOS.
  - Cross-sectional momentum in crypto (Drogen et al., SSRN 2023): coins
    ranked by 30d returns continue to outperform over 7 days.
  - Volatility management is essential (FMPM 2025): unmanaged momentum
    crashes during volatile periods.
"""
import numpy as np
from typing import Optional
from bot.logger import get_logger

log = get_logger("ranking")

try:
    from sklearn.linear_model import Ridge
    RIDGE_AVAILABLE = True
except ImportError:
    RIDGE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# HAND-BUILT SCORE (Choice A — default)
# ═══════════════════════════════════════════════════════════════

# Default weights: chosen by theory, not fitted.
# Positive weights = we WANT high values (momentum, persistence, breakout)
# Negative weights = we AVOID high values (risk, cost, choppiness)
DEFAULT_WEIGHTS = {
    "r_1h":              0.05,   # short-term momentum (small — noisy)
    "r_6h":              0.15,   # medium momentum
    "r_24h":             0.25,   # primary momentum horizon (strongest signal per research)
    "r_3d":              0.15,   # longer momentum
    "persistence":       0.10,   # path quality — clean moves
    "choppiness":       -0.10,   # path quality — penalize noise
    "breakout_distance": 0.10,   # above prior high = strong
    "volume_ratio":      0.05,   # volume confirmation
    "risk_penalty":     -0.15,   # penalize risky coins
    "cost_penalty":     -0.10,   # penalize expensive-to-trade coins
    "overshoot":        -0.05,   # negative overshoot = large drop = reversal candidate
}


class Ranker:
    """Ranks valid trade candidates by composite score."""

    def __init__(self, weights: dict[str, float] = None, ridge_model=None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.ridge_model = ridge_model
        self.use_ridge = ridge_model is not None

        # Feature order for ridge (must match training)
        self.ridge_features = list(DEFAULT_WEIGHTS.keys())

    def score_coin(self, zscored_features: dict) -> float:
        """Score a single coin using hand-built weights.

        All features should be z-scored cross-sectionally before calling this.
        """
        score = 0.0
        for feature, weight in self.weights.items():
            score += weight * zscored_features.get(feature, 0.0)
        return score

    def score_coin_ridge(self, zscored_features: dict) -> float:
        """Score using fitted ridge model."""
        if not self.use_ridge or self.ridge_model is None:
            return self.score_coin(zscored_features)

        feature_vec = np.array([
            zscored_features.get(f, 0.0) for f in self.ridge_features
        ]).reshape(1, -1)

        try:
            return float(self.ridge_model.predict(feature_vec)[0])
        except Exception:
            return self.score_coin(zscored_features)

    def rank(
        self,
        candidates: dict[str, dict],
        max_results: int = 12,
    ) -> list[tuple[str, float, dict]]:
        """Rank valid candidates by score, return top N.

        Args:
            candidates: {pair: zscored_features_dict, ...}
            max_results: max number of coins to return

        Returns list of (pair, score, features) sorted by score descending.
        """
        scored = []
        for pair, features in candidates.items():
            if self.use_ridge:
                s = self.score_coin_ridge(features)
            else:
                s = self.score_coin(features)
            scored.append((pair, s, features))

        scored.sort(key=lambda x: x[1], reverse=True)

        if max_results:
            scored = scored[:max_results]

        if scored:
            top = scored[0]
            log.debug(f"Top ranked: {top[0]} score={top[1]:.4f}")

        return scored

    def set_ridge_model(self, model):
        """Hot-swap to ridge model."""
        self.ridge_model = model
        self.use_ridge = True
        log.info("Ranker switched to ridge regression mode")

    def get_weights(self) -> dict:
        """Return current scoring weights (for logging/debugging)."""
        if self.use_ridge and self.ridge_model is not None:
            try:
                coefs = dict(zip(self.ridge_features, self.ridge_model.coef_))
                return {"mode": "ridge", "weights": coefs}
            except Exception:
                pass
        return {"mode": "hand_built", "weights": self.weights}
