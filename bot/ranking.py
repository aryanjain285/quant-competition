"""
Ranker v7: EWMA momentum ranking with optional Lasso boost.

Primary signal: EWMA momentum score (always available, always ranks coins).
  score = average(EWMA(log_returns, halflife=6h), EWMA(log_returns, halflife=24h))

ML boost: when Lasso on RELATIVE returns finds signal (R² > 0.5%),
  blend its cross-sectional prediction into the EWMA score.

Entry gate: r_24h > 0 AND volume_ratio > 0.8.
  Deliberately loose — ranking handles quality.
"""
import numpy as np
from typing import Optional
from bot.config import ML_BLEND_WEIGHT, ML_MIN_R2, LASSO_FEATURES
from bot.features import compute_ewma_momentum, check_entry_gate
from bot.logger import get_logger

log = get_logger("ranking")


class Ranker:
    """Ranks coins by EWMA momentum with optional Lasso boost."""

    def __init__(self):
        self.lasso_model = None
        self.lasso_r2: float = 0.0
        self.lasso_active: bool = False

    def set_lasso(self, model, r2: float):
        """Hot-swap Lasso. Only activates boost if R² exceeds threshold."""
        self.lasso_model = model
        self.lasso_r2 = r2
        self.lasso_active = r2 >= ML_MIN_R2
        if self.lasso_active:
            log.info(f"Lasso boost ACTIVE (R²={r2:.4f})")
        else:
            log.info(f"Lasso boost OFF (R²={r2:.4f} < {ML_MIN_R2})")

    def has_model(self) -> bool:
        """Whether the ranker can produce rankings (always True — EWMA is always available)."""
        return True

    def _lasso_predict_one(self, zscored: dict) -> Optional[float]:
        """Get Lasso predicted relative return for one coin."""
        if not self.lasso_active or self.lasso_model is None:
            return None
        x = [zscored.get(k, 0.0) for k in LASSO_FEATURES]
        if any(not np.isfinite(v) for v in x):
            return None
        return float(self.lasso_model.predict([x])[0])

    def rank(
        self,
        raw_features: dict[str, dict],
        zscored_features: dict[str, dict],
        held_pairs: set[str] = None,
        closes_dict: dict[str, np.ndarray] = None,
    ) -> list[tuple[str, float, dict]]:
        """Rank all coins by EWMA momentum and filter by entry gate.

        Args:
            raw_features: {pair: raw features} for gate checks
            zscored_features: {pair: z-scored features} for Lasso boost
            held_pairs: pairs currently held (skip re-entry)
            closes_dict: {pair: np.ndarray of closes} for EWMA computation

        Returns:
            List of (pair, score, raw_features) sorted by score desc.
        """
        held_pairs = held_pairs or set()
        closes_dict = closes_dict or {}
        candidates = []

        for pair in raw_features:
            if pair in held_pairs:
                continue

            raw = raw_features[pair]

            # Entry gate
            if not check_entry_gate(raw):
                continue

            # EWMA momentum score
            closes = closes_dict.get(pair)
            if closes is not None and len(closes) > 30:
                ewma_score = compute_ewma_momentum(closes)
            else:
                # Fallback: use raw 24h return as proxy
                ewma_score = raw.get("r_24h", 0.0)

            if ewma_score <= 0:
                continue

            # Optional Lasso boost on cross-sectional relative returns
            zs = zscored_features.get(pair, {})
            lasso_pred = self._lasso_predict_one(zs)
            if lasso_pred is not None:
                # Scale lasso prediction to similar magnitude as EWMA score
                # EWMA score is in log-return units (~0.001 to 0.01 typically)
                score = (1 - ML_BLEND_WEIGHT) * ewma_score + ML_BLEND_WEIGHT * lasso_pred
            else:
                score = ewma_score

            if score > 0:
                candidates.append((pair, score, raw))

        candidates.sort(key=lambda x: x[1], reverse=True)

        if candidates:
            log.info(
                f"Ranked {len(candidates)} candidates. "
                f"Top: {candidates[0][0]} (score={candidates[0][1]:.6f}) "
                f"{'[+Lasso]' if self.lasso_active else '[EWMA only]'}"
            )

        return candidates
