"""
Ranker v7: momentum composite ranking with optional ML boost.

Primary signal: weighted momentum score (always available, always produces rankings).
ML boost: if Lasso has signal (R² > threshold), blend its prediction into the score.

This ensures the bot always trades (momentum score always ranks coins),
while still benefiting from ML when it finds genuine signal.
"""
from typing import Optional
from bot.config import LASSO_FEATURES, ML_BLEND_WEIGHT, ML_MIN_R2
from bot.features import compute_momentum_score, check_entry_gate
from bot.logger import get_logger

log = get_logger("ranking")


class Ranker:
    """Ranks coins by momentum composite score with optional Lasso boost."""

    def __init__(self):
        self.lasso_model = None
        self.lasso_r2: float = 0.0
        self.lasso_active: bool = False

    def set_lasso(self, model, r2: float):
        """Hot-swap Lasso model. Only activates boost if R² exceeds threshold."""
        self.lasso_model = model
        self.lasso_r2 = r2
        self.lasso_active = r2 >= ML_MIN_R2
        if self.lasso_active:
            log.info(f"Lasso boost ACTIVE (R²={r2:.4f})")
        else:
            log.info(f"Lasso boost OFF (R²={r2:.4f} < {ML_MIN_R2})")

    def _lasso_predict(self, zscored: dict) -> Optional[float]:
        """Get Lasso predicted return for one coin."""
        if not self.lasso_active or self.lasso_model is None:
            return None
        import numpy as np
        x = [zscored.get(k, 0.0) for k in LASSO_FEATURES]
        if any(not np.isfinite(v) for v in x):
            return None
        return float(self.lasso_model.predict([x])[0])

    def rank(
        self,
        raw_features: dict[str, dict],
        zscored_features: dict[str, dict],
        held_pairs: set[str] = None,
    ) -> list[tuple[str, float, dict]]:
        """Rank all coins and filter by entry gate.

        Args:
            raw_features: {pair: {raw feature values}} for gate checks
            zscored_features: {pair: {z-scored feature values}} for scoring
            held_pairs: pairs currently held (skip re-entry)

        Returns:
            List of (pair, score, raw_features) sorted by score desc.
            Only includes coins that pass the entry gate AND are not held.
        """
        held_pairs = held_pairs or set()
        candidates = []

        for pair in zscored_features:
            # Skip if already holding
            if pair in held_pairs:
                continue

            raw = raw_features.get(pair)
            if raw is None:
                continue

            # Entry gate (simple binary conditions on raw features)
            if not check_entry_gate(raw):
                continue

            zs = zscored_features[pair]

            # Primary score: momentum composite
            momentum_score = compute_momentum_score(zs)

            # Optional ML boost
            lasso_pred = self._lasso_predict(zs)
            if lasso_pred is not None:
                # Blend: (1-w)*momentum + w*lasso_pred (normalized)
                # Normalize lasso_pred to similar scale as momentum_score
                # momentum_score is in z-score units (~[-3, 3])
                # lasso_pred is in return units (~[-0.05, 0.05])
                # Scale lasso by 20 to bring into similar range
                lasso_scaled = lasso_pred * 20
                score = (1 - ML_BLEND_WEIGHT) * momentum_score + ML_BLEND_WEIGHT * lasso_scaled
            else:
                score = momentum_score

            # Only keep positive-scoring candidates
            if score > 0:
                candidates.append((pair, score, raw))

        # Sort by score, highest first
        candidates.sort(key=lambda x: x[1], reverse=True)

        if candidates:
            top = candidates[0]
            log.info(
                f"Ranked {len(candidates)} candidates. "
                f"Top: {top[0]} (score={top[1]:.4f}) "
                f"{'[+Lasso]' if self.lasso_active else '[momentum only]'}"
            )

        return candidates