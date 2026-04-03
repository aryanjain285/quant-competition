"""
Ranker v6: Lasso-based cross-sectional ranking with entry gate.

Replaces the old multi-strategy ranker with a single model-derived score.
The Lasso predicted return serves dual purpose:
  1. GATE: predicted return > ENTRY_THRESHOLD → allowed to trade
  2. RANK: higher predicted return → higher priority for capital allocation

This replaces the old 7-condition conjunction filter (continuation/reversal)
with a data-driven threshold. The Lasso implicitly learns which conditions
matter and how much weight to give each.
"""
from typing import Optional
from bot.config import ENTRY_THRESHOLD, LASSO_FEATURES
from bot.logger import get_logger

log = get_logger("ranking")


class Ranker:
    """Ranks coins by Lasso predicted forward return."""

    def __init__(self):
        self.model = None

    def set_model(self, model):
        """Hot-swap the trained Lasso model."""
        self.model = model

    def has_model(self) -> bool:
        return self.model is not None

    def rank(
        self,
        zscored_features: dict[str, dict],
        predictions: dict[str, float],
        held_pairs: set[str] = None,
    ) -> list[tuple[str, float, dict]]:
        """Rank coins and apply entry gate.

        Args:
            zscored_features: {pair: {feature: z_value, ...}}
            predictions: {pair: predicted_24h_return} from LassoTrainer.predict()
            held_pairs: set of pairs currently held (skip re-entry)

        Returns:
            List of (pair, predicted_return, features) sorted by predicted return desc.
            Only includes coins that pass the entry gate AND are not currently held.
        """
        held_pairs = held_pairs or set()
        candidates = []

        for pair, pred_return in predictions.items():
            # Gate: only enter if predicted return > threshold
            if pred_return < ENTRY_THRESHOLD:
                continue

            # Don't re-enter positions we already hold
            if pair in held_pairs:
                continue

            features = zscored_features.get(pair, {})
            candidates.append((pair, pred_return, features))

        # Sort by predicted return, highest first
        candidates.sort(key=lambda x: x[1], reverse=True)

        if candidates:
            log.info(
                f"Ranked {len(candidates)} candidates above gate "
                f"({ENTRY_THRESHOLD:.4f}). "
                f"Top: {candidates[0][0]} ({candidates[0][1]:.4f})"
            )

        return candidates