"""
Ranker v7: EWMA ranking + Ridge veto.

EWMA drives the ordering (captures momentum — the signal that produces returns).
Ridge acts as a quality filter: vetoes EWMA picks where the cross-sectional
model predicts the coin will underperform the universe.

Why veto, not reorder or blend:
  - Score blending (0.70×EWMA + 0.30×Ridge) injected magnitude noise → hurt returns
  - Rank averaging (avg of two ranks) disrupted momentum ordering → worse
  - Veto preserves EWMA order and only REMOVES bad picks
  - The shootout proved Ridge can distinguish good from bad (p=0.025)
  - A veto uses exactly this: remove the bad, keep the good

Fallback: if Ridge vetoes ALL candidates, trade the top EWMA pick anyway.
This guarantees the bot always trades (competition activity requirement).
"""
import numpy as np
from bot.config import LASSO_FEATURES, ML_MIN_R2
from bot.features import compute_ewma_momentum, check_entry_gate
from bot.logger import get_logger

log = get_logger("ranking")


class Ranker:
    """EWMA ranking with Ridge veto on predicted underperformers."""

    def __init__(self):
        self.model = None
        self.model_r2: float = 0.0
        self.ridge_active: bool = False

    def set_model(self, model, r2: float):
        self.model = model
        self.model_r2 = r2
        self.ridge_active = r2 >= ML_MIN_R2
        status = "ACTIVE (veto mode)" if self.ridge_active else f"OFF (R²={r2:.4f} < {ML_MIN_R2})"
        log.info(f"Ridge veto {status}")

    def has_model(self) -> bool:
        return True

    # Compat properties for main.py logging
    @property
    def lasso_active(self) -> bool:
        return self.ridge_active

    @property
    def lasso_r2(self) -> float:
        return self.model_r2

    def _ridge_predict(self, zscored: dict) -> float:
        """Predict relative return. Only the SIGN matters for veto."""
        if self.model is None:
            return 0.0
        x = [zscored.get(k, 0.0) for k in LASSO_FEATURES]
        if any(not np.isfinite(v) for v in x):
            return 0.0
        return float(self.model.predict([x])[0])

    def rank(
        self,
        raw_features: dict[str, dict],
        zscored_features: dict[str, dict],
        held_pairs: set[str] = None,
        closes_dict: dict[str, np.ndarray] = None,
    ) -> list[tuple[str, float, dict]]:
        """Rank by EWMA, veto by Ridge.

        Returns (pair, ewma_score, raw_features) in EWMA order.
        Ridge only removes — never reorders or adjusts scores.
        """
        held_pairs = held_pairs or set()
        closes_dict = closes_dict or {}

        # Step 1: EWMA score for all eligible coins
        eligible = []
        for pair in raw_features:
            if pair in held_pairs:
                continue
            raw = raw_features[pair]
            if not check_entry_gate(raw):
                continue

            closes = closes_dict.get(pair)
            if closes is not None and len(closes) > 30:
                ewma = compute_ewma_momentum(closes)
            else:
                ewma = raw.get("r_24h", 0.0)

            if ewma <= 0:
                continue

            eligible.append((pair, ewma, raw))

        # Step 2: Sort by EWMA (highest momentum first)
        eligible.sort(key=lambda x: x[1], reverse=True)

        if not eligible:
            return []

        # Step 3: Ridge veto (if available)
        if self.ridge_active:
            survivors = []
            vetoed = []
            for pair, ewma, raw in eligible:
                zs = zscored_features.get(pair, {})
                ridge_pred = self._ridge_predict(zs)

                if ridge_pred >= 0:
                    # Ridge agrees or is neutral → keep
                    survivors.append((pair, ewma, raw))
                else:
                    # Ridge predicts underperformance → veto
                    vetoed.append(pair)

            if survivors:
                if vetoed:
                    log.info(f"Ridge vetoed {len(vetoed)}/{len(eligible)} candidates: {vetoed[:3]}")
                result = survivors
            else:
                # Ridge vetoed EVERYTHING → fallback to top EWMA pick
                log.info(f"Ridge vetoed all {len(eligible)} candidates — falling back to top EWMA")
                result = [eligible[0]]
        else:
            result = eligible

        if result:
            log.info(
                f"Ranked {len(result)} candidates. "
                f"Top: {result[0][0]} (ewma={result[0][1]:.6f})"
            )

        return result
