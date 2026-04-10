"""
Ranker v8: EWMA ranking with dynamic spread filter.

1. SPREAD FILTER (dynamic, per-cycle):
   Compute median spread across all coins from LIVE ticker data.
   Only trade coins with spread <= median.
   This automatically adapts:
     Calm markets → tight spreads → trade ~30 coins
     Volatile markets → wide spreads → narrow to ~15 liquid coins
   No hardcoded threshold — always the better half of the universe.

2. EWMA RANKING:
   Rank surviving coins by 0.8 * EWMA(6h) + 0.2 * EWMA(24h).
   Highest momentum first. No ML — cleanest signal we found.

3. ENTRY GATE:
   r_1h > 1%, volume_ratio > 0.8.
"""
import numpy as np
from bot.features import compute_ewma_momentum, check_entry_gate
from bot.logger import get_logger

log = get_logger("ranking")


class Ranker:
    """EWMA ranking with dynamic spread filter."""

    def __init__(self):
        # Ridge kept for potential future use but disabled (ML_ENABLED=False)
        self.model = None
        self.model_r2: float = 0.0
        self.ridge_active: bool = False
        self.last_median_spread: float = 0.0
        self.last_filtered_count: int = 0

    def set_model(self, model, r2: float):
        self.model = model
        self.model_r2 = r2

    def has_model(self) -> bool:
        return True

    # Compat properties for main.py logging
    @property
    def lasso_active(self) -> bool:
        return self.ridge_active

    @property
    def lasso_r2(self) -> float:
        return self.model_r2

    def rank(
        self,
        raw_features: dict[str, dict],
        zscored_features: dict[str, dict],
        held_pairs: set[str] = None,
        closes_dict: dict[str, np.ndarray] = None,
    ) -> list[tuple[str, float, dict]]:
        """Rank coins: dynamic spread filter → entry gate → EWMA sort.

        The spread filter adapts each cycle to current market conditions.
        Coins with spread above the median are excluded — they start
        every trade in a hole that the profit target can't recover.
        """
        held_pairs = held_pairs or set()
        closes_dict = closes_dict or {}

        # Step 1: Dynamic spread filter
        # Compute median spread across ALL coins (not just eligible ones)
        all_spreads = []
        for pair, raw in raw_features.items():
            spread = raw.get("spread_pct", 0)
            if spread > 0:
                all_spreads.append(spread)

        if all_spreads:
            median_spread = float(np.median(all_spreads))
        else:
            median_spread = 0.001  # fallback ~10 bps

        self.last_median_spread = median_spread

        # Step 2: Filter + gate + EWMA
        eligible = []
        spread_filtered = 0

        for pair in raw_features:
            if pair in held_pairs:
                continue

            raw = raw_features[pair]

            # Dynamic spread filter: only trade coins with spread <= median
            coin_spread = raw.get("spread_pct", 0)
            if coin_spread > median_spread and coin_spread > 0:
                spread_filtered += 1
                continue

            # Entry gate (r_1h > 1%, volume > 0.8)
            if not check_entry_gate(raw):
                continue

            # EWMA momentum score
            closes = closes_dict.get(pair)
            if closes is not None and len(closes) > 30:
                ewma = compute_ewma_momentum(closes)
            else:
                ewma = raw.get("r_24h", 0.0)

            if ewma <= 0:
                continue

            eligible.append((pair, ewma, raw))

        self.last_filtered_count = spread_filtered

        # Sort by EWMA (highest momentum first)
        eligible.sort(key=lambda x: x[1], reverse=True)

        if eligible:
            log.info(
                f"Ranked {len(eligible)} candidates "
                f"(spread filter removed {spread_filtered}, "
                f"median_spread={median_spread*10000:.1f}bps). "
                f"Top: {eligible[0][0]} (ewma={eligible[0][1]:.6f})"
            )

        return eligible
