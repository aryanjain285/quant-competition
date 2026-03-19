"""
Market regime detector v4: rule-based + HMM ensemble.

Primary: rule-based regime score from cross-sectional market features
  (breadth, trend strength, downside stress, cost stress).
  More interpretable than HMM, grounded in observable market structure.

Secondary: HMM on BTC as a confirmation signal.
  If both agree → high confidence. If they disagree → use the more conservative.

Three regime states:
  TREND_SUPPORTIVE: breadth high, trend positive, stress low
    → trade freely, full exposure budget
  SELECTIVE: mixed signals, moderate stress
    → trade cautiously, reduced exposure, only top-ranked candidates
  HOSTILE: breadth low, trend negative, stress high
    → minimal or no new longs, prefer cash

Output: exposure multiplier (0.2 to 1.0) fed to risk_manager.
"""
import warnings
import numpy as np
from typing import Optional
from bot.logger import get_logger

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
log = get_logger("regime")

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# Regime states
TREND_SUPPORTIVE = 0
SELECTIVE = 1
HOSTILE = 2
REGIME_NAMES = {0: "TREND_SUPPORTIVE", 1: "SELECTIVE", 2: "HOSTILE"}

# Exposure multipliers per state
EXPOSURE_MULTIPLIERS = {
    TREND_SUPPORTIVE: 1.0,
    SELECTIVE: 0.6,
    HOSTILE: 0.25,
}


class RegimeDetector:
    """Detects market regime from cross-sectional + BTC features."""

    def __init__(self):
        self.current_regime = SELECTIVE  # start cautious
        self.regime_confidence = 0.5
        self.rule_regime = SELECTIVE
        self.hmm_regime = SELECTIVE
        self.market_features = {}

        # HMM state
        self.hmm_model: Optional[GaussianHMM] = None
        self.hmm_fitted = False
        self._hmm_state_map = {}

    # ─── Rule-Based Regime (Primary) ─────────────────────────────

    def classify_regime_rules(self, market_features: dict) -> int:
        """Classify regime from market-level features.

        Uses a simple additive score:
          breadth helps, trend_strength helps,
          mkt_downside_vol hurts, cost_stress hurts.
        Then threshold into 3 states.
        """
        breadth = market_features.get("breadth", 0.5)
        trend = market_features.get("trend_strength", 0)
        downside = market_features.get("mkt_downside_vol", 0)
        cost = market_features.get("cost_stress", 0)

        # Normalize downside vol and cost to [0, 1] range roughly
        # These are medians across universe — typical values:
        #   downside_vol: 0.2-0.8 annualized
        #   cost_stress: 0.0001-0.001 (spread %)
        downside_norm = min(1.0, downside / 0.5) if downside > 0 else 0
        cost_norm = min(1.0, cost / 0.001) if cost > 0 else 0

        # Regime score: higher = more bullish
        # Breadth > 0.6 is bullish, < 0.4 is bearish
        # Trend > 0 is bullish
        score = (
            (breadth - 0.5) * 2.0       # [-1, +1] contribution
            + np.clip(trend * 20, -1, 1)  # scale small returns to [-1, +1]
            - downside_norm * 0.5         # penalty for stress
            - cost_norm * 0.3             # penalty for poor conditions
        )

        if score > 0.3:
            return TREND_SUPPORTIVE
        elif score < -0.3:
            return HOSTILE
        else:
            return SELECTIVE

    # ─── HMM Regime (Secondary) ──────────────────────────────────

    def fit_hmm(self, btc_closes: np.ndarray, lookback: int = 720):
        """Fit 2-state HMM on BTC. Resamples 5-min data to hourly."""
        if not HMM_AVAILABLE or len(btc_closes) < 200:
            return

        prices = btc_closes[-lookback:]
        # Resample to hourly if sub-hourly
        if len(prices) > 2000:
            trim = len(prices) % 12
            if trim > 0:
                prices = prices[trim:]
            prices = prices.reshape(-1, 12)[:, -1]

        log_returns = np.diff(np.log(prices))
        vol_window = 24
        rolling_vol = np.array([
            np.std(log_returns[max(0, i - vol_window):i]) if i > vol_window else np.std(log_returns[:i + 1])
            for i in range(len(log_returns))
        ])
        features = np.column_stack([log_returns, rolling_vol])
        mask = np.all(np.isfinite(features), axis=1)
        features = features[mask]

        if len(features) < 50:
            return

        try:
            self.hmm_model = GaussianHMM(n_components=2, covariance_type="full",
                                          n_iter=100, random_state=42)
            self.hmm_model.fit(features)

            vol_means = self.hmm_model.means_[:, 1]
            if vol_means[0] <= vol_means[1]:
                self._hmm_state_map = {0: TREND_SUPPORTIVE, 1: SELECTIVE}
            else:
                self._hmm_state_map = {0: SELECTIVE, 1: TREND_SUPPORTIVE}

            states = self.hmm_model.predict(features)
            self.hmm_regime = self._hmm_state_map.get(states[-1], SELECTIVE)
            self.hmm_fitted = True
            log.info(f"HMM fitted: state={REGIME_NAMES[self.hmm_regime]}")
        except Exception as e:
            log.debug(f"HMM fit failed: {e}")

    def update_hmm(self, btc_closes: np.ndarray):
        """Lightweight HMM predict (no refit)."""
        if not self.hmm_fitted or self.hmm_model is None:
            return

        if len(btc_closes) < 300:
            return

        recent = btc_closes[-360:]
        if len(recent) > 60:
            trim = len(recent) % 12
            if trim > 0:
                recent = recent[trim:]
            recent = recent.reshape(-1, 12)[:, -1]

        log_returns = np.diff(np.log(recent))
        vol_window = min(24, len(log_returns) - 1)
        rolling_vol = np.std(log_returns[-vol_window:]) if vol_window > 0 else 0

        try:
            features = np.column_stack([log_returns[-5:], [rolling_vol] * 5])
            if np.all(np.isfinite(features)):
                states = self.hmm_model.predict(features)
                self.hmm_regime = self._hmm_state_map.get(states[-1], SELECTIVE)
        except Exception:
            pass

    # ─── Combined Regime Decision ────────────────────────────────

    def update(self, market_features: dict, btc_closes: np.ndarray = None):
        """Update regime from market features + optional BTC HMM.

        The combination logic:
          - If both agree → high confidence, use that state
          - If they disagree → use the MORE CONSERVATIVE of the two
          This prevents the regime detector from being overly bullish.
        """
        self.market_features = market_features

        # Rule-based (primary)
        self.rule_regime = self.classify_regime_rules(market_features)

        # HMM (secondary)
        if btc_closes is not None and len(btc_closes) > 100:
            self.update_hmm(btc_closes)

        # Combine: use more conservative of the two
        if self.hmm_fitted:
            # Higher number = more conservative (HOSTILE > SELECTIVE > TREND_SUPPORTIVE)
            self.current_regime = max(self.rule_regime, self.hmm_regime)
            self.regime_confidence = 1.0 if self.rule_regime == self.hmm_regime else 0.6
        else:
            self.current_regime = self.rule_regime
            self.regime_confidence = 0.7

    def get_exposure_multiplier(self) -> float:
        return EXPOSURE_MULTIPLIERS.get(self.current_regime, 0.6)

    def should_trade(self) -> bool:
        """Whether the regime allows ANY new longs."""
        return self.current_regime != HOSTILE

    def get_status(self) -> dict:
        return {
            "regime": REGIME_NAMES.get(self.current_regime, "UNKNOWN"),
            "rule_regime": REGIME_NAMES.get(self.rule_regime, "UNKNOWN"),
            "hmm_regime": REGIME_NAMES.get(self.hmm_regime, "UNKNOWN"),
            "confidence": round(self.regime_confidence, 3),
            "exposure_mult": round(self.get_exposure_multiplier(), 3),
            "should_trade": self.should_trade(),
            "market_features": self.market_features,
            "hmm_fitted": self.hmm_fitted,
        }
