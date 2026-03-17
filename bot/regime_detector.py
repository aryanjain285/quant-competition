"""
Market regime detection using Hidden Markov Models.

Fits a 2-state Gaussian HMM on BTC returns + volatility to identify:
  State 0: Low-volatility trending regime  → be aggressive (breakout strategy)
  State 1: High-volatility choppy regime   → be defensive (mean reversion, smaller sizes)

Research basis:
- HMMs increase Sharpe by filtering trades in wrong-regime conditions
- QuantInsti: "train specialist models per regime"
- Academic: HMMs detect regime changes in cryptoasset markets (ResearchGate 2020)

The detector runs on BTC because BTC drives market-wide regime.
Individual altcoin signals still come from signals.py.
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
    log.warning("hmmlearn not installed — regime detection will use fallback volatility method")


class RegimeDetector:
    """Detects market regime using BTC data.

    Two regimes:
        TRENDING (0): low vol, directional — favor breakout, full exposure
        VOLATILE (1): high vol, choppy — favor mean-reversion, reduce exposure
    """

    TRENDING = 0
    VOLATILE = 1
    REGIME_NAMES = {0: "TRENDING", 1: "VOLATILE"}

    def __init__(self):
        self.model: Optional[GaussianHMM] = None
        self.is_fitted = False
        self.current_regime = self.TRENDING
        self.regime_confidence = 0.5
        self.regime_history: list[int] = []

        # Fallback volatility-based detection
        self.vol_baseline: float = 0.0

    def fit(self, btc_closes: np.ndarray, lookback: int = 720):
        """Fit HMM on BTC returns and volatility.

        Args:
            btc_closes: array of BTC close prices (hourly or 5-min)
            lookback: number of periods to use for fitting
        """
        if len(btc_closes) < max(lookback, 100):
            log.warning(f"Not enough BTC data for regime fit ({len(btc_closes)} bars, need {lookback})")
            self._fit_fallback(btc_closes)
            return

        # Resample to hourly if data is sub-hourly (e.g. 5-min candles)
        # Regime should operate on daily timescales, not intra-hour noise
        prices = btc_closes[-lookback:]
        if len(prices) > 2000:
            # Likely 5-min data (1000 bars = ~3.5 days). Resample to hourly.
            trim = len(prices) % 12
            if trim > 0:
                prices = prices[trim:]
            prices = prices.reshape(-1, 12)[:, -1]  # take last close per hour

        log_returns = np.diff(np.log(prices))

        # Feature matrix: [returns, rolling_vol]
        # Rolling volatility over 24-HOUR window (24 bars after resampling)
        vol_window = 24
        rolling_vol = np.array([
            np.std(log_returns[max(0, i - vol_window):i]) if i > vol_window else np.std(log_returns[:i+1])
            for i in range(len(log_returns))
        ])

        features = np.column_stack([log_returns, rolling_vol])

        # Remove any NaN/inf
        mask = np.all(np.isfinite(features), axis=1)
        features = features[mask]

        if len(features) < 50:
            log.warning("Not enough valid features for HMM")
            self._fit_fallback(btc_closes)
            return

        if not HMM_AVAILABLE:
            self._fit_fallback(btc_closes)
            return

        try:
            self.model = GaussianHMM(
                n_components=2,
                covariance_type="full",
                n_iter=100,
                random_state=42,
            )
            self.model.fit(features)

            # Identify which state is "trending" vs "volatile"
            # The state with lower volatility mean is the trending state
            vol_means = self.model.means_[:, 1]  # column 1 = rolling vol
            if vol_means[0] <= vol_means[1]:
                # State 0 is already low-vol (trending)
                self._state_map = {0: self.TRENDING, 1: self.VOLATILE}
            else:
                # Swap
                self._state_map = {0: self.VOLATILE, 1: self.TRENDING}

            # Get current regime
            states = self.model.predict(features)
            self.current_regime = self._state_map[states[-1]]

            # Confidence from posterior probabilities
            probs = self.model.predict_proba(features)
            last_probs = probs[-1]
            self.regime_confidence = float(np.max(last_probs))

            self.is_fitted = True
            log.info(
                f"HMM fitted: current regime={self.REGIME_NAMES[self.current_regime]} "
                f"confidence={self.regime_confidence:.2f} "
                f"vol_means={vol_means}"
            )

        except Exception as e:
            log.error(f"HMM fitting failed: {e}, using fallback")
            self._fit_fallback(btc_closes)

    def _fit_fallback(self, closes: np.ndarray):
        """Fallback: use simple volatility threshold for regime detection."""
        if len(closes) < 50:
            return

        log_returns = np.diff(np.log(closes[-288:]))  # last ~24h or available
        self.vol_baseline = float(np.std(log_returns))
        self.is_fitted = True
        log.info(f"Regime fallback: using vol baseline={self.vol_baseline:.6f}")

    def update(self, btc_closes: np.ndarray):
        """Update regime state with latest BTC data.

        Call this every cycle. Does NOT re-fit the model — just predicts
        the current state using the existing model.
        """
        if not self.is_fitted:
            self.fit(btc_closes)
            return

        if len(btc_closes) < 300:
            return

        # Resample to hourly for regime prediction (same as fit)
        recent = btc_closes[-360:]  # last 30 hours of 5-min data
        if len(recent) > 60:
            trim = len(recent) % 12
            if trim > 0:
                recent = recent[trim:]
            recent = recent.reshape(-1, 12)[:, -1]

        log_returns = np.diff(np.log(recent))
        vol_window = min(24, len(log_returns) - 1)
        rolling_vol = np.std(log_returns[-vol_window:]) if vol_window > 0 else 0

        if self.model is not None and HMM_AVAILABLE:
            try:
                features = np.column_stack([log_returns[-5:], [rolling_vol] * 5])
                if np.all(np.isfinite(features)):
                    states = self.model.predict(features)
                    self.current_regime = self._state_map[states[-1]]

                    probs = self.model.predict_proba(features)
                    self.regime_confidence = float(np.max(probs[-1]))
            except Exception as e:
                log.debug(f"HMM predict failed: {e}, using fallback")
                self._update_fallback(rolling_vol)
        else:
            self._update_fallback(rolling_vol)

        self.regime_history.append(self.current_regime)
        # Keep last 2000
        if len(self.regime_history) > 2000:
            self.regime_history = self.regime_history[-2000:]

    def _update_fallback(self, current_vol: float):
        """Fallback regime detection: vol > 1.5x baseline = volatile."""
        if self.vol_baseline > 0:
            if current_vol > 1.5 * self.vol_baseline:
                self.current_regime = self.VOLATILE
                self.regime_confidence = min(1.0, current_vol / (2 * self.vol_baseline))
            else:
                self.current_regime = self.TRENDING
                self.regime_confidence = min(1.0, 1 - current_vol / (1.5 * self.vol_baseline))

    def get_exposure_multiplier(self) -> float:
        """Get recommended exposure multiplier based on regime.

        TRENDING: 1.0 (full exposure)
        VOLATILE: 0.4-0.6 (reduced exposure)
        """
        if self.current_regime == self.VOLATILE:
            # Scale by confidence: more confident it's volatile = lower exposure
            return max(0.3, 0.6 - 0.3 * self.regime_confidence)
        else:
            return 1.0

    def get_strategy_preference(self) -> str:
        """Get which strategy should be prioritized in current regime.

        TRENDING: 'breakout' (catch momentum)
        VOLATILE: 'mean_rev' (fade extremes)
        """
        return "breakout" if self.current_regime == self.TRENDING else "mean_rev"

    def should_skip_breakout(self) -> bool:
        """Whether to suppress breakout entries (volatile regime, high confidence)."""
        return self.current_regime == self.VOLATILE and self.regime_confidence > 0.7

    def get_status(self) -> dict:
        return {
            "regime": self.REGIME_NAMES.get(self.current_regime, "UNKNOWN"),
            "confidence": round(self.regime_confidence, 3),
            "exposure_mult": round(self.get_exposure_multiplier(), 3),
            "strategy_pref": self.get_strategy_preference(),
            "is_fitted": self.is_fitted,
            "method": "hmm" if (self.model is not None and HMM_AVAILABLE) else "fallback",
        }
