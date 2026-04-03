"""
Market regime detector v6: PCA-HMM on top-K principal components.

Key changes from v4:
- HMM observes K=3 PC score vectors (not PC1 + rolling vol)
- K is fixed at 3 to keep HMM parameter count stable (~24 params)
- State mapping uses trace(covariance) to rank states by volatility
- Exposure multipliers flipped: LOW_VOL=1.0 (full sizing in calm trends)
- PCA loadings cached; full refit every 6h, cheap projection between refits

Parameter budget:
  3-state HMM, 3D observations, full covariance:
  Means: 3×3 = 9 params
  Covariances: 3×(3×4/2) = 18 params
  Transition: 6 free params
  Initial: 2 free params
  Total: ~35 params → stable with 800+ observations
"""
import warnings
import time as _time
import numpy as np
from typing import Optional
from bot.config import (
    HMM_N_PCS, HMM_N_STATES, PCA_REFIT_INTERVAL_HOURS, REGIME_EXPOSURE,
)
from bot.logger import get_logger

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
log = get_logger("regime")

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    log.warning("hmmlearn not installed. HMM regime detection disabled.")

try:
    from sklearn.decomposition import PCA
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False
    log.warning("scikit-learn not installed. PCA disabled.")

# Regime labels
LOW_VOL = 0
MID_VOL = 1
HI_VOL = 2
REGIME_NAMES = {0: "LOW_VOL", 1: "MID_VOL", 2: "HI_VOL"}


class RegimeDetector:
    """Detects market regime from PCA-HMM on top-K principal components."""

    def __init__(self):
        self.current_regime = MID_VOL
        self.regime_confidence = 0.5

        # HMM state
        self.hmm_model: Optional[GaussianHMM] = None
        self.hmm_fitted = False
        self._hmm_state_map: dict[int, int] = {}
        self.hmm_regime = MID_VOL

        # PCA cache
        self._pca_model: Optional[PCA] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_std: Optional[np.ndarray] = None
        self._pca_explained_var: float = 0.0
        self._pca_last_fit: float = 0.0
        self._pca_refit_interval: float = PCA_REFIT_INTERVAL_HOURS * 3600

        # Latest PC scores for HMM update
        self._latest_pc_scores: Optional[np.ndarray] = None

    def compute_pc_scores(self, price_matrix: np.ndarray) -> Optional[np.ndarray]:
        """Compute top-K PC scores from price matrix.

        Args:
            price_matrix: shape (T, N_assets) of close prices

        Returns:
            pc_scores: shape (T-1, K) of standardized PC scores, or None
        """
        if not PCA_AVAILABLE:
            return None
        if price_matrix is None or price_matrix.ndim != 2:
            return None
        if price_matrix.shape[0] < 50 or price_matrix.shape[1] < HMM_N_PCS:
            return None

        # Log returns: (T-1, N)
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.log(price_matrix[1:] / price_matrix[:-1])
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        now = _time.time()
        need_refit = (
            self._pca_model is None
            or now - self._pca_last_fit > self._pca_refit_interval
            or returns.shape[1] != len(self._pca_mean or [])
        )

        if need_refit:
            mu = returns.mean(axis=0)
            sigma = returns.std(axis=0)
            sigma[sigma == 0] = 1.0
            returns_z = (returns - mu) / sigma

            pca = PCA(n_components=HMM_N_PCS)
            pc_scores = pca.fit_transform(returns_z)

            self._pca_model = pca
            self._pca_mean = mu
            self._pca_std = sigma
            self._pca_explained_var = float(np.sum(pca.explained_variance_ratio_))
            self._pca_last_fit = now

            log.info(
                f"PCA refit: {HMM_N_PCS} PCs explain "
                f"{self._pca_explained_var:.1%} of variance"
            )
        else:
            # Cheap projection onto cached loadings
            std = self._pca_std.copy()
            std[std == 0] = 1.0
            returns_z = (returns - self._pca_mean) / std
            returns_z = np.nan_to_num(returns_z, nan=0.0, posinf=0.0, neginf=0.0)
            pc_scores = returns_z @ self._pca_model.components_.T

        pc_scores = np.nan_to_num(pc_scores, nan=0.0, posinf=0.0, neginf=0.0)
        self._latest_pc_scores = pc_scores
        return pc_scores

    def fit_hmm(self, pc_scores: np.ndarray, lookback: int = 1440):
        """Fit 3-state HMM on PC score vectors.

        Args:
            pc_scores: shape (T, K) — the multi-dimensional observation sequence
            lookback: max number of recent observations to use for fitting
        """
        if not HMM_AVAILABLE:
            return
        if pc_scores is None or len(pc_scores) < 100:
            return

        obs = pc_scores[-lookback:]

        # Filter out any non-finite rows
        mask = np.all(np.isfinite(obs), axis=1)
        obs = obs[mask]
        if len(obs) < 100:
            return

        try:
            self.hmm_model = GaussianHMM(
                n_components=HMM_N_STATES,
                covariance_type="full",
                n_iter=300,
                random_state=42,
            )
            self.hmm_model.fit(obs)

            # Map states by total variance: trace(covariance) for each state
            # Higher trace = more volatile regime
            state_variance = np.array([
                np.trace(self.hmm_model.covars_[i])
                for i in range(HMM_N_STATES)
            ])
            sorted_ids = np.argsort(state_variance)

            self._hmm_state_map = {
                sorted_ids[0]: LOW_VOL,   # lowest variance
                sorted_ids[1]: MID_VOL,   # middle variance
                sorted_ids[2]: HI_VOL,    # highest variance
            }

            # Predict current state
            states = self.hmm_model.predict(obs)
            self.hmm_regime = self._hmm_state_map.get(states[-1], MID_VOL)
            self.hmm_fitted = True

            var_str = ", ".join(f"{v:.4f}" for v in state_variance[sorted_ids])
            log.info(
                f"HMM fitted on {HMM_N_PCS} PCs: "
                f"state={REGIME_NAMES[self.hmm_regime]}, "
                f"variances=[{var_str}]"
            )

        except Exception as e:
            log.warning(f"HMM fit failed: {e}")

    def update_hmm(self, pc_scores: np.ndarray):
        """Cheap HMM update: predict on recent observations without refitting.

        Uses last 30 observations to predict current state.
        """
        if not self.hmm_fitted or self.hmm_model is None:
            return
        if pc_scores is None or len(pc_scores) < 30:
            return

        recent = pc_scores[-30:]
        mask = np.all(np.isfinite(recent), axis=1)
        recent = recent[mask]
        if len(recent) < 10:
            return

        try:
            states = self.hmm_model.predict(recent)
            self.hmm_regime = self._hmm_state_map.get(states[-1], MID_VOL)
        except Exception:
            pass

    def update(self, pc_scores: np.ndarray = None):
        """Update regime from PC scores. Called every cycle."""
        if pc_scores is not None and len(pc_scores) > 30:
            self.update_hmm(pc_scores)

        if self.hmm_fitted:
            self.current_regime = self.hmm_regime
            self.regime_confidence = 1.0
        else:
            self.current_regime = MID_VOL
            self.regime_confidence = 0.0

    def get_exposure_multiplier(self) -> float:
        """Get exposure multiplier for current regime.

        LOW_VOL=1.0 (calm → full size, trends are clean)
        MID_VOL=0.7 (normal)
        HI_VOL=0.0 (crisis → sit out)
        """
        name = REGIME_NAMES.get(self.current_regime, "MID_VOL")
        return REGIME_EXPOSURE.get(name, 0.7)

    def should_trade(self) -> bool:
        return self.current_regime != HI_VOL

    def get_status(self) -> dict:
        return {
            "regime": REGIME_NAMES.get(self.current_regime, "UNKNOWN"),
            "confidence": round(self.regime_confidence, 3),
            "exposure_mult": round(self.get_exposure_multiplier(), 3),
            "should_trade": self.should_trade(),
            "hmm_fitted": self.hmm_fitted,
            "pca_explained_var": round(self._pca_explained_var, 3),
            "n_pcs": HMM_N_PCS,
        }