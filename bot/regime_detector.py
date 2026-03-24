"""
Market regime detector v4: PCA-HMM Only.

- Full PCA refit every 6 hours (expensive: SVD on T×N matrix)
- Between refits: project new returns onto cached loadings (cheap: dot product)
- HMM refit every 24 hours on the PC1 series
"""
import warnings
import time as _time
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

try:
    from sklearn.decomposition import PCA
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False

# Regime states
LOW_VOL = 0
MID_VOL = 1
HI_VOL = 2
REGIME_NAMES = {0: "LOW_VOL", 1: "MID_VOL", 2: "HI_VOL"}

EXPOSURE_MULTIPLIERS = {
    LOW_VOL: 0.5,
    MID_VOL: 1.0,
    HI_VOL: 0.0,
}

class RegimeDetector:
    """Detects market regime from PCA-HMM."""

    def __init__(self):
        self.current_regime = MID_VOL
        self.regime_confidence = 0.5

        # HMM
        self.hmm_model: Optional[GaussianHMM] = None
        self.hmm_fitted = False
        self._hmm_state_map = {}
        self.hmm_regime = MID_VOL

        # PCA cache — avoid recomputing every cycle
        self._pca_loadings: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_std: Optional[np.ndarray] = None
        self._pca_explained_var: float = 0.0
        self._pca_last_fit: float = 0.0
        self._pca_refit_interval: float = 6 * 3600
        self._pc1_series: Optional[np.ndarray] = None
        
    def compute_pc1_market_proxy(self, price_matrix: np.ndarray) -> tuple:
        if not PCA_AVAILABLE:
            return None, None, 0.0, None

        if price_matrix is None or price_matrix.ndim != 2:
            return None, None, 0.0, None
        if price_matrix.shape[0] < 10 or price_matrix.shape[1] < 2:
            return None, None, 0.0, None

        returns = np.log(price_matrix[1:] / price_matrix[:-1])
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        now = _time.time()
        need_refit = (
            self._pca_loadings is None
            or now - self._pca_last_fit > self._pca_refit_interval
            or self._pca_loadings.shape[0] != returns.shape[1]
        )

        if need_refit:
            mu = returns.mean(axis=0, keepdims=True)
            sigma = returns.std(axis=0, keepdims=True)
            sigma[sigma == 0] = 1.0
            returns_z = (returns - mu) / sigma

            pca = PCA(n_components=1)
            pc1_scores = pca.fit_transform(returns_z).flatten()
            explained_var = float(pca.explained_variance_ratio_[0])
            loadings = pca.components_[0].copy()

            if loadings.mean() < 0:
                pc1_scores = -pc1_scores
                loadings = -loadings

            self._pca_loadings = loadings
            self._pca_mean = mu.flatten()
            self._pca_std = sigma.flatten()
            self._pca_explained_var = explained_var
            self._pca_last_fit = now

        else:
            std = self._pca_std.copy()
            std[std == 0] = 1.0
            with np.errstate(all='ignore'):
                returns_z = (returns - self._pca_mean) / std
                returns_z = np.nan_to_num(returns_z, nan=0.0, posinf=0.0, neginf=0.0)
                pc1_scores = returns_z @ self._pca_loadings
            pc1_scores = np.nan_to_num(pc1_scores, nan=0.0, posinf=0.0, neginf=0.0)
            loadings = self._pca_loadings
            explained_var = self._pca_explained_var

        synthetic_series = np.exp(np.insert(np.cumsum(pc1_scores), 0, 0.0))
        self._pc1_series = synthetic_series

        return synthetic_series, pc1_scores, explained_var, loadings

    def fit_hmm(self, pc1_series: np.ndarray, lookback: int = 1440):
        if not HMM_AVAILABLE or len(pc1_series) < 50:
            return

        prices = pc1_series[-lookback:]
        log_returns = np.diff(np.log(np.clip(prices, 1e-10, None)))

        vol_window = 20
        rolling_vol = np.array([
            np.std(log_returns[max(0, i - vol_window):i])
            if i > vol_window else np.std(log_returns[:i + 1])
            for i in range(len(log_returns))
        ])

        features = np.column_stack([log_returns, rolling_vol])
        mask = np.all(np.isfinite(features), axis=1)
        features = features[mask]

        if len(features) < 50:
            return

        try:
            # 1. Change to 3 states to capture Extreme/HI_VOL environments
            self.hmm_model = GaussianHMM(
                n_components=3, covariance_type="full",
                n_iter=300, random_state=70,
            )
            self.hmm_model.fit(features)

            # 2. Extract the volatility means of the 3 clusters
            vol_means = self.hmm_model.means_[:, 1]
            
            # 3. Sort the clusters from lowest volatility to highest volatility
            # argsort returns the cluster IDs (0, 1, 2) in ascending order of their volatility
            sorted_state_ids = np.argsort(vol_means)

            # 4. Map the sorted clusters to your 3 regimes
            self._hmm_state_map = {
                sorted_state_ids[0]: LOW_VOL,  # Lowest vol   
                sorted_state_ids[1]: MID_VOL,         # Middle vol  
                sorted_state_ids[2]: HI_VOL            # Highest vol  - Choose not to trade in this regime. 
            }

            states = self.hmm_model.predict(features)
            self.hmm_regime = self._hmm_state_map.get(states[-1], MID_VOL)
            self.hmm_fitted = True
            log.info(f"HMM fitted on PC1: state={REGIME_NAMES[self.hmm_regime]}")
            
        except Exception as e:
                # Now properly logging as a warning so you can see convergence failures!
                log.warning(f"HMM fit failed to converge: {e}")

    def update_hmm(self, pc1_series: np.ndarray):
        if not self.hmm_fitted or self.hmm_model is None:
            return
        if len(pc1_series) < 30:
            return

        prices = pc1_series[-30:]
        log_returns = np.diff(np.log(np.clip(prices, 1e-10, None)))
        vol_window = min(24, len(log_returns) - 1)
        rolling_vol = np.std(log_returns[-vol_window:]) if vol_window > 0 else 0

        try:
            features = np.column_stack([log_returns[-5:], [rolling_vol] * 5])
            if np.all(np.isfinite(features)):
                states = self.hmm_model.predict(features)
                self.hmm_regime = self._hmm_state_map.get(states[-1], MID_VOL)
        except Exception:
            pass

    def update(self, pc1_series: np.ndarray = None):
        """Update regime using ONLY the PCA-HMM."""
        if pc1_series is not None and len(pc1_series) > 30:
            self.update_hmm(pc1_series)

        if self.hmm_fitted:
            self.current_regime = self.hmm_regime
            self.regime_confidence = 1.0
        else:
            self.current_regime = MID_VOL
            self.regime_confidence = 0.0

    def get_exposure_multiplier(self) -> float:
        return EXPOSURE_MULTIPLIERS.get(self.current_regime, 0.6)

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
        }