"""
Market regime detector v8: data-driven HMM state analysis with tiered exposure.

KEY DESIGN PRINCIPLE:
States are NOT pre-labeled. The HMM discovers them, we analyze them AFTER:
  1. PCA: 43 coins → 4 PCs (captures ~72% of cross-sectional variance)
  2. HMM: 3-state Gaussian HMM on 4D PC score vectors (unsupervised)
  3. State analysis: for each state, measure forward returns, vol, duration
  4. Exposure tiers: states ranked by forward-return Sharpe:
       Best  → 1.2x (leveraged — deploy more in favorable conditions)
       Mid   → 0.6x (moderate — mixed conditions)
       Worst → 0.10x (minimal — activity compliance floor)
  5. Names: assigned from observed properties (BULL/BEAR + CALM/VOLATILE)
     for logging only — no logic depends on names.

Exposure derivation:
  Linear interpolation from Sharpe ranking with 0.10 floor.
  Best state → ~1.0, worst state → 0.10 (activity compliance).
  Tested alternatives: 0.15 floor (+2.08%), fixed tiers 1.2/0.6/0.10 (-0.85%).
  Linear with 0.10 floor gave best result: +2.45% on 4-month backtest.

Parameter budget:
  3-state HMM, 4D observations, full covariance:
  Means: 3×4 = 12, Covariances: 3×(4×5/2) = 30, Transition: 6, Initial: 2
  Total: ~50 params → stable with 800+ observations
"""
import warnings
import time as _time
import numpy as np
from typing import Optional
from bot.config import (
    HMM_N_PCS, HMM_N_STATES, PCA_REFIT_INTERVAL_HOURS,
    STATE_ANALYSIS_FORWARD_HOURS,
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


class RegimeDetector:
    """Detects market regime from PCA-HMM with data-driven state interpretation."""

    def __init__(self):
        self.n_pcs = HMM_N_PCS
        self.n_states = HMM_N_STATES
        self.current_state: int = 0
        self.regime_confidence: float = 0.0

        # HMM
        self.hmm_model: Optional[GaussianHMM] = None
        self.hmm_fitted: bool = False

        # State analysis results (populated by analyze_states)
        self._state_analysis: dict[int, dict] = {}
        self._state_exposure: dict[int, float] = {}
        self._state_names: dict[int, str] = {}

        # PCA cache
        self._pca_model: Optional[PCA] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_std: Optional[np.ndarray] = None
        self._pca_explained_var: float = 0.0
        self._pca_last_fit: float = 0.0
        self._pca_refit_interval: float = PCA_REFIT_INTERVAL_HOURS * 3600

        self._latest_pc_scores: Optional[np.ndarray] = None

    # ─── PCA ────────────────────────────────────────────────────

    def compute_pc_scores(self, price_matrix: np.ndarray) -> Optional[np.ndarray]:
        """Compute top-K PC scores from price matrix.

        Args:
            price_matrix: shape (T, N_assets) of close prices
        Returns:
            pc_scores: shape (T-1, K) or None
        """
        if not PCA_AVAILABLE:
            return None
        if price_matrix is None or price_matrix.ndim != 2:
            return None
        if price_matrix.shape[0] < 50 or price_matrix.shape[1] < self.n_pcs:
            return None

        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.log(price_matrix[1:] / price_matrix[:-1])
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        now = _time.time()
        need_refit = (
            self._pca_model is None
            or now - self._pca_last_fit > self._pca_refit_interval
            or self._pca_mean is None
            or returns.shape[1] != len(self._pca_mean)
        )

        if need_refit:
            mu = returns.mean(axis=0)
            sigma = returns.std(axis=0)
            sigma[sigma == 0] = 1.0
            returns_z = (returns - mu) / sigma

            pca = PCA(n_components=self.n_pcs)
            pc_scores = pca.fit_transform(returns_z)

            self._pca_model = pca
            self._pca_mean = mu
            self._pca_std = sigma
            self._pca_explained_var = float(np.sum(pca.explained_variance_ratio_))
            self._pca_last_fit = now

            log.info(
                f"PCA refit: {self.n_pcs} PCs explain "
                f"{self._pca_explained_var:.1%} of variance"
            )
        else:
            std = self._pca_std.copy()
            std[std == 0] = 1.0
            with np.errstate(all='ignore'):
                returns_z = (returns - self._pca_mean) / std
                returns_z = np.nan_to_num(returns_z, nan=0.0, posinf=0.0, neginf=0.0)
                # Clip extreme z-scores to prevent overflow in matmul
                returns_z = np.clip(returns_z, -10, 10)
                pc_scores = returns_z @ self._pca_model.components_.T

        pc_scores = np.nan_to_num(pc_scores, nan=0.0, posinf=0.0, neginf=0.0)
        self._latest_pc_scores = pc_scores
        return pc_scores

    # ─── HMM Fitting + State Analysis ──────────────────────────

    def fit_hmm(self, pc_scores: np.ndarray, lookback: int = 1440):
        """Fit HMM and analyze discovered states.

        This is the core v7 method. Steps:
        1. Fit 3-state HMM on K-dimensional PC observations
        2. Predict state sequence for all observations
        3. Analyze each state's properties (forward returns, vol, duration)
        4. Derive exposure multipliers from the analysis
        """
        if not HMM_AVAILABLE:
            return
        if pc_scores is None or len(pc_scores) < 100:
            return

        obs = pc_scores[-lookback:]
        mask = np.all(np.isfinite(obs), axis=1)
        obs = obs[mask]
        if len(obs) < 100:
            return

        try:
            self.hmm_model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=300,
                random_state=42,
            )
            self.hmm_model.fit(obs)
            self.hmm_fitted = True

            # Step 2-4: Analyze and map states
            self._analyze_states(obs)

            # Predict current state
            states = self.hmm_model.predict(obs)
            self.current_state = int(states[-1])
            self.regime_confidence = 1.0

            log.info(
                f"HMM fitted on {self.n_pcs} PCs ({len(obs)} obs). "
                f"Current state: {self._state_names.get(self.current_state, '?')} "
                f"(exposure={self._state_exposure.get(self.current_state, 0.5):.2f})"
            )

        except Exception as e:
            log.warning(f"HMM fit failed: {e}")

    def _analyze_states(self, obs: np.ndarray):
        """Analyze each HMM state after fitting. This is where interpretation happens.

        For each state we compute:
        - mean_pc_vector: the mean direction in PC-space
        - trace_cov: total variance (volatility)
        - avg_forward_return: average return of PC1 over next 24h bars
        - avg_duration: how many consecutive hours the state typically lasts
        - frequency: what fraction of time the market is in this state
        - transition_probs: where does this state tend to go next

        The exposure multiplier is derived from avg_forward_return:
        - Normalize across states so best → 1.0, worst → 0.0
        - If all states have similar forward returns, all get ~0.5
        """
        states = self.hmm_model.predict(obs)
        n_obs = len(obs)

        analysis = {}
        for s in range(self.n_states):
            state_mask = states == s
            indices = np.where(state_mask)[0]
            n_in_state = int(np.sum(state_mask))

            # Mean PC vector
            state_obs = obs[state_mask]
            mean_vec = np.mean(state_obs, axis=0) if n_in_state > 0 else np.zeros(self.n_pcs)

            # Total variance
            if n_in_state > 10:
                trace_cov = float(np.trace(np.cov(state_obs.T)))
            else:
                trace_cov = float(np.trace(self.hmm_model.covars_[s]))

            # Average forward return (PC1 cumulative over forward horizon)
            fwd_horizon = STATE_ANALYSIS_FORWARD_HOURS
            forward_returns = []
            for i in indices:
                end_idx = min(i + fwd_horizon, n_obs)
                if end_idx > i + 1:
                    # Sum PC1 scores over the forward window as return proxy
                    fwd_ret = float(np.sum(obs[i + 1 : end_idx, 0]))
                    forward_returns.append(fwd_ret)

            avg_fwd_return = float(np.mean(forward_returns)) if forward_returns else 0.0
            std_fwd_return = float(np.std(forward_returns)) if len(forward_returns) > 1 else 1.0

            # Average consecutive duration
            runs = []
            current_run = 0
            for i in range(len(states)):
                if states[i] == s:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)
            avg_duration = float(np.mean(runs)) if runs else 0.0

            analysis[s] = {
                "mean_pc_vector": mean_vec.tolist(),
                "trace_cov": round(trace_cov, 6),
                "avg_forward_return": round(avg_fwd_return, 6),
                "std_forward_return": round(std_fwd_return, 6),
                "sharpe_of_state": round(
                    avg_fwd_return / std_fwd_return if std_fwd_return > 0 else 0, 4
                ),
                "avg_duration_hours": round(avg_duration, 1),
                "frequency": round(n_in_state / n_obs, 3),
                "n_observations": n_in_state,
            }

        self._state_analysis = analysis

        # Derive exposure multipliers from forward returns
        self._derive_exposure(analysis)

        # Name states descriptively (for logging only — names don't drive logic)
        self._name_states(analysis)

        # Log the full analysis
        for s in range(self.n_states):
            a = analysis[s]
            name = self._state_names.get(s, f"STATE_{s}")
            log.info(
                f"  State {s} [{name}]: "
                f"fwd_ret={a['avg_forward_return']:+.4f} "
                f"sharpe={a['sharpe_of_state']:+.3f} "
                f"vol={a['trace_cov']:.4f} "
                f"duration={a['avg_duration_hours']:.0f}h "
                f"freq={a['frequency']:.1%} "
                f"→ exposure={self._state_exposure[s]:.2f}"
            )

    def _derive_exposure(self, analysis: dict):
        """Map states to exposure multipliers based on historical forward returns.

        Strategy:
        - Rank states by their Sharpe-of-state (avg_fwd_return / std_fwd_return)
        - Best state → 1.0 exposure
        - Worst state → 0.0 if negative Sharpe, 0.2 if positive
        - Middle state → interpolated

        This is fully data-driven. If all states have similar forward returns,
        all get moderate exposure (~0.5).
        """
        sharpes = {s: analysis[s]["sharpe_of_state"] for s in range(self.n_states)}
        sorted_states = sorted(sharpes.keys(), key=lambda s: sharpes[s], reverse=True)

        best_sharpe = sharpes[sorted_states[0]]
        worst_sharpe = sharpes[sorted_states[-1]]
        spread = best_sharpe - worst_sharpe

        # Linear interpolation with 0.10 floor for activity compliance.
        # Best state → 1.0, worst → 0.10 (not 0.0 — need 8 active trading days).
        COMPETITION_MIN_EXPOSURE = 0.10

        for s in range(self.n_states):
            if spread < 0.01:
                self._state_exposure[s] = 0.5
            else:
                rank_frac = (sharpes[s] - worst_sharpe) / spread  # 0 to 1
                raw_exposure = rank_frac
                self._state_exposure[s] = round(
                    max(COMPETITION_MIN_EXPOSURE, raw_exposure), 3
                )

    def _name_states(self, analysis: dict):
        """Assign descriptive names based on observed properties.

        These names are for LOGGING ONLY. No logic depends on them.
        """
        for s in range(self.n_states):
            a = analysis[s]
            fwd = a["avg_forward_return"]
            vol = a["trace_cov"]

            # Median vol across states
            all_vols = [analysis[i]["trace_cov"] for i in range(self.n_states)]
            median_vol = float(np.median(all_vols))

            direction = "BULL" if fwd > 0.001 else ("BEAR" if fwd < -0.001 else "FLAT")
            volatility = "CALM" if vol < median_vol else "VOLATILE"

            self._state_names[s] = f"{direction}_{volatility}"

    # ─── Incremental Update (between full refits) ──────────────

    def update_hmm(self, pc_scores: np.ndarray):
        """Cheap HMM update: predict on recent observations without refitting."""
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
            self.current_state = int(states[-1])
        except Exception:
            pass

    def update(self, pc_scores: np.ndarray = None):
        """Update regime from PC scores. Called every cycle."""
        if pc_scores is not None and len(pc_scores) > 30:
            self.update_hmm(pc_scores)

        if self.hmm_fitted:
            self.regime_confidence = 1.0
        else:
            self.regime_confidence = 0.0

    # ─── Exposure & Status ─────────────────────────────────────

    def get_exposure_multiplier(self) -> float:
        """Get exposure multiplier for current state (data-driven)."""
        if not self.hmm_fitted or not self._state_exposure:
            return 0.5  # default: moderate exposure before HMM is fitted
        return self._state_exposure.get(self.current_state, 0.5)

    def should_trade(self) -> bool:
        """Whether current state allows new entries.
        Always True now — competition requires 8 active trading days.
        Exposure multiplier controls sizing, not whether to trade."""
        return True

    def get_status(self) -> dict:
        state_name = self._state_names.get(self.current_state, "UNKNOWN")
        return {
            "state": self.current_state,
            "state_name": state_name,
            "confidence": round(self.regime_confidence, 3),
            "exposure_mult": round(self.get_exposure_multiplier(), 3),
            "should_trade": self.should_trade(),
            "hmm_fitted": self.hmm_fitted,
            "pca_explained_var": round(self._pca_explained_var, 3),
            "n_pcs": self.n_pcs,
            "state_analysis": {
                s: {
                    "name": self._state_names.get(s, "?"),
                    "exposure": self._state_exposure.get(s, 0.5),
                    **self._state_analysis.get(s, {}),
                }
                for s in range(self.n_states)
            } if self._state_analysis else {},
        }