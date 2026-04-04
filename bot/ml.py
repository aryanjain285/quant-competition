"""
ML model trainer — Finals v7: Ridge regression on relative returns.

Ridge is the PRIMARY ranker (not optional boost). Shootout results:
  Ridge: Spearman +0.020, TopK return +0.47%, p=0.025 vs EWMA
  EWMA:  Spearman -0.050 (negative — ranks coins backwards)

Target: coin_return - median(all_coin_returns) over 24h forward.
This removes market direction and focuses on cross-sectional spread.

Ridge over Lasso: Lasso zeros all features (too aggressive sparsity).
Ridge keeps all features with shrunk coefficients — better for small
sample sizes with correlated features (typical in crypto).
"""
import numpy as np
from typing import Optional
from bot.logger import get_logger
from bot.config import LASSO_FEATURES, ML_LOOKBACK_HOURS, ML_FORWARD_HORIZON, ML_SAMPLE_INTERVAL
from bot.features import compute_coin_features, zscore_universe

log = get_logger("ml")

try:
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not installed. Ridge disabled.")


class RidgeTrainer:
    """Trains Ridge regression on cross-sectional relative returns."""

    def __init__(self, active_pairs: list[str]):
        self.active_pairs = active_pairs
        self.last_model = None
        self.last_r2: float = 0.0
        self.last_nonzero_features: list[str] = []

    def train(
        self,
        historical_data: dict[str, list[dict]],
        lookback: int = ML_LOOKBACK_HOURS,
        forward_horizon: int = ML_FORWARD_HORIZON,
        sample_interval: int = ML_SAMPLE_INTERVAL,
    ) -> tuple[Optional[object], float]:
        """Train Ridge on relative returns. Returns (model, R²)."""
        if not SKLEARN_AVAILABLE:
            return None, 0.0

        feature_cols = list(LASSO_FEATURES)

        min_len = min(
            (len(historical_data[p]) for p in self.active_pairs if p in historical_data),
            default=0,
        )
        min_required = lookback + forward_horizon + 100
        if min_len < min_required:
            log.info(f"ML: insufficient data ({min_len}/{min_required} bars). Skipping.")
            return None, 0.0

        X_train, y_train = [], []
        start_t = min_len - lookback - forward_horizon
        end_t = min_len - forward_horizon

        for t in range(start_t, end_t, sample_interval):
            raw_features = {}
            for pair in self.active_pairs:
                if pair not in historical_data:
                    continue
                hist = historical_data[pair][:t + 1]
                if len(hist) < 100:
                    continue
                closes = np.array([x["close"] for x in hist])
                highs = np.array([x["high"] for x in hist])
                lows = np.array([x["low"] for x in hist])
                volumes = np.array([x["volume"] for x in hist])
                feats = compute_coin_features(closes, highs, lows, volumes, closes[-1], closes[-1])
                if feats:
                    raw_features[pair] = feats

            if len(raw_features) < 5:
                continue

            zscored = zscore_universe(raw_features, feature_keys=feature_cols)

            # Compute forward relative returns
            forward_returns = {}
            for pair in zscored:
                if pair not in historical_data:
                    continue
                future_idx = t + forward_horizon
                if future_idx >= len(historical_data[pair]):
                    continue
                cp = historical_data[pair][t]["close"]
                fp = historical_data[pair][future_idx]["close"]
                if cp > 0:
                    forward_returns[pair] = (fp - cp) / cp

            if len(forward_returns) < 5:
                continue

            median_ret = float(np.median(list(forward_returns.values())))

            for pair, fz in zscored.items():
                if pair not in forward_returns:
                    continue
                target = forward_returns[pair] - median_ret
                x = [fz.get(k, 0.0) for k in feature_cols]
                if any(not np.isfinite(v) for v in x) or not np.isfinite(target):
                    continue
                X_train.append(x)
                y_train.append(target)

        if len(X_train) < 50:
            log.info(f"ML: only {len(X_train)} samples. Skipping.")
            return None, 0.0

        X = np.array(X_train)
        y = np.array(y_train)

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        r2 = float(model.score(X, y))
        self.last_model = model
        self.last_r2 = r2

        # Log coefficients
        coefs = [(feature_cols[i], round(float(model.coef_[i]), 6))
                 for i in range(len(feature_cols))]
        coefs_sorted = sorted(coefs, key=lambda x: abs(x[1]), reverse=True)
        self.last_nonzero_features = [name for name, c in coefs if abs(c) > 1e-8]

        log.info(f"Ridge trained: {len(X_train)} samples, R²={r2:.4f}")
        log.info(f"  Top coefs: {coefs_sorted[:5]}")

        return model, r2
