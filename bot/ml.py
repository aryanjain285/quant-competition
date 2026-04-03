"""
ML model trainer — Finals v7: optional Lasso running in background.

The Lasso is NOT the primary signal. It runs every ML_RETRAIN_INTERVAL cycles,
and if it finds genuine signal (R² > ML_MIN_R2), it boosts the momentum score.
If it finds nothing (R² ≈ 0, all features zeroed), the bot still trades normally
using the momentum composite.

This addresses the v6 failure where Lasso zeroing all features caused the bot
to hold cash for 22 consecutive days.
"""
import numpy as np
from typing import Optional
from bot.logger import get_logger
from bot.config import LASSO_FEATURES, ML_LOOKBACK_HOURS, ML_FORWARD_HORIZON, ML_SAMPLE_INTERVAL
from bot.features import compute_coin_features, zscore_universe

log = get_logger("ml")

try:
    from sklearn.linear_model import LassoCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not installed. Lasso disabled.")


class LassoTrainer:
    """Trains optional Lasso model for score boosting."""

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
        """Train Lasso, return (model, R²).

        Returns (None, 0.0) if insufficient data or training fails.
        The caller checks R² to decide whether to activate the boost.
        """
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
                hist = historical_data[pair][: t + 1]
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
            for pair, fz in zscored.items():
                if pair not in historical_data:
                    continue
                future_idx = t + forward_horizon
                if future_idx >= len(historical_data[pair]):
                    continue
                current_price = historical_data[pair][t]["close"]
                future_price = historical_data[pair][future_idx]["close"]
                if current_price <= 0:
                    continue

                target = (future_price - current_price) / current_price
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

        model = LassoCV(
            alphas=np.logspace(-5, -1, 100),
            cv=5,
            max_iter=10000,
            random_state=42,
        )
        model.fit(X, y)

        r2 = float(model.score(X, y))
        nonzero = [
            (feature_cols[i], round(float(model.coef_[i]), 6))
            for i in range(len(feature_cols))
            if abs(model.coef_[i]) > 1e-8
        ]
        zeroed = [
            feature_cols[i]
            for i in range(len(feature_cols))
            if abs(model.coef_[i]) <= 1e-8
        ]

        self.last_model = model
        self.last_r2 = r2
        self.last_nonzero_features = [name for name, _ in nonzero]

        log.info(
            f"ML: {len(X_train)} samples, alpha={model.alpha_:.6f}, R²={r2:.4f}"
        )
        if nonzero:
            log.info(f"  Active: {nonzero}")
        if zeroed:
            log.info(f"  Zeroed: {zeroed}")

        return model, r2