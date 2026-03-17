"""
ML confidence gate using LightGBM.

Acts as a meta-model on top of rule-based signals:
  Rule fires → ML predicts P(profitable) → only enter if P > threshold

CRITICAL DESIGN PRINCIPLE — NO TRAIN-SERVE SKEW:
  All features are computed identically between training and live inference.
  Features that depend on candle timeframe (EMA, ATR, RSI) are computed
  by resampling to hourly resolution first, since training always uses
  hourly candles. Features that are scale-invariant (returns, ratios)
  use explicit bars_per_hour (bph) scaling.

  Training (pretrain_ml.py):  hourly candles, bph=1  → indicators on hourly
  Live (main.py):             5-min candles, bph=12  → resample to hourly → same indicators

Supports ensemble prediction (average of multiple models) for robustness.
"""
import os
import numpy as np
from typing import Optional
from bot.logger import get_logger

log = get_logger("ml_model")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    log.warning("lightgbm not installed — ML confidence gate disabled")


# ─── Indicators (self-contained, no import from signals.py) ────
# Duplicated here intentionally to break the dependency on signals.py
# and guarantee identical computation between training and inference.

def _ema(prices: np.ndarray, span: int) -> np.ndarray:
    if len(prices) < 2:
        return prices.copy()
    alpha = 2.0 / (span + 1)
    out = np.empty_like(prices, dtype=float)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 2:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
    losses = np.mean(-deltas[deltas < 0]) if np.any(deltas < 0) else 0.0
    if losses == 0:
        return 100.0
    rs = gains / losses
    return float(100 - (100 / (1 + rs)))


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 0.0
    if len(highs) >= period + 1 and len(lows) >= period + 1:
        tr = np.maximum(
            highs[-period:] - lows[-period:],
            np.maximum(
                np.abs(highs[-period:] - closes[-period - 1:-1]),
                np.abs(lows[-period:] - closes[-period - 1:-1]),
            ),
        )
    else:
        tr = np.abs(np.diff(closes[-(period + 1):]))
    return float(np.mean(tr))


def _resample_to_hourly(arr: np.ndarray, bph: int, agg: str = "last") -> np.ndarray:
    """Resample a bar-level array to hourly resolution.

    Args:
        arr: raw bar-level data (e.g. 5-min closes)
        bph: bars per hour (12 for 5-min, 1 for hourly = no-op)
        agg: aggregation method — 'last' (closes), 'max' (highs), 'min' (lows)

    Trims from the START so the last block always includes the most recent bar.
    """
    if bph <= 1:
        return arr
    trim_start = len(arr) % bph
    trimmed = arr[trim_start:] if trim_start > 0 else arr
    blocks = trimmed.reshape(-1, bph)
    if agg == "last":
        return blocks[:, -1]
    elif agg == "max":
        return blocks.max(axis=1)
    elif agg == "min":
        return blocks.min(axis=1)
    return blocks[:, -1]


class MLConfidenceGate:
    """LightGBM ensemble confidence gate for trading signals."""

    FEATURE_NAMES = [
        "rsi", "ema_dist_fast", "ema_dist_slow", "atr_ratio",
        "volume_ratio", "ret_1h", "ret_6h", "ret_24h",
        "realized_vol", "downside_vol",
        "funding_zscore", "oi_change_pct",
        "btc_1h_return", "regime",
        "breakout_strength", "spread_pct",
    ]
    NUM_FEATURES = len(FEATURE_NAMES)

    def __init__(self, model_path: Optional[str] = None):
        self.models: list = []
        self.is_trained = False
        self.threshold = 0.55

        # Online training accumulator
        self.training_features: list[np.ndarray] = []
        self.training_labels: list[int] = []
        self.min_training_samples = 50

        if model_path:
            self._load_models(model_path)

    def _load_models(self, base_path: str):
        """Load ensemble models. Tries _ens0, _ens1, _ens2, falls back to single."""
        if not LGB_AVAILABLE:
            return

        loaded = []
        for i in range(3):
            ens_path = base_path.replace(".txt", f"_ens{i}.txt")
            if os.path.exists(ens_path):
                try:
                    loaded.append(lgb.Booster(model_file=ens_path))
                except Exception as e:
                    log.debug(f"Failed to load ensemble model {i}: {e}")

        if not loaded and os.path.exists(base_path):
            try:
                loaded.append(lgb.Booster(model_file=base_path))
            except Exception as e:
                log.warning(f"Failed to load ML model: {e}")

        if loaded:
            self.models = loaded
            self.is_trained = True
            log.info(f"ML ensemble loaded: {len(self.models)} models from {base_path}")

    # ─── Feature Engineering ────────────────────────────────────

    def build_features(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        signal: dict,
        deriv_signal: dict,
        btc_1h_return: float,
        regime: int,
        bars_per_hour: int = 12,
    ) -> Optional[np.ndarray]:
        """Build feature vector with exact train-serve parity.

        STRATEGY FOR EACH FEATURE:
          - RSI, EMA, ATR: resample to hourly, compute with period 14/21/55.
            These indicators have timeframe-dependent behavior — you can't just
            scale the period because the price dynamics within each bar differ.
          - Returns (1h, 6h, 24h): use bph scaling on raw data.
            Point-to-point returns are scale-invariant.
          - Volume ratio: use bph scaling on raw data.
            Averaging over bph*N bars ≈ averaging over N hours.
          - Volatility: compute on raw data with bph-aware annualization.
            std(5min returns) * sqrt(288*365) ≈ std(1h returns) * sqrt(24*365).
          - Breakout strength: read from signal dict (already uses 72h lookback).
          - Derivatives, regime, spread: read from signal/deriv dict (no timeframe dependency).

        Args:
            bars_per_hour: 12 for 5-min candles, 1 for hourly. Controls resampling.
        """
        bph = bars_per_hour
        min_bars = 25 * bph  # need at least 25 hours of data

        if len(closes) < min_bars:
            return None

        price = closes[-1]
        if price <= 0:
            return None

        # ── Resample to hourly for indicator computation ──
        # Trim from START so last block includes most recent bar
        hourly_c = _resample_to_hourly(closes, bph, "last")
        hourly_h = _resample_to_hourly(highs, bph, "max")
        hourly_l = _resample_to_hourly(lows, bph, "min")

        if len(hourly_c) < 16:  # need 14+2 for RSI
            return None

        # ── RSI on hourly (training: rsi(hourly, 14) = 14-hour RSI) ──
        rsi_val = _rsi(hourly_c, 14)

        # ── EMA distances on hourly ──
        # Training: ema(hourly, 21) and ema(hourly, 55)
        # We compute on hourly resampled data with same periods
        ef = _ema(hourly_c[-min(len(hourly_c), 60):], 21)
        es = _ema(hourly_c[-min(len(hourly_c), 100):], 55)
        ema_dist_fast = (price - ef[-1]) / price if len(ef) > 0 else 0.0
        ema_dist_slow = (price - es[-1]) / price if len(es) > 0 else 0.0

        # ── ATR on hourly (training: atr(hourly_OHLC, 14) = 14-hour ATR) ──
        atr_val = _atr(hourly_h, hourly_l, hourly_c, 14)
        atr_ratio = atr_val / price if atr_val > 0 else 0.0

        # ── Volume ratio: 50-HOUR average (bph scaling on raw data) ──
        vol_lookback = 50 * bph
        if len(volumes) >= vol_lookback and np.mean(volumes[-vol_lookback:-1]) > 0:
            vol_ratio = float(volumes[-1] / np.mean(volumes[-vol_lookback:-1]))
        else:
            vol_ratio = 1.0

        # ── Returns at fixed hour intervals (bph scaling) ──
        ret_1h = (closes[-1] / closes[-1 * bph - 1] - 1) if len(closes) > 1 * bph + 1 else 0
        ret_6h = (closes[-1] / closes[-6 * bph - 1] - 1) if len(closes) > 6 * bph + 1 else 0
        ret_24h = (closes[-1] / closes[-24 * bph - 1] - 1) if len(closes) > 24 * bph + 1 else 0

        # ── Volatility: 24-hour window, bph-aware annualization ──
        # std(5min_rets) * sqrt(288*365) ≈ std(hourly_rets) * sqrt(24*365)
        # This is mathematically equivalent due to variance scaling with time.
        vol_window = 24 * bph
        if len(closes) > vol_window + 1:
            log_rets = np.diff(np.log(closes[-(vol_window + 1):]))
            # Annualize: periods_per_year = len(log_rets) * (365 * 24 / 24)
            # For hourly (24 rets): sqrt(24 * 365) ≈ 93.6
            # For 5-min (288 rets): sqrt(288 * 365) ≈ 324.2
            # Both produce same annualized vol because std scales inversely with sqrt(bph)
            ann_factor = np.sqrt(len(log_rets) * 365)
            real_vol = float(np.std(log_rets)) * ann_factor
            neg_lr = log_rets[log_rets < 0]
            down_vol = float(np.std(neg_lr)) * ann_factor if len(neg_lr) > 0 else 0.3
        else:
            real_vol, down_vol = 0.5, 0.3

        # ── Derivatives (0 from training, real values live — model ignores) ──
        funding_z = deriv_signal.get("funding_zscore", 0.0)
        oi_change = deriv_signal.get("oi_change_pct", 0.0)

        # ── Regime: binary 0/1 matching training heuristic ──
        regime_binary = 1.0 if regime >= 1 else 0.0

        # ── Breakout strength: from signal dict (72h lookback, scale-invariant ratio) ──
        breakout_str = signal.get("breakout_strength", 0.0)

        # ── Spread: from signal dict (0 in training, real live — model ignores) ──
        spread = signal.get("spread", 0.0)

        features = np.array([
            rsi_val, ema_dist_fast, ema_dist_slow, atr_ratio,
            vol_ratio, ret_1h, ret_6h, ret_24h,
            real_vol, down_vol,
            funding_z, oi_change,
            btc_1h_return, regime_binary,
            breakout_str, spread,
        ], dtype=np.float64)

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # ─── Prediction ─────────────────────────────────────────────

    def predict_confidence(self, features: np.ndarray) -> float:
        """Predict P(profitable) using ensemble average."""
        if not self.is_trained or not self.models or not LGB_AVAILABLE:
            return 0.5

        try:
            preds = [m.predict(features.reshape(1, -1))[0] for m in self.models]
            return float(np.clip(np.mean(preds), 0, 1))
        except Exception as e:
            log.debug(f"ML predict failed: {e}")
            return 0.5

    def should_enter(self, features: np.ndarray) -> tuple[bool, float]:
        """Check if ML ensemble approves this entry."""
        conf = self.predict_confidence(features)
        return conf >= self.threshold, conf

    # ─── Online Training ────────────────────────────────────────

    def record_trade(self, features: np.ndarray, profitable: bool):
        if features is None or len(features) != self.NUM_FEATURES:
            return
        self.training_features.append(features.copy())
        self.training_labels.append(1 if profitable else 0)

    def retrain_if_ready(self) -> bool:
        """Retrain from accumulated live trades. Appends to ensemble."""
        if not LGB_AVAILABLE:
            return False
        if len(self.training_features) < self.min_training_samples:
            return False

        try:
            X = np.array(self.training_features)
            y = np.array(self.training_labels)

            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            if len(np.unique(y_train)) < 2:
                return False

            train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.FEATURE_NAMES)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            params = {
                "objective": "binary", "metric": "binary_logloss",
                "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
                "min_child_samples": 10, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 5,
                "verbose": -1, "is_unbalance": True, "feature_pre_filter": False,
            }

            new_model = lgb.train(
                params, train_data, num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
            )

            # Append to ensemble (don't replace pre-trained models)
            self.models.append(new_model)
            self.is_trained = True

            importance = dict(zip(self.FEATURE_NAMES, new_model.feature_importance()))
            top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            log.info(f"ML retrained on {len(X)} live samples, ensemble now "
                     f"{len(self.models)} models. Top features: {top}")
            return True

        except Exception as e:
            log.error(f"ML retraining failed: {e}")
            return False

    def save_model(self, path: str):
        if self.models and self.is_trained:
            try:
                self.models[-1].save_model(path)
                log.info(f"ML model saved to {path}")
            except Exception as e:
                log.warning(f"Failed to save ML model: {e}")

    def get_status(self) -> dict:
        return {
            "is_trained": self.is_trained,
            "ensemble_size": len(self.models),
            "training_samples": len(self.training_features),
            "threshold": self.threshold,
            "available": LGB_AVAILABLE,
        }