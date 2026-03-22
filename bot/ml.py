# bot/ml_model.py
import numpy as np
from bot.logger import get_logger
from bot.features import compute_coin_features, zscore_universe

log = get_logger("ml_model")

try:
    from sklearn.linear_model import RidgeCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not installed. Ridge regression disabled.")

class RidgeTrainer:
    def __init__(self, active_pairs):
        self.active_pairs = active_pairs

    def train(self, historical_data: dict, lookback=400, forward_horizon=24):
        if not SKLEARN_AVAILABLE:
            return None, None

        log.info(f"Training Ridge ML Model (Lookback: {lookback}h, Target: +{forward_horizon}h)...")
        
        X_train, y_train = [], []
        feature_cols = None
        
        # Ensure we have enough data
        min_len = min((len(historical_data[p]) for p in self.active_pairs if p in historical_data), default=0)
        if min_len < lookback + forward_horizon + 50:
            log.warning(f"Insufficient data for ML training. Have {min_len} bars.")
            return None, None
            
        start_train = min_len - lookback - forward_horizon
        end_train = min_len - forward_horizon
        
        for t in range(start_train, end_train, 6): # Sample every 6 hours
            raw_features = {}
            for pair in self.active_pairs:
                if pair not in historical_data: continue
                hist = historical_data[pair][:t+1]
                closes = np.array([x["close"] for x in hist])
                if len(closes) < 100: continue
                
                highs = np.array([x["high"] for x in hist])
                lows = np.array([x["low"] for x in hist])
                volumes = np.array([x["volume"] for x in hist])
                
                feats = compute_coin_features(closes, highs, lows, volumes, closes[-1], closes[-1])
                if feats: raw_features[pair] = feats
                    
            zscored = zscore_universe(raw_features)
            
            for pair, f_z in zscored.items():
                if feature_cols is None:
                    # Capture valid float features to map exactly to ranking.py expectations
                    feature_cols = [k for k, v in f_z.items() if isinstance(v, float) and not k.startswith('_')]
                    
                target_return = (historical_data[pair][t + forward_horizon]["close"] - historical_data[pair][t]["close"]) / historical_data[pair][t]["close"]
                
                X_train.append([f_z.get(k, 0.0) for k in feature_cols])
                y_train.append(target_return)
                
        if len(X_train) < 50:
            return None, None
            
        model = RidgeCV(alphas=np.logspace(-3, 1, 100))
        model.fit(X_train, y_train)
        return model, feature_cols