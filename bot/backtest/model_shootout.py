#!/usr/bin/env python3
"""
Model shootout: which ML model (if any) can rank coins better than EWMA alone?

Tests 6 approaches on the SAME data with temporal walk-forward evaluation:
  1. EWMA only (baseline — no ML)
  2. Ridge regression
  3. ElasticNet
  4. Lasso
  5. Random Forest
  6. XGBoost (gradient boosting)

Evaluation metric: NOT R² (which is always ~0 for absolute returns).
Instead we measure RANKING QUALITY:
  - Spearman rank correlation: does predicted ranking match actual ranking?
  - Top-3 hit rate: what fraction of predicted top-3 coins are in actual top-3?
  - Information Coefficient (IC): correlation between predicted and actual returns

Target: RELATIVE return (coin_return - median_return) to focus on cross-sectional spread.

Walk-forward: train on 400h rolling window, predict next 24h, step by 24h.
No future leakage.

Run: venv/bin/python -m bot.backtest.model_shootout
"""
import sys
import os
import time as pytime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import requests
from scipy.stats import spearmanr
from bot.config import BINANCE_BASE_URL, BINANCE_SYMBOL_MAP, TRADEABLE_COINS, LASSO_FEATURES
from bot.features import compute_coin_features, zscore_universe, compute_ewma_momentum

# Models
from sklearn.linear_model import Ridge, ElasticNet, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def download_klines(symbol, session, months=4):
    end_ms = int(pytime.time() * 1000)
    start_ms = end_ms - (months * 30 * 24 * 3_600_000)
    candles = []
    cur = start_ms
    while cur < end_ms:
        try:
            r = session.get(f"{BINANCE_BASE_URL}/api/v3/klines",
                            params={"symbol": symbol, "interval": "1h",
                                    "startTime": cur, "endTime": end_ms, "limit": 1000},
                            timeout=15)
            r.raise_for_status()
            raw = r.json()
        except Exception:
            break
        if not raw or not isinstance(raw, list):
            break
        for k in raw:
            candles.append({"close": float(k[4]), "high": float(k[2]),
                            "low": float(k[3]), "volume": float(k[5])})
        cur = raw[-1][0] + 3_600_000
        if len(raw) < 1000:
            break
        pytime.sleep(0.05)
    return candles


def build_features_and_targets(all_data, pairs, t, forward=24):
    """Build z-scored features and relative return targets at time t."""
    raw_features = {}
    for pair in pairs:
        hist = all_data[pair][:t + 1]
        if len(hist) < 100:
            continue
        closes = np.array([x["close"] for x in hist])
        highs = np.array([x["high"] for x in hist])
        lows = np.array([x["low"] for x in hist])
        volumes = np.array([x["volume"] for x in hist])
        feats = compute_coin_features(closes, highs, lows, volumes, closes[-1], closes[-1])
        if feats:
            # Also compute EWMA score
            feats["ewma_score"] = compute_ewma_momentum(closes)
            raw_features[pair] = feats

    if len(raw_features) < 10:
        return None, None, None, None

    zscored = zscore_universe(raw_features, feature_keys=LASSO_FEATURES)

    # Compute forward relative returns
    forward_returns = {}
    for pair in raw_features:
        future_idx = t + forward
        if future_idx >= len(all_data[pair]):
            continue
        cp = all_data[pair][t]["close"]
        fp = all_data[pair][future_idx]["close"]
        if cp > 0:
            forward_returns[pair] = (fp - cp) / cp

    if len(forward_returns) < 10:
        return None, None, None, None

    median_ret = np.median(list(forward_returns.values()))
    relative_returns = {p: r - median_ret for p, r in forward_returns.items()}

    # Build aligned arrays
    valid_pairs = [p for p in raw_features if p in relative_returns and p in zscored]
    X = np.array([[zscored[p].get(k, 0.0) for k in LASSO_FEATURES] for p in valid_pairs])
    y = np.array([relative_returns[p] for p in valid_pairs])
    ewma_scores = np.array([raw_features[p]["ewma_score"] for p in valid_pairs])

    return valid_pairs, X, y, ewma_scores


def evaluate_ranking(predicted_scores, actual_returns, pairs, top_k=3):
    """Evaluate ranking quality."""
    if len(predicted_scores) < 5:
        return {}

    # Spearman rank correlation
    spearman_corr, spearman_p = spearmanr(predicted_scores, actual_returns)

    # Information coefficient (Pearson correlation)
    ic = float(np.corrcoef(predicted_scores, actual_returns)[0, 1])

    # Top-K hit rate: what fraction of predicted top-K are in actual top-K?
    pred_top = set(np.argsort(predicted_scores)[-top_k:])
    actual_top = set(np.argsort(actual_returns)[-top_k:])
    hit_rate = len(pred_top & actual_top) / top_k

    # Top-K average relative return (how much did our picks actually return?)
    pred_top_indices = np.argsort(predicted_scores)[-top_k:]
    top_k_return = float(np.mean(actual_returns[pred_top_indices]))

    return {
        "spearman": round(float(spearman_corr), 4),
        "ic": round(ic, 4),
        "hit_rate": round(hit_rate, 4),
        "top_k_return": round(top_k_return, 6),
    }


def main():
    print("=" * 70)
    print("  MODEL SHOOTOUT: which model ranks coins best?")
    print("  Target: relative returns (cross-sectional spread)")
    print("  Metric: Spearman correlation + Top-3 hit rate")
    print("=" * 70)

    # Download data
    session = requests.Session()
    session.timeout = 15
    all_data = {}
    pairs_to_use = [p for p in TRADEABLE_COINS if p in BINANCE_SYMBOL_MAP]

    print(f"\n  Downloading 4 months of 1h data for {len(pairs_to_use)} pairs...")
    for i, pair in enumerate(pairs_to_use):
        symbol = BINANCE_SYMBOL_MAP[pair]
        candles = download_klines(symbol, session, months=4)
        if len(candles) >= 600:
            all_data[pair] = candles
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(pairs_to_use)} loaded...")

    pairs = list(all_data.keys())
    min_len = min(len(v) for v in all_data.values())
    print(f"  Loaded {len(pairs)} pairs, {min_len} bars (~{min_len//24}d)")

    # Define models
    models = {
        "EWMA_only": None,  # baseline
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
        "Lasso": LassoCV(alphas=np.logspace(-5, -1, 50), cv=3, max_iter=10000),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=4,
                                               min_samples_leaf=20, random_state=42),
        "GradientBoost": GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                    learning_rate=0.05, random_state=42),
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                               learning_rate=0.05, verbosity=0,
                                               random_state=42)

    # Walk-forward evaluation
    train_window = 400
    forward = 24
    step = 24  # non-overlapping evaluation windows
    start = 600  # warmup

    results = {name: [] for name in models}

    n_windows = (min_len - start - forward) // step
    print(f"\n  Walk-forward: {n_windows} evaluation windows (train={train_window}h, fwd={forward}h)")
    print(f"  {'Model':<15} {'Spearman':>10} {'IC':>8} {'Hit@3':>8} {'TopK Ret':>10}")
    print(f"  {'─' * 55}")

    for window_idx in range(n_windows):
        t = start + window_idx * step

        # Build test data (at time t, predict t+24)
        test_pairs, X_test, y_test, ewma_test = build_features_and_targets(
            all_data, pairs, t, forward
        )
        if test_pairs is None:
            continue

        # Build training data (rolling window before t)
        X_train_all, y_train_all = [], []
        for train_t in range(max(100, t - train_window), t, step):
            tp, Xt, yt, _ = build_features_and_targets(all_data, pairs, train_t, forward)
            if tp is not None:
                X_train_all.append(Xt)
                y_train_all.append(yt)

        if not X_train_all:
            continue
        X_train = np.vstack(X_train_all)
        y_train = np.concatenate(y_train_all)

        if len(X_train) < 50:
            continue

        for name, model in models.items():
            if name == "EWMA_only":
                # Baseline: rank by EWMA score
                predicted = ewma_test
            else:
                try:
                    m = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                    m.fit(X_train, y_train)
                    predicted = m.predict(X_test)
                except Exception:
                    predicted = np.zeros(len(X_test))

            metrics = evaluate_ranking(predicted, y_test, test_pairs, top_k=3)
            if metrics:
                results[name].append(metrics)

    # Print results
    print(f"\n{'=' * 70}")
    print(f"  RESULTS (averaged over {n_windows} windows)")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<15} {'Spearman':>10} {'IC':>8} {'Hit@3':>8} {'TopK Ret':>10} {'Windows':>8}")
    print(f"  {'─' * 60}")

    best_model = None
    best_spearman = -1

    for name in models:
        r = results[name]
        if not r:
            print(f"  {name:<15} {'N/A':>10}")
            continue
        avg_sp = np.mean([x["spearman"] for x in r])
        avg_ic = np.mean([x["ic"] for x in r])
        avg_hr = np.mean([x["hit_rate"] for x in r])
        avg_tr = np.mean([x["top_k_return"] for x in r])

        is_best = avg_sp > best_spearman
        if is_best:
            best_spearman = avg_sp
            best_model = name

        marker = " ◀ BEST" if is_best else ""
        print(f"  {name:<15} {avg_sp:>+9.4f} {avg_ic:>+7.4f} {avg_hr:>7.1%} {avg_tr:>+9.6f} {len(r):>8}{marker}")

    # Statistical significance: is the best model significantly better than EWMA?
    print(f"\n  {'─' * 60}")
    if best_model and best_model != "EWMA_only":
        ewma_sp = [x["spearman"] for x in results["EWMA_only"]]
        best_sp = [x["spearman"] for x in results[best_model]]
        # Paired t-test
        from scipy.stats import ttest_rel
        min_n = min(len(ewma_sp), len(best_sp))
        if min_n > 5:
            t_stat, p_val = ttest_rel(best_sp[:min_n], ewma_sp[:min_n])
            print(f"  {best_model} vs EWMA: t={t_stat:.3f}, p={p_val:.4f}")
            if p_val < 0.05:
                print(f"  → {best_model} is SIGNIFICANTLY better than EWMA (p<0.05)")
            else:
                print(f"  → NOT significantly better (p={p_val:.2f}). EWMA is fine.")
    else:
        print(f"  EWMA is the best or tied. No ML model adds value.")

    print(f"\n  Random baseline: Spearman=0.000, Hit@3=~{3/len(pairs)*100:.0f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
