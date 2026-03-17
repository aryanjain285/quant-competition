#!/usr/bin/env python3
"""
Pre-train ML confidence gate on 6 months of Binance data.

Fixes applied:
  1. TEMPORAL CV: dataset is sorted by timestamp across ALL coins,
     so walk-forward splits are truly temporal (no future leakage).
  2. HOUR-BASED FEATURES: all lookbacks use explicit hour counts,
     matching the live bot's bars_per_hour=1 for hourly training data.
  3. REGIME: binary 0/1 matching both training and live inference.
  4. ENSEMBLE: all 3 models saved and loaded at inference.
  5. LABELING: realistic trade simulation with stops and targets.

Run: venv/bin/python -m bot.backtest.pretrain_ml
"""
import sys
import os
import time
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import requests

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed"); sys.exit(1)

from sklearn.metrics import roc_auc_score
from bot.config import BINANCE_BASE_URL, BINANCE_SYMBOL_MAP
from bot.signals import ema, rsi, atr
from bot.ml_model import MLConfidenceGate
from bot.logger import get_logger

log = get_logger("pretrain")

MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_model.txt")

TRAINING_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD", "XRP/USD",
    "DOGE/USD", "ADA/USD", "AVAX/USD", "LINK/USD", "DOT/USD",
    "SUI/USD", "NEAR/USD", "LTC/USD", "UNI/USD", "FET/USD",
    "HBAR/USD", "FIL/USD", "APT/USD", "ARB/USD", "AAVE/USD",
    "WLD/USD", "PEPE/USD", "TRX/USD", "ZEC/USD",
]

LGB_BASE = {"feature_pre_filter": False, "verbose": -1, "is_unbalance": True}
FEATURE_NAMES = MLConfidenceGate.FEATURE_NAMES


# ═══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════

def download_klines(symbol, interval="1h", months=6):
    session = requests.Session()
    session.timeout = 15
    end_ms = int(time.time() * 1000)
    step = {"1h": 3_600_000, "5m": 300_000}[interval]
    start_ms = end_ms - (months * 30 * 24 * 3_600_000)
    candles = []
    cur = start_ms
    while cur < end_ms:
        try:
            r = session.get(f"{BINANCE_BASE_URL}/api/v3/klines",
                            params={"symbol": symbol, "interval": interval,
                                    "startTime": cur, "endTime": end_ms, "limit": 1000})
            r.raise_for_status()
            raw = r.json()
        except Exception as e:
            print(f"    Error {symbol}: {e}"); break
        if not raw: break
        for k in raw:
            candles.append({"t": k[0], "c": float(k[4]), "h": float(k[2]),
                            "l": float(k[3]), "v": float(k[5])})
        cur = raw[-1][0] + step
        if len(raw) < 1000: break
        time.sleep(0.08)
    return candles


def load_all_data(pairs, interval="1h", months=6):
    data = {}
    print(f"\n  Downloading {months}mo of {interval} data for {len(pairs)} pairs...")
    for i, pair in enumerate(pairs):
        sym = BINANCE_SYMBOL_MAP.get(pair)
        if not sym: continue
        candles = download_klines(sym, interval, months)
        if len(candles) < 500:
            print(f"    [{i+1}/{len(pairs)}] {pair}: {len(candles)} bars — skip"); continue
        data[pair] = {
            "timestamps": np.array([x["t"] for x in candles]),
            "c": np.array([x["c"] for x in candles]),
            "h": np.array([x["h"] for x in candles]),
            "l": np.array([x["l"] for x in candles]),
            "v": np.array([x["v"] for x in candles]),
        }
        print(f"    [{i+1}/{len(pairs)}] {pair}: {len(candles)} bars ({len(candles)//24}d)")
    return data


# ═══════════════════════════════════════════════════════════════
# FEATURE COMPUTATION (hour-based, matches ml_model.py exactly)
# ═══════════════════════════════════════════════════════════════

def compute_features(c, h, l, v, btc_c, idx, bph=1):
    """Compute features at index. bph=1 for hourly, bph=12 for 5-min."""
    min_bars = 25 * bph
    if idx < min_bars or idx >= len(c):
        return None
    price = c[idx]
    if price <= 0:
        return None

    # RSI (always uses last 14+1 bars regardless of timeframe)
    rsi_val = rsi(c[:idx+1], 14)

    # EMA distances
    ef = ema(c[max(0, idx-60*bph):idx+1], 21*bph)
    es = ema(c[max(0, idx-100*bph):idx+1], 55*bph)
    ema_dist_fast = (price - ef[-1]) / price if len(ef) > 0 else 0
    ema_dist_slow = (price - es[-1]) / price if len(es) > 0 else 0

    # ATR ratio
    a = atr(h[:idx+1], l[:idx+1], c[:idx+1], 14)
    atr_ratio = a / price if a > 0 else 0

    # Volume ratio: 50 HOURS
    vl = 50 * bph
    if idx >= vl and np.mean(v[idx-vl:idx]) > 0:
        vol_ratio = float(v[idx] / np.mean(v[idx-vl:idx]))
    else:
        vol_ratio = 1.0

    # Returns at fixed hour intervals
    ret_1h = (c[idx] / c[idx - 1*bph] - 1) if idx >= 1*bph else 0
    ret_6h = (c[idx] / c[idx - 6*bph] - 1) if idx >= 6*bph else 0
    ret_24h = (c[idx] / c[idx - 24*bph] - 1) if idx >= 24*bph else 0

    # Volatility over 24 hours
    vw = 24 * bph
    if idx >= vw + 1:
        lr = np.diff(np.log(c[idx-vw:idx+1]))
        real_vol = float(np.std(lr)) * np.sqrt(len(lr) * 365) if len(lr) > 0 else 0.5
        neg_lr = lr[lr < 0]
        down_vol = float(np.std(neg_lr)) * np.sqrt(len(lr) * 365) if len(neg_lr) > 0 else 0.3
    else:
        real_vol, down_vol = 0.5, 0.3

    # BTC return (1 hour)
    btc_ret = 0.0
    if idx < len(btc_c) and idx >= 1*bph:
        btc_ret = (btc_c[idx] / btc_c[idx - 1*bph] - 1)

    # Regime: binary based on BTC vol (matches clipped HMM output)
    regime = 0.0
    if idx < len(btc_c) and idx >= 72*bph:
        blr = np.diff(np.log(btc_c[idx-72*bph:idx+1]))
        if len(blr) >= 24:
            regime = 1.0 if np.std(blr[-24:]) > 1.5 * np.std(blr) else 0.0

    # Breakout strength (72 hours)
    bl = 72 * bph
    if idx >= bl:
        prior_high = np.max(h[idx-bl:idx])
        breakout_str = max(0, (price - prior_high) / prior_high)
    else:
        breakout_str = 0.0

    features = np.array([
        rsi_val, ema_dist_fast, ema_dist_slow, atr_ratio,
        vol_ratio, ret_1h, ret_6h, ret_24h,
        real_vol, down_vol,
        0.0, 0.0,  # funding/OI: not available in historical
        btc_ret, regime,
        breakout_str, 0.0,  # spread: not in historical
    ], dtype=np.float64)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# ═══════════════════════════════════════════════════════════════
# LABELING
# ═══════════════════════════════════════════════════════════════

def label_bar(c, idx, window=6):
    """Simulate entry at c[idx], check if profitable exit exists within window bars."""
    if idx + window >= len(c):
        return None
    entry = c[idx]
    forward = c[idx+1:idx+window+1]
    max_ret = (np.max(forward) / entry) - 1
    # Walk bar-by-bar: check if -4% stop hits before profit
    peak = entry
    for p in forward:
        if (p - entry) / entry <= -0.04:
            return 0  # hard stop hit first
        peak = max(peak, p)
        # Trailing stop in profit zone
        if (peak - entry) / entry > 0.02 and (peak - p) / peak >= 0.03:
            return 1  # locked in profit via trailing
    # End of window: profitable if > 0.5% (covers commission)
    return 1 if (forward[-1] / entry - 1) > 0.005 else 0


# ═══════════════════════════════════════════════════════════════
# DATASET (TEMPORALLY SORTED across coins)
# ═══════════════════════════════════════════════════════════════

def generate_dataset(all_data, btc_c, sample_every=3):
    """Generate features interleaved by timestamp across all coins.

    This ensures walk-forward CV splits are truly temporal:
    fold k contains ALL coins at time T, not all times for coin C.
    """
    # Collect (timestamp, pair, idx) tuples
    entries = []
    for pair, d in all_data.items():
        if pair == "BTC/USD":
            continue
        ts = d["timestamps"]
        for idx in range(200, len(ts) - 8, sample_every):
            entries.append((ts[idx], pair, idx))

    # Sort by timestamp (THE KEY FIX)
    entries.sort(key=lambda x: x[0])
    print(f"  Total candidate bars (sorted by time): {len(entries):,}")

    X_list, y_list = [], []
    for _, pair, idx in entries:
        d = all_data[pair]
        feat = compute_features(d["c"], d["h"], d["l"], d["v"], btc_c, idx, bph=1)
        if feat is None:
            continue
        lbl = label_bar(d["c"], idx, window=6)
        if lbl is None:
            continue
        X_list.append(feat)
        y_list.append(lbl)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  Final dataset: {len(X):,} samples, {np.mean(y):.1%} positive")
    return X, y


# ═══════════════════════════════════════════════════════════════
# PURGED WALK-FORWARD CV
# ═══════════════════════════════════════════════════════════════

def walk_forward_cv(X, y, n_folds=5, purge=120):
    fold_sz = len(X) // n_folds
    results = []
    print(f"\n  Purged walk-forward CV ({n_folds} folds, ~{fold_sz:,} each, {purge}-sample purge):")
    print(f"  {'Fold':<5} {'Train':>8} {'Val':>7} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6} {'Iters':>6}")
    print(f"  {'─' * 55}")

    for k in range(1, n_folds):
        te = k * fold_sz
        vs = te + purge
        ve = min((k+1) * fold_sz, len(X))
        if vs >= ve: continue
        Xt, yt = X[:te], y[:te]
        Xv, yv = X[vs:ve], y[vs:ve]
        if len(Xv) < 50 or len(np.unique(yt)) < 2: continue

        td = lgb.Dataset(Xt, label=yt, feature_name=FEATURE_NAMES)
        vd = lgb.Dataset(Xv, label=yv, reference=td)
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6,
            "min_child_samples": 50, "feature_fraction": 0.7,
            "bagging_fraction": 0.7, "bagging_freq": 5,
            "lambda_l1": 0.1, "lambda_l2": 1.0, **LGB_BASE,
        }
        model = lgb.train(params, td, num_boost_round=1000,
                          valid_sets=[vd],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        preds = model.predict(Xv)
        pl = (preds > 0.5).astype(int)
        acc = np.mean(pl == yv)
        tp = np.sum((pl == 1) & (yv == 1))
        fp = np.sum((pl == 1) & (yv == 0))
        fn = np.sum((pl == 0) & (yv == 1))
        prec = tp/(tp+fp) if tp+fp > 0 else 0
        rec = tp/(tp+fn) if tp+fn > 0 else 0
        try: auc = roc_auc_score(yv, preds)
        except: auc = 0.5
        results.append({"acc": acc, "prec": prec, "rec": rec, "auc": auc,
                         "iters": model.best_iteration})
        print(f"  {k:<5} {len(Xt):>8,} {len(Xv):>7,} {acc:>5.1%} {prec:>5.1%} "
              f"{rec:>5.1%} {auc:>5.3f} {model.best_iteration:>6}")

    if results:
        avg_auc = np.mean([r["auc"] for r in results])
        avg_acc = np.mean([r["acc"] for r in results])
        print(f"  {'─' * 55}")
        print(f"  AVG  acc={avg_acc:.1%}  auc={avg_auc:.3f}")
    return results


# ═══════════════════════════════════════════════════════════════
# ENSEMBLE TRAINING
# ═══════════════════════════════════════════════════════════════

def train_ensemble(X, y):
    split = int(len(X) * 0.9)
    Xt, Xv = X[:split], X[split:]
    yt, yv = y[:split], y[split:]
    td = lgb.Dataset(Xt, label=yt, feature_name=FEATURE_NAMES)
    vd = lgb.Dataset(Xv, label=yv, reference=td)

    configs = [
        ("Conservative", {"learning_rate": 0.01, "num_leaves": 15, "max_depth": 4,
                          "min_child_samples": 80, "lambda_l1": 1.0, "lambda_l2": 10.0,
                          "feature_fraction": 0.5, "bagging_fraction": 0.5}, 1500),
        ("Balanced",     {"learning_rate": 0.03, "num_leaves": 31, "max_depth": 6,
                          "min_child_samples": 50, "lambda_l1": 0.1, "lambda_l2": 1.0,
                          "feature_fraction": 0.7, "bagging_fraction": 0.7}, 800),
        ("Expressive",   {"learning_rate": 0.05, "num_leaves": 63, "max_depth": 8,
                          "min_child_samples": 30, "lambda_l1": 0.01, "lambda_l2": 0.1,
                          "feature_fraction": 0.8, "bagging_fraction": 0.8}, 500),
    ]

    models = []
    print(f"\n  Ensemble:")
    print(f"  {'Name':<15} {'Best':>5} {'Acc':>6} {'AUC':>6}")
    print(f"  {'─' * 36}")

    for name, extra, rounds in configs:
        p = {"objective": "binary", "metric": "binary_logloss",
             "bagging_freq": 5, **LGB_BASE, **extra}
        m = lgb.train(p, td, num_boost_round=rounds, valid_sets=[vd],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        preds = m.predict(Xv)
        acc = np.mean((preds > 0.5).astype(int) == yv)
        try: auc = roc_auc_score(yv, preds)
        except: auc = 0.5
        models.append(m)
        print(f"  {name:<15} {m.best_iteration:>5} {acc:>5.1%} {auc:>5.3f}")

    ens_p = np.mean([m.predict(Xv) for m in models], axis=0)
    ens_acc = np.mean((ens_p > 0.5).astype(int) == yv)
    try: ens_auc = roc_auc_score(yv, ens_p)
    except: ens_auc = 0.5
    print(f"  {'─' * 36}")
    print(f"  {'ENSEMBLE':<15} {'':>5} {ens_acc:>5.1%} {ens_auc:>5.3f}")
    return models


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  ML PRE-TRAINING (FIXED: temporal CV, hour-based features)")
    print("=" * 70)

    all_data = load_all_data(TRAINING_PAIRS, "1h", months=6)
    if len(all_data) < 10:
        print("ERROR: Not enough data"); sys.exit(1)

    btc_c = all_data.get("BTC/USD", {}).get("c", np.array([]))
    total = sum(len(d["c"]) for d in all_data.values())
    print(f"\n  {len(all_data)} pairs, {total:,} total bars, BTC={len(btc_c)//24}d")

    print("\n" + "=" * 70)
    print("  GENERATING TEMPORALLY-SORTED DATASET")
    print("=" * 70)
    X, y = generate_dataset(all_data, btc_c, sample_every=3)

    print("\n" + "=" * 70)
    print("  PURGED WALK-FORWARD CV (truly temporal)")
    print("=" * 70)
    cv = walk_forward_cv(X, y, n_folds=5, purge=200)

    print("\n" + "=" * 70)
    print("  ENSEMBLE TRAINING")
    print("=" * 70)
    models = train_ensemble(X, y)

    # Feature importance
    primary = models[1]
    imp = dict(zip(FEATURE_NAMES, primary.feature_importance()))
    si = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    mx = max(v for _, v in si) or 1
    print(f"\n  Feature importance:")
    for name, val in si:
        print(f"    {name:<20} {val:>5}  {'█' * int(40 * val / mx)}")

    # Threshold analysis
    split = int(len(X) * 0.9)
    Xt, yt = X[split:], y[split:]
    ens_p = np.mean([m.predict(Xt) for m in models], axis=0)
    base = np.mean(yt)
    print(f"\n  Threshold analysis (test={len(Xt):,}, base={base:.1%}):")
    print(f"  {'Thresh':>7} {'Pass':>6} {'Hit%':>6} {'Lift':>5}")
    print(f"  {'─' * 28}")
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = ens_p >= t
        if np.sum(mask) > 10:
            print(f"  {t:>7.2f} {np.sum(mask):>6} {np.mean(yt[mask]):>5.1%} "
                  f"{np.mean(yt[mask])/base:>4.1f}x")

    # Save all 3 ensemble models
    for i, m in enumerate(models):
        m.save_model(MODEL_SAVE_PATH.replace(".txt", f"_ens{i}.txt"))
    # Also save primary as default
    primary.save_model(MODEL_SAVE_PATH)

    meta = {
        "pairs": list(all_data.keys()), "samples": len(X),
        "positive_rate": float(np.mean(y)),
        "cv_avg_auc": float(np.mean([r["auc"] for r in cv])) if cv else 0,
        "features": FEATURE_NAMES, "interval": "1h",
        "fixes": ["temporal_cv", "hour_based_features", "binary_regime", "ensemble_inference"],
    }
    with open(MODEL_SAVE_PATH.replace(".txt", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    avg_auc = meta["cv_avg_auc"]
    print(f"\n  Saved: {MODEL_SAVE_PATH} + 3 ensemble models")
    print(f"  CV AUC: {avg_auc:.3f} | Samples: {len(X):,}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
