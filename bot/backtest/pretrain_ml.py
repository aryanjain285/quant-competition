#!/usr/bin/env python3
"""
Pre-train ML confidence gate with REAL positioning features.

The previous model failed (AUC 0.55) because price-derived indicators
(RSI, EMA, returns) are already priced in — no information asymmetry.

This version trains on features that capture OTHER PEOPLE'S POSITIONS:
  - Funding rates (cost of leverage → crowding signal)
  - Open interest changes (new money entering/leaving)
  - Long/short account ratio (retail crowd positioning → contrarian)
  - Taker buy/sell ratio (aggressive order flow imbalance)
  - Top trader positions (whale vs retail divergence)

These ARE information-asymmetric: they reflect private positioning
data that hasn't been fully incorporated into spot prices.

Data: All from Binance Futures API, hourly, free, no auth.
History: ~21 days for OI/ratios, ~66 days for funding (we paginate for more).

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
from bot.config import BINANCE_BASE_URL, BINANCE_FUTURES_URL, BINANCE_SYMBOL_MAP
from bot.ml_model import MLConfidenceGate
from bot.signals import ema, rsi, atr
from bot.logger import get_logger

log = get_logger("pretrain")

MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "ml_model.txt")

# Pairs that have Binance USDT perpetual futures (not all altcoins do)
TRAINING_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD", "XRP/USD",
    "DOGE/USD", "ADA/USD", "AVAX/USD", "LINK/USD", "DOT/USD",
    "SUI/USD", "NEAR/USD", "LTC/USD", "UNI/USD", "FET/USD",
    "FIL/USD", "APT/USD", "ARB/USD", "AAVE/USD",
    "PEPE/USD", "TRX/USD", "ZEC/USD",
]

LGB_BASE = {"feature_pre_filter": False, "verbose": -1, "is_unbalance": True}

# New feature set: positioning + minimal price context
FEATURE_NAMES = [
    # Positioning features (the alpha)
    "funding_rate",              # raw funding rate
    "funding_rate_zscore",       # z-score vs 30-period mean
    "oi_change_1h",              # OI % change last 1h
    "oi_change_6h",              # OI % change last 6h
    "oi_change_24h",             # OI % change last 24h
    "ls_ratio",                  # long/short account ratio
    "ls_ratio_change",           # change in L/S ratio (delta)
    "taker_buy_sell_ratio",      # aggressive order flow
    "taker_ratio_change",        # change in taker ratio
    "top_trader_ls_ratio",       # whale positioning
    "top_vs_retail_divergence",  # whale L/S minus retail L/S
    # Price context (minimal — just enough to condition on regime)
    "ret_1h",                    # 1h return
    "ret_6h",                    # 6h return
    "ret_24h",                   # 24h return
    "rsi_14",                    # RSI
    "vol_24h",                   # 24h realized vol (annualized)
    # Cross-asset context
    "btc_ret_1h",                # BTC 1h return
    "btc_funding",               # BTC funding rate
    "btc_oi_change_6h",          # BTC OI change
    "btc_taker_ratio",           # BTC taker buy/sell
]

NUM_FEATURES = len(FEATURE_NAMES)


# ═══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════

def _fetch_json(url, params, session):
    try:
        r = session.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None


def download_spot_klines(symbol, session, months=3):
    """Download spot 1h candles via pagination."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (months * 30 * 24 * 3_600_000)
    candles = []
    cur = start_ms
    while cur < end_ms:
        raw = _fetch_json(f"{BINANCE_BASE_URL}/api/v3/klines",
                          {"symbol": symbol, "interval": "1h",
                           "startTime": cur, "endTime": end_ms, "limit": 1000}, session)
        if not raw: break
        for k in raw:
            candles.append({"t": k[0], "c": float(k[4]), "h": float(k[2]),
                            "l": float(k[3]), "v": float(k[5])})
        cur = raw[-1][0] + 3_600_000
        if len(raw) < 1000: break
        time.sleep(0.05)
    return candles


def download_funding(symbol, session, limit=1000):
    """Download funding rate history (every 8h, up to ~66 days)."""
    data = _fetch_json(f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate",
                       {"symbol": symbol, "limit": limit}, session)
    if not data or not isinstance(data, list):
        return []
    return [{"t": d["fundingTime"], "rate": float(d["fundingRate"]),
             "price": float(d.get("markPrice", 0))} for d in data]


def download_futures_data(symbol, endpoint, session, limit=500):
    """Download OI / long-short / taker data (hourly, up to ~21 days)."""
    data = _fetch_json(f"{BINANCE_FUTURES_URL}/futures/data/{endpoint}",
                       {"symbol": symbol, "period": "1h", "limit": limit}, session)
    if not data or not isinstance(data, list):
        return []
    return data


def load_all_data(pairs):
    """Download all data sources for all pairs."""
    session = requests.Session()
    session.timeout = 15
    all_data = {}

    print(f"\n  Downloading data for {len(pairs)} pairs...")
    for i, pair in enumerate(pairs):
        sym = BINANCE_SYMBOL_MAP.get(pair)
        if not sym: continue

        d = {}

        # Spot candles (3 months)
        candles = download_spot_klines(sym, session, months=3)
        if len(candles) < 500:
            print(f"    [{i+1}/{len(pairs)}] {pair}: only {len(candles)} candles, skip")
            continue
        d["spot_t"] = np.array([c["t"] for c in candles])
        d["spot_c"] = np.array([c["c"] for c in candles])
        d["spot_h"] = np.array([c["h"] for c in candles])
        d["spot_l"] = np.array([c["l"] for c in candles])

        # Funding rates
        funding = download_funding(sym, session)
        d["funding"] = funding

        # OI history
        oi = download_futures_data(sym, "openInterestHist", session)
        d["oi"] = oi

        # Long/short ratio
        ls = download_futures_data(sym, "globalLongShortAccountRatio", session)
        d["ls"] = ls

        # Taker buy/sell
        taker = download_futures_data(sym, "takerlongshortRatio", session)
        d["taker"] = taker

        # Top trader positions
        top = download_futures_data(sym, "topLongShortPositionRatio", session)
        d["top_trader"] = top

        all_data[pair] = d
        print(f"    [{i+1}/{len(pairs)}] {pair}: candles={len(candles)} "
              f"funding={len(funding)} oi={len(oi)} ls={len(ls)} "
              f"taker={len(taker)} top={len(top)}")
        time.sleep(0.1)

    return all_data


# ═══════════════════════════════════════════════════════════════
# ALIGN DATA BY TIMESTAMP
# ═══════════════════════════════════════════════════════════════

def build_aligned_hourly(data: dict) -> dict | None:
    """Align all data sources to common hourly timestamps.

    Returns dict with numpy arrays all of the same length, or None if insufficient.
    """
    spot_t = data["spot_t"]
    spot_c = data["spot_c"]

    # Build lookup dicts: timestamp -> value
    def _build_lookup(entries, ts_key, val_keys):
        lookup = {}
        for e in entries:
            t = e.get(ts_key, e.get("timestamp", 0))
            # Round to nearest hour
            t_hour = (t // 3_600_000) * 3_600_000
            vals = {}
            for k in val_keys:
                try:
                    vals[k] = float(e[k])
                except (KeyError, ValueError, TypeError):
                    vals[k] = 0.0
            lookup[t_hour] = vals
        return lookup

    funding_lookup = _build_lookup(data["funding"], "t",
                                    ["rate"])
    oi_lookup = _build_lookup(data["oi"], "timestamp",
                               ["sumOpenInterest"])
    ls_lookup = _build_lookup(data["ls"], "timestamp",
                               ["longShortRatio"])
    taker_lookup = _build_lookup(data["taker"], "timestamp",
                                  ["buySellRatio"])
    top_lookup = _build_lookup(data["top_trader"], "timestamp",
                                ["longShortRatio"])

    # Find common timestamps where we have ALL data
    # Use spot candle timestamps as anchor
    spot_hours = set((t // 3_600_000) * 3_600_000 for t in spot_t)
    oi_hours = set(oi_lookup.keys())
    common = sorted(spot_hours & oi_hours)  # at minimum need spot + OI

    if len(common) < 100:
        return None

    # Build aligned arrays
    n = len(common)
    ts_map = {t: i for i, t in enumerate(common)}

    # Spot data: map each common timestamp to nearest spot candle
    spot_t_hours = (spot_t // 3_600_000) * 3_600_000
    spot_idx = {t: i for i, t in enumerate(spot_t_hours)}

    closes = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    for j, t in enumerate(common):
        si = spot_idx.get(t)
        if si is not None:
            closes[j] = spot_c[si]
            highs[j] = data["spot_h"][si]
            lows[j] = data["spot_l"][si]

    # Derivatives arrays
    funding_rates = np.zeros(n)
    oi_values = np.zeros(n)
    ls_ratios = np.zeros(n)
    taker_ratios = np.zeros(n)
    top_ratios = np.zeros(n)

    # Forward-fill: for each hour, use latest available data
    last_funding = 0.0
    last_ls = 1.0
    last_taker = 1.0
    last_top = 1.0

    for j, t in enumerate(common):
        # Funding: 8h intervals, forward-fill
        f = funding_lookup.get(t)
        if f:
            last_funding = f["rate"]
        funding_rates[j] = last_funding

        # OI
        o = oi_lookup.get(t)
        oi_values[j] = o["sumOpenInterest"] if o else (oi_values[j-1] if j > 0 else 0)

        # Long/short
        l = ls_lookup.get(t)
        if l:
            last_ls = l["longShortRatio"]
        ls_ratios[j] = last_ls

        # Taker
        tk = taker_lookup.get(t)
        if tk:
            last_taker = tk["buySellRatio"]
        taker_ratios[j] = last_taker

        # Top trader
        tp = top_lookup.get(t)
        if tp:
            last_top = tp["longShortRatio"]
        top_ratios[j] = last_top

    return {
        "timestamps": np.array(common),
        "closes": closes, "highs": highs, "lows": lows,
        "funding": funding_rates, "oi": oi_values,
        "ls_ratio": ls_ratios, "taker_ratio": taker_ratios,
        "top_trader_ratio": top_ratios,
    }


# ═══════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_features_at(aligned, btc_aligned, idx):
    """Compute feature vector at index. All positioning + minimal price context."""
    if idx < 30:
        return None

    c = aligned["closes"]
    fr = aligned["funding"]
    oi = aligned["oi"]
    ls = aligned["ls_ratio"]
    tk = aligned["taker_ratio"]
    top = aligned["top_trader_ratio"]
    price = c[idx]
    if price <= 0 or oi[idx] <= 0:
        return None

    # ── Positioning features ──
    funding_rate = fr[idx]

    # Funding z-score (vs last 30 periods)
    if idx >= 30:
        fr_window = fr[idx-30:idx+1]
        fr_mean = np.mean(fr_window)
        fr_std = np.std(fr_window)
        funding_zscore = (fr[idx] - fr_mean) / fr_std if fr_std > 0 else 0
    else:
        funding_zscore = 0

    # OI changes
    oi_1h = (oi[idx] / oi[idx-1] - 1) if idx >= 1 and oi[idx-1] > 0 else 0
    oi_6h = (oi[idx] / oi[idx-6] - 1) if idx >= 6 and oi[idx-6] > 0 else 0
    oi_24h = (oi[idx] / oi[idx-24] - 1) if idx >= 24 and oi[idx-24] > 0 else 0

    # Long/short ratio and its change
    ls_ratio = ls[idx]
    ls_change = (ls[idx] - ls[idx-6]) if idx >= 6 else 0

    # Taker buy/sell ratio and change
    taker_ratio = tk[idx]
    taker_change = (tk[idx] - tk[idx-6]) if idx >= 6 else 0

    # Top trader ratio
    top_ratio = top[idx]

    # Whale vs retail divergence (top trader long/short minus global long/short)
    # Positive = whales more long than retail = bullish signal
    top_vs_retail = top_ratio - ls_ratio

    # ── Price context (minimal) ──
    ret_1h = (c[idx] / c[idx-1] - 1) if idx >= 1 else 0
    ret_6h = (c[idx] / c[idx-6] - 1) if idx >= 6 else 0
    ret_24h = (c[idx] / c[idx-24] - 1) if idx >= 24 else 0

    rsi_val = rsi(c[:idx+1], 14)

    # 24h vol
    if idx >= 25:
        lr = np.diff(np.log(c[idx-24:idx+1]))
        vol_24h = float(np.std(lr)) * np.sqrt(24 * 365) if len(lr) > 0 else 0.5
    else:
        vol_24h = 0.5

    # ── BTC cross-asset context ──
    btc_ret = 0
    btc_funding = 0
    btc_oi_change = 0
    btc_taker = 1.0
    if btc_aligned is not None and idx < len(btc_aligned["closes"]):
        bc = btc_aligned["closes"]
        btc_ret = (bc[idx] / bc[idx-1] - 1) if idx >= 1 and bc[idx-1] > 0 else 0
        btc_funding = btc_aligned["funding"][idx]
        btc_oi_change = (btc_aligned["oi"][idx] / btc_aligned["oi"][idx-6] - 1) \
            if idx >= 6 and btc_aligned["oi"][idx-6] > 0 else 0
        btc_taker = btc_aligned["taker_ratio"][idx]

    features = np.array([
        funding_rate, funding_zscore,
        oi_1h, oi_6h, oi_24h,
        ls_ratio, ls_change,
        taker_ratio, taker_change,
        top_ratio, top_vs_retail,
        ret_1h, ret_6h, ret_24h,
        rsi_val, vol_24h,
        btc_ret, btc_funding, btc_oi_change, btc_taker,
    ], dtype=np.float64)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# ═══════════════════════════════════════════════════════════════
# LABELING
# ═══════════════════════════════════════════════════════════════

def label_bar(closes, idx, window=6):
    if idx + window >= len(closes): return None
    entry = closes[idx]
    if entry <= 0: return None
    forward = closes[idx+1:idx+window+1]
    peak = entry
    for p in forward:
        if (p - entry) / entry <= -0.04: return 0
        peak = max(peak, p)
        if (peak - entry) / entry > 0.02 and (peak - p) / peak >= 0.03: return 1
    return 1 if (forward[-1] / entry - 1) > 0.005 else 0


# ═══════════════════════════════════════════════════════════════
# DATASET GENERATION (temporally sorted)
# ═══════════════════════════════════════════════════════════════

def generate_dataset(all_aligned, btc_aligned, sample_every=2):
    entries = []
    for pair, aligned in all_aligned.items():
        if pair == "BTC/USD": continue
        ts = aligned["timestamps"]
        for idx in range(30, len(ts) - 8, sample_every):
            entries.append((ts[idx], pair, idx))

    entries.sort(key=lambda x: x[0])  # TEMPORAL SORT
    print(f"  Candidate bars (sorted by time): {len(entries):,}")

    X_list, y_list = [], []
    for _, pair, idx in entries:
        aligned = all_aligned[pair]
        feat = compute_features_at(aligned, btc_aligned, idx)
        if feat is None: continue
        lbl = label_bar(aligned["closes"], idx, window=6)
        if lbl is None: continue
        X_list.append(feat)
        y_list.append(lbl)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  Dataset: {len(X):,} samples, {np.mean(y):.1%} positive, {NUM_FEATURES} features")
    return X, y


# ═══════════════════════════════════════════════════════════════
# CV + TRAINING
# ═══════════════════════════════════════════════════════════════

def walk_forward_cv(X, y, n_folds=5, purge=100):
    fold_sz = len(X) // n_folds
    results = []
    print(f"\n  Purged walk-forward CV ({n_folds} folds, ~{fold_sz:,} each, {purge} purge):")
    print(f"  {'Fold':<5} {'Train':>8} {'Val':>7} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6} {'Iters':>6}")
    print(f"  {'─' * 55}")
    for k in range(1, n_folds):
        te = k * fold_sz
        vs = te + purge
        ve = min((k+1) * fold_sz, len(X))
        if vs >= ve: continue
        Xt, yt = X[:te], y[:te]
        Xv, yv = X[vs:ve], y[vs:ve]
        if len(Xv) < 30 or len(np.unique(yt)) < 2: continue

        td = lgb.Dataset(Xt, label=yt, feature_name=FEATURE_NAMES)
        vd = lgb.Dataset(Xv, label=yv, reference=td)
        p = {"objective": "binary", "metric": "binary_logloss",
             "learning_rate": 0.03, "num_leaves": 31, "max_depth": 6,
             "min_child_samples": 30, "feature_fraction": 0.7,
             "bagging_fraction": 0.7, "bagging_freq": 5,
             "lambda_l1": 0.1, "lambda_l2": 1.0, **LGB_BASE}
        model = lgb.train(p, td, num_boost_round=1000, valid_sets=[vd],
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
        print(f"  {'─' * 55}")
        print(f"  AVG AUC: {avg_auc:.3f}")
    return results


def train_ensemble(X, y):
    split = int(len(X) * 0.9)
    Xt, Xv = X[:split], X[split:]
    yt, yv = y[:split], y[split:]
    td = lgb.Dataset(Xt, label=yt, feature_name=FEATURE_NAMES)
    vd = lgb.Dataset(Xv, label=yv, reference=td)

    configs = [
        ("Conservative", {"learning_rate": 0.01, "num_leaves": 15, "max_depth": 4,
                          "min_child_samples": 50, "lambda_l1": 1.0, "lambda_l2": 10.0,
                          "feature_fraction": 0.5, "bagging_fraction": 0.5}, 1500),
        ("Balanced",     {"learning_rate": 0.03, "num_leaves": 31, "max_depth": 6,
                          "min_child_samples": 30, "lambda_l1": 0.1, "lambda_l2": 1.0,
                          "feature_fraction": 0.7, "bagging_fraction": 0.7}, 800),
        ("Expressive",   {"learning_rate": 0.05, "num_leaves": 63, "max_depth": 8,
                          "min_child_samples": 20, "lambda_l1": 0.01, "lambda_l2": 0.1,
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
    print("  ML PRE-TRAINING v3: POSITIONING FEATURES")
    print("  Funding + OI + Long/Short + Taker + Top Trader")
    print("=" * 70)

    # 1. Download
    all_data = load_all_data(TRAINING_PAIRS)
    if len(all_data) < 10:
        print("ERROR: Not enough data"); sys.exit(1)

    # 2. Align to hourly
    print(f"\n  Aligning data to common hourly timestamps...")
    all_aligned = {}
    for pair, d in all_data.items():
        aligned = build_aligned_hourly(d)
        if aligned is not None:
            all_aligned[pair] = aligned
            print(f"    {pair}: {len(aligned['timestamps'])} aligned hours")

    btc_aligned = all_aligned.get("BTC/USD")
    print(f"  Aligned: {len(all_aligned)} pairs")

    if len(all_aligned) < 8:
        print("ERROR: Not enough aligned data"); sys.exit(1)

    # 3. Generate dataset
    print("\n" + "=" * 70)
    print("  GENERATING DATASET")
    print("=" * 70)
    X, y = generate_dataset(all_aligned, btc_aligned, sample_every=1)
    if len(X) < 300:
        print("ERROR: Too few samples"); sys.exit(1)

    # 4. CV
    print("\n" + "=" * 70)
    print("  WALK-FORWARD CV")
    print("=" * 70)
    cv = walk_forward_cv(X, y, n_folds=5, purge=50)

    # 5. Ensemble
    print("\n" + "=" * 70)
    print("  ENSEMBLE")
    print("=" * 70)
    models = train_ensemble(X, y)

    # 6. Feature importance
    primary = models[1]
    imp = dict(zip(FEATURE_NAMES, primary.feature_importance()))
    si = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    mx = max(v for _, v in si) or 1
    print(f"\n  Feature importance:")
    for name, val in si:
        print(f"    {name:<28} {val:>5}  {'█' * int(40 * val / mx)}")

    # 7. Threshold analysis
    split = int(len(X) * 0.9)
    Xt, yt = X[split:], y[split:]
    ens_p = np.mean([m.predict(Xt) for m in models], axis=0)
    base = np.mean(yt) if len(yt) > 0 else 0.5
    print(f"\n  Threshold analysis (test={len(Xt):,}, base={base:.1%}):")
    print(f"  {'Thresh':>7} {'Pass':>6} {'Hit%':>6} {'Lift':>5}")
    print(f"  {'─' * 28}")
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = ens_p >= t
        if np.sum(mask) > 5:
            print(f"  {t:>7.2f} {np.sum(mask):>6} {np.mean(yt[mask]):>5.1%} "
                  f"{np.mean(yt[mask])/base:>4.1f}x")

    # 8. Save
    for i, m in enumerate(models):
        m.save_model(MODEL_SAVE_PATH.replace(".txt", f"_ens{i}.txt"))
    primary.save_model(MODEL_SAVE_PATH)

    meta = {
        "version": "v3_positioning",
        "pairs": list(all_aligned.keys()), "samples": len(X),
        "positive_rate": float(np.mean(y)),
        "cv_avg_auc": float(np.mean([r["auc"] for r in cv])) if cv else 0,
        "features": FEATURE_NAMES, "num_features": NUM_FEATURES,
    }
    with open(MODEL_SAVE_PATH.replace(".txt", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n  Model saved: {MODEL_SAVE_PATH}")
    print(f"  CV AUC: {meta['cv_avg_auc']:.3f} | Samples: {len(X):,}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
