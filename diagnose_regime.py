"""
Diagnostic: is the PCA-HMM regime layer actually being selective?

Pulls real Binance hourly history (same way the live bot does), runs the
real RegimeDetector, and reports:
  - PCA explained variance
  - Per-state forward-return Sharpe, frequency, exposure mult
  - Current state and exposure
  - Time series of state assignments over the last N bars (sanity check)
  - Distribution of historical exposure mult (was it ever actually <1?)

Run: venv/bin/python diagnose_regime.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from collections import Counter

from bot.binance_data import BinanceData
from bot.regime_detector import RegimeDetector
from bot.config import TRADEABLE_COINS

print("=" * 70)
print("REGIME DETECTOR DIAGNOSTIC")
print("=" * 70)

# 1. Load history (same as live)
print("\n[1/4] Loading 1h Binance history for 43 coins (1000 bars each)...")
bd = BinanceData()
bd.load_history(interval="1h", limit=1000)

# 2. Build price matrix (T x N)
print("\n[2/4] Building price matrix...")
closes_per_pair = {}
min_len = None
for pair in TRADEABLE_COINS:
    closes = bd.get_closes(pair)
    if len(closes) > 0:
        closes_per_pair[pair] = closes
        min_len = len(closes) if min_len is None else min(min_len, len(closes))

pairs_used = sorted(closes_per_pair.keys())
price_matrix = np.column_stack([closes_per_pair[p][-min_len:] for p in pairs_used])
print(f"  shape: {price_matrix.shape} (T={min_len} hours, N={len(pairs_used)} coins)")

# 3. Run regime detector
print("\n[3/4] Running PCA + HMM fit...")
rd = RegimeDetector()
pc_scores = rd.compute_pc_scores(price_matrix)
print(f"  PC scores shape: {pc_scores.shape if pc_scores is not None else None}")
print(f"  PCA explained variance: {rd._pca_explained_var:.1%}")

rd.fit_hmm(pc_scores, lookback=1440)
print(f"  HMM fitted: {rd.hmm_fitted}")
print(f"  Current state: {rd.current_state}")
print(f"  Current exposure mult: {rd.get_exposure_multiplier():.3f}")

# 4. Per-state report
print("\n[4/4] Per-state analysis:")
print(f"  {'State':<8}{'Name':<18}{'fwd_ret':>10}{'sharpe':>10}{'freq':>8}{'dur(h)':>9}{'exposure':>11}")
for s in sorted(rd._state_analysis.keys()):
    a = rd._state_analysis[s]
    name = rd._state_names.get(s, "?")
    exp = rd._state_exposure.get(s, 0)
    print(f"  {s:<8}{name:<18}{a['avg_forward_return']:>+10.4f}{a['sharpe_of_state']:>+10.3f}"
          f"{a['frequency']:>7.1%}{a['avg_duration_hours']:>9.0f}{exp:>10.3f}")

# 5. Walk through last N bars and see how exposure has varied
print("\n[5/5] Historical state walk (last 200 hours):")
if rd.hmm_fitted and pc_scores is not None:
    obs = pc_scores[-200:]
    mask = np.all(np.isfinite(obs), axis=1)
    obs_clean = obs[mask]
    states_hist = rd.hmm_model.predict(obs_clean)
    exposures_hist = np.array([rd._state_exposure.get(int(s), 0.5) for s in states_hist])

    state_counts = Counter(states_hist.tolist())
    print(f"  State distribution over last {len(states_hist)} bars:")
    for s, c in sorted(state_counts.items()):
        name = rd._state_names.get(s, "?")
        exp = rd._state_exposure.get(s, 0)
        print(f"    state {s} [{name}]: {c} bars ({c/len(states_hist):.1%}) → exposure {exp:.2f}")

    print(f"\n  Exposure mult statistics over last 200h:")
    print(f"    mean:   {exposures_hist.mean():.3f}")
    print(f"    median: {np.median(exposures_hist):.3f}")
    print(f"    min:    {exposures_hist.min():.3f}")
    print(f"    max:    {exposures_hist.max():.3f}")
    print(f"    std:    {exposures_hist.std():.3f}")

    # How often does exposure actually meaningfully throttle?
    pct_low = (exposures_hist < 0.3).mean()
    pct_mid = ((exposures_hist >= 0.3) & (exposures_hist < 0.7)).mean()
    pct_high = (exposures_hist >= 0.7).mean()
    print(f"\n  Time spent at each exposure tier:")
    print(f"    <0.30 (cautious):  {pct_low:.1%}")
    print(f"    0.30-0.70 (mid):   {pct_mid:.1%}")
    print(f"    >=0.70 (aggressive): {pct_high:.1%}")

    # Show last 50 bars as a strip
    print(f"\n  Last 50 bars (state stream):")
    strip = "".join(str(int(s)) for s in states_hist[-50:])
    print(f"    {strip}")

# 6. Verdict
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
if rd.hmm_fitted:
    sharpes = sorted([rd._state_analysis[s]["sharpe_of_state"] for s in rd._state_analysis])
    spread = sharpes[-1] - sharpes[0]
    exposures_set = sorted({round(rd._state_exposure[s], 2) for s in rd._state_exposure})

    print(f"Sharpe spread between best/worst state: {spread:.3f}")
    print(f"Distinct exposure tiers: {exposures_set}")
    if spread < 0.1:
        print("⚠ States are nearly indistinguishable — regime layer is NOT being selective.")
        print("  Forward returns are too similar across states; HMM has no useful signal.")
    elif len(exposures_set) < 2:
        print("⚠ All states map to same exposure — regime layer is inactive.")
    elif max(exposures_set) - min(exposures_set) < 0.3:
        print("⚠ Exposure tiers are tightly bunched — regime is mildly selective at best.")
    else:
        print("✓ Regime layer has meaningful spread between states.")

    if rd.get_exposure_multiplier() >= 0.9:
        print(f"✓ Currently in a permissive regime (exposure {rd.get_exposure_multiplier():.2f}).")
    elif rd.get_exposure_multiplier() <= 0.2:
        print(f"⚠ Currently in a defensive regime (exposure {rd.get_exposure_multiplier():.2f}).")
    else:
        print(f"  Currently in middle regime (exposure {rd.get_exposure_multiplier():.2f}).")
else:
    print("⚠ HMM did not fit. No regime signal at all.")
