"""
Diagnostic: is the ranking layer actually being selective about candidates?

Mirrors the live pipeline:
  1. Pull 1h Binance history for all coins (signal source)
  2. Pull live Roostoo ticker for spreads (filter source)
  3. Compute features per coin
  4. Run the real Ranker
  5. Report:
     - How many coins survive each filter stage
     - EWMA score distribution of eligible coins
     - Top-10 vs bottom-10 score gap (selectivity proxy)
     - How many coins would be NEW entries given current MAX_NEW_ENTRIES

Run: venv/bin/python diagnose_ranking.py
"""
import sys
sys.path.insert(0, ".")

import os
# Reroute to mirror that isn't blocked from this IP
os.environ["BINANCE_BASE_URL"] = "https://data-api.binance.vision"

import numpy as np

from bot.binance_data import BinanceData
from bot.roostoo_client import RoostooClient
from bot.features import compute_coin_features, zscore_universe, check_entry_gate, compute_ewma_momentum
from bot.ranking import Ranker
from bot.config import TRADEABLE_COINS, BREAKOUT_LOOKBACK, MAX_NEW_ENTRIES_PER_CYCLE, MAX_POSITIONS

print("=" * 72)
print("RANKING DIAGNOSTIC")
print("=" * 72)

# 1. Binance history (with retry/backoff for rate limiting)
print("\n[1/5] Loading 1h Binance history (with backoff)...")
import time
bd = BinanceData()
for attempt in range(4):
    bd.load_history(interval="1h", limit=500)
    loaded = sum(1 for p in TRADEABLE_COINS if len(bd.get_closes(p)) > 0)
    if loaded >= 35:
        break
    print(f"  attempt {attempt+1}: only {loaded}/43 loaded, backing off 15s...")
    time.sleep(15)

# 2. Roostoo ticker for spreads
print("\n[2/5] Fetching live Roostoo ticker for bid/ask spreads...")
client = RoostooClient()
ticker = client.ticker() or {}
print(f"  Got ticker for {len(ticker)} pairs")

# 3. Compute features per coin
print("\n[3/5] Computing per-coin features...")
raw_features = {}
closes_dict = {}
for pair in TRADEABLE_COINS:
    closes = bd.get_closes(pair)
    highs = bd.get_highs(pair)
    lows = bd.get_lows(pair)
    volumes = bd.get_volumes(pair)
    if len(closes) < 100:
        continue

    tk = ticker.get(pair, {})
    bid = tk.get("MaxBid", 0) or 0
    ask = tk.get("MinAsk", 0) or 0
    if bid <= 0 or ask <= 0:
        last = tk.get("LastPrice", closes[-1])
        bid = last * 0.999
        ask = last * 1.001

    feats = compute_coin_features(
        closes=closes, highs=highs, lows=lows,
        volumes=volumes, bid=bid, ask=ask,
        breakout_lookback=BREAKOUT_LOOKBACK,
    )
    if feats:
        raw_features[pair] = feats
        closes_dict[pair] = closes

print(f"  Built features for {len(raw_features)} pairs")

# 4. Run Ranker
print("\n[4/5] Running ranker...")
zscored = zscore_universe(raw_features)
ranker = Ranker()
eligible = ranker.rank(
    raw_features=raw_features,
    zscored_features=zscored,
    held_pairs=set(),
    closes_dict=closes_dict,
)

# 5. Detailed funnel analysis
print("\n[5/5] Selection funnel:")

n_total = len(raw_features)

# spread filter analysis
all_spreads = [f.get("spread_pct", 0) for f in raw_features.values() if f.get("spread_pct", 0) > 0]
median_spread = float(np.median(all_spreads)) if all_spreads else 0.001
n_spread_pass = sum(1 for f in raw_features.values()
                    if f.get("spread_pct", 0) <= median_spread or f.get("spread_pct", 0) <= 0)

# entry gate analysis
n_gate_pass = sum(1 for f in raw_features.values() if check_entry_gate(f))

# r_24h positive
n_r24_pos = sum(1 for f in raw_features.values() if f.get("r_24h", 0) > 0)

# volume gate
n_vol_pass = sum(1 for f in raw_features.values() if f.get("volume_ratio", 0) >= 0.8)

# EWMA positive among gate-pass
n_ewma_pos = 0
for pair, f in raw_features.items():
    if not check_entry_gate(f):
        continue
    closes = closes_dict.get(pair)
    if closes is not None and len(closes) > 30:
        if compute_ewma_momentum(closes) > 0:
            n_ewma_pos += 1

print(f"  Total coins with features:          {n_total:>3}")
print(f"  Pass spread filter (<= median):     {n_spread_pass:>3}  (median spread {median_spread*10000:.1f}bps)")
print(f"  r_24h > 0 (positive 24h momentum):  {n_r24_pos:>3}")
print(f"  volume_ratio >= 0.8:                {n_vol_pass:>3}")
print(f"  Pass full entry gate:               {n_gate_pass:>3}")
print(f"  Pass gate AND ewma > 0:             {n_ewma_pos:>3}")
print(f"  Final eligible (after all filters): {len(eligible):>3}")

# Show top 15 with scores
print(f"\n  Top 15 ranked candidates (out of {len(eligible)}):")
print(f"  {'rank':<6}{'pair':<14}{'ewma':>12}{'r_24h':>10}{'spread bps':>13}")
for i, (pair, score, raw) in enumerate(eligible[:15]):
    print(f"  {i+1:<6}{pair:<14}{score:>+12.6f}{raw.get('r_24h', 0):>+10.4f}"
          f"{raw.get('spread_pct', 0)*10000:>13.1f}")

# Score distribution
if len(eligible) > 0:
    scores = np.array([s for _, s, _ in eligible])
    print(f"\n  EWMA score distribution (eligible only):")
    print(f"    n:        {len(scores)}")
    print(f"    mean:     {scores.mean():+.6f}")
    print(f"    median:   {np.median(scores):+.6f}")
    print(f"    max:      {scores.max():+.6f}")
    print(f"    min:      {scores.min():+.6f}")
    print(f"    p90:      {np.percentile(scores, 90):+.6f}")
    print(f"    p10:      {np.percentile(scores, 10):+.6f}")

    # Selectivity gap: how much better is the top of the list vs bottom?
    if len(scores) >= 6:
        top3 = scores[:3].mean()
        bot3 = scores[-3:].mean()
        gap = top3 - bot3
        print(f"\n  Top-3 vs bottom-3 mean: {top3:+.6f} vs {bot3:+.6f}  (gap: {gap:+.6f})")

print(f"\n  Bot would take up to {MAX_NEW_ENTRIES_PER_CYCLE} new entries this cycle.")
print(f"  MAX_POSITIONS cap: {MAX_POSITIONS}")

# Verdict
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)

if len(eligible) == 0:
    print("⚠ Zero eligible candidates — bot would not trade this cycle.")
elif len(eligible) < 5:
    print(f"✓ Very selective: only {len(eligible)} coins survived all filters.")
    print("  This is GOOD selectivity — bot only trades highest-conviction names.")
elif len(eligible) < 15:
    print(f"~ Moderately selective: {len(eligible)} coins eligible.")
    print("  Bot has real choice but not too broad.")
else:
    print(f"⚠ Loose: {len(eligible)} coins eligible out of {n_total}.")
    print("  The funnel is barely filtering — most coins pass.")

if len(eligible) >= 6:
    scores = np.array([s for _, s, _ in eligible])
    if scores[:3].mean() / max(scores[-3:].mean(), 1e-9) < 2:
        print("⚠ Top vs bottom score gap is small — ranking is not strongly differentiating.")
    else:
        print("✓ Clear score gap between top and bottom — ranking is differentiating.")

# Compare against earlier finding: regime says BEAR with neg fwd_ret
print("\nNote: regime diagnostic showed all states have NEGATIVE forward returns.")
print("So even if the ranker picks the 'best' looking coins, they're being chosen")
print("from a universe that has been losing money on average. EWMA momentum positive")
print("does NOT mean 'safe to buy' in a market-wide bear regime.")
