"""
Loads historical candle data from Binance for backtesting.
Shared across all backtest scripts.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from bot.binance_data import BinanceData

# All coins available on Roostoo that have Binance equivalents
ALL_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD", "XRP/USD", "DOGE/USD",
    "ADA/USD", "AVAX/USD", "LINK/USD", "DOT/USD", "SUI/USD", "NEAR/USD",
    "LTC/USD", "TON/USD", "UNI/USD", "FET/USD", "HBAR/USD", "XLM/USD",
    "FIL/USD", "APT/USD", "ARB/USD", "SEI/USD", "PEPE/USD", "SHIB/USD",
    "FLOKI/USD", "WIF/USD", "BONK/USD", "TRX/USD", "ICP/USD", "AAVE/USD",
    "WLD/USD", "ONDO/USD", "CRV/USD", "PENDLE/USD", "ENA/USD", "TAO/USD",
    "POL/USD", "ZEC/USD", "TRUMP/USD", "EIGEN/USD", "VIRTUAL/USD",
    "CAKE/USD", "PAXG/USD",
]

# Subset for faster testing
CORE_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "LINK/USD", "AVAX/USD",
    "DOT/USD", "NEAR/USD", "SUI/USD", "FET/USD", "ADA/USD", "DOGE/USD",
    "LTC/USD", "FIL/USD", "UNI/USD", "HBAR/USD", "ZEC/USD", "TRX/USD",
    "ARB/USD", "AAVE/USD", "WLD/USD", "PEPE/USD", "APT/USD", "PAXG/USD",
]


def load_data(
    pairs: list[str] = None,
    interval: str = "1h",
    limit: int = 1000,
    min_bars: int = 200,
) -> dict:
    """Load candle data from Binance.

    Returns dict with keys:
        closes, highs, lows, volumes: dict[pair -> np.ndarray]
        pairs: list of pairs with enough data
        min_len: shortest array length across all pairs
    """
    pairs = pairs or CORE_PAIRS
    bd = BinanceData()
    bd.load_history(pairs, interval=interval, limit=limit)

    closes, highs, lows, volumes = {}, {}, {}, {}
    min_len = 999999

    for pair in pairs:
        c = bd.get_closes(pair)
        h = bd.get_highs(pair)
        l = bd.get_lows(pair)
        v = bd.get_volumes(pair)
        if len(c) >= min_bars:
            closes[pair] = c
            highs[pair] = h
            lows[pair] = l
            volumes[pair] = v
            min_len = min(min_len, len(c))

    active_pairs = list(closes.keys())
    print(f"Loaded {len(active_pairs)}/{len(pairs)} pairs, "
          f"{min_len} bars ({interval}), ~{min_len // (24 if interval == '1h' else 288)} days")

    return {
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
        "pairs": active_pairs,
        "min_len": min_len,
    }


def compute_metrics(portfolio_history: list[float], initial: float, daily_sample_rate: int = 24) -> dict:
    """Compute all competition metrics from a portfolio value series.

    Args:
        portfolio_history: list of portfolio values (one per bar)
        initial: starting portfolio value
        daily_sample_rate: bars per day (24 for hourly, 288 for 5-min)

    Returns dict with all metrics.
    """
    arr = np.array(portfolio_history)
    if len(arr) < 2:
        return {}

    final = arr[-1]
    total_ret = (final - initial) / initial

    # Max drawdown
    peak_arr = np.maximum.accumulate(arr)
    drawdowns = (peak_arr - arr) / peak_arr
    max_dd = float(np.max(drawdowns))

    # Daily returns
    daily_vals = arr[::daily_sample_rate]
    daily_rets = np.diff(daily_vals) / daily_vals[:-1]
    days = len(daily_rets)

    # Sharpe
    sharpe = 0.0
    if len(daily_rets) > 1 and np.std(daily_rets) > 0:
        sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(365))

    # Sortino
    neg_rets = daily_rets[daily_rets < 0]
    sortino = 0.0
    if len(neg_rets) > 0 and np.std(neg_rets) > 0:
        sortino = float(np.mean(daily_rets) / np.std(neg_rets) * np.sqrt(365))
    elif np.mean(daily_rets) > 0:
        sortino = 10.0  # no negative days = excellent

    # Calmar
    calmar = 0.0
    if max_dd > 0 and days > 0:
        annual_ret = total_ret * (365 / days)
        calmar = float(annual_ret / max_dd)
    elif total_ret > 0:
        calmar = 10.0

    # Competition composite
    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    # Win/loss days
    up_days = int(np.sum(daily_rets > 0))
    down_days = int(np.sum(daily_rets < 0))
    flat_days = int(np.sum(daily_rets == 0))

    return {
        "initial": initial,
        "final": round(final, 2),
        "total_return_pct": round(total_ret * 100, 3),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "composite": round(composite, 3),
        "days": days,
        "up_days": up_days,
        "down_days": down_days,
        "flat_days": flat_days,
        "daily_return_avg_pct": round(float(np.mean(daily_rets)) * 100, 4) if len(daily_rets) > 0 else 0,
        "daily_return_std_pct": round(float(np.std(daily_rets)) * 100, 4) if len(daily_rets) > 0 else 0,
    }


def print_metrics(metrics: dict, title: str = "RESULTS"):
    """Pretty-print backtest metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Initial:          ${metrics['initial']:>14,.2f}")
    print(f"  Final:            ${metrics['final']:>14,.2f}")
    print(f"  Total Return:     {metrics['total_return_pct']:>+13.3f}%")
    print(f"  Max Drawdown:     {metrics['max_drawdown_pct']:>13.3f}%")
    print(f"  ─────────────────────────────────────")
    print(f"  Sharpe:           {metrics['sharpe']:>13.3f}")
    print(f"  Sortino:          {metrics['sortino']:>13.3f}")
    print(f"  Calmar:           {metrics['calmar']:>13.3f}")
    print(f"  COMPOSITE SCORE:  {metrics['composite']:>13.3f}")
    print(f"  ─────────────────────────────────────")
    print(f"  Days:             {metrics['days']:>13d}")
    print(f"  Up / Down / Flat: {metrics['up_days']} / {metrics['down_days']} / {metrics['flat_days']}")
    print(f"  Avg Daily Return: {metrics['daily_return_avg_pct']:>+13.4f}%")
    print(f"  Daily Volatility: {metrics['daily_return_std_pct']:>13.4f}%")
    print(f"{'=' * 60}")
