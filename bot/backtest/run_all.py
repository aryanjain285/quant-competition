#!/usr/bin/env python3
"""
Run all backtests sequentially.

Usage:
    venv/bin/python -m bot.backtest.run_all          # run all
    venv/bin/python -m bot.backtest.run_all market    # run just market analysis
    venv/bin/python -m bot.backtest.run_all signals   # run just signal comparison
    venv/bin/python -m bot.backtest.run_all portfolio # run just full portfolio
    venv/bin/python -m bot.backtest.run_all params    # run just parameter sensitivity
"""
import sys
import time


TESTS = {
    "market": ("Market Structure Analysis", "bot.backtest.bt_market_analysis"),
    "signals": ("Signal Comparison", "bot.backtest.bt_signal_comparison"),
    "portfolio": ("Full Portfolio Simulation", "bot.backtest.bt_full_portfolio"),
    "params": ("Parameter Sensitivity", "bot.backtest.bt_param_sensitivity"),
}


def run_test(name: str, module_path: str):
    print(f"\n\n{'#' * 80}")
    print(f"#  RUNNING: {name}")
    print(f"{'#' * 80}\n")

    start = time.time()
    mod = __import__(module_path, fromlist=["main"])
    mod.main()
    elapsed = time.time() - start
    print(f"\n  [{name} completed in {elapsed:.1f}s]")


def main():
    requested = sys.argv[1] if len(sys.argv) > 1 else None

    if requested:
        if requested in TESTS:
            name, mod = TESTS[requested]
            run_test(name, mod)
        else:
            print(f"Unknown test: {requested}")
            print(f"Available: {', '.join(TESTS.keys())}")
            sys.exit(1)
    else:
        for key, (name, mod) in TESTS.items():
            run_test(name, mod)

        print(f"\n\n{'#' * 80}")
        print(f"#  ALL BACKTESTS COMPLETE")
        print(f"{'#' * 80}")


if __name__ == "__main__":
    main()
