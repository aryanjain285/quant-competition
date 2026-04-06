"""
Pre-redeployment state saver.
Run this BEFORE stopping the bot to preserve trailing stop state.

Queries Roostoo API for:
  1. Current wallet positions
  2. Recent order history to find entry prices
  3. Current prices for trailing high

Saves to bot/state.json — the bot loads this on startup.

Usage: venv/bin/python save_state.py
"""
import sys
import json
import time
sys.path.insert(0, ".")

from bot.roostoo_client import RoostooClient

client = RoostooClient()

# 1. Get current balance
wallet = client.balance()
if wallet is None:
    print("ERROR: Cannot fetch balance")
    sys.exit(1)

positions = {}
for coin, bal in wallet.items():
    if coin == "USD":
        continue
    free = bal.get("Free", 0)
    if free > 0:
        pair = f"{coin}/USD"
        positions[pair] = free

print(f"Current positions: {len(positions)}")
for pair, qty in positions.items():
    print(f"  {pair}: {qty}")

if not positions:
    print("No positions to save.")
    state = {"trailing_stops": {}, "saved_at": time.time()}
    with open("bot/state.json", "w") as f:
        json.dump(state, f, indent=2)
    print("Saved empty state to bot/state.json")
    sys.exit(0)

# 2. Get current prices
ticker = client.ticker()

# 3. Query recent orders to find entry prices
trailing_stops = {}
for pair in positions:
    # Query filled BUY orders for this pair
    orders = client.query_order(pair=pair, limit=50)

    # Find the most recent filled BUY order
    entry_price = None
    if orders:
        for order in orders:
            if (order.get("Side") == "BUY" and
                order.get("Status") == "FILLED" and
                order.get("FilledAverPrice", 0) > 0):
                entry_price = order["FilledAverPrice"]
                break  # most recent first

    # Current price for trailing high
    current_price = 0
    if ticker and pair in ticker:
        current_price = ticker[pair].get("LastPrice", 0)

    if entry_price is None:
        # Fallback: use current price as entry estimate
        entry_price = current_price
        print(f"  {pair}: no BUY order found, using current price {current_price} as entry")
    else:
        print(f"  {pair}: entry={entry_price}, current={current_price}")

    # Trailing high = max of entry and current
    high = max(entry_price, current_price) if current_price > 0 else entry_price

    trailing_stops[pair] = {
        "high": high,
        "entry_price": entry_price,
        "entry_time": time.time() - 3600,  # assume entered ~1h ago (conservative)
        "partial_taken": False,
    }

# 4. Save state
state = {
    "trailing_stops": trailing_stops,
    "positions": {p: q for p, q in positions.items()},
    "saved_at": time.time(),
}

with open("bot/state.json", "w") as f:
    json.dump(state, f, indent=2)

print(f"\nState saved to bot/state.json")
print(f"Trailing stops for {len(trailing_stops)} positions preserved.")
print(f"\nNow safe to: stop bot → pull changes → restart bot")
