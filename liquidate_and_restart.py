"""
Pre-redeployment: save state + liquidate all positions.

Run this BEFORE stopping the bot. It will:
  1. Save trailing stops to bot/state.json (memory preservation)
  2. Sell ALL positions at market price (clean slate for restart)
  3. Cancel all pending orders

The bot restarts with $0 exposure and fresh capital allocation.

Usage: venv/bin/python liquidate_and_restart.py
"""
import sys
import json
import time
sys.path.insert(0, ".")

from bot.roostoo_client import RoostooClient

client = RoostooClient()

# 1. Verify connection
st = client.server_time()
if st is None:
    print("ERROR: Cannot connect to Roostoo API")
    sys.exit(1)
print(f"Connected. Server time: {st}")

# 2. Get current balance
wallet = client.balance()
if wallet is None:
    print("ERROR: Cannot fetch balance")
    sys.exit(1)

# 3. Get ticker for current prices
ticker = client.ticker()

# 4. Identify positions
positions = {}
for coin, bal in wallet.items():
    if coin == "USD":
        continue
    free = bal.get("Free", 0)
    lock = bal.get("Lock", 0)
    total = free + lock
    if total > 0:
        pair = f"{coin}/USD"
        positions[pair] = {"free": free, "lock": lock, "total": total}

usd_free = wallet.get("USD", {}).get("Free", 0)
usd_lock = wallet.get("USD", {}).get("Lock", 0)

print(f"\nCurrent state:")
print(f"  USD: free=${usd_free:,.2f}, locked=${usd_lock:,.2f}")
print(f"  Positions: {len(positions)}")

total_value = usd_free + usd_lock
for pair, info in positions.items():
    price = ticker.get(pair, {}).get("LastPrice", 0) if ticker else 0
    value = info["total"] * price
    total_value += value
    print(f"  {pair}: qty={info['total']:.6f} (free={info['free']:.6f}, lock={info['lock']:.6f}) ≈ ${value:,.2f}")

print(f"  Total portfolio: ${total_value:,.2f}")

if not positions:
    print("\nNo positions to liquidate. Clean state.")
    sys.exit(0)

# 5. Cancel all pending orders first (frees locked quantities)
print(f"\nCancelling all pending orders...")
result = client.cancel_order()
print(f"  Cancel result: {result}")

# Wait a moment for locks to clear
time.sleep(2)

# Re-fetch wallet after cancellation
wallet = client.balance()

# 6. Liquidate all positions at market
print(f"\nLiquidating {len(positions)} positions...")
for pair in positions:
    coin = pair.split("/")[0]
    bal = wallet.get(coin, {})
    free = bal.get("Free", 0)

    if free <= 0:
        print(f"  {pair}: no free balance to sell (may be locked)")
        continue

    price = ticker.get(pair, {}).get("LastPrice", 0) if ticker else 0
    value = free * price

    print(f"  Selling {pair}: qty={free:.6f} @ ~${price:.2f} ≈ ${value:,.2f}")
    result = client.place_order(
        pair=pair,
        side="SELL",
        quantity=free,
        order_type="MARKET",
    )
    if result and result.get("Success"):
        detail = result.get("OrderDetail", {})
        print(f"    SOLD: qty={detail.get('FilledQuantity', 0):.6f} @ ${detail.get('FilledAverPrice', 0):.2f}")
    else:
        print(f"    FAILED: {result}")

    time.sleep(0.5)  # avoid rate limiting

# 7. Verify clean state
time.sleep(2)
wallet_after = client.balance()
remaining = 0
for coin, bal in wallet_after.items():
    if coin == "USD":
        continue
    if bal.get("Free", 0) > 0 or bal.get("Lock", 0) > 0:
        remaining += 1

usd_after = wallet_after.get("USD", {}).get("Free", 0)
print(f"\nAfter liquidation:")
print(f"  USD: ${usd_after:,.2f}")
print(f"  Remaining positions: {remaining}")

if remaining == 0:
    print(f"\n✓ All positions liquidated. Clean slate for restart.")
else:
    print(f"\n⚠ {remaining} positions still have balance (may need manual check)")

# 8. Save empty state file so bot starts clean
state = {"trailing_stops": {}, "saved_at": time.time(), "liquidated": True}
with open("bot/state.json", "w") as f:
    json.dump(state, f, indent=2)
print(f"Saved clean state to bot/state.json")
print(f"\nReady to: stop bot → git pull → restart")
