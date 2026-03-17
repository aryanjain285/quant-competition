"""
Main trading bot loop.
Orchestrates: data polling -> signal generation -> risk checks -> execution -> logging.
"""
import time
import sys
import signal
import traceback
from datetime import datetime, timezone

from bot.config import (
    POLL_INTERVAL_SECONDS, TRADEABLE_COINS,
    MAX_TOTAL_EXPOSURE_PCT, USE_LIMIT_ORDERS,
)
from bot.roostoo_client import RoostooClient
from bot.binance_data import BinanceData
from bot.signals import compute_signal
from bot.risk_manager import RiskManager
from bot.executor import Executor
from bot.metrics import PerformanceTracker
from bot.logger import get_logger, log_cycle

log = get_logger("main")

# Graceful shutdown
_running = True


def _shutdown(signum, frame):
    global _running
    log.info(f"Shutdown signal received ({signum}), finishing current cycle...")
    _running = False


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self):
        log.info("Initializing trading bot...")

        # API client
        self.client = RoostooClient()

        # Verify connectivity
        server_time = self.client.server_time()
        if server_time is None:
            log.error("Cannot connect to Roostoo API!")
            sys.exit(1)
        log.info(f"Connected to Roostoo. Server time: {server_time}")

        # Load exchange info
        info = self.client.exchange_info()
        if info is None:
            log.error("Cannot fetch exchange info!")
            sys.exit(1)
        self.trade_pairs = info.get("TradePairs", {})
        log.info(f"Exchange has {len(self.trade_pairs)} tradeable pairs")

        # Filter to pairs we want to trade (that exist on exchange)
        self.active_pairs = [p for p in TRADEABLE_COINS if p in self.trade_pairs]
        log.info(f"Active pairs: {len(self.active_pairs)} — {self.active_pairs[:10]}...")

        # Get initial balance
        wallet = self.client.balance()
        if wallet is None:
            log.error("Cannot fetch balance!")
            sys.exit(1)
        usd_free = wallet.get("USD", {}).get("Free", 0)
        usd_lock = wallet.get("USD", {}).get("Lock", 0)
        initial_value = usd_free + usd_lock
        # Add value of any existing coin holdings
        ticker_data = self.client.ticker()
        if ticker_data:
            for coin, bal in wallet.items():
                if coin == "USD":
                    continue
                free = bal.get("Free", 0) + bal.get("Lock", 0)
                if free > 0:
                    pair = f"{coin}/USD"
                    if pair in ticker_data:
                        initial_value += free * ticker_data[pair].get("LastPrice", 0)

        log.info(f"Initial portfolio value: ${initial_value:,.2f}")

        # Binance data feed
        self.binance = BinanceData()

        # Risk manager
        self.risk_mgr = RiskManager(initial_value)

        # Executor
        self.executor = Executor(self.client, self.trade_pairs)

        # Performance tracker
        self.perf = PerformanceTracker(initial_value)

        # Track our positions: pair -> quantity held
        self.positions: dict[str, float] = {}
        self._sync_positions_from_wallet(wallet)

        self.cycle_count = 0

    def _sync_positions_from_wallet(self, wallet: dict):
        """Sync internal position tracker with actual wallet."""
        self.positions.clear()
        for coin, bal in wallet.items():
            if coin == "USD":
                continue
            free = bal.get("Free", 0)
            if free > 0:
                pair = f"{coin}/USD"
                if pair in self.trade_pairs:
                    self.positions[pair] = free

    def _compute_portfolio_value(self, wallet: dict, ticker_data: dict) -> float:
        """Compute total portfolio value in USD."""
        value = wallet.get("USD", {}).get("Free", 0) + wallet.get("USD", {}).get("Lock", 0)
        for coin, bal in wallet.items():
            if coin == "USD":
                continue
            total = bal.get("Free", 0) + bal.get("Lock", 0)
            if total > 0:
                pair = f"{coin}/USD"
                if pair in ticker_data:
                    value += total * ticker_data[pair].get("LastPrice", 0)
        return value

    def _get_total_exposure_usd(self, ticker_data: dict) -> float:
        """Get total USD value of all crypto positions."""
        exposure = 0.0
        for pair, qty in self.positions.items():
            if qty > 0 and pair in ticker_data:
                exposure += qty * ticker_data[pair].get("LastPrice", 0)
        return exposure

    def load_historical_data(self):
        """Load historical candle data from Binance on startup."""
        log.info("Loading historical data from Binance...")
        self.binance.load_history(self.active_pairs, interval="5m", limit=1000)
        log.info("Historical data loaded.")

    def run_cycle(self):
        """Execute one trading cycle."""
        self.cycle_count += 1
        cycle_start = time.time()

        # 1. Fetch fresh data
        ticker_data = self.client.ticker()
        if not ticker_data:
            log.warning("Failed to fetch ticker data, skipping cycle")
            return

        wallet = self.client.balance()
        if not wallet:
            log.warning("Failed to fetch balance, skipping cycle")
            return

        # Sync positions
        self._sync_positions_from_wallet(wallet)

        # Update Binance candles
        self.binance.update_latest(self.active_pairs)

        # 2. Compute portfolio value
        portfolio_value = self._compute_portfolio_value(wallet, ticker_data)
        self.risk_mgr.update_portfolio_value(portfolio_value)
        self.perf.record(portfolio_value)

        # 3. Check drawdown breakers
        dd_check = self.risk_mgr.check_drawdown_breakers()
        if dd_check["action"] == "liquidate":
            self._liquidate_all(ticker_data)
            self.executor.cancel_all_pending()
            self._log_cycle(ticker_data, portfolio_value, {}, dd_check)
            return

        if self.risk_mgr.is_paused:
            log.info(f"Trading paused (drawdown breaker). Resuming in {self.risk_mgr.get_status()['pause_remaining_min']:.0f} min")
            self._log_cycle(ticker_data, portfolio_value, {}, dd_check)
            return

        # 4. Manage pending orders (cancel stale)
        self.executor.manage_pending_orders()

        # 5. Compute signals for all active pairs
        signals = {}
        for pair in self.active_pairs:
            closes = self.binance.get_closes(pair)
            highs = self.binance.get_highs(pair)
            lows = self.binance.get_lows(pair)
            volumes = self.binance.get_volumes(pair)
            if len(closes) < 80:
                continue

            tick = ticker_data.get(pair, {})
            bid = tick.get("MaxBid", 0)
            ask = tick.get("MinAsk", 0)

            sig = compute_signal(closes, highs, lows, volumes, bid, ask)
            signals[pair] = sig

        # 6. Check trailing stops on existing positions
        for pair, qty in list(self.positions.items()):
            if qty <= 0:
                continue
            tick = ticker_data.get(pair, {})
            price = tick.get("LastPrice", 0)
            if price <= 0:
                continue

            self.risk_mgr.update_trailing_stop(pair, price)
            if self.risk_mgr.check_trailing_stop(pair, price):
                # Trailing stop hit — sell immediately via market
                log.info(f"TRAILING STOP EXIT: {pair} @ {price}")
                bid = tick.get("MaxBid", 0)
                ask = tick.get("MinAsk", 0)
                self.executor.sell(pair, qty, price, bid, ask, use_limit=False)
                self.risk_mgr.clear_trailing_stop(pair)
                # Remove from positions (will resync next cycle)
                self.positions[pair] = 0

        # 7. Execute signal-based trades
        total_exposure = self._get_total_exposure_usd(ticker_data)
        num_positions = sum(1 for q in self.positions.values() if q > 0)

        # Sort by signal strength (strongest first)
        buy_candidates = [
            (pair, sig) for pair, sig in signals.items()
            if sig["action"] == "BUY" and self.positions.get(pair, 0) <= 0
        ]
        buy_candidates.sort(key=lambda x: x[1]["strength"], reverse=True)

        for pair, sig in buy_candidates:
            tick = ticker_data.get(pair, {})
            price = tick.get("LastPrice", 0)
            bid = tick.get("MaxBid", 0)
            ask = tick.get("MinAsk", 0)

            if price <= 0:
                continue

            # Get position size from risk manager
            size_usd = self.risk_mgr.position_size_usd(
                pair, sig["real_vol"], total_exposure, num_positions,
            )
            if size_usd < 10:  # minimum $10 order
                continue

            # Scale position size by signal strength
            size_usd *= sig["strength"]

            log.info(f"BUY SIGNAL [{sig['strategy']}]: {pair} "
                     f"strength={sig['strength']:.2f} rsi={sig['rsi']:.0f} "
                     f"breakout={sig['breakout']} vol_confirm={sig['volume_confirm']} "
                     f"size=${size_usd:,.0f}")

            result = self.executor.buy(
                pair, size_usd, price, bid, ask, use_limit=USE_LIMIT_ORDERS,
            )
            if result and result.get("Success"):
                detail = result.get("OrderDetail", {})
                filled_qty = detail.get("FilledQuantity", 0)
                if filled_qty > 0:
                    self.positions[pair] = self.positions.get(pair, 0) + filled_qty
                    self.risk_mgr.update_trailing_stop(
                        pair, price,
                        strategy=sig["strategy"],
                        entry_price=detail.get("FilledAverPrice", price),
                    )
                    total_exposure += size_usd
                    num_positions = sum(1 for q in self.positions.values() if q > 0)

        # Handle sell signals for held positions
        for pair, sig in signals.items():
            if sig["action"] == "SELL" and self.positions.get(pair, 0) > 0:
                tick = ticker_data.get(pair, {})
                price = tick.get("LastPrice", 0)
                bid = tick.get("MaxBid", 0)
                ask = tick.get("MinAsk", 0)
                qty = self.positions[pair]

                if dd_check["action"] == "reduce":
                    qty *= dd_check["reduce_pct"]
                    qty = max(qty, 0)

                # Breakdowns exit urgently via market; mean-rev profit-takes can use limit
                urgent = sig["strategy"] == "breakout" and sig["breakdown"]
                log.info(f"SELL SIGNAL [{sig['strategy']}]: {pair} "
                         f"rsi={sig['rsi']:.0f} breakdown={sig['breakdown']} qty={qty}")
                self.executor.sell(pair, qty, price, bid, ask, use_limit=not urgent)
                self.risk_mgr.clear_trailing_stop(pair)

        # 8. Log cycle data
        self._log_cycle(ticker_data, portfolio_value, signals, dd_check)

        # Periodic performance summary
        if self.cycle_count % 12 == 0:  # every hour
            metrics = self.perf.summary()
            risk_status = self.risk_mgr.get_status()
            log.info(
                f"HOURLY SUMMARY | Return: {metrics['total_return_pct']:.2f}% | "
                f"MaxDD: {metrics['max_drawdown_pct']:.2f}% | "
                f"Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f} | "
                f"Calmar: {metrics['calmar']:.2f} | Composite: {metrics['composite']:.2f} | "
                f"Positions: {num_positions} | Exposure: ${total_exposure:,.0f}"
            )

        elapsed = time.time() - cycle_start
        log.debug(f"Cycle {self.cycle_count} completed in {elapsed:.2f}s")

    def _liquidate_all(self, ticker_data: dict):
        """Emergency liquidation: sell all positions at market."""
        log.warning("LIQUIDATING ALL POSITIONS")
        for pair, qty in self.positions.items():
            if qty <= 0:
                continue
            tick = ticker_data.get(pair, {})
            price = tick.get("LastPrice", 0)
            bid = tick.get("MaxBid", 0)
            ask = tick.get("MinAsk", 0)
            self.executor.sell(pair, qty, price, bid, ask, use_limit=False)
            self.risk_mgr.clear_trailing_stop(pair)
        self.positions.clear()

    def _log_cycle(self, ticker_data: dict, portfolio_value: float, signals: dict, dd_check: dict):
        """Log cycle state for compliance and debugging."""
        log_cycle({
            "cycle": self.cycle_count,
            "portfolio_value": round(portfolio_value, 2),
            "positions": {p: q for p, q in self.positions.items() if q > 0},
            "signals": {
                p: {
                    "action": s["action"], "strategy": s["strategy"],
                    "strength": s["strength"], "rsi": s["rsi"],
                    "breakout": s["breakout"], "breakdown": s["breakdown"],
                }
                for p, s in signals.items()
            },
            "risk": self.risk_mgr.get_status(),
            "drawdown_action": dd_check["action"],
            "metrics": self.perf.summary(),
        })

    def run(self):
        """Main loop: run cycles at the configured interval."""
        global _running

        log.info("=" * 60)
        log.info("TRADING BOT STARTING")
        log.info(f"Poll interval: {POLL_INTERVAL_SECONDS}s")
        log.info(f"Active pairs: {len(self.active_pairs)}")
        log.info("=" * 60)

        # Load historical data on startup
        self.load_historical_data()

        while _running:
            try:
                self.run_cycle()
            except Exception as e:
                log.error(f"Cycle error: {e}\n{traceback.format_exc()}")

            # Sleep until next cycle
            next_cycle = time.time() + POLL_INTERVAL_SECONDS
            while _running and time.time() < next_cycle:
                time.sleep(1)

        log.info("Bot shutting down gracefully.")
        # Final metrics
        metrics = self.perf.summary()
        log.info(f"FINAL METRICS: {metrics}")


def main():
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
