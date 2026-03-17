"""
Main trading bot loop v2.
Orchestrates: data → regime → signals → derivatives → ML gate → risk → execution → logging.

Architecture:
  1. Poll data (Roostoo ticker + Binance candles + derivatives + sentiment)
  2. Detect market regime (HMM on BTC)
  3. Compute per-coin signals (breakout + mean reversion)
  4. Overlay derivatives signals (funding + OI)
  5. ML confidence gate (approve/reject entries)
  6. Risk management (REDD sizing, trailing stops, drawdown breakers)
  7. Execute orders
  8. Log everything + feed results back to ML
"""
import os
import time
import sys
import signal
import traceback
import numpy as np
from datetime import datetime, timezone

from bot.config import (
    POLL_INTERVAL_SECONDS, TRADEABLE_COINS,
    MAX_TOTAL_EXPOSURE_PCT, USE_LIMIT_ORDERS, BREAKOUT_LOOKBACK,
)
from bot.roostoo_client import RoostooClient
from bot.binance_data import BinanceData
from bot.derivatives_data import DerivativesData
from bot.regime_detector import RegimeDetector
from bot.sentiment import SentimentAnalyzer
from bot.ml_model import MLConfidenceGate
from bot.signals import compute_signal
from bot.risk_manager import RiskManager
from bot.executor import Executor
from bot.metrics import PerformanceTracker
from bot.logger import get_logger, log_cycle

log = get_logger("main")

_running = True


def _shutdown(signum, frame):
    global _running
    log.info(f"Shutdown signal received ({signum}), finishing current cycle...")
    _running = False


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


class TradingBot:
    """Main trading bot orchestrator with full signal stack."""

    def __init__(self):
        log.info("=" * 60)
        log.info("INITIALIZING TRADING BOT v2")
        log.info("=" * 60)

        # ── Core infrastructure ──
        self.client = RoostooClient()

        server_time = self.client.server_time()
        if server_time is None:
            log.error("Cannot connect to Roostoo API!")
            sys.exit(1)
        log.info(f"Connected to Roostoo. Server time: {server_time}")

        info = self.client.exchange_info()
        if info is None:
            log.error("Cannot fetch exchange info!")
            sys.exit(1)
        self.trade_pairs = info.get("TradePairs", {})
        log.info(f"Exchange has {len(self.trade_pairs)} tradeable pairs")

        self.active_pairs = [p for p in TRADEABLE_COINS if p in self.trade_pairs]
        log.info(f"Active pairs: {len(self.active_pairs)}")

        # Get initial balance
        wallet = self.client.balance()
        if wallet is None:
            log.error("Cannot fetch balance!")
            sys.exit(1)

        ticker_data = self.client.ticker() or {}
        initial_value = self._compute_portfolio_value(wallet, ticker_data)
        log.info(f"Initial portfolio value: ${initial_value:,.2f}")

        # ── Data feeds ──
        self.binance = BinanceData()
        self.derivatives = DerivativesData()

        # ── Intelligence layers ──
        self.regime = RegimeDetector()
        self.sentiment = SentimentAnalyzer()
        # Load pre-trained ML model if available
        ml_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bot", "ml_model.txt")
        self.ml_gate = MLConfidenceGate(model_path=ml_model_path)

        # ── Risk & execution ──
        self.risk_mgr = RiskManager(initial_value)
        self.executor = Executor(self.client, self.trade_pairs)
        self.perf = PerformanceTracker(initial_value)

        # ── State ──
        self.positions: dict[str, float] = {}
        self._sync_positions_from_wallet(wallet)
        # Track entry features for ML feedback: pair -> features array
        self.entry_features: dict[str, np.ndarray] = {}

        self.cycle_count = 0
        self.deriv_update_interval = 6  # update derivatives every 6 cycles (30 min)
        self.regime_refit_interval = 288  # refit HMM every ~24h

    # ─── Helpers ────────────────────────────────────────────────

    def _sync_positions_from_wallet(self, wallet: dict):
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
        exposure = 0.0
        for pair, qty in self.positions.items():
            if qty > 0 and pair in ticker_data:
                exposure += qty * ticker_data[pair].get("LastPrice", 0)
        return exposure

    # ─── Startup ────────────────────────────────────────────────

    def load_historical_data(self):
        """Load all historical data on startup."""
        log.info("Loading historical data...")

        # Binance spot candles
        self.binance.load_history(self.active_pairs, interval="5m", limit=1000)

        # Derivatives data (funding rates + OI)
        log.info("Loading derivatives data...")
        self.derivatives.load_all(self.active_pairs)

        # Fit regime detector on BTC
        btc_closes = self.binance.get_closes("BTC/USD")
        if len(btc_closes) > 100:
            self.regime.fit(btc_closes)

        # Fetch sentiment
        self.sentiment.update_fear_greed()

        log.info("All historical data loaded.")

    # ─── Main Cycle ─────────────────────────────────────────────

    def run_cycle(self):
        self.cycle_count += 1
        cycle_start = time.time()

        # ── 1. Fetch fresh data ──
        ticker_data = self.client.ticker()
        if not ticker_data:
            log.warning("Failed to fetch ticker, skipping cycle")
            return

        wallet = self.client.balance()
        if not wallet:
            log.warning("Failed to fetch balance, skipping cycle")
            return

        self._sync_positions_from_wallet(wallet)
        self.binance.update_latest(self.active_pairs)

        # ── 2. Portfolio value + risk check ──
        portfolio_value = self._compute_portfolio_value(wallet, ticker_data)
        self.risk_mgr.update_portfolio_value(portfolio_value)
        self.perf.record(portfolio_value)

        dd_check = self.risk_mgr.check_drawdown_breakers()
        if dd_check["action"] == "liquidate":
            self._liquidate_all(ticker_data)
            self.executor.cancel_all_pending()
            self._log_cycle(ticker_data, portfolio_value, {}, dd_check)
            return

        if self.risk_mgr.is_paused:
            log.info(f"Trading paused. Resume in {self.risk_mgr.get_status()['pause_remaining_min']:.0f} min")
            self._log_cycle(ticker_data, portfolio_value, {}, dd_check)
            return

        # ── 3. Update intelligence layers ──
        # Regime (every cycle — lightweight predict, heavy refit periodically)
        btc_closes = self.binance.get_closes("BTC/USD")
        self.regime.update(btc_closes)
        if self.cycle_count % self.regime_refit_interval == 0:
            self.regime.fit(btc_closes)

        self.risk_mgr.set_regime_multiplier(self.regime.get_exposure_multiplier())

        # Derivatives (every N cycles — API rate limits)
        if self.cycle_count % self.deriv_update_interval == 0:
            self.derivatives.load_all(self.active_pairs)

        spot_prices = {p: t.get("LastPrice", 0) for p, t in ticker_data.items()}
        deriv_signals = self.derivatives.compute_signals(self.active_pairs, spot_prices)

        # Sentiment (once per hour)
        if self.cycle_count % 12 == 1:
            self.sentiment.update_fear_greed()
        self.sentiment.update_btc_context(btc_closes.tolist() if len(btc_closes) > 0 else [])

        self.risk_mgr.set_sentiment_multiplier(self.sentiment.get_fg_exposure_multiplier())

        # ── 4. Manage pending orders ──
        self.executor.manage_pending_orders()

        # ── 5. Compute signals for all pairs ──
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

            sig = compute_signal(closes, highs, lows, volumes, bid, ask, BREAKOUT_LOOKBACK)

            # ── Overlay derivatives signal ──
            dsig = deriv_signals.get(pair, {})
            deriv_score = dsig.get("composite_deriv_score", 0)

            # Derivatives can boost or suppress signals
            if sig["action"] == "BUY":
                if deriv_score < -0.5:
                    # Strong bearish derivatives → suppress buy
                    sig["action"] = "HOLD"
                    sig["strategy"] = "deriv_suppress"
                elif deriv_score > 0.3:
                    # Bullish derivatives → boost strength
                    sig["strength"] = min(1.0, sig["strength"] + 0.15)

                # OI-price divergence = strong contrarian buy
                if dsig.get("oi_price_divergence") and dsig.get("oi_signal", 0) > 0:
                    if sig["action"] == "HOLD" and sig.get("rsi", 50) < 40:
                        sig["action"] = "BUY"
                        sig["strategy"] = "oi_divergence"
                        sig["strength"] = 0.7

            # ── Regime filter ──
            if sig["action"] == "BUY" and sig["strategy"] == "breakout":
                if self.regime.should_skip_breakout():
                    sig["action"] = "HOLD"
                    sig["strategy"] = "regime_suppress"

            # ── BTC lead-lag filter ──
            if sig["action"] == "BUY" and pair != "BTC/USD":
                if self.sentiment.should_skip_altcoin_buys():
                    sig["action"] = "HOLD"
                    sig["strategy"] = "btc_crash_filter"

                # BTC momentum boost for breakout entries
                if sig["strategy"] == "breakout":
                    sig["strength"] = min(1.0, sig["strength"] + self.sentiment.get_btc_signal_boost())

            sig["deriv_score"] = round(deriv_score, 3)
            sig["deriv_divergence"] = dsig.get("oi_price_divergence", False)
            sig["funding_zscore"] = dsig.get("funding_zscore", 0)
            signals[pair] = sig

        # ── 6. Check trailing stops ──
        for pair, qty in list(self.positions.items()):
            if qty <= 0:
                continue
            tick = ticker_data.get(pair, {})
            price = tick.get("LastPrice", 0)
            if price <= 0:
                continue

            self.risk_mgr.update_trailing_stop(pair, price)
            should_exit, reason = self.risk_mgr.check_trailing_stop(pair, price)
            if should_exit:
                log.info(f"STOP EXIT [{reason}]: {pair} @ {price}")
                bid = tick.get("MaxBid", 0)
                ask = tick.get("MinAsk", 0)
                self.executor.sell(pair, qty, price, bid, ask, use_limit=False)

                # Feed result to ML
                stop_info = self.risk_mgr.clear_trailing_stop(pair)
                if stop_info and pair in self.entry_features:
                    pnl = (price - stop_info["entry_price"]) / stop_info["entry_price"]
                    self.ml_gate.record_trade(self.entry_features.pop(pair), profitable=(pnl > 0))

                self.positions[pair] = 0

        # ── 7. Execute buy signals ──
        total_exposure = self._get_total_exposure_usd(ticker_data)
        num_positions = sum(1 for q in self.positions.values() if q > 0)

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

            # ── ML confidence gate ──
            # bars_per_hour=12 for 5-min candles (matches hour-based feature semantics)
            ml_features = self.ml_gate.build_features(
                self.binance.get_closes(pair),
                self.binance.get_highs(pair),
                self.binance.get_lows(pair),
                self.binance.get_volumes(pair),
                sig,
                deriv_signals.get(pair, {}),
                self.sentiment.btc_1h_return,
                self.regime.current_regime,
                bars_per_hour=12,
            )
            if ml_features is not None:
                ml_approved, ml_conf = self.ml_gate.should_enter(ml_features)
                if not ml_approved and self.ml_gate.is_trained:
                    log.debug(f"ML REJECTED: {pair} conf={ml_conf:.2f}")
                    continue

            # ── Position sizing ──
            deriv_score = sig.get("deriv_score", 0)
            size_usd = self.risk_mgr.position_size_usd(
                pair, sig["real_vol"], total_exposure, num_positions,
                signal_strength=sig["strength"],
                deriv_score=deriv_score,
            )
            if size_usd < 50:
                continue

            log.info(
                f"BUY [{sig['strategy']}]: {pair} "
                f"str={sig['strength']:.2f} rsi={sig['rsi']:.0f} "
                f"deriv={deriv_score:+.2f} regime={self.regime.REGIME_NAMES[self.regime.current_regime]} "
                f"size=${size_usd:,.0f}"
            )

            result = self.executor.buy(pair, size_usd, price, bid, ask, use_limit=USE_LIMIT_ORDERS)
            if result and result.get("Success"):
                detail = result.get("OrderDetail", {})
                filled_qty = detail.get("FilledQuantity", 0)
                if filled_qty > 0:
                    self.positions[pair] = self.positions.get(pair, 0) + filled_qty
                    self.risk_mgr.update_trailing_stop(
                        pair, price,
                        strategy=sig["strategy"],
                        entry_price=detail.get("FilledAverPrice", price),
                        entry_features=ml_features,
                    )
                    if ml_features is not None:
                        self.entry_features[pair] = ml_features
                    total_exposure += size_usd
                    num_positions = sum(1 for q in self.positions.values() if q > 0)

        # ── 8. Execute sell signals ──
        for pair, sig in signals.items():
            if sig["action"] == "SELL" and self.positions.get(pair, 0) > 0:
                tick = ticker_data.get(pair, {})
                price = tick.get("LastPrice", 0)
                bid = tick.get("MaxBid", 0)
                ask = tick.get("MinAsk", 0)
                qty = self.positions[pair]

                if dd_check["action"] == "reduce":
                    qty *= dd_check["reduce_pct"]

                urgent = sig.get("breakdown", False)
                log.info(f"SELL [{sig['strategy']}]: {pair} rsi={sig['rsi']:.0f} qty={qty}")
                self.executor.sell(pair, qty, price, bid, ask, use_limit=not urgent)

                stop_info = self.risk_mgr.clear_trailing_stop(pair)
                if stop_info and pair in self.entry_features:
                    pnl = (price - stop_info["entry_price"]) / stop_info["entry_price"]
                    self.ml_gate.record_trade(self.entry_features.pop(pair), profitable=(pnl > 0))

        # ── 9. Periodic ML retraining ──
        if self.cycle_count % 288 == 0:  # every ~24h
            if self.ml_gate.retrain_if_ready():
                log.info("ML model retrained")

        # ── 10. Logging ──
        self._log_cycle(ticker_data, portfolio_value, signals, dd_check)

        if self.cycle_count % 12 == 0:
            metrics = self.perf.summary()
            log.info(
                f"HOURLY | Ret: {metrics['total_return_pct']:.2f}% | "
                f"DD: {metrics['max_drawdown_pct']:.2f}% | "
                f"Sharpe: {metrics['sharpe']:.2f} | Sort: {metrics['sortino']:.2f} | "
                f"Calm: {metrics['calmar']:.2f} | Comp: {metrics['composite']:.2f} | "
                f"Pos: {num_positions} | Exp: ${total_exposure:,.0f} | "
                f"Regime: {self.regime.REGIME_NAMES[self.regime.current_regime]} | "
                f"FnG: {self.sentiment.fear_greed_value}"
            )

        elapsed = time.time() - cycle_start
        log.debug(f"Cycle {self.cycle_count} in {elapsed:.2f}s")

    # ─── Helpers ────────────────────────────────────────────────

    def _liquidate_all(self, ticker_data: dict):
        log.warning("LIQUIDATING ALL POSITIONS")
        for pair, qty in self.positions.items():
            if qty <= 0:
                continue
            tick = ticker_data.get(pair, {})
            price = tick.get("LastPrice", 0)
            bid = tick.get("MaxBid", 0)
            ask = tick.get("MinAsk", 0)
            self.executor.sell(pair, qty, price, bid, ask, use_limit=False)

            stop_info = self.risk_mgr.clear_trailing_stop(pair)
            if stop_info and pair in self.entry_features:
                pnl = (price - stop_info["entry_price"]) / stop_info["entry_price"]
                self.ml_gate.record_trade(self.entry_features.pop(pair), profitable=(pnl > 0))

        self.positions.clear()

    def _log_cycle(self, ticker_data: dict, portfolio_value: float, signals: dict, dd_check: dict):
        log_cycle({
            "cycle": self.cycle_count,
            "portfolio_value": round(portfolio_value, 2),
            "positions": {p: q for p, q in self.positions.items() if q > 0},
            "signals_summary": {
                p: {
                    "action": s["action"], "strategy": s["strategy"],
                    "strength": s["strength"], "rsi": s["rsi"],
                    "breakout": s["breakout"], "deriv_score": s.get("deriv_score", 0),
                }
                for p, s in signals.items() if s["action"] != "HOLD"
            },
            "regime": self.regime.get_status(),
            "sentiment": self.sentiment.get_status(),
            "risk": self.risk_mgr.get_status(),
            "ml": self.ml_gate.get_status(),
            "drawdown_action": dd_check["action"],
            "metrics": self.perf.summary(),
        })

    def run(self):
        global _running

        log.info("=" * 60)
        log.info("TRADING BOT v2 STARTING")
        log.info(f"Poll interval: {POLL_INTERVAL_SECONDS}s")
        log.info(f"Active pairs: {len(self.active_pairs)}")
        log.info(f"Modules: regime=HMM, derivatives=funding+OI, sentiment=FnG+BTC, ml=LightGBM")
        log.info("=" * 60)

        self.load_historical_data()

        while _running:
            try:
                self.run_cycle()
            except Exception as e:
                log.error(f"Cycle error: {e}\n{traceback.format_exc()}")

            next_cycle = time.time() + POLL_INTERVAL_SECONDS
            while _running and time.time() < next_cycle:
                time.sleep(1)

        log.info("Bot shutting down.")
        metrics = self.perf.summary()
        log.info(f"FINAL METRICS: {metrics}")


def main():
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
