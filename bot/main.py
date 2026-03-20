"""
Main trading bot loop v4: integrated pipeline.

Architecture (maps to teammate's 5-step brief):

  Step 1 — REGIME FILTER: Should we trade at all?
    Cross-sectional market features (breadth, trend strength, stress)
    + HMM on BTC. Sets exposure budget. Can disable all new longs.

  Step 2 — EVENT FILTER: Any valid setups?
    Continuation (volume-confirmed breakout with clean path)
    + Reversal (extreme overshoot, stabilizing price)
    Both use z-scored features with risk/cost penalties.

  Step 3 — VALID TRADES: Collect survivors.
    Only setups that passed regime + event + derivatives overlay.
    If empty → hold cash. No forced trades.

  Step 4 — RANKING: Which are best?
    Cross-sectional score: weighted sum of z-scored features.
    Top N selected for execution. Ready for ridge drop-in.

  Step 5 — EXECUTION: Size, enter, manage, exit.
    Vol-parity × REDD × regime multiplier.
    Partial exits, trailing stops, limit orders.
"""
import os
import time
import sys
import signal
import traceback
import numpy as np
from typing import Optional
from datetime import datetime, timezone

from bot.config import (
    POLL_INTERVAL_SECONDS, TRADEABLE_COINS,
    MAX_TOTAL_EXPOSURE_PCT, USE_LIMIT_ORDERS, BREAKOUT_LOOKBACK,
    MAX_POSITIONS,
)
from bot.roostoo_client import RoostooClient
from bot.binance_data import BinanceData
from bot.derivatives_data import DerivativesData
from bot.features import compute_coin_features, zscore_universe, compute_submodel_scores, compute_market_features
from bot.signals import compute_signal
from bot.ranking import Ranker
from bot.regime_detector import RegimeDetector
from bot.sentiment import SentimentAnalyzer
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
    """Main trading bot orchestrator — v4 integrated pipeline."""

    def __init__(self):
        log.info("=" * 60)
        log.info("INITIALIZING TRADING BOT v4 (integrated pipeline)")
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
        # self.sentiment = SentimentAnalyzer()
        self.ranker = Ranker()

        # ── Risk & execution ──
        self.risk_mgr = RiskManager(initial_value)
        self.executor = Executor(self.client, self.trade_pairs)
        self.perf = PerformanceTracker(initial_value)

        # ── State ──
        self.positions: dict[str, float] = {}
        self._sync_positions_from_wallet(wallet)

        self.cycle_count = 0
        self.deriv_update_interval = 6    # every 30 min
        self.regime_refit_interval = 288  # every 24h

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
    
    def _build_price_matrix(self, min_bars: int = 100) -> Optional[np.ndarray]:
        """Build time × assets close-price matrix from loaded candles.

        Returns np.ndarray of shape (time, assets) or None.
        """
        close_series = []
        for pair in self.active_pairs:
            closes = self.binance.get_closes(pair)
            if closes is None or len(closes) < min_bars:
                continue
            arr = np.asarray(closes, dtype=float)
            if np.any(~np.isfinite(arr)):
                continue
            close_series.append(arr)

        if len(close_series) < 2:
            return None

        min_len = min(len(x) for x in close_series)
        return np.column_stack([x[-min_len:] for x in close_series])
        
    # ─── Startup ────────────────────────────────────────────────

    # def load_historical_data(self):
    #     log.info("Loading historical data...")
    #     self.binance.load_history(self.active_pairs, interval="5m", limit=1000)
    #     log.info("Loading derivatives data...")
    #     self.derivatives.load_all(self.active_pairs)

    #     btc_closes = self.binance.get_closes("BTC/USD")
    #     if len(btc_closes) > 100:
    #         self.regime.fit_hmm(btc_closes)

    #     self.sentiment.update_fear_greed()
    #     log.info("All historical data loaded.")

    def load_historical_data(self):
        log.info("Loading historical data (1h candles)...")
        self.binance.load_history(self.active_pairs, interval="1h", limit=1000)
        log.info("Loading derivatives data...")
        self.derivatives.load_all(self.active_pairs)

        # Build price matrix and fit PCA + HMM
        price_matrix = self._build_price_matrix()
        if price_matrix is not None:
            pc1_series, _, pc1_var, _ = self.regime.compute_pc1_market_proxy(price_matrix)
            if pc1_series is not None and len(pc1_series) > 100:
                self.regime.fit_hmm(pc1_series)
                log.info(f"PCA-HMM fitted: PC1 explains {pc1_var:.1%} of variance")

        log.info("All historical data loaded.")

    # ─── Main Cycle ─────────────────────────────────────────────

    def run_cycle(self):
        self.cycle_count += 1
        cycle_start = time.time()

        # ══════════════════════════════════════════════════════
        # STEP 0: Fetch fresh data
        # ══════════════════════════════════════════════════════
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

        portfolio_value = self._compute_portfolio_value(wallet, ticker_data)
        self.risk_mgr.update_portfolio_value(portfolio_value)
        self.perf.record(portfolio_value)

        # Drawdown breakers (emergency only)
        dd_check = self.risk_mgr.check_drawdown_breakers()
        if dd_check["action"] == "liquidate":
            self._liquidate_all(ticker_data)
            self.executor.cancel_all_pending()
            self._log_cycle(ticker_data, portfolio_value, {}, {}, dd_check)
            return

        if self.risk_mgr.is_paused:
            log.info(f"Trading paused. Resume in {self.risk_mgr.get_status()['pause_remaining_min']:.0f} min")
            self._log_cycle(ticker_data, portfolio_value, {}, {}, dd_check)
            return

        # ══════════════════════════════════════════════════════
        # STEP 1: REGIME FILTER — Should we trade at all?
        # ══════════════════════════════════════════════════════

        # Compute features for ALL coins first (needed for market-level features)
        all_raw_features = {}
        for pair in self.active_pairs:
            closes = self.binance.get_closes(pair)
            highs = self.binance.get_highs(pair)
            lows = self.binance.get_lows(pair)
            volumes = self.binance.get_volumes(pair)
            if len(closes) < 100:
                continue
            tick = ticker_data.get(pair, {})
            bid = tick.get("MaxBid", 0)
            ask = tick.get("MinAsk", 0)

            feats = compute_coin_features(closes, highs, lows, volumes, bid, ask)
            if feats:
                all_raw_features[pair] = feats

        # Market-level features for regime
        market_feats = compute_market_features(all_raw_features)

        # PCA market proxy → regime detection
        # PCA loadings cached internally, refit every 6h (not every cycle)
        price_matrix = self._build_price_matrix()
        pc1_series = None
        if price_matrix is not None:
            pc1_result = self.regime.compute_pc1_market_proxy(price_matrix)
            pc1_series = pc1_result[0]
            if pc1_result[2] > 0:
                market_feats["pc1_explained_var"] = pc1_result[2]

        # Update regime: rules + PCA-HMM
        self.regime.update(market_feats, pc1_series)

        # Refit HMM on PC1 every 24h (heavy operation)
        if self.cycle_count % self.regime_refit_interval == 0 and pc1_series is not None:
            self.regime.fit_hmm(pc1_series)

        self.risk_mgr.set_regime_multiplier(self.regime.get_exposure_multiplier())

        # Derivatives (every N cycles)
        if self.cycle_count % self.deriv_update_interval == 0:
            self.derivatives.load_all(self.active_pairs)
        spot_prices = {p: t.get("LastPrice", 0) for p, t in ticker_data.items()}
        deriv_signals = self.derivatives.compute_signals(self.active_pairs, spot_prices)

        # Manage pending orders
        self.executor.manage_pending_orders()

        # If regime is hostile, skip all new entries
        if not self.regime.should_trade():
            log.info(f"REGIME HOSTILE — no new entries. "
                     f"Market: breadth={market_feats.get('breadth', 0):.2f} "
                     f"trend={market_feats.get('trend_strength', 0):.4f}")
            self._check_exits(ticker_data, dd_check)
            self._log_cycle(ticker_data, portfolio_value, all_raw_features, {}, dd_check)
            return

        # ══════════════════════════════════════════════════════
        # STEP 2: EVENT FILTER — Any valid setups?
        # ══════════════════════════════════════════════════════

        # Z-score features cross-sectionally
        all_zscored = zscore_universe(all_raw_features)

        # Add submodel scores (risk_penalty, cost_penalty) to z-scored
        for pair in all_zscored:
            compute_submodel_scores(all_zscored[pair])

        # Compute signals for each coin
        signals = {}
        for pair in all_raw_features:
            closes = self.binance.get_closes(pair)
            lows = self.binance.get_lows(pair)

            sig = compute_signal(
                all_raw_features[pair], all_zscored[pair],
                closes, lows, BREAKOUT_LOOKBACK,
            )

            # Derivatives overlay
            dsig = deriv_signals.get(pair, {})
            deriv_score = dsig.get("composite_deriv_score", 0)

            if sig["action"] == "BUY":
                if deriv_score < -0.5:
                    sig["action"] = "HOLD"
                    sig["strategy"] = "deriv_suppress"
                elif deriv_score > 0.3:
                    sig["strength"] = min(1.0, sig["strength"] + 0.15)

                # OI-price divergence contrarian
                if dsig.get("oi_price_divergence") and dsig.get("oi_signal", 0) > 0:
                    if sig["action"] == "HOLD" and all_raw_features[pair].get("overshoot", 0) < -1.0:
                        sig["action"] = "BUY"
                        sig["strategy"] = "oi_divergence"
                        sig["strength"] = 0.7

            # # BTC crash filter
            # if sig["action"] == "BUY" and pair != "BTC/USD":
            #     if self.sentiment.should_skip_altcoin_buys():
            #         sig["action"] = "HOLD"
            #         sig["strategy"] = "btc_crash_filter"
            #     elif sig["strategy"] == "continuation":
            #         sig["strength"] = min(1.0, sig["strength"] + self.sentiment.get_btc_signal_boost())

            sig["deriv_score"] = round(deriv_score, 3)
            sig["deriv_divergence"] = dsig.get("oi_price_divergence", False)
            sig["funding_zscore"] = dsig.get("funding_zscore", 0)
            signals[pair] = sig

        # ══════════════════════════════════════════════════════
        # STEP 3: VALID TRADES — Collect survivors
        # ══════════════════════════════════════════════════════

        valid_candidates = {}
        for pair, sig in signals.items():
            if sig["action"] == "BUY" and self.positions.get(pair, 0) <= 0:
                valid_candidates[pair] = all_zscored.get(pair, {})
                valid_candidates[pair]["_signal"] = sig

        if not valid_candidates:
            # No valid trades — just manage exits and hold cash
            self._check_exits(ticker_data, dd_check)
            self._log_cycle(ticker_data, portfolio_value, all_raw_features, signals, dd_check)
            return

        # ══════════════════════════════════════════════════════
        # STEP 4: RANKING — Which are best?
        # ══════════════════════════════════════════════════════

        ranked = self.ranker.rank(valid_candidates, max_results=MAX_POSITIONS)

        # ══════════════════════════════════════════════════════
        # STEP 5: EXECUTION — Size, enter, manage, exit
        # ══════════════════════════════════════════════════════

        # First check exits on existing positions
        self._check_exits(ticker_data, dd_check)

        # Execute buys for top-ranked candidates
        total_exposure = self._get_total_exposure_usd(ticker_data)
        num_positions = sum(1 for q in self.positions.values() if q > 0)

        for pair, score, features in ranked:
            if num_positions >= MAX_POSITIONS:
                break

            sig = features.get("_signal", {})
            tick = ticker_data.get(pair, {})
            price = tick.get("LastPrice", 0)
            bid = tick.get("MaxBid", 0)
            ask = tick.get("MinAsk", 0)
            if price <= 0:
                continue

            real_vol = all_raw_features.get(pair, {}).get("realized_vol", 0.5)
            deriv_score = sig.get("deriv_score", 0)

            size_usd = self.risk_mgr.position_size_usd(
                pair, real_vol, total_exposure, num_positions,
                signal_strength=sig.get("strength", 0.5),
                deriv_score=deriv_score,
            )
            if size_usd < 50:
                continue

            log.info(
                f"BUY [{sig.get('strategy', '?')}]: {pair} "
                f"rank_score={score:.3f} strength={sig.get('strength', 0):.2f} "
                f"deriv={deriv_score:+.2f} "
                f"regime={self.regime.get_status()['regime']} "
                f"size=${size_usd:,.0f}"
            )

            result = self.executor.buy(pair, size_usd, price, bid, ask, use_limit=USE_LIMIT_ORDERS)
            if result and result.get("Success"):
                detail = result.get("OrderDetail", {})
                filled_qty = detail.get("FilledQuantity", 0)
                if filled_qty > 0:
                    self.positions[pair] = self.positions.get(pair, 0) + filled_qty
                    strategy = sig.get("strategy", "continuation")
                    self.risk_mgr.update_trailing_stop(
                        pair, price,
                        strategy="mean_rev" if strategy == "reversal" else "breakout",
                        entry_price=detail.get("FilledAverPrice", price),
                    )
                    total_exposure += size_usd
                    num_positions = sum(1 for q in self.positions.values() if q > 0)

        # Handle sell signals
        for pair, sig in signals.items():
            if sig["action"] == "SELL" and self.positions.get(pair, 0) > 0:
                tick = ticker_data.get(pair, {})
                price = tick.get("LastPrice", 0)
                bid = tick.get("MaxBid", 0)
                ask = tick.get("MinAsk", 0)
                qty = self.positions[pair]

                urgent = sig.get("breakdown", False)
                log.info(f"SELL [{sig.get('strategy', '?')}]: {pair} qty={qty} "
                         f"{'MARKET' if urgent else 'LIMIT'}")
                self.executor.sell(pair, qty, price, bid, ask, use_limit=not urgent)
                self.risk_mgr.clear_trailing_stop(pair)

        # ══════════════════════════════════════════════════════
        # LOGGING
        # ══════════════════════════════════════════════════════
        self._log_cycle(ticker_data, portfolio_value, all_raw_features, signals, dd_check)

        if self.cycle_count % 12 == 0:
            metrics = self.perf.summary()
            log.info(
                f"HOURLY | Ret: {metrics['total_return_pct']:.2f}% | "
                f"DD: {metrics['max_drawdown_pct']:.2f}% | "
                f"Sharpe: {metrics['sharpe']:.2f} | Sort: {metrics['sortino']:.2f} | "
                f"Calm: {metrics['calmar']:.2f} | Comp: {metrics['composite']:.2f} | "
                f"Pos: {num_positions} | Exp: ${total_exposure:,.0f} | "
                f"Regime: {self.regime.get_status()['regime']} | "
                f"Breadth: {market_feats.get('breadth', 0):.2f}"
            )

        elapsed = time.time() - cycle_start
        log.debug(f"Cycle {self.cycle_count} in {elapsed:.2f}s")

    # ─── Exit Management ─────────────────────────────────────

    def _check_exits(self, ticker_data: dict, dd_check: dict):
        """Check trailing stops and partial exits on all positions."""
        for pair, qty in list(self.positions.items()):
            if qty <= 0:
                continue
            tick = ticker_data.get(pair, {})
            price = tick.get("LastPrice", 0)
            if price <= 0:
                continue

            self.risk_mgr.update_trailing_stop(pair, price)
            should_exit, reason, exit_fraction = self.risk_mgr.check_trailing_stop(pair, price)
            if should_exit:
                sell_qty = qty * exit_fraction
                bid = tick.get("MaxBid", 0)
                ask = tick.get("MinAsk", 0)

                urgent = reason in ("hard_stop",)
                log.info(f"STOP EXIT [{reason}]: {pair} @ {price} "
                         f"({'PARTIAL 50%' if exit_fraction < 1 else 'FULL'} "
                         f"{'MARKET' if urgent else 'LIMIT'})")
                self.executor.sell(pair, sell_qty, price, bid, ask, use_limit=not urgent)

                if exit_fraction >= 1.0:
                    self.risk_mgr.clear_trailing_stop(pair)
                    self.positions[pair] = 0
                else:
                    self.positions[pair] = qty - sell_qty

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
        self.positions.clear()

    def _log_cycle(self, ticker_data, portfolio_value, raw_feats, signals, dd_check):
        log_cycle({
            "cycle": self.cycle_count,
            "portfolio_value": round(portfolio_value, 2),
            "positions": {p: q for p, q in self.positions.items() if q > 0},
            "signals_summary": {
                p: {
                    "action": s["action"], "strategy": s.get("strategy", "none"),
                    "strength": s.get("strength", 0),
                }
                for p, s in signals.items() if s.get("action") != "HOLD"
            } if signals else {},
            "regime": self.regime.get_status(),
            #"sentiment": self.sentiment.get_status(),
            "risk": self.risk_mgr.get_status(),
            "drawdown_action": dd_check.get("action", "none"),
            "metrics": self.perf.summary(),
        })

    def run(self):
        global _running

        log.info("=" * 60)
        log.info("TRADING BOT v4 STARTING")
        log.info(f"Poll interval: {POLL_INTERVAL_SECONDS}s")
        log.info(f"Active pairs: {len(self.active_pairs)}")
        log.info(f"Pipeline: regime → events → valid trades → ranking → execution")
        log.info(f"Regime: rule-based + HMM | Signals: continuation + reversal")
        log.info(f"Ranking: hand-built score (ridge-ready)")
        log.info("=" * 60)

        self.load_historical_data()

        while _running:
            try:
                log.info(f"About to run cycle {self.cycle_count + 1}")
                self.run_cycle()
                log.info(f"Finished cycle {self.cycle_count}")
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
