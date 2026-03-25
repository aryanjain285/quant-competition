"""
Main trading bot loop v5: integrated pipeline.

Architecture:

  Step 1 — REGIME FILTER: Should we trade at all?
    HMM on Principal component 1.

  Step 2 — EVENT FILTER: Any valid setups?
    Continuation (volume-confirmed breakout with clean path)
    + Reversal (extreme overshoot, stabilizing price)
    Both use z-scored features with risk/cost penalties.

  Step 3 — VALID TRADES: Collect survivors.
    Only setups that passed regime + event + derivatives overlay.
    If empty → hold cash. No forced trades.

  Step 4 — RANKING: Which are best?
    Cross-sectional score: Using Ridge CV for ranking. 

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
from datetime import datetime, timezone, timedelta

from bot.config import (
    POLL_INTERVAL_SECONDS, TRADEABLE_COINS,
    MAX_TOTAL_EXPOSURE_PCT, USE_LIMIT_ORDERS, BREAKOUT_LOOKBACK, 
    MAX_POSITIONS
)
from bot.roostoo_client import RoostooClient
from bot.binance_data import BinanceData
from bot.features import compute_coin_features, zscore_universe, compute_submodel_scores
from bot.signals import compute_signal
from bot.ranking import Ranker
from bot.regime_detector import RegimeDetector
from bot.risk_manager import RiskManager
from bot.executor import Executor
from bot.metrics import PerformanceTracker
from bot.logger import get_logger, log_cycle
from bot.ml import RidgeTrainer

log = get_logger("main")

_running = True


def _shutdown(signum, frame):
    global _running
    log.info(f"Shutdown signal received ({signum}), finishing current cycle...")
    _running = False


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


CONTINUATION_SOFT_FLOOR = 0.0010
REVERSAL_SOFT_FLOOR = 0.0015

TOP_K_CONTINUATION = 3
TOP_K_REVERSAL = 2


def _next_hour_boundary(delay_seconds: int = 5) -> float:
    now = datetime.now(timezone.utc)
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return (next_hour + timedelta(seconds=delay_seconds)).timestamp()

def _sleep_until(ts: float):
    global _running
    while _running:
        remaining = ts - time.time()
        if remaining <= 0:
            break
        time.sleep(min(1.0, remaining))

class TradingBot:
    """Main trading bot orchestrator — v5 integrated pipeline."""

    def __init__(self):
        log.info("=" * 60)
        log.info("INITIALIZING TRADING BOT v5 (integrated pipeline)")
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

        ticker_data = self.client.ticker()
        if ticker_data is None:
            log.error("Cannot fetch ticker at init — portfolio value would be wrong!")
            sys.exit(1)
        initial_value = self._compute_portfolio_value(wallet, ticker_data)
        log.info(f"Initial portfolio value: ${initial_value:,.2f}")

        # ── Data feeds ──
        self.binance = BinanceData()

        # ── Intelligence layers ──
        self.regime = RegimeDetector()
        self.ranker = Ranker()
        self.ridge_trainer = RidgeTrainer(self.active_pairs)

        # ── Risk & execution ──
        self.risk_mgr = RiskManager(initial_value)
        self.executor = Executor(self.client, self.trade_pairs)
        self.perf = PerformanceTracker(initial_value)

        # ── State ──
        self.positions: dict[str, float] = {}
        self._sync_positions_from_wallet(wallet)

        self.cycle_count = 0
        self.regime_refit_interval = 12   # refit every 12 hours 

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
    
    def _build_historical_data_for_ml(self) -> dict[str, list[dict]]:
        """Convert BinanceData arrays into the dict-of-bars format expected by RidgeTrainer."""
        historical_data = {}
        for pair in self.active_pairs:
            closes = self.binance.get_closes(pair)
            highs = self.binance.get_highs(pair)
            lows = self.binance.get_lows(pair)
            volumes = self.binance.get_volumes(pair)

            min_len = min(len(closes), len(highs), len(lows), len(volumes))
            if min_len == 0:
                continue

            historical_data[pair] = [
                {
                    "close": float(closes[i]),
                    "high": float(highs[i]),
                    "low": float(lows[i]),
                    "volume": float(volumes[i]),
                }
                for i in range(min_len)
            ]
        return historical_data

    def _refresh_ridge_model(self, lookback_hours: int = 400, forward_horizon: int = 24):
        """Train Ridge via RidgeTrainer and hot-swap it into the ranker."""
        historical_data = self._build_historical_data_for_ml()
        model, feature_cols = self.ridge_trainer.train(
            historical_data=historical_data,
            lookback=lookback_hours,
            forward_horizon=forward_horizon,
            feature_cols=self.ranker.ridge_features,
        )

        if model is None:
            log.warning("Ridge model training skipped or failed; keeping current ranker mode.")
            return

        if feature_cols and feature_cols != self.ranker.ridge_features:
            log.warning("Ridge feature order mismatch detected; keeping ranker feature order.")

        self.ranker.set_ridge_model(model)
        log.info("Ridge model refreshed and passed to Ranker.")

    # ─── Startup ────────────────────────────────────────────────

    def load_historical_data(self):
        log.info("Loading historical data (1h candles)...")
        self.binance.load_history(self.active_pairs, interval="1h", limit=1000)

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

        # STEP 0: Fetch fresh data
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

        # LIVE MACHINE LEARNING INJECTION
        # Train the model every 6 hours (cycle 1, 7, 13, ...) to keep it fresh
        if self.cycle_count % 6 == 1:
            try:
                self._refresh_ridge_model()
            except Exception as e:
                log.error(f"ML Training failed: {e}")

        # Drawdown breakers (emergency only)
        dd_check = self.risk_mgr.check_drawdown_breakers()
        if dd_check["action"] == "liquidate":
            self._liquidate_all(ticker_data)
            self.executor.cancel_all_pending()
            self._log_cycle(ticker_data, portfolio_value, {}, {}, dd_check)
            return

        if self.risk_mgr.is_paused:
            pause_min = self.risk_mgr.get_status()["pause_remaining_min"]
            log.info(f"Trading paused. Resume in {pause_min:.0f} min")
            self._log_cycle(ticker_data, portfolio_value, {}, {}, dd_check)
            return

        # STEP 1: REGIME FILTER
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

        price_matrix = self._build_price_matrix()
        pc1_series = None
        if price_matrix is not None:
            pc1_result = self.regime.compute_pc1_market_proxy(price_matrix)
            pc1_series = pc1_result[0]

        # Refit first, then update so the new model is used immediately on refit cycles
        if self.cycle_count % self.regime_refit_interval == 0 and pc1_series is not None:
            self.regime.fit_hmm(pc1_series)

        self.regime.update(pc1_series=pc1_series)
        self.risk_mgr.set_regime_multiplier(self.regime.get_exposure_multiplier())

        # Process pending limit orders — capture fills to update positions
        fill_events = self.executor.manage_pending_orders()
        for fill in fill_events:
            pair = fill["pair"]
            filled_qty = fill["filled_qty"]

            if fill["side"] == "BUY" and filled_qty > 0:
                self.positions[pair] = self.positions.get(pair, 0) + filled_qty
                self.risk_mgr.update_trailing_stop(
                    pair,
                    fill["filled_avg_price"],
                    strategy="breakout",
                    entry_price=fill["filled_avg_price"],
                )
                log.info(
                    f"PENDING FILL [BUY]: {pair} qty={filled_qty:.6f} "
                    f"@ {fill['filled_avg_price']:.2f}"
                )
            elif fill["side"] == "SELL" and filled_qty > 0:
                self.positions[pair] = max(0, self.positions.get(pair, 0) - filled_qty)
                log.info(f"PENDING FILL [SELL]: {pair} qty={filled_qty:.6f}")

        regime_name = self.regime.get_status()["regime"]
        if not self.regime.should_trade():
            log.info(f"REGIME HOSTILE — no new entries. Regime: {regime_name}")
            self._check_exits(ticker_data, dd_check)
            self._log_cycle(ticker_data, portfolio_value, all_raw_features, {}, dd_check)
            return

        # STEP 2: EVENT FILTER
        all_zscored = zscore_universe(all_raw_features)
        for pair in all_zscored:
            compute_submodel_scores(all_zscored[pair])

        signals = {}
        for pair in all_raw_features:
            closes = self.binance.get_closes(pair)
            lows = self.binance.get_lows(pair)
            signals[pair] = compute_signal(
                all_raw_features[pair],
                all_zscored[pair],
                closes,
                lows,
                BREAKOUT_LOOKBACK,
            )

        # STEP 3: VALID TRADES — Collect survivors
        valid_candidates = {}
        for pair, sig in signals.items():
            if sig["action"] == "BUY" and self.positions.get(pair, 0) <= 0:
                valid_candidates[pair] = all_zscored.get(pair, {})
                valid_candidates[pair]["_signal"] = sig

        if not valid_candidates:
            log.info(
                f"CAN TRADE, BUT NO VALID CANDIDATES | "
                f"Raw signals: {len(signals)} | Regime: {regime_name}"
            )
            self._check_exits(ticker_data, dd_check)
            self._log_cycle(ticker_data, portfolio_value, all_raw_features, signals, dd_check)
            return

        # STEP 4: RANKING
        ranked = self.ranker.rank(valid_candidates, max_results=len(valid_candidates))

        # STEP 5: EXECUTION — Size, enter, manage, exit
        self._check_exits(ticker_data, dd_check)

        total_exposure = self._get_total_exposure_usd(ticker_data)
        num_positions = sum(1 for q in self.positions.values() if q > 0)

        continuation_candidates = []
        reversal_candidates = []

        for pair, score, features in ranked:
            sig = features.get("_signal", {})
            strategy = sig.get("strategy", "none")

            if strategy == "continuation":
                if score >= CONTINUATION_SOFT_FLOOR:
                    continuation_candidates.append((pair, score, features))
                else:
                    log.info(
                        f"Skipping {pair}: score {score:.4f} below continuation floor "
                        f"({CONTINUATION_SOFT_FLOOR:.4f})"
                    )
            elif strategy == "reversal":
                if score >= REVERSAL_SOFT_FLOOR:
                    reversal_candidates.append((pair, score, features))
                else:
                    log.info(
                        f"Skipping {pair}: score {score:.4f} below reversal floor "
                        f"({REVERSAL_SOFT_FLOOR:.4f})"
                    )

        continuation_candidates.sort(key=lambda x: x[1], reverse=True)
        reversal_candidates.sort(key=lambda x: x[1], reverse=True)

        top_continuations = continuation_candidates[:TOP_K_CONTINUATION]
        top_reversals = reversal_candidates[:TOP_K_REVERSAL]

        selected = top_continuations + top_reversals
        selected.sort(key=lambda x: x[1], reverse=True)

        for idx, (pair, score, features) in enumerate(selected):
            if num_positions >= MAX_POSITIONS:
                break
            if self.positions.get(pair, 0) > 0:
                continue

            sig = features.get("_signal", {})
            tick = ticker_data.get(pair, {})
            price = tick.get("LastPrice", 0)
            bid = tick.get("MaxBid", 0)
            ask = tick.get("MinAsk", 0)
            if price <= 0:
                continue

            real_vol = all_raw_features.get(pair, {}).get("realized_vol", 0.5)

            rank_mult = 1.0
            if regime_name == "MID_VOL":
                if idx == 0:
                    rank_mult = 1.30
                elif idx == 1:
                    rank_mult = 1.10
                elif idx == len(selected) - 1: 
                    rank_mult = 0.75

            size_usd = self.risk_mgr.position_size_usd(
                pair,
                real_vol,
                total_exposure,
                num_positions,
                signal_strength=sig.get("strength", 0.5),
                rank_multiplier=rank_mult,
            )
            if size_usd < 50:
                continue

            log.info(
                f"BUY [{sig.get('strategy', '?')}]: {pair} "
                f"rank_score={score:.4f} strength={sig.get('strength', 0):.2f} "
                f"regime={regime_name} size=${size_usd:,.0f}"
            )

            result = self.executor.buy(
                pair,
                size_usd,
                price,
                bid,
                ask,
                use_limit=USE_LIMIT_ORDERS,
            )
            if result and result.get("Success"):
                detail = result.get("OrderDetail", {})
                filled_qty = detail.get("FilledQuantity", 0)
                if filled_qty > 0:
                    self.positions[pair] = self.positions.get(pair, 0) + filled_qty
                    strategy = sig.get("strategy", "continuation")
                    self.risk_mgr.update_trailing_stop(
                        pair,
                        price,
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
                log.info(
                    f"SELL [{sig.get('strategy', '?')}]: {pair} qty={qty} "
                    f"{'MARKET' if urgent else 'LIMIT'}"
                )
                self.executor.sell(pair, qty, price, bid, ask, use_limit=not urgent)
                self.risk_mgr.clear_trailing_stop(pair)
                self.positions[pair] = 0

        self._log_cycle(ticker_data, portfolio_value, all_raw_features, signals, dd_check)

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
        metrics = self.perf.summary()
        num_positions = sum(1 for q in self.positions.values() if q > 0)
        total_exposure = self._get_total_exposure_usd(ticker_data)

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
            "metrics": metrics,
        })

        log.info(
            f"HOURLY | Ret: {metrics['total_return_pct']:.2f}% | "
            f"DD: {metrics['max_drawdown_pct']:.2f}% | "
            f"Sharpe: {metrics['sharpe']:.2f} | Sort: {metrics['sortino']:.2f} | "
            f"Calm: {metrics['calmar']:.2f} | Comp: {metrics['composite']:.2f} | "
            f"Pos: {num_positions} | Exp: ${total_exposure:,.0f} | "
            f"Regime: {self.regime.get_status()['regime']}"
        )

    def run(self):
        global _running

        log.info("=" * 60)
        log.info("TRADING BOT v5 STARTING")
        log.info(f"Poll interval: {POLL_INTERVAL_SECONDS}s")
        log.info(f"Active pairs: {len(self.active_pairs)}")
        log.info(f"Pipeline: regime → events → valid trades → ranking → execution")
        log.info(f"Regime: HMM on PCA | Signals: continuation + reversal")
        log.info(f"Ranking: Ridge CV")
        log.info("=" * 60)

        self.load_historical_data()

        while _running:
            try:
                log.info(f"About to run cycle {self.cycle_count + 1}")
                self.run_cycle()
                log.info(f"Finished cycle {self.cycle_count}")
            except Exception as e:
                log.error(f"Cycle error: {e}\n{traceback.format_exc()}")

            wake_ts = _next_hour_boundary(delay_seconds=5)
            _sleep_until(wake_ts)

        log.info("Bot shutting down.")
        metrics = self.perf.summary()
        log.info(f"FINAL METRICS: {metrics}")


def main():
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
