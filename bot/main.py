"""
Main trading bot — v8 finals.

Pipeline (every hour):
  1. REGIME:   PCA (4 PCs) → 3-state HMM → data-driven exposure (0.15-1.0)
  2. FEATURES: 12 per-coin features, z-scored cross-sectionally
  3. RANK:     EWMA momentum (avg of 6h + 24h halflife) → sorted descending
  4. GATE:     r_24h > 0, volume_ratio > 0.8 (loose — ranking handles quality)
  5. SIZE:     vol-parity × REDD × regime exposure × rank multiplier
  6. EXIT:     unified stops (hard -3.5%, partial +3%, trail 3.5%/4.5%, time 60h)

Key decisions:
  - EWMA over arbitrary weights: no magic numbers, one parameter per horizon
  - No ML in ranking: tested Ridge, Lasso, ElasticNet, RF, XGBoost in shootout.
    Ridge had p=0.025 Spearman but EVERY integration (blend, rank avg, veto)
    made the full pipeline worse. EWMA alone is the cleanest signal.
  - Data-driven regime: states analyzed post-fit, exposure from forward returns.
    Min floor 0.15 for competition activity compliance.
  - ML code kept for documentation (shows rigorous testing, honest disabling).
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
    MAX_POSITIONS, MAX_NEW_ENTRIES_PER_CYCLE,
    ML_ENABLED, ML_RETRAIN_INTERVAL,
    HMM_REFIT_INTERVAL_HOURS,
)
from bot.roostoo_client import RoostooClient
from bot.binance_data import BinanceData
from bot.features import compute_coin_features, zscore_universe, check_breakdown
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
    """Main trading bot orchestrator — v7 momentum-driven pipeline."""

    def __init__(self):
        log.info("=" * 60)
        log.info("INITIALIZING TRADING BOT v7 (momentum + data-driven regime)")
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
            log.error("Cannot fetch ticker at init!")
            sys.exit(1)
        initial_value = self._compute_portfolio_value(wallet, ticker_data)
        log.info(f"Initial portfolio value: ${initial_value:,.2f}")

        # ── Data feeds ──
        self.binance = BinanceData()

        # ── Intelligence layers ──
        self.regime = RegimeDetector()
        self.ranker = Ranker()
        self.ridge_trainer = RidgeTrainer(self.active_pairs) if ML_ENABLED else None

        # ── Risk & execution ──
        self.risk_mgr = RiskManager(initial_value)
        self.executor = Executor(self.client, self.trade_pairs)
        self.perf = PerformanceTracker(initial_value)

        # ── State ──
        self.positions: dict[str, float] = {}
        self._sync_positions_from_wallet(wallet)
        self.cycle_count = 0

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
        historical_data = {}
        for pair in self.active_pairs:
            closes = self.binance.get_closes(pair)
            highs = self.binance.get_highs(pair)
            lows = self.binance.get_lows(pair)
            volumes = self.binance.get_volumes(pair)
            if len(closes) == 0:
                continue
            min_len = min(len(closes), len(highs), len(lows), len(volumes))
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

    def _refresh_ridge(self):
        """Train Ridge and set as primary ranker."""
        if self.ridge_trainer is None:
            return
        historical_data = self._build_historical_data_for_ml()
        model, r2 = self.ridge_trainer.train(historical_data)
        if model is not None:
            self.ranker.set_model(model, r2)

    # ─── Startup ────────────────────────────────────────────────

    def load_historical_data(self):
        log.info("Loading historical data (1h candles)...")
        self.binance.load_history(self.active_pairs, interval="1h", limit=1000)

        # Build price matrix and fit PCA + HMM
        price_matrix = self._build_price_matrix()
        if price_matrix is not None:
            pc_scores = self.regime.compute_pc_scores(price_matrix)
            if pc_scores is not None and len(pc_scores) > 100:
                self.regime.fit_hmm(pc_scores)

        # Initial Lasso training (optional boost)
        if ML_ENABLED:
            try:
                self._refresh_ridge()
            except Exception as e:
                log.error(f"Initial Lasso training failed: {e}")

        log.info("Historical data loaded. Bot ready.")

    # ─── Main Cycle ─────────────────────────────────────────────

    def run_cycle(self):
        self.cycle_count += 1
        cycle_start = time.time()

        # ── STEP 0: Fetch fresh data ──
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

        # Optional Lasso retrain
        if ML_ENABLED and self.cycle_count % ML_RETRAIN_INTERVAL == 1:
            try:
                self._refresh_ridge()
            except Exception as e:
                log.error(f"Lasso training failed: {e}")

        # Drawdown breakers
        dd_check = self.risk_mgr.check_drawdown_breakers()
        if dd_check["action"] == "liquidate":
            self._liquidate_all(ticker_data)
            self.executor.cancel_all_pending()
            self._log_cycle(ticker_data, portfolio_value, dd_check)
            return

        if self.risk_mgr.is_paused:
            pause_min = self.risk_mgr.get_status()["pause_remaining_min"]
            log.info(f"Trading paused. Resume in {pause_min:.0f} min")
            self._log_cycle(ticker_data, portfolio_value, dd_check)
            return

        # ── STEP 1: REGIME ──
        price_matrix = self._build_price_matrix()
        pc_scores = None
        if price_matrix is not None:
            pc_scores = self.regime.compute_pc_scores(price_matrix)

        # Refit HMM periodically (includes state analysis)
        if self.cycle_count % HMM_REFIT_INTERVAL_HOURS == 0 and pc_scores is not None:
            self.regime.fit_hmm(pc_scores)

        self.regime.update(pc_scores=pc_scores)
        self.risk_mgr.set_regime_multiplier(self.regime.get_exposure_multiplier())

        # Process pending limit orders
        fill_events = self.executor.manage_pending_orders()
        for fill in fill_events:
            pair = fill["pair"]
            filled_qty = fill["filled_qty"]
            if fill["side"] == "BUY" and filled_qty > 0:
                self.positions[pair] = self.positions.get(pair, 0) + filled_qty
                self.risk_mgr.update_trailing_stop(
                    pair, fill["filled_avg_price"],
                    entry_price=fill["filled_avg_price"],
                )
                log.info(
                    f"FILL [BUY]: {pair} qty={filled_qty:.6f} "
                    f"@ {fill['filled_avg_price']:.2f}"
                )
            elif fill["side"] == "SELL" and filled_qty > 0:
                self.positions[pair] = max(0, self.positions.get(pair, 0) - filled_qty)
                log.info(f"FILL [SELL]: {pair} qty={filled_qty:.6f}")

        regime_status = self.regime.get_status()

        if not self.regime.should_trade():
            log.info(
                f"REGIME: state={regime_status['state_name']} "
                f"exposure={regime_status['exposure_mult']:.2f} — no new entries"
            )
            self._check_exits(ticker_data)
            self._log_cycle(ticker_data, portfolio_value, dd_check)
            return

        # ── STEP 2: FEATURES ──
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

        all_zscored = zscore_universe(all_raw_features)

        # ── STEP 3+4: RANK + GATE ──
        # Build closes dict for EWMA computation
        closes_dict = {}
        for pair in all_raw_features:
            closes = self.binance.get_closes(pair)
            if closes is not None and len(closes) > 30:
                closes_dict[pair] = closes

        held_pairs = {p for p, q in self.positions.items() if q > 0}
        ranked = self.ranker.rank(all_raw_features, all_zscored, held_pairs, closes_dict)

        # ── STEP 6 (before entries): EXIT MANAGEMENT ──
        self._check_exits(ticker_data)

        # Breakdown sell signals
        for pair in list(self.positions.keys()):
            if self.positions.get(pair, 0) <= 0:
                continue
            closes = self.binance.get_closes(pair)
            lows = self.binance.get_lows(pair)
            if len(closes) < BREAKOUT_LOOKBACK + 1:
                continue
            if check_breakdown(closes, lows, BREAKOUT_LOOKBACK):
                tick = ticker_data.get(pair, {})
                price = tick.get("LastPrice", 0)
                bid = tick.get("MaxBid", 0)
                ask = tick.get("MinAsk", 0)
                qty = self.positions[pair]
                log.info(f"BREAKDOWN SELL: {pair} qty={qty:.6f} @ {price:.2f}")
                self.executor.sell(pair, qty, price, bid, ask, use_limit=False)
                self.risk_mgr.clear_trailing_stop(pair)
                self.positions[pair] = 0

        # ── STEP 5: SIZE + ENTER ──
        if not ranked:
            log.info(
                f"No candidates passed gate | "
                f"Universe: {len(all_raw_features)} | "
                f"State: {regime_status['state_name']}"
            )
            self._log_cycle(ticker_data, portfolio_value, dd_check)
            return

        total_exposure = self._get_total_exposure_usd(ticker_data)
        num_positions = sum(1 for q in self.positions.values() if q > 0)
        new_entries = 0

        for idx, (pair, score, raw_feats) in enumerate(ranked):
            if num_positions >= MAX_POSITIONS:
                break
            if new_entries >= MAX_NEW_ENTRIES_PER_CYCLE:
                break
            if self.positions.get(pair, 0) > 0:
                continue

            tick = ticker_data.get(pair, {})
            price = tick.get("LastPrice", 0)
            bid = tick.get("MaxBid", 0)
            ask = tick.get("MinAsk", 0)
            if price <= 0:
                continue

            real_vol = raw_feats.get("realized_vol", 0.5)

            # Rank-based sizing: top gets more capital
            if idx == 0:
                rank_mult = 1.3
            elif idx == 1:
                rank_mult = 1.15
            elif idx >= len(ranked) - 1 and len(ranked) > 3:
                rank_mult = 0.8
            else:
                rank_mult = 1.0

            # Signal strength from score (clamp to reasonable range)
            signal_strength = min(1.3, max(0.3, 0.5 + score * 0.5))

            size_usd = self.risk_mgr.position_size_usd(
                pair, real_vol, total_exposure, num_positions,
                signal_strength=signal_strength,
                rank_multiplier=rank_mult,
            )
            if size_usd < 50:
                continue

            log.info(
                f"BUY: {pair} | score={score:.4f} rank={idx + 1}/{len(ranked)} | "
                f"state={regime_status['state_name']} | size=${size_usd:,.0f}"
            )

            result = self.executor.buy(
                pair, size_usd, price, bid, ask,
                use_limit=USE_LIMIT_ORDERS,
            )
            if result and result.get("Success"):
                detail = result.get("OrderDetail", {})
                filled_qty = detail.get("FilledQuantity", 0)
                if filled_qty > 0:
                    self.positions[pair] = self.positions.get(pair, 0) + filled_qty
                    self.risk_mgr.update_trailing_stop(
                        pair, price,
                        entry_price=detail.get("FilledAverPrice", price),
                    )
                    total_exposure += size_usd
                    num_positions = sum(1 for q in self.positions.values() if q > 0)
                    new_entries += 1

        self._log_cycle(ticker_data, portfolio_value, dd_check)
        elapsed = time.time() - cycle_start
        log.debug(f"Cycle {self.cycle_count} completed in {elapsed:.2f}s")

    # ─── Exit Management ─────────────────────────────────────

    def _check_exits(self, ticker_data: dict):
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
                urgent = reason == "hard_stop"

                log.info(
                    f"EXIT [{reason}]: {pair} @ {price:.2f} | "
                    f"{'PARTIAL' if exit_fraction < 1 else 'FULL'} | "
                    f"{'MARKET' if urgent else 'LIMIT'}"
                )
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

    def _log_cycle(self, ticker_data, portfolio_value, dd_check):
        metrics = self.perf.summary()
        num_positions = sum(1 for q in self.positions.values() if q > 0)
        total_exposure = self._get_total_exposure_usd(ticker_data)

        log_cycle({
            "cycle": self.cycle_count,
            "portfolio_value": round(portfolio_value, 2),
            "positions": {p: q for p, q in self.positions.items() if q > 0},
            "regime": self.regime.get_status(),
            "risk": self.risk_mgr.get_status(),
            "drawdown_action": dd_check.get("action", "none"),
            "ridge_active": self.ranker.using_ridge,
            "ridge_r2": round(self.ranker.model_r2, 4),
            "metrics": metrics,
        })

        regime_status = self.regime.get_status()
        log.info(
            f"HOURLY | Ret: {metrics['total_return_pct']:.2f}% | "
            f"DD: {metrics['max_drawdown_pct']:.2f}% | "
            f"Sharpe: {metrics['sharpe']:.2f} | Sort: {metrics['sortino']:.2f} | "
            f"Calm: {metrics['calmar']:.2f} | Comp: {metrics['composite']:.2f} | "
            f"Pos: {num_positions} | Exp: ${total_exposure:,.0f} | "
            f"State: {regime_status['state_name']} "
            f"(exp={regime_status['exposure_mult']:.2f})"
        )

    def run(self):
        global _running

        log.info("=" * 60)
        log.info("TRADING BOT v7 — FINALS")
        log.info("Pipeline: regime(HMM) → features → momentum rank → gate → size → exit")
        log.info(f"Regime: HMM on {self.regime.n_pcs} PCs, states analyzed post-fit")
        log.info(f"Signal: momentum composite (optional Lasso boost)")
        log.info(f"Active pairs: {len(self.active_pairs)}")
        log.info(f"ML enabled: {ML_ENABLED}")
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