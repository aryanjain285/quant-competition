"""
Backtest engine v6: mirrors main.py's run_cycle() exactly.

Design: import and call the SAME modules as live trading:
  - features.compute_coin_features (same features)
  - features.zscore_universe (same z-scoring)
  - features.check_breakdown (same exit signal)
  - ranking.Ranker with Lasso (same ranking + gate)
  - regime_detector.RegimeDetector (same PCA-HMM regime)
  - risk_manager.RiskManager (same unified stops, REDD, sizing)
  - ml.LassoTrainer (same training pipeline)

The ONLY difference is SimExchange instead of RoostooClient.

Pipeline (matches main.py exactly):
  1. Regime: PCA → 3 PCs → HMM → trade or sit out
  2. Features: compute per-coin, z-score cross-sectionally
  3. Rank: Lasso predicted 24h return for all coins
  4. Gate: predicted return > ENTRY_THRESHOLD
  5. Size: vol-parity × REDD × regime × strength × rank_mult
  6. Exit: unified trailing stops + breakdown detection
"""
import time as _time_module
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

from bot.backtest.sim_exchange import SimExchange, SimClock
from bot.config import (
    BREAKOUT_LOOKBACK, MAX_POSITIONS, MAX_TOTAL_EXPOSURE_PCT,
    USE_LIMIT_ORDERS, LIMIT_ORDER_TIMEOUT_SECONDS, TRADEABLE_COINS,
    LASSO_FEATURES, MAX_NEW_ENTRIES_PER_CYCLE,
    ML_ENABLED, ML_RETRAIN_INTERVAL, HMM_REFIT_INTERVAL_HOURS,
)
from bot.features import compute_coin_features, zscore_universe, check_breakdown
from bot.ranking import Ranker
from bot.regime_detector import RegimeDetector
from bot.risk_manager import RiskManager
from bot.metrics import PerformanceTracker
from bot.ml import LassoTrainer
from bot.logger import get_logger

log = get_logger("bt_engine")

INITIAL_CASH = 1_000_000.0


@dataclass
class BacktestResult:
    portfolio_history: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    final_value: float = 0.0
    total_trades: int = 0


class BacktestEngine:
    """Runs the exact v6 pipeline against simulated exchange data."""

    def __init__(
        self,
        all_candles: dict[str, list[dict]],
        exchange_info: dict[str, dict],
    ):
        self.all_candles = all_candles
        self.exchange_info = exchange_info
        self.active_pairs = [
            p for p in TRADEABLE_COINS
            if p in self.all_candles and p in self.exchange_info
        ]

    def _build_price_matrix(self, t: int, min_bars: int = 100) -> Optional[np.ndarray]:
        series = []
        for pair in self.active_pairs:
            candles = self.all_candles[pair][:t + 1]
            if len(candles) < min_bars:
                continue
            closes = np.array([c["close"] for c in candles], dtype=float)
            if np.any(~np.isfinite(closes)):
                continue
            series.append(closes)
        if len(series) < 2:
            return None
        min_len = min(len(s) for s in series)
        return np.column_stack([s[-min_len:] for s in series])

    def _build_historical_data(self, t: int) -> dict[str, list[dict]]:
        """Build dict-of-bars for LassoTrainer up to bar t."""
        data = {}
        for pair in self.active_pairs:
            candles = self.all_candles[pair][:t + 1]
            if len(candles) >= 100:
                data[pair] = candles
        return data

    def run(
        self,
        start: int,
        end: int,
        initial_cash: float = INITIAL_CASH,
    ) -> BacktestResult:
        """Run backtest from bar `start` to `end`, mirroring main.py exactly."""

        clock = SimClock()
        original_time = _time_module.time
        _time_module.time = clock.time

        try:
            exchange = SimExchange(self.all_candles, self.exchange_info, clock)
            exchange.reset(initial_cash)

            regime = RegimeDetector()
            ranker = Ranker()
            risk_mgr = RiskManager(initial_cash)
            perf = PerformanceTracker(initial_cash)
            lasso_trainer = LassoTrainer(self.active_pairs)

            from bot.executor import Executor
            executor = Executor(exchange, self.exchange_info)

            positions: dict[str, float] = {}
            result = BacktestResult()
            cycle_count = 0

            # Initial PCA-HMM fit
            pm = self._build_price_matrix(start)
            if pm is not None:
                pc_scores = regime.compute_pc_scores(pm)
                if pc_scores is not None and len(pc_scores) > 100:
                    regime.fit_hmm(pc_scores)

            # Initial Lasso training (optional boost)
            if ML_ENABLED:
                try:
                    hist = self._build_historical_data(start)
                    model, r2 = lasso_trainer.train(hist)
                    if model is not None:
                        ranker.set_lasso(model, r2)
                except Exception:
                    pass

            for t in range(start, end):
                cycle_count += 1
                exchange.set_bar(t)

                # ── Step 0: Data ──
                ticker_data = exchange.ticker()
                wallet = exchange.balance()

                positions.clear()
                for coin, bal in wallet.items():
                    if coin != "USD" and bal.get("Free", 0) > 0:
                        positions[f"{coin}/USD"] = bal["Free"]

                pv = exchange.portfolio_value(ticker_data)
                risk_mgr.update_portfolio_value(pv)
                perf.record(pv)
                result.portfolio_history.append(pv)

                # ── Optional Lasso retrain ──
                if ML_ENABLED and cycle_count % ML_RETRAIN_INTERVAL == 1:
                    try:
                        hist = self._build_historical_data(t)
                        model, r2 = lasso_trainer.train(hist)
                        if model is not None:
                            ranker.set_lasso(model, r2)
                    except Exception:
                        pass

                # ── Drawdown breakers ──
                dd_check = risk_mgr.check_drawdown_breakers()
                if dd_check["action"] == "liquidate":
                    for pair, qty in list(positions.items()):
                        tick = ticker_data.get(pair, {})
                        executor.sell(pair, qty, tick.get("LastPrice", 0),
                                      tick.get("MaxBid", 0), tick.get("MinAsk", 0),
                                      use_limit=False)
                        result.trade_log.append({"bar": t, "pair": pair, "reason": "dd_breaker"})
                        result.total_trades += 1
                    positions.clear()
                    executor.cancel_all_pending()
                    continue

                if risk_mgr.is_paused:
                    continue

                # ── Step 1: Regime ──
                pm = self._build_price_matrix(t)
                pc_scores = None
                if pm is not None:
                    pc_scores = regime.compute_pc_scores(pm)

                if cycle_count % HMM_REFIT_INTERVAL_HOURS == 0 and pc_scores is not None:
                    regime.fit_hmm(pc_scores)

                regime.update(pc_scores=pc_scores)
                risk_mgr.set_regime_multiplier(regime.get_exposure_multiplier())

                # ── Process pending fills ──
                fill_events = exchange.advance_pending_orders()
                for oid in list(executor.pending_orders.keys()):
                    if oid not in exchange.pending_orders:
                        executor.pending_orders.pop(oid, None)

                for fill in fill_events:
                    pair = fill["pair"]
                    fqty = fill["filled_qty"]
                    if fill["side"] == "BUY" and fqty > 0:
                        positions[pair] = positions.get(pair, 0) + fqty
                        risk_mgr.update_trailing_stop(pair, fill["filled_avg_price"],
                                                       entry_price=fill["filled_avg_price"])
                    elif fill["side"] == "SELL" and fqty > 0:
                        positions[pair] = max(0, positions.get(pair, 0) - fqty)

                # Stale order cancellation
                now = clock.time()
                for oid in list(executor.pending_orders.keys()):
                    info = executor.pending_orders.get(oid)
                    if info and now - info["time_placed"] > LIMIT_ORDER_TIMEOUT_SECONDS:
                        exchange.cancel_order(oid)
                        executor.pending_orders.pop(oid, None)
                        if info["side"] == "SELL":
                            exchange.place_order(info["pair"], "SELL", info["quantity"], "MARKET")

                regime_name = regime.get_status().get("regime", "UNKNOWN")

                if not regime.should_trade():
                    self._check_exits(positions, ticker_data, risk_mgr, executor, result, t)
                    continue

                # ── Step 2: Features ──
                all_raw = {}
                for pair in self.active_pairs:
                    candles = self.all_candles[pair][:t + 1]
                    if len(candles) < 100:
                        continue
                    closes = np.array([c["close"] for c in candles])
                    highs = np.array([c["high"] for c in candles])
                    lows = np.array([c["low"] for c in candles])
                    volumes = np.array([c["volume"] for c in candles])
                    tick = ticker_data.get(pair, {})
                    feats = compute_coin_features(
                        closes, highs, lows, volumes,
                        tick.get("MaxBid", 0), tick.get("MinAsk", 0),
                    )
                    if feats:
                        all_raw[pair] = feats

                all_z = zscore_universe(all_raw)

                # ── Step 3+4: Rank (momentum composite + gate) ──
                held = {p for p, q in positions.items() if q > 0}
                ranked = ranker.rank(all_raw, all_z, held)

                # ── Step 6 (before entries): Exits ──
                self._check_exits(positions, ticker_data, risk_mgr, executor, result, t)

                # Breakdown detection
                for pair in list(positions.keys()):
                    if positions.get(pair, 0) <= 0:
                        continue
                    candles = self.all_candles[pair][:t + 1]
                    closes = np.array([c["close"] for c in candles])
                    lows_arr = np.array([c["low"] for c in candles])
                    if len(closes) < BREAKOUT_LOOKBACK + 1:
                        continue
                    if check_breakdown(closes, lows_arr, BREAKOUT_LOOKBACK):
                        tick = ticker_data.get(pair, {})
                        executor.sell(pair, positions[pair], tick.get("LastPrice", 0),
                                      tick.get("MaxBid", 0), tick.get("MinAsk", 0),
                                      use_limit=False)
                        risk_mgr.clear_trailing_stop(pair)
                        positions[pair] = 0
                        result.trade_log.append({"bar": t, "pair": pair, "reason": "breakdown"})
                        result.total_trades += 1

                # ── Steps 4+5: Enter + Size ──
                if not ranked:
                    continue

                total_exposure = sum(
                    qty * ticker_data.get(p, {}).get("LastPrice", 0)
                    for p, qty in positions.items()
                )
                num_pos = sum(1 for q in positions.values() if q > 0)
                new_entries = 0

                for idx, (pair, score, raw_feats) in enumerate(ranked):
                    if num_pos >= MAX_POSITIONS:
                        break
                    if new_entries >= MAX_NEW_ENTRIES_PER_CYCLE:
                        break
                    if positions.get(pair, 0) > 0:
                        continue

                    tick = ticker_data.get(pair, {})
                    price = tick.get("LastPrice", 0)
                    if price <= 0:
                        continue

                    real_vol = raw_feats.get("realized_vol", 0.5)

                    # Rank multiplier (matches main.py)
                    if idx == 0:
                        rank_mult = 1.3
                    elif idx == 1:
                        rank_mult = 1.15
                    elif idx >= len(ranked) - 1 and len(ranked) > 3:
                        rank_mult = 0.8
                    else:
                        rank_mult = 1.0

                    signal_strength = min(1.3, max(0.3, 0.5 + score * 0.5))

                    size_usd = risk_mgr.position_size_usd(
                        pair, real_vol, total_exposure, num_pos,
                        signal_strength=signal_strength,
                        rank_multiplier=rank_mult,
                    )
                    if size_usd < 50:
                        continue

                    res = executor.buy(
                        pair, size_usd, price,
                        tick.get("MaxBid", 0), tick.get("MinAsk", 0),
                        use_limit=USE_LIMIT_ORDERS,
                    )
                    if res and res.get("Success"):
                        detail = res.get("OrderDetail", {})
                        filled_qty = detail.get("FilledQuantity", 0)
                        if filled_qty > 0:
                            positions[pair] = positions.get(pair, 0) + filled_qty
                            risk_mgr.update_trailing_stop(
                                pair, price,
                                entry_price=detail.get("FilledAverPrice", price),
                            )
                            total_exposure += size_usd
                            num_pos += 1
                            new_entries += 1
                            result.trade_log.append({
                                "bar": t, "pair": pair, "reason": "buy",
                                "score": score, "size": size_usd,
                            })
                            result.total_trades += 1

            # Close remaining
            ticker_data = exchange.ticker()
            for pair, qty in list(exchange.holdings.items()):
                if qty > 0:
                    tick = ticker_data.get(pair, {})
                    executor.sell(pair, qty, tick.get("LastPrice", 0),
                                  tick.get("MaxBid", 0), tick.get("MinAsk", 0),
                                  use_limit=False)

            result.final_value = exchange.portfolio_value()
            result.metrics = perf.summary()
            return result

        finally:
            _time_module.time = original_time

    def _check_exits(self, positions, ticker_data, risk_mgr, executor, result, bar):
        """Unified trailing stops — matches main.py _check_exits exactly."""
        for pair, qty in list(positions.items()):
            if qty <= 0:
                continue
            tick = ticker_data.get(pair, {})
            price = tick.get("LastPrice", 0)
            if price <= 0:
                continue

            risk_mgr.update_trailing_stop(pair, price)
            should_exit, reason, frac = risk_mgr.check_trailing_stop(pair, price)
            if should_exit:
                sell_qty = qty * frac
                urgent = reason == "hard_stop"
                executor.sell(pair, sell_qty, price,
                              tick.get("MaxBid", 0), tick.get("MinAsk", 0),
                              use_limit=not urgent)
                if frac >= 1.0:
                    risk_mgr.clear_trailing_stop(pair)
                    positions[pair] = 0
                else:
                    positions[pair] = qty - sell_qty
                result.trade_log.append({"bar": bar, "pair": pair, "reason": reason})
                result.total_trades += 1
