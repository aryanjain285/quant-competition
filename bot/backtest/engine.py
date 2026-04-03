"""
Backtest engine that mirrors main.py's run_cycle() exactly.

Design: import and call the SAME modules as live trading:
  - features.compute_coin_features (same features)
  - features.zscore_universe (same z-scoring)
  - signals.compute_signal (same signal logic)
  - ranking.Ranker (same ranking)
  - regime_detector.RegimeDetector (same regime)
  - risk_manager.RiskManager (same stops, REDD, sizing)
  - executor.Executor (swapped to use SimExchange)

The ONLY difference is the data source: SimExchange instead of RoostooClient.
Strategy logic is a pluggable function: pass any strategy and the engine
runs it against the same infrastructure.

Usage:
    from bot.backtest.engine import BacktestEngine
    from bot.backtest.sim_exchange import SimExchange

    engine = BacktestEngine(candles, exchange_info)
    results = engine.run(start=100, end=340)  # 240 bars = 10 days
"""
import time as _time_module
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass, field

from bot.backtest.sim_exchange import SimExchange, SimClock
from bot.config import (
    BREAKOUT_LOOKBACK, MAX_POSITIONS, MAX_TOTAL_EXPOSURE_PCT,
    USE_LIMIT_ORDERS, LIMIT_ORDER_TIMEOUT_SECONDS, TRADEABLE_COINS,
)
from bot.features import compute_coin_features, zscore_universe, compute_submodel_scores
from bot.signals import compute_signal
from bot.ranking import Ranker
from bot.regime_detector import RegimeDetector
from bot.risk_manager import RiskManager
from bot.metrics import PerformanceTracker
from bot.executor import Executor
from bot.ml import RidgeTrainer
from bot.logger import get_logger

log = get_logger("bt_engine")

# Live main.py constants (imported to match exactly)
CONTINUATION_SOFT_FLOOR = 0.0010
REVERSAL_SOFT_FLOOR = 0.0015
TOP_K_CONTINUATION = 3
TOP_K_REVERSAL = 2
INITIAL_CASH = 1_000_000.0


@dataclass
class BacktestResult:
    portfolio_history: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    final_value: float = 0.0
    total_trades: int = 0
    wins: int = 0


class BacktestEngine:
    """Runs the exact same pipeline as main.py against simulated exchange data.

    The engine handles:
      - SimExchange setup and time management
      - Data slicing (provides candle history up to current bar)
      - Calling the same feature/signal/ranking/regime/risk modules
      - Collecting results

    Strategy parameters can be overridden via the run() method.
    """

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

    def _build_historical_data_for_ml(self, t: int) -> dict[str, list[dict]]:
        """Build dict-of-bars format for RidgeTrainer, up to bar t."""
        data = {}
        for pair in self.active_pairs:
            candles = self.all_candles[pair][:t + 1]
            if len(candles) < 100:
                continue
            data[pair] = candles
        return data

    def run(
        self,
        start: int,
        end: int,
        regime_refit_interval: int = 12,
        ridge_refit_interval: int = 6,
        initial_cash: float = INITIAL_CASH,
    ) -> BacktestResult:
        """Run backtest from bar `start` to bar `end`.

        Mirrors main.py run_cycle() exactly, including:
          - PCA-HMM regime detection
          - Ridge regression ranking (retrained periodically)
          - Continuation + reversal signal filters
          - Top-K selection with soft floors
          - Rank multiplier for MID_VOL
          - Vol-parity sizing × REDD × regime
          - Trailing stops with partial exits
          - Fill event processing from pending orders
        """
        # ── Setup (mirrors TradingBot.__init__) ──
        clock = SimClock()

        # Monkeypatch time.time so risk_manager uses simulated clock
        original_time = _time_module.time
        _time_module.time = clock.time

        try:
            exchange = SimExchange(self.all_candles, self.exchange_info, clock)
            exchange.reset(initial_cash)

            regime = RegimeDetector()
            ranker = Ranker()
            risk_mgr = RiskManager(initial_cash)
            perf = PerformanceTracker(initial_cash)
            ridge_trainer = RidgeTrainer(self.active_pairs)

            # Build a thin executor that wraps SimExchange with the same interface
            # We use the live Executor class but swap client to SimExchange
            executor = Executor(exchange, self.exchange_info)

            positions: dict[str, float] = {}
            result = BacktestResult()
            cycle_count = 0

            # Initial HMM fit
            price_matrix = self._build_price_matrix(start)
            if price_matrix is not None:
                pc1 = regime.compute_pc1_market_proxy(price_matrix)
                if pc1[0] is not None and len(pc1[0]) > 100:
                    regime.fit_hmm(pc1[0])

            for t in range(start, end):
                cycle_count += 1
                exchange.set_bar(t)

                # ── STEP 0: Data ──
                ticker_data = exchange.ticker()
                wallet = exchange.balance()

                # Sync positions from wallet
                positions.clear()
                for coin, bal in wallet.items():
                    if coin != "USD" and bal.get("Free", 0) > 0:
                        positions[f"{coin}/USD"] = bal["Free"]

                portfolio_value = exchange.portfolio_value(ticker_data)
                risk_mgr.update_portfolio_value(portfolio_value)
                perf.record(portfolio_value)
                result.portfolio_history.append(portfolio_value)

                # ── Ridge training (every N cycles, matches live) ──
                if cycle_count % ridge_refit_interval == 1:
                    try:
                        hist_data = self._build_historical_data_for_ml(t)
                        model, cols = ridge_trainer.train(
                            historical_data=hist_data,
                            lookback=400, forward_horizon=24,
                            feature_cols=ranker.ridge_features,
                        )
                        if model is not None:
                            ranker.set_ridge_model(model)
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
                    positions.clear()
                    executor.cancel_all_pending()
                    continue

                if risk_mgr.is_paused:
                    continue

                # ── STEP 1: Regime ──
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

                price_matrix = self._build_price_matrix(t)
                pc1_series = None
                if price_matrix is not None:
                    pc1_result = regime.compute_pc1_market_proxy(price_matrix)
                    pc1_series = pc1_result[0]

                if cycle_count % regime_refit_interval == 0 and pc1_series is not None:
                    regime.fit_hmm(pc1_series)

                regime.update(pc1_series=pc1_series)
                risk_mgr.set_regime_multiplier(regime.get_exposure_multiplier())

                # ── Process pending fills ──
                fill_events = exchange.advance_pending_orders()
                # Also handle executor's pending tracking
                for oid in list(executor.pending_orders.keys()):
                    # Check if SimExchange already filled it
                    if oid not in exchange.pending_orders:
                        executor.pending_orders.pop(oid, None)

                for fill in fill_events:
                    pair = fill["pair"]
                    fqty = fill["filled_qty"]
                    if fill["side"] == "BUY" and fqty > 0:
                        positions[pair] = positions.get(pair, 0) + fqty
                        risk_mgr.update_trailing_stop(
                            pair, fill["filled_avg_price"],
                            strategy="breakout",
                            entry_price=fill["filled_avg_price"],
                        )
                    elif fill["side"] == "SELL" and fqty > 0:
                        positions[pair] = max(0, positions.get(pair, 0) - fqty)

                # Cancel stale pending orders (same as live executor)
                now = clock.time()
                for oid in list(executor.pending_orders.keys()):
                    info = executor.pending_orders.get(oid)
                    if info and now - info["time_placed"] > LIMIT_ORDER_TIMEOUT_SECONDS:
                        exchange.cancel_order(oid)
                        executor.pending_orders.pop(oid, None)
                        if info["side"] == "SELL":
                            exchange.place_order(info["pair"], "SELL", info["quantity"], "MARKET")

                regime_name = regime.get_status().get("regime", "UNKNOWN")

                # If regime hostile, only manage exits
                if not regime.should_trade():
                    self._check_exits(positions, ticker_data, risk_mgr, executor, result, t)
                    continue

                # ── STEP 2: Event filter ──
                all_z = zscore_universe(all_raw)
                for pair in all_z:
                    compute_submodel_scores(all_z[pair])

                signals = {}
                for pair in all_raw:
                    candles = self.all_candles[pair][:t + 1]
                    closes = np.array([c["close"] for c in candles])
                    lows = np.array([c["low"] for c in candles])
                    signals[pair] = compute_signal(
                        all_raw[pair], all_z[pair], closes, lows, BREAKOUT_LOOKBACK,
                    )

                # ── STEP 3: Valid trades ──
                valid = {}
                for pair, sig in signals.items():
                    if sig["action"] == "BUY" and positions.get(pair, 0) <= 0:
                        valid[pair] = all_z.get(pair, {})
                        valid[pair]["_signal"] = sig

                if not valid:
                    self._check_exits(positions, ticker_data, risk_mgr, executor, result, t)
                    continue

                # ── STEP 4: Ranking (matches live TOP_K logic) ──
                ranked = ranker.rank(valid, max_results=len(valid))

                cont_candidates = []
                rev_candidates = []
                for pair, score, features in ranked:
                    sig = features.get("_signal", {})
                    strategy = sig.get("strategy", "none")
                    if strategy == "continuation" and score >= CONTINUATION_SOFT_FLOOR:
                        cont_candidates.append((pair, score, features))
                    elif strategy == "reversal" and score >= REVERSAL_SOFT_FLOOR:
                        rev_candidates.append((pair, score, features))

                cont_candidates.sort(key=lambda x: x[1], reverse=True)
                rev_candidates.sort(key=lambda x: x[1], reverse=True)
                selected = cont_candidates[:TOP_K_CONTINUATION] + rev_candidates[:TOP_K_REVERSAL]
                selected.sort(key=lambda x: x[1], reverse=True)

                # ── STEP 5: Execution ──
                self._check_exits(positions, ticker_data, risk_mgr, executor, result, t)

                total_exposure = sum(
                    qty * ticker_data.get(p, {}).get("LastPrice", 0)
                    for p, qty in positions.items()
                )
                num_pos = sum(1 for q in positions.values() if q > 0)

                for idx, (pair, score, features) in enumerate(selected):
                    if num_pos >= MAX_POSITIONS:
                        break
                    if positions.get(pair, 0) > 0:
                        continue

                    sig = features.get("_signal", {})
                    tick = ticker_data.get(pair, {})
                    price = tick.get("LastPrice", 0)
                    if price <= 0:
                        continue

                    real_vol = all_raw.get(pair, {}).get("realized_vol", 0.5)

                    # Rank multiplier (matches live)
                    rank_mult = 1.0
                    if regime_name == "MID_VOL":
                        if idx == 0:
                            rank_mult = 1.5
                        elif idx == 1:
                            rank_mult = 1.25
                        elif idx == len(selected) - 1:
                            rank_mult = 0.75

                    size_usd = risk_mgr.position_size_usd(
                        pair, real_vol, total_exposure, num_pos,
                        signal_strength=sig.get("strength", 0.5),
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
                            strategy = sig.get("strategy", "continuation")
                            risk_mgr.update_trailing_stop(
                                pair, price,
                                strategy="mean_rev" if strategy == "reversal" else "breakout",
                                entry_price=detail.get("FilledAverPrice", price),
                            )
                            total_exposure += size_usd
                            num_pos += 1
                            result.trade_log.append({
                                "bar": t, "pair": pair, "reason": "buy",
                                "strategy": strategy, "size": size_usd, "score": score,
                            })
                            result.total_trades += 1

                # Handle sell signals
                for pair, sig in signals.items():
                    if sig["action"] == "SELL" and positions.get(pair, 0) > 0:
                        tick = ticker_data.get(pair, {})
                        qty = positions[pair]
                        urgent = sig.get("breakdown", False)
                        executor.sell(
                            pair, qty, tick.get("LastPrice", 0),
                            tick.get("MaxBid", 0), tick.get("MinAsk", 0),
                            use_limit=not urgent,
                        )
                        risk_mgr.clear_trailing_stop(pair)
                        positions[pair] = 0
                        result.trade_log.append({
                            "bar": t, "pair": pair, "reason": "signal_sell",
                        })
                        result.total_trades += 1

            # Close remaining positions
            ticker_data = exchange.ticker()
            for pair, qty in list(exchange.holdings.items()):
                if qty > 0:
                    tick = ticker_data.get(pair, {})
                    executor.sell(
                        pair, qty, tick.get("LastPrice", 0),
                        tick.get("MaxBid", 0), tick.get("MinAsk", 0),
                        use_limit=False,
                    )

            result.final_value = exchange.portfolio_value()
            result.metrics = perf.summary()
            return result

        finally:
            # Restore real time.time
            _time_module.time = original_time

    def _check_exits(
        self, positions, ticker_data, risk_mgr, executor, result, bar,
    ):
        """Check trailing stops — mirrors main.py _check_exits exactly."""
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
                urgent = reason in ("hard_stop",)
                executor.sell(
                    pair, sell_qty, price,
                    tick.get("MaxBid", 0), tick.get("MinAsk", 0),
                    use_limit=not urgent,
                )
                if frac >= 1.0:
                    risk_mgr.clear_trailing_stop(pair)
                    positions[pair] = 0
                else:
                    positions[pair] = qty - sell_qty

                result.trade_log.append({
                    "bar": bar, "pair": pair, "reason": reason,
                })
                result.total_trades += 1
                if reason in ("profit_target", "partial_exit"):
                    result.wins += 1
