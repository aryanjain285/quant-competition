"""
Live performance metrics: Sharpe, Sortino, Calmar.
Tracks portfolio returns and computes risk-adjusted metrics in real-time.
"""
import numpy as np
from datetime import datetime, timezone
from bot.logger import get_logger

log = get_logger("metrics")


class PerformanceTracker:
    """Tracks portfolio value over time and computes competition metrics."""

    def __init__(self, initial_value: float):
        self.initial_value = initial_value
        self.values: list[tuple[float, float]] = []  # (timestamp, value)
        self.daily_values: list[tuple[str, float]] = []  # (date_str, eod_value)
        self._last_date: str = ""

    def record(self, value: float, timestamp: float = None):
        """Record a portfolio value snapshot."""
        ts = timestamp or datetime.now(timezone.utc).timestamp()
        self.values.append((ts, value))

        # Track daily values for Sharpe/Sortino/Calmar
        date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        if date_str != self._last_date:
            if self._last_date:
                # Store the last value of the previous day
                self.daily_values.append((self._last_date, self.values[-2][1] if len(self.values) > 1 else value))
            self._last_date = date_str

    def _daily_returns(self) -> np.ndarray:
        """Compute daily returns from daily_values."""
        if len(self.daily_values) < 2:
            # Fall back to sub-daily returns if not enough daily data
            if len(self.values) < 2:
                return np.array([])
            vals = np.array([v for _, v in self.values])
            # Sample every ~288 points (one day of 5-min bars)
            step = max(1, len(vals) // max(1, len(vals) // 288))
            sampled = vals[::step]
            if len(sampled) < 2:
                return np.array([])
            return np.diff(sampled) / sampled[:-1]

        vals = np.array([v for _, v in self.daily_values])
        return np.diff(vals) / vals[:-1]

    @property
    def total_return(self) -> float:
        """Total portfolio return since inception."""
        if not self.values:
            return 0.0
        current = self.values[-1][1]
        return (current - self.initial_value) / self.initial_value

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a positive fraction."""
        if not self.values:
            return 0.0
        vals = np.array([v for _, v in self.values])
        peak = np.maximum.accumulate(vals)
        drawdowns = (peak - vals) / peak
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (assuming 0 risk-free rate)."""
        rets = self._daily_returns()
        if len(rets) < 2:
            return 0.0
        std = np.std(rets)
        if std == 0:
            return 0.0
        return float(np.mean(rets) / std * np.sqrt(365))

    @property
    def sortino_ratio(self) -> float:
        """Annualized Sortino ratio (penalizes only downside vol)."""
        rets = self._daily_returns()
        if len(rets) < 2:
            return 0.0
        neg_rets = rets[rets < 0]
        if len(neg_rets) == 0:
            # No negative returns — perfect, return high value
            return 10.0 if np.mean(rets) > 0 else 0.0
        downside_std = np.std(neg_rets)
        if downside_std == 0:
            return 0.0
        return float(np.mean(rets) / downside_std * np.sqrt(365))

    @property
    def calmar_ratio(self) -> float:
        """Calmar ratio = annualized return / max drawdown."""
        mdd = self.max_drawdown
        if mdd == 0:
            return 10.0 if self.total_return > 0 else 0.0
        # Annualize return based on days elapsed
        if len(self.values) < 2:
            return 0.0
        days_elapsed = (self.values[-1][0] - self.values[0][0]) / 86400
        if days_elapsed < 1:
            days_elapsed = 1
        annual_return = self.total_return * (365 / days_elapsed)
        return float(annual_return / mdd)

    @property
    def composite_score(self) -> float:
        """Competition composite: 0.4*Sortino + 0.3*Sharpe + 0.3*Calmar."""
        return 0.4 * self.sortino_ratio + 0.3 * self.sharpe_ratio + 0.3 * self.calmar_ratio

    def summary(self) -> dict:
        """Return all metrics as a dict."""
        return {
            "total_return_pct": round(self.total_return * 100, 3),
            "max_drawdown_pct": round(self.max_drawdown * 100, 3),
            "sharpe": round(self.sharpe_ratio, 3),
            "sortino": round(self.sortino_ratio, 3),
            "calmar": round(self.calmar_ratio, 3),
            "composite": round(self.composite_score, 3),
            "num_snapshots": len(self.values),
            "num_daily": len(self.daily_values),
        }
