"""
Configuration for the trading bot — Finals v6.
All tunable parameters in one place. Override via environment variables.

ALL LOOKBACKS AND WINDOWS ARE IN 1-HOUR BARS.
The bot fetches 1h candles from Binance. Every bar = 1 hour.

PARAMETER JUSTIFICATION:
- Commission: 0.05% maker, 0.1% taker → round trip = 0.10-0.20%
- ENTRY_THRESHOLD: 0.003 (30 bps) > round-trip cost → only trade when expected profit covers costs
- TARGET_RISK_PER_TRADE: 0.005 → at BTC vol=0.4, this gives ~$238K sizing on $1M portfolio
- Trailing stops: unified at 3.5% → calibrated to 1h bar noise (hourly vol ~2% annualized → 3.5% ≈ 1.7σ daily)
- HMM_N_COMPONENTS: 3 PCs fixed → 24 HMM params, stable with 1000 observations
"""
import os

# Load .env file if present
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

# --- API Configuration ---
API_KEY = os.getenv("ROOSTOO_API_KEY", "")
API_SECRET = os.getenv("ROOSTOO_API_SECRET", "")
BASE_URL = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")

# --- Binance ---
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
BINANCE_FUTURES_URL = os.getenv("BINANCE_FUTURES_URL", "https://fapi.binance.com")

# --- Timing (1h bars → 1h cycle) ---
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL", "3600"))  # 1 hour
LIMIT_ORDER_TIMEOUT_SECONDS = 90

# --- Signal Parameters (ALL IN 1-HOUR BARS) ---
MOMENTUM_LOOKBACKS = {
    "6h": 6,      # 6 bars = 6 hours
    "24h": 24,    # 24 bars = 24 hours (1 day)
    "3d": 72,     # 72 bars = 72 hours (3 days)
}
# Note: r_1h dropped — too noisy for cross-sectional prediction (< 1 bar of signal)

BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "72"))  # 72 bars = 72h = 3 days

# --- Volatility (1h bars) ---
VOL_LOOKBACK = 24              # 24 bars = 24 hours for realized vol
VOL_LONG_LOOKBACK = 168        # 168 bars = 7 days for regime baseline
ANNUALIZATION_FACTOR = 93.6    # sqrt(24 * 365) = sqrt(8760) for 1h bars

# --- Regime Detection ---
# Fixed K=3 PCs for HMM observation vector.
# Rationale: 3 PCs typically capture 70-85% of variance in a 43-coin crypto universe.
# 3-state HMM with 3D observations: ~24 covariance params. Stable with 1000+ bars.
# Increasing K beyond 3 risks overfitting the HMM (params scale as K²).
HMM_N_PCS = 3
HMM_N_STATES = 3
HMM_REFIT_INTERVAL_HOURS = 12  # refit HMM every 12 cycles
PCA_REFIT_INTERVAL_HOURS = 6   # refit PCA loadings every 6 hours

# Regime exposure multipliers:
# LOW_VOL = 1.0: calm markets → clean trends → full sizing (vol-parity already scales per-coin)
# MID_VOL = 0.7: normal conditions → moderate sizing
# HI_VOL  = 0.0: crisis → sit out entirely
# Rationale: vol-parity handles coin-level vol. Regime multiplier handles systemic risk.
# Halving in LOW_VOL (old design) was double-penalizing low vol. Flipped for Sortino optimization.
REGIME_EXPOSURE = {
    "LOW_VOL": 1.0,
    "MID_VOL": 0.7,
    "HI_VOL": 0.0,
}

# --- ML Model ---
# Entry gate: predicted 24h return must exceed this to open a position.
# 30 bps > round-trip commission (10-20 bps) + slippage buffer.
# This replaces the old 7-condition conjunction filter with a single model-derived gate.
ENTRY_THRESHOLD = 0.003

# Lasso training parameters
ML_LOOKBACK_HOURS = 400        # training window
ML_FORWARD_HORIZON = 24        # predict 24h forward returns
ML_SAMPLE_INTERVAL = 24        # sample every 24h → non-overlapping targets
ML_RETRAIN_INTERVAL = 6        # retrain every 6 cycles (hours)

# Feature columns for Lasso — explicit and fixed.
# Rationale for each:
#   r_6h, r_24h, r_3d: multi-horizon momentum (Jegadeesh & Titman cross-sectional momentum)
#   persistence: trend quality — higher = more aligned sub-returns
#   choppiness: path noise — lower = cleaner trend
#   realized_vol: vol predicts vol, vol predicts cross-sectional returns
#   downside_vol: Sortino-relevant risk measure
#   jump_proxy: tail risk — max|return|/vol
#   breakout_distance: distance above prior high (trend strength)
#   volume_ratio: volume confirmation (proven in v3 backtest)
#   overshoot: mean-reversion signal (z-scored drop magnitude)
#   spread_pct: liquidity/cost proxy
LASSO_FEATURES = [
    "r_6h", "r_24h", "r_3d",
    "persistence", "choppiness",
    "realized_vol", "downside_vol", "jump_proxy",
    "breakout_distance", "volume_ratio",
    "overshoot",
    "spread_pct",
]

# --- Risk Management ---
MAX_POSITION_PCT = 0.15
MAX_TOTAL_EXPOSURE_PCT = 0.80
TARGET_RISK_PER_TRADE = 0.005
MAX_POSITIONS = 12

# Unified trailing stop parameters (no strategy branching):
# Hard stop: -3.5% from entry
#   Rationale: BTC hourly vol ≈ 0.4 ann → daily vol ≈ 2.1% → 3.5% ≈ 1.7σ daily move.
#   Gives room for normal noise but caps losses before they compound.
HARD_STOP_PCT = 0.035

# Partial exit: sell 50% at +3% from entry
#   Rationale: locks in profit, reduces Sortino denominator (less variance in outcomes).
#   Remaining 50% rides with wider trailing stop.
PARTIAL_EXIT_PCT = 0.03
PARTIAL_EXIT_FRACTION = 0.5

# Trailing stop: 3.5% from high (4.5% after partial exit)
#   Rationale: after partial, remaining position has "house money" character → wider trail.
TRAILING_STOP_PCT = 0.035
TRAILING_STOP_WIDE_PCT = 0.045  # after partial exit

# Time stop: 60h with < 1% gain
#   Rationale: opportunity cost. If position hasn't moved in 2.5 days, capital is better elsewhere.
TIME_STOP_HOURS = 60
TIME_STOP_MIN_GAIN = 0.01

# Drawdown circuit breakers (REDD handles smooth scaling; these are emergency)
DRAWDOWN_LEVEL_1 = 0.035
DRAWDOWN_LEVEL_2 = 0.06
DRAWDOWN_LEVEL_3 = 0.10
DRAWDOWN_PAUSE_HOURS = {1: 0, 2: 4, 3: 12}
REDD_MAX_DD = 0.10  # REDD goes to 0 at this drawdown level

# --- Execution ---
USE_LIMIT_ORDERS = True
LIMIT_ORDER_OFFSET_BPS = 1

# --- Coin Filtering ---
TRADEABLE_COINS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD", "XRP/USD",
    "DOGE/USD", "ADA/USD", "AVAX/USD", "LINK/USD", "DOT/USD",
    "SUI/USD", "NEAR/USD", "LTC/USD", "TON/USD", "UNI/USD",
    "FET/USD", "HBAR/USD", "XLM/USD", "FIL/USD", "APT/USD",
    "ARB/USD", "SEI/USD", "PEPE/USD", "SHIB/USD", "FLOKI/USD",
    "WIF/USD", "BONK/USD", "TRX/USD", "ICP/USD", "AAVE/USD",
    "WLD/USD", "ONDO/USD", "CRV/USD", "PENDLE/USD", "ENA/USD",
    "TAO/USD", "POL/USD", "ZEC/USD", "TRUMP/USD", "EIGEN/USD",
    "VIRTUAL/USD", "CAKE/USD", "PAXG/USD",
]

BINANCE_SYMBOL_MAP = {
    "BTC/USD": "BTCUSDT", "ETH/USD": "ETHUSDT", "SOL/USD": "SOLUSDT",
    "BNB/USD": "BNBUSDT", "XRP/USD": "XRPUSDT", "DOGE/USD": "DOGEUSDT",
    "ADA/USD": "ADAUSDT", "AVAX/USD": "AVAXUSDT", "LINK/USD": "LINKUSDT",
    "DOT/USD": "DOTUSDT", "SUI/USD": "SUIUSDT", "NEAR/USD": "NEARUSDT",
    "LTC/USD": "LTCUSDT", "TON/USD": "TONUSDT", "UNI/USD": "UNIUSDT",
    "FET/USD": "FETUSDT", "HBAR/USD": "HBARUSDT", "XLM/USD": "XLMUSDT",
    "FIL/USD": "FILUSDT", "APT/USD": "APTUSDT", "ARB/USD": "ARBUSDT",
    "SEI/USD": "SEIUSDT", "PEPE/USD": "PEPEUSDT", "SHIB/USD": "SHIBUSDT",
    "FLOKI/USD": "FLOKIUSDT", "WIF/USD": "WIFUSDT", "BONK/USD": "BONKUSDT",
    "TRX/USD": "TRXUSDT", "ICP/USD": "ICPUSDT", "AAVE/USD": "AAVEUSDT",
    "WLD/USD": "WLDUSDT", "ONDO/USD": "ONDOUSDT", "CRV/USD": "CRVUSDT",
    "PENDLE/USD": "PENDLEUSDT", "ENA/USD": "ENAUSDT", "TAO/USD": "TAOUSDT",
    "POL/USD": "POLUSDT", "ZEC/USD": "ZECUSDT", "TRUMP/USD": "TRUMPUSDT",
    "EIGEN/USD": "EIGENUSDT", "VIRTUAL/USD": "VIRTUALUSDT",
    "CAKE/USD": "CAKEUSDT", "PAXG/USD": "PAXGUSDT",
}

# --- Logging ---
LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")