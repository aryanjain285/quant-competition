"""
Configuration for the trading bot — Finals v7.
All tunable parameters in one place. Override via environment variables.

ALL LOOKBACKS AND WINDOWS ARE IN 1-HOUR BARS.

v7 DESIGN PHILOSOPHY:
- HMM discovers latent states; we analyze them AFTER fitting to derive exposure
- No ML prediction gate (Lasso R²≈0.000 on cross-sectional crypto features)
- Momentum composite with domain-knowledge weights (crypto momentum literature)
- Active trading to satisfy 8 active trading days rule
- Every parameter justified by cost, stats, or literature

PARAMETER SOURCES:
- Momentum weights: Liu, Tsyvinski & Wu (2019) "Risks and Returns of Cryptocurrency"
  → 1-week (24h-168h) momentum is the strongest cross-sectional predictor in crypto
- Commission: 0.05% maker, 0.10% taker → 10-20 bps round trip
- Vol-parity: standard risk budgeting (Roncalli 2013)
- Trailing stops: calibrated to hourly crypto vol (~2% daily → 3.5% ≈ 1.7σ)
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

# --- Timing ---
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL", "3600"))
LIMIT_ORDER_TIMEOUT_SECONDS = 90

# --- Momentum Lookbacks (1h bars) ---
# r_1h intentionally excluded: noise at 1-bar resolution
MOMENTUM_LOOKBACKS = {
    "6h": 6,
    "24h": 24,
    "3d": 72,
}

BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "72"))

# --- Volatility ---
VOL_LOOKBACK = 24
VOL_LONG_LOOKBACK = 168
ANNUALIZATION_FACTOR = 93.6  # sqrt(8760) for 1h bars

# --- Regime Detection (HMM on PCs) ---
# 4 PCs: typically 80-90% of variance in 43-coin crypto universe.
# 3-state HMM with 4D obs: ~50 params. Stable with 800+ observations.
# States are NOT pre-labeled. They are analyzed post-fit to derive exposure.
HMM_N_PCS = 4
HMM_N_STATES = 3
HMM_REFIT_INTERVAL_HOURS = 12
PCA_REFIT_INTERVAL_HOURS = 6

# Forward return horizon for state analysis (how far ahead to look when
# characterizing each state's profitability).
STATE_ANALYSIS_FORWARD_HOURS = 24

# --- EWMA Momentum ---
# Two EWMA horizons averaged. No arbitrary weight allocation.
# EWMA smooths noise, weights recent hours exponentially, decays older hours.
# Halflife IS the parameter — grounded in "we care about the last 6-24 hours."
# Equal average of two horizons = least assumptive choice.
EWMA_HALFLIFE_SHORT = 6    # 6-hour EWMA — captures short-term shifts
EWMA_HALFLIFE_LONG = 24    # 24-hour EWMA — captures daily momentum

# Entry gate (simple binary conditions on RAW features, not z-scored):
#   r_24h > 0: only buy coins with positive 24h momentum
#   volume_ratio > 0.8: minimum volume confirmation
# These are deliberately loose — we want active trading.
# The ranking score handles quality; the gate just eliminates obvious bad entries.
GATE_MIN_R24H = 0.0
GATE_MIN_VOLUME_RATIO = 0.8

# Max new entries per cycle — prevents overtrading in a single hour
MAX_NEW_ENTRIES_PER_CYCLE = 3

# --- Risk Management ---
MAX_POSITION_PCT = 0.15
MAX_TOTAL_EXPOSURE_PCT = 0.80
TARGET_RISK_PER_TRADE = 0.005
MAX_POSITIONS = 12

# Unified trailing stops (no strategy branching)
HARD_STOP_PCT = 0.035
PARTIAL_EXIT_PCT = 0.03
PARTIAL_EXIT_FRACTION = 0.5
TRAILING_STOP_PCT = 0.035
TRAILING_STOP_WIDE_PCT = 0.045
TIME_STOP_HOURS = 60
TIME_STOP_MIN_GAIN = 0.01

# Drawdown circuit breakers
DRAWDOWN_LEVEL_1 = 0.035
DRAWDOWN_LEVEL_2 = 0.06
DRAWDOWN_LEVEL_3 = 0.10
DRAWDOWN_PAUSE_HOURS = {1: 0, 2: 4, 3: 12}
REDD_MAX_DD = 0.10

# --- Execution ---
USE_LIMIT_ORDERS = True
LIMIT_ORDER_OFFSET_BPS = 1

# --- Optional ML (Lasso on RELATIVE returns — cross-sectional ranking) ---
# Target: coin_return - median(all_coin_returns) over forward horizon.
# This removes market direction (unpredictable) and focuses on cross-sectional spread.
# When Lasso finds signal (R² > 0.5%), it boosts the EWMA score.
ML_ENABLED = True
ML_LOOKBACK_HOURS = 400
ML_FORWARD_HORIZON = 24
ML_SAMPLE_INTERVAL = 24
ML_RETRAIN_INTERVAL = 6
ML_BLEND_WEIGHT = 0.3
ML_MIN_R2 = 0.005

# Feature columns for Lasso (when enabled)
LASSO_FEATURES = [
    "r_6h", "r_24h", "r_3d",
    "persistence", "choppiness",
    "realized_vol", "downside_vol", "jump_proxy",
    "breakout_distance", "volume_ratio",
    "overshoot",
    "spread_pct",
]

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