"""
Configuration for the trading bot.
All tunable parameters in one place. Override via environment variables.

ALL LOOKBACKS AND WINDOWS ARE IN 1-HOUR BARS.
The bot fetches 1h candles from Binance. Every bar = 1 hour.
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
EMA_FAST = 21    # 21-hour EMA
EMA_SLOW = 55    # 55-hour EMA

MOMENTUM_LOOKBACKS = {
    "1h": 1,      # 1 bar  = 1 hour
    "6h": 6,      # 6 bars = 6 hours
    "24h": 24,    # 24 bars = 24 hours (1 day)
    "3d": 72,     # 72 bars = 72 hours (3 days)
}
MOMENTUM_WEIGHTS = {
    "1h": 0.15,
    "6h": 0.25,
    "24h": 0.35,
    "3d": 0.25,
}

BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "72"))  # 72 bars = 72 hours = 3 days

# --- Volatility (1h bars) ---
VOL_LOOKBACK = 24              # 24 bars = 24 hours for realized vol
VOL_LONG_LOOKBACK = 168        # 168 bars = 7 days for regime baseline
ANNUALIZATION_FACTOR = 93.6    # sqrt(24 * 365) = sqrt(8760) for 1h bars

# --- Risk Management ---
# Position sizing: vol-parity should be the binding constraint, not the cap.
# With TARGET_RISK=0.005 and BTC vol=0.4: size = 0.005*1M / (0.4/19.1) = $238K
# With DOGE vol=0.8: size = 0.005*1M / (0.8/19.1) = $119K
# Cap at 15% = $150K only binds for very low vol coins. Vol-parity drives sizing.
MAX_POSITION_PCT = 0.15
MAX_TOTAL_EXPOSURE_PCT = 0.80
TARGET_RISK_PER_TRADE = 0.005    # was 0.025 — too high, cap always bound, vol-parity was off
MAX_POSITIONS = 12

# Trailing stops
TRAILING_STOP_PCT = 0.03
MEAN_REV_TAKE_PROFIT = 0.03    # +3% profit target for reversal trades
MEAN_REV_STOP_LOSS = -0.03     # -3% hard stop for reversal trades

# Drawdown circuit breakers (REDD handles smooth scaling; these are emergency)
DRAWDOWN_LEVEL_1 = 0.035
DRAWDOWN_LEVEL_2 = 0.06
DRAWDOWN_LEVEL_3 = 0.10
DRAWDOWN_PAUSE_HOURS = {1: 0, 2: 4, 3: 12}

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
