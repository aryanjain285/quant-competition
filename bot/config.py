"""
Configuration for the trading bot.
All tunable parameters in one place. Override via environment variables.

API keys must be set via environment variables or a .env file.
NEVER hardcode keys here — they will leak into git history.
"""
import os

# Load .env file if present (keeps keys out of source code)
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

# --- Binance (public, no auth needed) ---
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
BINANCE_FUTURES_URL = os.getenv("BINANCE_FUTURES_URL", "https://fapi.binance.com")

# --- Timing ---
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL", "3600"))  # 1 hour — matches 1h candle data
LIMIT_ORDER_TIMEOUT_SECONDS = 90

# --- Signal Parameters ---
EMA_FAST = 21
EMA_SLOW = 55
MOMENTUM_LOOKBACKS = {
    "1h": 12,    # 12 x 5min = 1 hour
    "6h": 72,    # 72 x 5min = 6 hours
    "24h": 288,  # 288 x 5min = 24 hours
    "3d": 864,   # 864 x 5min = 3 days
}
MOMENTUM_WEIGHTS = {
    "1h": 0.15,
    "6h": 0.25,
    "24h": 0.35,
    "3d": 0.25,
}

# --- Signal Parameters (v3) ---
BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "864"))  # 72h in 5-min bars

# RSI mean reversion (tightened in v3)
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "25"))      # was 30 — fewer, better entries
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "65"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))

# --- Risk Management ---
# v3 COMPETITION TUNING: this is a mock portfolio with zero real downside.
# Institutional-level conservatism (v1: 60% max, 8 pos, 1.5% risk) left
# too much capital idle. Competition rewards being in the market.
MAX_POSITION_PCT = 0.15          # 15% per coin (was 20% — more positions, less concentration)
MAX_TOTAL_EXPOSURE_PCT = 0.80    # 80% invested (was 60% — deploy more capital)
TARGET_RISK_PER_TRADE = 0.025    # 2.5% risk per trade (was 1.5% — bigger bets)
MAX_POSITIONS = 12               # 12 positions (was 8 — more diversification)

# Trailing stop
TRAILING_STOP_PCT = 0.03
TRAILING_STOP_ATR_MULT = 2.0

# Drawdown circuit breakers (WIDENED for crypto volatility)
# REDD handles smooth scaling; these are emergency-only.
# v3: widened from 2%/4%/7% to avoid premature liquidation
DRAWDOWN_LEVEL_1 = 0.035  # -3.5%: warning only (REDD already reducing sizing)
DRAWDOWN_LEVEL_2 = 0.06   # -6%: liquidate + 4h pause (was 12h)
DRAWDOWN_LEVEL_3 = 0.10   # -10%: liquidate + 12h pause (was 48h)

DRAWDOWN_PAUSE_HOURS = {
    1: 0,     # level 1: no pause, REDD handles it
    2: 4,     # level 2: 4h pause (was 12h — too long, misses recovery)
    3: 12,    # level 3: 12h pause (was 48h — risked failing 8-day activity rule)
}

# --- Volatility ---
VOL_LOOKBACK_PERIODS = 288
VOL_REGIME_BASELINE_PERIODS = 8640

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

# Map Roostoo pair -> Binance symbol
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