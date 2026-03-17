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
# Set these via .env file or environment variables before running the bot.
API_KEY = os.getenv("ROOSTOO_API_KEY", "")
API_SECRET = os.getenv("ROOSTOO_API_SECRET", "")
BASE_URL = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")

# --- Binance (public, no auth needed) ---
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://data-api.binance.vision")

# --- Timing ---
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL", "300"))  # 5 minutes
LIMIT_ORDER_TIMEOUT_SECONDS = 90  # cancel unfilled limits after this

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

# --- Signal Parameters (v2: dual engine) ---
# Breakout: 72h lookback on 5-min candles = 864 periods
# Using hourly-equivalent: 72 periods on 5-min data ≈ 6h breakout window
# But we also load 1h data separately, so actual lookback is configurable
BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "864"))  # 72h in 5-min bars

# RSI mean reversion
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "30"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "65"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))

# --- Risk Management ---
MAX_POSITION_PCT = 0.20          # max 20% of portfolio in one coin
MAX_TOTAL_EXPOSURE_PCT = 0.60    # max 60% invested, 40% cash
TARGET_RISK_PER_TRADE = 0.015    # risk 1.5% of portfolio per position
MAX_POSITIONS = 8                # hold at most 8 coins at once

# Trailing stop: adaptive based on ATR, fallback to fixed %
TRAILING_STOP_PCT = 0.03         # -3% trailing stop per position (wider to avoid whipsaw)
TRAILING_STOP_ATR_MULT = 2.0     # alternative: 2x ATR trailing stop

# Drawdown circuit breakers (from portfolio peak)
DRAWDOWN_LEVEL_1 = 0.02   # -2%: cut positions 50%
DRAWDOWN_LEVEL_2 = 0.04   # -4%: liquidate all, pause 12h
DRAWDOWN_LEVEL_3 = 0.07   # -7%: cash only for 48h

DRAWDOWN_PAUSE_HOURS = {
    1: 0,     # level 1: no pause, just reduce
    2: 12,    # level 2: pause 12 hours
    3: 48,    # level 3: pause 48 hours
}

# --- Volatility ---
VOL_LOOKBACK_PERIODS = 288  # 24h of 5-min candles for realized vol
VOL_REGIME_BASELINE_PERIODS = 8640  # 30 days of 5-min candles

# --- Execution ---
USE_LIMIT_ORDERS = True
LIMIT_ORDER_OFFSET_BPS = 1  # place limit 1 bps inside spread

# --- Coin Filtering ---
# Focus on liquid, well-known coins with Binance data available.
# Exclude obscure tokens where Binance data won't match.
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
