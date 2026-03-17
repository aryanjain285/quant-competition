# **Overview**

This document outlines three key sources for cryptocurrency market data, ordered by our recommended free options first, down to enterprise solutions.

These sources are only a starting point. You are by no means limited to these sources. There shall be a wealth of publicly available data in the market – we strongly encourage you to conduct your own research to identify the best possible data for your bots.

---

## 1. Binance Public Data *(Highly Recommended)*

**Pricing:** 100% Free

**Link:** [Binance Vision Data](https://data.binance.vision/)

**Overview:**
Binance Vision is the official public data repository for the Binance exchange. It allows developers and researchers to bypass rate limits by directly downloading bulk archives of Binance's historical market data.

**Key Features & Data Types:**

-
- **Complete Binance History:** Provides tick-level and aggregated data for all Spot, USD-Margined Futures, and COIN-Margined Futures traded on Binance.
- **Klines (Candlesticks):** Available in intervals from 1-second (`1s`) up to 1-month (`1mo`).
- **Trade Data:** Raw trades and Aggregated Trades (`AggTrades`).
- **No Authentication Needed:** You do not need a Binance account or an API key to download this data.

**Delivery Methods:**

- Direct `.zip` downloads via the web browser.
- Programmatic downloads using `wget` or `curl` commands.
- Open-source Python packages (like `binance-historical-data` on PyPI) or GitHub shell scripts that automate fetching daily and monthly data dumps.

**Best For:**
Heavy backtesting, data science, and training Machine Learning models where the team needs massive amounts of highly accurate historical data for free, and is okay with relying strictly on Binance's market liquidity.

---

## 2. CryptoDataDownload ([www.cryptodatadownload.com](http://www.cryptodatadownload.com/))

**Pricing:** Free
**Link:** [CryptoDataDownload](https://www.cryptodatadownload.com/data/)

**Overview:**
CryptoDataDownload (CDD) is a widely used aggregator that provides easy-to-use historical cryptocurrency data. It is highly popular among academic researchers, students, and analysts who need immediate, clean data without writing complex API scripts.

**Key Features & Data Types:**

- **Multi-Exchange CSVs:** Offers pre-formatted, easy-to-download CSV files covering major exchanges like Coinbase, Binance, Kraken, Gemini, and more.
- **Beyond Raw Prices:** Their API (CDD API v1) provides derived financial metrics, including:
    - **Market Breadth Indicators:** Advance/Decline counts, 52-week highs/lows.
    - **Risk Metrics:** Value at Risk (VaR) by symbol, trading correlations, and Black-Scholes option pricing.
    - **On-Chain & Macro Data:** Blockchain metrics (transaction counts, BTC market cap) and US Treasury yield data.
    - **Derivatives Data:** Deribit options/futures and funding rates.

**Delivery Methods:**

- Direct CSV downloads from the website.
- Public API endpoints (CDD API v1).
- Available via AWS Data Exchange for cloud integration.

**Best For:**
Quick, ad-hoc analysis, importing data directly into Excel/Pandas, and fetching broader market risk or on-chain metrics without paying premium data provider fees.

---

## 3. CoinAPI ([docs.coinapi.io](http://docs.coinapi.io/))

**Pricing:** Paid (Enterprise & Pro tiers), but offers a free tier/free credits for initial testing and development.
**Link:** [CoinAPI Market Data](https://docs.coinapi.io/market-data/)

**Overview:**
CoinAPI is an institutional-grade, unified data provider. Instead of connecting to dozens of different exchanges with different data formats, CoinAPI standardizes the data from over 400+ cryptocurrency exchanges into one single format.

**Key Features & Data Types:**

- **Massive Coverage:** Covers 400+ exchanges, 18,000+ assets, and over 900k symbols.
- **Real-Time & Historical Data:** Offers everything from live streaming data to years of historical records.
- **Tick-by-Tick Data:** Highly granular data capturing every single trade execution.
- **Order Book Depth:** Full Level 2 (L2) and Level 3 (L3) order book snapshots and incremental updates to analyze market liquidity.
- **OHLCV Data:** Standard candlestick data ranging from 1-second intervals up to years.

**Delivery Methods:**

-- REST API (for querying historical or latest data)
- WebSocket / FIX API (for ultra-low latency, real-time streaming)
- S3-compatible "Flat Files" (for bulk downloading massive historical datasets as compressed CSVs)