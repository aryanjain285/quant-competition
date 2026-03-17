 Competition Summary: AI Web3 Trading Bot Hackathon                      
                                                                          
  What You're Building                                                    
                                                                          
  An autonomous trading bot that interacts with Roostoo's mock exchange   
  via REST APIs. It must make buy/hold/sell decisions without manual      
  intervention. You start with a $1,000,000 mock portfolio.               
                                                                
  Key Rules & Constraints                                       

  1. No HFT, market-making, or arbitrage — excessive requests = failed    
  APIs
  2. Spot trading only — no leverage, no short selling                    
  3. Commission fees: 0.1% taker (market orders), 0.05% maker (limit
  orders)                                                                 
  4. Open-source repo required with clean commit history
  5. No manual API calls — everything must be traceable to your bot's     
  strategy                                                                
  6. Must deploy on AWS EC2 (provided by Roostoo)                         
  7. At least 8 active trading days with enough trades each day           
                                                                          
  Timeline (You're in it now)                                             
                                                                          
  - Mar 16–20: Preparation — build & test bot                             
  - Mar 21–31: Round 1 live trading (10 days)                             
  - Before Mar 28: Submit repo link                                       
  - Apr 2: Top 16 finalists announced                                     
  - Apr 4–14: Round 2 (if you qualify)                                    
                                                                          
  Evaluation Criteria (in order)                                          
                                                                          
  1. Screen 1 — Rule Compliance (pass/fail): Consistent autonomous trades,
   traceable commit history, no manual API calls
  2. Screen 2 — Portfolio Returns: Top 20 by return advance               
  3. Screen 3 — Risk-Adjusted Score (40%): 0.4 × Sortino + 0.3 × Sharpe + 
  0.3 × Calmar                                                            
  4. Screen 4 — Code & Strategy Review (60%): Strategy logic (30%), code  
  quality (20%), runnable on Roostoo (10%)                                
                  
  API Details (https://mock-api.roostoo.com)                              
                  
  ┌───────────────────┬────────┬───────────┬───────────────────────────┐  
  │     Endpoint      │ Method │   Auth    │          Purpose          │
  ├───────────────────┼────────┼───────────┼───────────────────────────┤  
  │ /v3/serverTime    │ GET    │ None      │ Server time               │
  ├───────────────────┼────────┼───────────┼───────────────────────────┤
  │ /v3/exchangeInfo  │ GET    │ None      │ Available pairs,          │  
  │                   │        │           │ precision rules           │  
  ├───────────────────┼────────┼───────────┼───────────────────────────┤  
  │ /v3/ticker        │ GET    │ Timestamp │ Market prices, bid/ask,   │  
  │                   │        │           │ 24h change                │  
  ├───────────────────┼────────┼───────────┼───────────────────────────┤
  │ /v3/balance       │ GET    │ Signed    │ Wallet balances           │  
  │                   │        │           │ (Free/Lock)               │  
  ├───────────────────┼────────┼───────────┼───────────────────────────┤
  │ /v3/pending_count │ GET    │ Signed    │ Pending order count       │  
  ├───────────────────┼────────┼───────────┼───────────────────────────┤  
  │ /v3/place_order   │ POST   │ Signed    │ Place LIMIT/MARKET orders │
  ├───────────────────┼────────┼───────────┼───────────────────────────┤  
  │ /v3/query_order   │ POST   │ Signed    │ Query order history       │
  ├───────────────────┼────────┼───────────┼───────────────────────────┤  
  │ /v3/cancel_order  │ POST   │ Signed    │ Cancel pending orders     │
  └───────────────────┴────────┴───────────┴───────────────────────────┘  
                  
  Auth: Signed endpoints require RST-API-KEY + MSG-SIGNATURE (HMAC SHA256 
  of sorted params using secret key) in headers. POST requests need
  Content-Type: application/x-www-form-urlencoded.                        
                  
  Your API Keys                                                           
   
  - General/Testing: API Key l5zxW7... + Secret                           
  - Round 1 Competition: API Key GAENnm... + Secret
                                                                          
  Data Sources Available                                                  
   
  1. Binance Vision (free) — bulk historical klines, trades, no auth      
  needed          
  2. CryptoDataDownload (free) — multi-exchange CSVs, risk metrics,       
  on-chain data                                                           
  3. CoinAPI (freemium) — institutional-grade, 400+ exchanges, real-time +
   historical                                                             
                  
  Strategy Considerations                                                 
                  
  - Sortino/Sharpe/Calmar weighting means you need to minimize downside   
  risk, not just maximize returns
  - Sortino has the highest weight (0.4) — penalizes downside volatility  
  specifically                                                            
  - Calmar ratio = return / max drawdown — avoid large drawdowns
  - Strategies are open-ended: LLMs, RL (PPO), traditional quant, or      
  hybrid all allowed                                                      
  - Log everything internally for debugging and compliance                
                                                                          
  ---             
  What would you like to do next? I can help you design the bot           
  architecture, set up the project structure, implement the API client, or
   plan a trading strategy.