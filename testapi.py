import os
from bot.roostoo_client import RoostooClient

def test_api():
    print("Initiating Roostoo API Connection Test...")
    
    # Initialize your existing client
    client = RoostooClient()
    
    # Optional: If you want to hardcode an HTTP proxy instead of using a system VPN
    # os.environ['HTTP_PROXY'] = "http://YOUR_PROXY_IP:PORT"
    # os.environ['HTTPS_PROXY'] = "http://YOUR_PROXY_IP:PORT"
    
    print("-" * 40)
    print("1. Testing Public Endpoint (serverTime)")
    # Testing the server_time method which requires no authentication
    server_time = client.server_time()
    
    if server_time:
        print(f"✅ SUCCESS: Connected to Roostoo. Server Time: {server_time}")
    else:
        print("❌ FAILED: 451 Error likely persisting. The API blocked the request before authentication.")
        return # Stop execution if the public ping fails
        
    print("-" * 40)
    print("2. Testing Private Auth Endpoint (balance)")
    # Testing the balance method which utilizes your HMAC-SHA256 signing
    wallet = client.balance()
    
    if wallet is not None:
        print("✅ SUCCESS: API Keys are valid and balance was retrieved.")
        usd_balance = wallet.get("USD", {}).get("Free", "0")
        print(f"   Available USD: ${usd_balance}")
    else:
        print("❌ FAILED: Connected to server, but authentication or balance fetch failed.")

if __name__ == "__main__":
    test_api()