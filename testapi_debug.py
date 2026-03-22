import traceback
from urllib.parse import parse_qsl

import requests

from bot.roostoo_client import RoostooClient, _sign


def print_response(label: str, response: requests.Response) -> None:
    print(f"\n[{label}] HTTP {response.status_code}")
    print(f"URL: {response.url}")
    print("Response headers:")
    for k, v in response.headers.items():
        print(f"  {k}: {v}")

    text = response.text
    print("Response body (first 1000 chars):")
    print(text[:1000] if text else "<empty>")


def test_public_endpoints(client: RoostooClient) -> None:
    print("-" * 60)
    print("1. Testing public endpoints")

    for name, path in [
        ("serverTime", "/v3/serverTime"),
        ("exchangeInfo", "/v3/exchangeInfo"),
    ]:
        url = f"{client.base}{path}"
        print(f"\nTesting public endpoint: {name} -> {url}")
        try:
            resp = client.session.get(url, timeout=10)
            print_response(f"public:{name}", resp)
        except Exception as e:
            print(f"Exception on {name}: {type(e).__name__}: {e}")
            traceback.print_exc()


def test_balance_original(client: RoostooClient) -> None:
    print("-" * 60)
    print("2. Testing balance endpoint using current client logic")

    params = {"timestamp": "PLACEHOLDER"}
    headers, _ = _sign(params)
    url = f"{client.base}/v3/balance"

    print("Request headers:")
    for k, v in headers.items():
        print(f"  {k}: {v}")
    print("Request params:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    try:
        resp = client.session.get(url, headers=headers, params=params, timeout=10)
        print_response("balance:current_logic", resp)
    except Exception as e:
        print(f"Exception on current balance logic: {type(e).__name__}: {e}")
        traceback.print_exc()


def test_balance_exact_signed_query(client: RoostooClient) -> None:
    print("-" * 60)
    print("3. Testing balance endpoint with exact signed query string")

    params = {}
    headers, total_params = _sign(params)
    signed_params = dict(parse_qsl(total_params, keep_blank_values=True))
    url = f"{client.base}/v3/balance"

    print(f"Signed query string: {total_params}")
    print("Request headers:")
    for k, v in headers.items():
        print(f"  {k}: {v}")
    print("Signed params sent:")
    for k, v in signed_params.items():
        print(f"  {k}: {v}")

    try:
        resp = client.session.get(url, headers=headers, params=signed_params, timeout=10)
        print_response("balance:exact_signed_query", resp)
    except Exception as e:
        print(f"Exception on exact signed query balance test: {type(e).__name__}: {e}")
        traceback.print_exc()


def test_api() -> None:
    print("Initiating Roostoo API Connection Diagnostic Test...")
    client = RoostooClient()
    print(f"Base URL: {client.base}")
    print(f"Session proxies resolved by requests: {client.session.proxies}")

    test_public_endpoints(client)
    test_balance_original(client)
    test_balance_exact_signed_query(client)


if __name__ == "__main__":
    test_api()
