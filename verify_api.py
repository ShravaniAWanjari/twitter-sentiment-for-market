import requests
import time

BASE_URL = "http://localhost:8000/api"

def test_backtest():
    print("Testing /api/backtest...")
    res = requests.post(f"{BASE_URL}/backtest", json={"model": "modernbert", "strategy": "RSI", "threshold": 0.4})
    if res.status_code == 200:
        print("SUCCESS: Backtest run complete.")
        data = res.json()
        print(f"Metrics: {data['metrics']}")
    else:
        print(f"FAILED: {res.status_code} {res.text}")

def test_latest():
    print("Testing /api/backtest/latest...")
    res = requests.get(f"{BASE_URL}/backtest/latest")
    if res.status_code == 200:
        print("SUCCESS: Retrieved latest run.")
    else:
        print(f"FAILED: {res.status_code}")

def test_analyze():
    print("Testing /api/analyze...")
    res = requests.post(f"{BASE_URL}/analyze", json={"model": "modernbert"})
    if res.status_code == 200:
        print("SUCCESS: Headline analysis complete.")
        data = res.json()
        print(f"Analyzed {len(data)} headlines.")
        if len(data) > 0:
            print(f"First headline sentiment: {data[0]['sentiment']}")
    else:
        print(f"FAILED: {res.status_code}")

def test_chat():
    print("Testing /api/chat...")
    res = requests.post(f"{BASE_URL}/chat", json={"query": "How was the performance?"})
    if res.status_code == 200:
        print(f"SUCCESS: Chat response: {res.json()['response']}")
    else:
        print(f"FAILED: {res.status_code}")

    res = requests.post(f"{BASE_URL}/chat", json={"query": "Should i buy btc?"})
    print(f"Financial advice check: {res.json()['response']}")

if __name__ == "__main__":
    test_backtest()
    test_latest()
    test_analyze()
    test_chat()
    print("Verification complete.")
