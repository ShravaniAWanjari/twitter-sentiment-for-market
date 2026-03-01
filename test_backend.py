import requests
import json

url = "http://localhost:8000/api/chat"
payload = {"query": "Tell me about the backtest results.", "run_artifact": {}}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Body: {response.text}")
except Exception as e:
    print(f"Error: {e}")
