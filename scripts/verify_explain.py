import requests
import json

BASE_URL = "http://localhost:8000/api"

def test_explain():
    payload = {
        "model_id": "modernbert",
        "text": "Crypto surges"
    }
    print(f"Testing /api/explain with: {payload['text']}")
    try:
        res = requests.post(f"{BASE_URL}/explain", json=payload)
        res.raise_for_status()
        data = res.json()
        
        print("\n[SUCCESS] Response received.")
        print(f"Method: {data.get('method')}")
        print(f"Tokens: {len(data.get('tokens', []))}")
        print(f"Stability Score: {data.get('stability', {}).get('score_0_1')}")
        print(f"Counterfactual Found: {data.get('counterfactual', {}).get('found')}")
        
        if data.get('counterfactual', {}).get('found'):
            print(f"CF Flip: {data['counterfactual']['edited_text']} -> {data['counterfactual']['flipped_label']}")
            
        # Verify schema
        required_keys = ["method", "tokens", "top_positive", "top_negative", "highlighted_html", "stability", "counterfactual"]
        for key in required_keys:
            if key not in data:
                print(f"[ERROR] Missing key in response: {key}")
            else:
                print(f"[OK] Found key: {key}")
                
    except Exception as e:
        print(f"[FAILED] Error: {e}")

if __name__ == "__main__":
    test_explain()
