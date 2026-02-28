import os
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GEMINI_API_KEY")
print(f"Key: {key[:10]}...")

try:
    from google import genai
    client = genai.Client(api_key=key)
    models = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-flash-8b']
    for m in models:
        try:
            print(f"Testing {m}...")
            response = client.models.generate_content(model=m, contents="Say 'OK'")
            print(f"{m} SUCCESS: {response.text}")
        except Exception as e:
            print(f"{m} FAILED: {e}")
except Exception as e:
    print(f"New SDK import failed: {e}")
    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=key)
        for m in ['gemini-1.5-flash', 'gemini-pro']:
            try:
                print(f"Testing legacy {m}...")
                model = genai_legacy.GenerativeModel(m)
                response = model.generate_content("Say 'OK'")
                print(f"{m} SUCCESS (legacy): {response.text}")
            except Exception as e2:
                print(f"{m} FAILED (legacy): {e2}")
    except Exception as e3:
        print(f"Legacy SDK check failed: {e3}")
