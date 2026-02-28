import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

try:
    from google import genai
    client = genai.Client(api_key=key)
    print("Testing gemini-flash-latest...")
    response = client.models.generate_content(model='gemini-flash-latest', contents="Say 'OK'")
    print(f"SUCCESS: {response.text}")
except Exception as e:
    print(f"FAILED: {e}")
