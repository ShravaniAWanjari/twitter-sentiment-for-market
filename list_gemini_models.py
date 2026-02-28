import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

try:
    from google import genai
    client = genai.Client(api_key=key)
    print("Listing models from new SDK:")
    for m in client.models.list():
        print(f"Name: {m.name}, DisplayName: {m.display_name}, Supported: {m.supported_actions}")
except Exception as e:
    print(f"New SDK list failed: {e}")

try:
    import google.generativeai as genai_legacy
    genai_legacy.configure(api_key=key)
    print("\nListing models from legacy SDK:")
    for m in genai_legacy.list_models():
        print(f"Name: {m.name}, DisplayName: {m.display_name}, Supported: {m.supported_methods}")
except Exception as e:
    print(f"Legacy SDK list failed: {e}")
