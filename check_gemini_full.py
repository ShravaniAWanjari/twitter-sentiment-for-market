import os
from pathlib import Path
from dotenv import load_dotenv

repo_root = Path('.').resolve()
env_path = repo_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"Loaded .env from {env_path}")
else:
    print("No .env found")

key = os.getenv('GEMINI_API_KEY')
print(f"Using key: {key[:8]}...")

try:
    import google.generativeai as genai
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content('Hello, are you online?')
    print(f"Gemini Response: {response.text}")
except Exception as e:
    print(f"Gemini Error: {e}")
