import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

repo_root = Path('.').resolve()
load_dotenv(dotenv_path=repo_root / '.env', override=True)
key = os.getenv('GEMINI_API_KEY')
print(f"Using key: {key[:8]}...")

genai.configure(api_key=key)
# Try gemini-2.0-flash as it's definitely in the list
model = genai.GenerativeModel('gemini-2.0-flash')
try:
    response = model.generate_content('Are you available?')
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
