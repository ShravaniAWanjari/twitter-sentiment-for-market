import os
from dotenv import load_dotenv

print(f"Pre-load key: {os.environ.get('GEMINI_API_KEY', 'MISSING')[:8]}")
load_dotenv()
print(f"Post-load key: {os.environ.get('GEMINI_API_KEY', 'MISSING')[:8]}")

if os.path.exists('.env'):
    with open('.env', 'r') as f:
        print(f"Raw .env line 1: {f.readline().strip()[:20]}")
