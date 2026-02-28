import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

out = []

try:
    from google import genai
    client = genai.Client(api_key=key)
    out.append("Listing models from new SDK:")
    for m in client.models.list():
        out.append(f"Name: {m.name}, DisplayName: {m.display_name}")
except Exception as e:
    out.append(f"New SDK list failed: {e}")

try:
    import google.generativeai as genai_legacy
    genai_legacy.configure(api_key=key)
    out.append("\nListing models from legacy SDK:")
    for m in genai_legacy.list_models():
        out.append(f"Name: {m.name}, DisplayName: {m.display_name}")
except Exception as e:
    out.append(f"Legacy SDK list failed: {e}")

with open("full_model_list.txt", "w") as f:
    f.write("\n".join(out))
