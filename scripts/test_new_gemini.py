import os
import logging
from dotenv import load_dotenv
from google import genai

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("No GEMINI_API_KEY found")
    exit(1)

try:
    client = genai.Client(api_key=api_key)
    print("Client initialized")
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents="Research test query: What is a Sortino ratio?"
    )
    print(f"Full Response text: {response.text}")
        
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
