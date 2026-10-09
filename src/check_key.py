import os
from dotenv import load_dotenv

load_dotenv(override=True)
key = os.getenv("GEMINI_API_KEY")
print(f"Key repr: {repr(key)}")
print(f"Key length: {len(key) if key else 0}")