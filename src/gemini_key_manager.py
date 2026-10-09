#src/gemini_key_manager.py
"""
Rotates across multiple Gemini API keys, so request volume splits between
them instead of hammering a single free-tier key and hitting 429s.

.env format — comma-separated, no quotes needed:
    GEMINI_API_KEY=key1,key2,key3
A single key also still works fine:
    GEMINI_API_KEY=key1
"""
import os
import itertools
from dotenv import load_dotenv

load_dotenv(override=True)

_raw_keys = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]

if not GEMINI_API_KEYS:
    raise RuntimeError("No GEMINI_API_KEY found in .env — set at least one, comma-separated for multiple.")

print(f"[INFO] Loaded {len(GEMINI_API_KEYS)} Gemini API key(s) for rotation.")

_key_cycle = itertools.cycle(GEMINI_API_KEYS)


def get_next_key() -> str:
    """Round-robins through all configured keys — one call, one key, next call, next key."""
    return next(_key_cycle)


def build_gemini_url(model: str = "gemini-3.5-flash") -> str:
    """Builds a fresh Gemini URL with the next key in rotation."""
    key = get_next_key()
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

# --- Pro key (dedicated, non-rotating) — used for the new cloud-only pipeline ---
GEMINI_PRO_API_KEY = os.getenv("GEMINI_PRO_API_KEY", "").strip()
if not GEMINI_PRO_API_KEY:
    print("[WARN] No GEMINI_PRO_API_KEY found in .env — pro-tier pipeline will fail if used.")


def build_gemini_pro_url(model: str = "gemini-pro-latest") -> str:
    """Builds a Gemini URL using the dedicated pro key (no rotation — this key has proper quota)."""
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_PRO_API_KEY}"


def build_gemini_embedding_url(model: str = "gemini-embedding-2") -> str:
    """Builds a Gemini embedding URL using the dedicated pro key."""
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={GEMINI_PRO_API_KEY}"