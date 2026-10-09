#cache_utils.py
import json
import os
import re

OUTPUT_DIR = "data/overviews_cache"


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_").lower()


def cache_path_for(company_id, display_name: str) -> str:
    filename = f"{company_id}_{sanitize_filename(display_name)}.json"
    return os.path.join(OUTPUT_DIR, filename)


def load_cached_result(company_id, display_name: str):
    """Returns the cached result dict, or None if nothing's cached yet."""
    path = cache_path_for(company_id, display_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cached_result(company_id, display_name: str, result: dict) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = cache_path_for(company_id, display_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return path