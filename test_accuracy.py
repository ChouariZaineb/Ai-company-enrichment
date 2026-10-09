import json
from src.accuracy import score_against_ground_truth

with open("data/overviews_cache/1002_monzo.json", encoding="utf-8") as f:
    monzo = json.load(f)

print("=== MISTRAL ===")
mistral_scores = score_against_ground_truth(
    company_id=1002,
    matched_sector_ids=monzo.get("matched_sector_ids"),
    topics_by_sector=monzo.get("topics_by_sector"),
)
print(json.dumps(mistral_scores, indent=2))

print("\n=== GEMINI ===")
gemini_scores = score_against_ground_truth(
    company_id=1002,
    matched_sector_ids=monzo.get("matched_sector_ids_gemini"),
    topics_by_sector=monzo.get("topics_by_sector_gemini"),
)
print(json.dumps(gemini_scores, indent=2))