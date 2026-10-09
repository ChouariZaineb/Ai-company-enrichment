import json
from src.loader import build_taxonomy
from src.sector_matcher_gemini import match_sectors_to_taxonomy_gemini

taxonomy = build_taxonomy(
    sectors_path="data/sectors_formatted.txt",
    topics_path="data/topics_formatted.txt",
)

with open("data/overviews_cache/1002_monzo.json", encoding="utf-8") as f:
    monzo = json.load(f)

result = match_sectors_to_taxonomy_gemini(
    company_name=monzo["company_name"],
    business_summary=monzo["business_summary"],
    generated_sectors=monzo["sectors"],
    sectors_prompt_text=taxonomy["sectors_prompt_text"],
    valid_sector_ids=taxonomy["valid_sector_ids"],
)

print("matched_sector_ids:", result["matched_sector_ids"])
print("reasoning:", result["reasoning"])
print("\nMistral known result: [7, 16]")