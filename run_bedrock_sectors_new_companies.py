import json
from src.loader import build_taxonomy
from src.sector_matcher_bedrock import match_sectors_to_taxonomy_bedrock
from src.cache_utils import save_cached_result

taxonomy = build_taxonomy(
    sectors_path="data/sectors_formatted.txt",
    topics_path="data/topics_formatted.txt",
)
sector_id_to_name = {s.sector_id: s.sector_name for s in taxonomy["sectors"]}

companies = [
    (1010, "data/overviews_cache/1010_babylon_health.json"),
    (1015, "data/overviews_cache/1015_hometree_group.json"),
]

for company_id, path in companies:
    with open(path, encoding="utf-8") as f:
        result = json.load(f)

    display_name = result.get("company_name")
    print(f"\n=== {display_name.upper()} ===")

    sector_result = match_sectors_to_taxonomy_bedrock(
        company_name=display_name,
        business_summary=result["business_summary"],
        generated_sectors=result.get("sectors", []),
        sectors_prompt_text=taxonomy["sectors_prompt_text"],
        valid_sector_ids=taxonomy["valid_sector_ids"],
    )
    print("Matched sector ids:", sector_result["matched_sector_ids"])
    print("Reasoning:", sector_result["reasoning"])

    matched_sector_names = [
        f"{sid} | {sector_id_to_name.get(sid, 'UNKNOWN ID')}"
        for sid in sector_result["matched_sector_ids"]
    ]

    result["matched_sector_ids_bedrock"] = sector_result["matched_sector_ids"]
    result["matched_sector_names_bedrock"] = matched_sector_names
    result["sector_matching_reasoning_bedrock"] = sector_result["reasoning"]

    save_cached_result(company_id, display_name, result)
    print(f"Saved to cache for company_id {company_id}")