import json
import glob
import os
from src.loader import build_taxonomy
from src.sector_matcher_bedrock import match_sectors_to_taxonomy_bedrock
from src.topic_matcher_bedrock_v2 import match_topics_flat_bedrock_v2

taxonomy = build_taxonomy(
    sectors_path="data/sectors_formatted.txt",
    topics_path="data/topics_formatted.txt",
)
sector_id_to_name = {s.sector_id: s.sector_name for s in taxonomy["sectors"]}

target_paths = []
for pattern in ["*monzo*.json", "*darktrace*.json", "*octopus*.json"]:
    target_paths.extend(glob.glob(os.path.join("data", "overviews_cache", pattern)))

for path in target_paths:
    with open(path, encoding="utf-8") as f:
        result = json.load(f)

    display_name = result.get("company_name") or os.path.basename(path)
    print(f"\n=== {display_name.upper()} (v2 - enriched embeddings) ===")

    sector_result = match_sectors_to_taxonomy_bedrock(
        company_name=result["company_name"],
        business_summary=result["business_summary"],
        generated_sectors=result.get("sectors", []),
        sectors_prompt_text=taxonomy["sectors_prompt_text"],
        valid_sector_ids=taxonomy["valid_sector_ids"],
    )
    print("Matched sector ids:", sector_result["matched_sector_ids"])

    matched_sectors_input = [
        {"sector_id": sid, "sector_name": sector_id_to_name.get(sid, "UNKNOWN ID")}
        for sid in sector_result["matched_sector_ids"]
    ]

    company_data_extra = {
        "industry": result.get("industry", []),
        "sectors": result.get("sectors", []),
        "products_offered": result.get("products_offered", []),
        "services_offered": result.get("services_offered", []),
        "unique_selling_propositions": result.get("unique_selling_propositions", []),
    }

    topic_result = match_topics_flat_bedrock_v2(
        company_name=result["company_name"],
        business_summary=result["business_summary"],
        matched_sectors=matched_sectors_input,
        topics=taxonomy["topics"],
        company_data_extra=company_data_extra,
    )
    print("Final topics (v2):")
    for t in topic_result["topics"]:
        print(f"  - {t['topic_id']} | {t['topic_name']}")
    print("Reasoning:", topic_result["reasoning"])

    # Saved under DIFFERENT keys (_v2 suffix) so original _bedrock results are untouched
    result["matched_sector_ids_bedrock_v2"] = sector_result["matched_sector_ids"]
    result["matched_topics_flat_bedrock_v2"] = topic_result["topics"]
    result["topic_matching_reasoning_bedrock_v2"] = topic_result["reasoning"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {path}")