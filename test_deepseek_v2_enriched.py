import json
import glob
import os
from src.loader import build_taxonomy
from src.sector_matcher_deepseek import match_sectors_to_taxonomy_deepseek
from src.topic_matcher_deepseek import match_topics_flat_deepseek

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
    print(f"\n=== {display_name.upper()} (DeepSeek - Enriched Embeddings) ===")

    sector_result = match_sectors_to_taxonomy_deepseek(
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

    # Uses the enriched-embedding candidate retrieval already built into
    # topic_matcher_deepseek.py — same underlying embeddings as Claude v2.
    topic_result = match_topics_flat_deepseek(
        company_name=result["company_name"],
        business_summary=result["business_summary"],
        matched_sectors=matched_sectors_input,
        topics=taxonomy["topics"],
        company_data_extra=company_data_extra,
    )
    print("Final topics (v2 - enriched):")
    for t in topic_result["topics"]:
        print(f"  - {t['topic_id']} | {t['topic_name']}")
    print("Reasoning:", topic_result["reasoning"])

    # Saved under DISTINCT _v2 keys so the original _deepseek results are untouched
    result["matched_sector_ids_deepseek_v2"] = sector_result["matched_sector_ids"]
    result["sector_matching_reasoning_deepseek_v2"] = sector_result["reasoning"]
    result["matched_topics_flat_deepseek_v2"] = topic_result["topics"]
    result["topic_matching_reasoning_deepseek_v2"] = topic_result["reasoning"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {path}")