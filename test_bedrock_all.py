import json
import glob
import os
from src.loader import build_taxonomy
from src.sector_matcher_bedrock import match_sectors_to_taxonomy_bedrock
from src.topic_matcher_bedrock import match_topics_flat_bedrock, refine_matched_topics_bedrock

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
    print(f"\n=== {display_name.upper()} ===")

    sector_result = match_sectors_to_taxonomy_bedrock(
        company_name=result["company_name"],
        business_summary=result["business_summary"],
        generated_sectors=result.get("sectors", []),
        sectors_prompt_text=taxonomy["sectors_prompt_text"],
        valid_sector_ids=taxonomy["valid_sector_ids"],
    )
    print("Matched sector ids:", sector_result["matched_sector_ids"])
    print("Sector reasoning:", sector_result["reasoning"])

    matched_sectors_input = [
        {"sector_id": sid, "sector_name": sector_id_to_name.get(sid, "UNKNOWN ID")}
        for sid in sector_result["matched_sector_ids"]
    ]

    topic_result = match_topics_flat_bedrock(
        company_name=result["company_name"],
        business_summary=result["business_summary"],
        matched_sectors=matched_sectors_input,
        topics=taxonomy["topics"],
    )
    print("Initial topics:", [(t["topic_id"], t["topic_name"]) for t in topic_result["topics"]])

    refined = refine_matched_topics_bedrock(
        company_name=result["company_name"],
        business_summary=result["business_summary"],
        matched_sectors=matched_sectors_input,
        initial_topics=topic_result["topics"],
    )
    print("Refined (final) topics:")
    for t in refined["topics"]:
        print(f"  - {t['topic_id']} | {t['topic_name']}")
    print("Refinement reasoning:", refined["reasoning"])

    result["matched_sector_ids_bedrock"] = sector_result["matched_sector_ids"]
    result["sector_matching_reasoning_bedrock"] = sector_result["reasoning"]
    result["matched_topics_flat_bedrock"] = refined["topics"]
    result["topic_matching_reasoning_bedrock"] = refined["reasoning"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {path}")