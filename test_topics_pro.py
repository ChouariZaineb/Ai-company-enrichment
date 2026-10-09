import json
from src.loader import build_taxonomy
from src.topic_matcher_gemini import match_topics_flat_gemini

taxonomy = build_taxonomy(
    sectors_path="data/sectors_formatted.txt",
    topics_path="data/topics_formatted.txt",
)

with open("data/overviews_cache/1002_monzo.json", encoding="utf-8") as f:
    monzo = json.load(f)

sector_id_to_name = {s.sector_id: s.sector_name for s in taxonomy["sectors"]}
matched_sector_ids = monzo["matched_sector_ids"]  # [7, 16] from mistral, already validated
matched_sectors_input = [
    {"sector_id": sid, "sector_name": sector_id_to_name.get(sid, "UNKNOWN ID")}
    for sid in matched_sector_ids
]

print("Matched sectors:", matched_sectors_input)
print("Starting embedding build + flat topic matching (this may take a few minutes on first run)...\n")

result = match_topics_flat_gemini(
    company_name=monzo["company_name"],
    business_summary=monzo["business_summary"],
    matched_sectors=matched_sectors_input,
    topics=taxonomy["topics"],
)

print("\n=== FLAT MATCHED TOPICS (Gemini Pro) ===")
for t in result["topics"]:
    print(f"  - {t['topic_id']} | {t['topic_name']}")
print("\nReasoning:", result["reasoning"])

print("\n=== GROUND TRUTH (Monzo official) ===")
print("  16 -> 1870 Digital Banking, 8619 Neobanking, 8747 Banking Services, 1465 FinTech")
print("  7  -> 722 SaaS")