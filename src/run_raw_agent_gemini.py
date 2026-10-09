# src/run_raw_agent_gemini.py
"""
Runs the raw-text tool-loop agent using Gemini Pro.
Saves results under *_raw_agent_gemini keys.
Run from project root: python src/run_raw_agent_gemini.py
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raw_agent_core import run_raw_agent
from gemini_pro_adapter import GeminiProLLMClient
from taxonomy_suggester import suggest_missing_topics

_BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_BASE, "data", "overviews_cache")

COMPANIES = [
    (14362, "McKenzie Intelligence Services", "14362_mckenzie_intelligence_services.json"),
    (30336, "Operations1",                    "30336_operations1.json"),
    (30344, "Oxane Partners",                 "30344_oxane_partners.json"),
    (30345, "Centor Rx",                      "30345_centor_rx.json"),
    (30346, "Axel Health",                    "30346_axel_health.json"),
    (30350, "Talent Funnel",                  "30350_talent_funnel.json"),
    (30352, "Teknoice",                       "30352_teknoice.json"),
]

GROUND_TRUTH = {
    14362: {"sector_ids": {7, 16, 21}, "topic_ids": {2282, 2984, 4204, 4541, 5751, 6897, 8640, 8791, 8793}},
    30336: {"sector_ids": {7},         "topic_ids": {722, 1859, 7980}},
    30344: {"sector_ids": {7, 16},     "topic_ids": {776, 1719, 1783, 1858, 1864}},
    30345: {"sector_ids": {20},        "topic_ids": {2828, 8399, 9241}},
    30346: {"sector_ids": {7, 19},     "topic_ids": {1956, 2282, 9025}},
    30350: {"sector_ids": {7, 17},     "topic_ids": {1719, 2931, 5477}},
    30352: {"sector_ids": {18, 20},    "topic_ids": {4777, 8136, 8206}},
}


def _score_recall(predicted: set, gt: set) -> dict:
    if not gt:
        return {"recall": 0.0, "hits": 0, "total": 0}
    hits = len(predicted & gt)
    return {"recall": round(hits / len(gt), 2), "hits": hits, "total": len(gt)}


client = GeminiProLLMClient()

for company_id, company_name, filename in COMPANIES:
    path = os.path.join(CACHE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        cached = json.load(f)

    raw_text = cached.get("raw_scraped_content", "")
    if not raw_text:
        print(f"\n[SKIP] {company_name} — no raw_scraped_content. Run scrape_and_store_raw.py first.")
        continue

    gt            = GROUND_TRUTH.get(company_id, {})
    gt_sector_ids = gt.get("sector_ids", set())
    gt_topic_ids  = gt.get("topic_ids", set())

    print(f"\n{'='*65}")
    print(f"COMPANY: {company_name}  (id={company_id})")
    print(f"{'='*65}")

    result = run_raw_agent(company_name, raw_text, client)

    cached["matched_sector_ids_raw_agent_gemini"]      = result["matched_sector_ids"]
    cached["matched_topics_flat_raw_agent_gemini"]     = result["matched_topics_flat"]
    cached["company_description_raw_agent_gemini"]     = result["company_description"]
    cached["topic_reasoning_raw_agent_gemini"]         = json.dumps(
        [{"topic_id": t["topic_id"], "topic_name": t["topic_name"], "evidence": t["evidence"]}
         for t in result["matched_topics_flat"]],
        ensure_ascii=False, indent=2,
    )
    cached["raw_agent_gemini_status"] = result["agent_status"]
    cached["raw_agent_gemini_steps"]  = result["steps_used"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    returned_sector_ids = set(result["matched_sector_ids"])
    returned_topic_ids  = {t["topic_id"] for t in result["matched_topics_flat"]}

    sector_recall = _score_recall(returned_sector_ids, gt_sector_ids)
    topic_recall  = _score_recall(returned_topic_ids, gt_topic_ids)

    print(f"  Steps used:    {result['steps_used']}")
    print(f"  Sector recall: {sector_recall['recall']:.2f} ({sector_recall['hits']}/{sector_recall['total']})")
    print(f"  Topic recall:  {topic_recall['recall']:.2f} ({topic_recall['hits']}/{topic_recall['total']})")
    print(f"  Sectors matched: {result['matched_sector_ids']}")
    for t in result["matched_topics_flat"]:
        hit = "✅" if t["topic_id"] in gt_topic_ids else "  "
        print(f"    {hit} {t['topic_id']} | {t['topic_name']}")
        print(f"         {t['evidence'][:80]}")

    print("  Checking for missing taxonomy terms...")
    suggestions = suggest_missing_topics(
        company_id=company_id,
        company_name=company_name,
        structured_overview_text=raw_text[:5000],
        matched_topics=result["matched_topics_flat"],
    )
    cached["taxonomy_suggestions_raw_agent_gemini"] = suggestions
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    if suggestions:
        print(f"  → {len(suggestions)} new suggestion(s):")
        for s in suggestions:
            print(f"    [{s['confidence']}] {s['suggested_term']}")

print(f"\n{'='*65}")
print("DONE — Gemini raw agent pipeline complete")
print(f"{'='*65}")