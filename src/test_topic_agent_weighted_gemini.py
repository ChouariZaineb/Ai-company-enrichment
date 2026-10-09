# src/test_topic_agent_weighted_gemini.py
"""
Runs the weighted topic agent using Gemini Pro.
Uses matched_sector_ids_gemini_pro from each company's cache.
Saves results under *_agent_weighted_gemini keys.
Run from project root: python src/test_topic_agent_weighted_gemini.py
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from structuring_agent import structure_company_overview, structured_overview_to_text
from topic_agent_weighted_gemini import run_topic_agent
from gemini_pro_adapter import GeminiProLLMClient
from topic_agent_weighted_gemini import run_topic_agent

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "overviews_cache")

COMPANIES = [
    ("1002_monzo.json",          "Monzo"),
    ("1004_darktrace.json",      "Darktrace"),
    ("1006_octopus_energy.json", "Octopus Energy"),
    ("1010_babylon_health.json", "Babylon Health"),
]

GROUND_TRUTH = {
    "Monzo":          {1870, 8619, 8747, 1465, 722},
    "Darktrace":      {5358, 8640, 2777, 1719, 1661},
    "Octopus Energy": {1619, 7268, 4110, 8364, 3137, 2475, 7335, 5163, 2474, 2054},
    "Babylon Health": {2978, 6099, 6261, 5787},
}

client = GeminiProLLMClient()

for filename, company_name in COMPANIES:
    path = os.path.join(CACHE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        cached = json.load(f)

    business_summary = cached.get("business_summary", "")
    industry_tags    = cached.get("industry", [])
    products         = cached.get("products_offered", [])
    services         = cached.get("services_offered", [])
    usps             = cached.get("unique_selling_propositions", [])

    rich_input = business_summary
    if industry_tags:
        rich_input += f"\n\nIndustry tags: {', '.join(industry_tags)}"
    if products:
        rich_input += "\n\nProducts offered:\n" + "\n".join(f"- {p}" for p in products)
    if services:
        rich_input += "\n\nServices offered:\n" + "\n".join(f"- {s}" for s in services)
    if usps:
        rich_input += "\n\nKey differentiators:\n" + "\n".join(f"- {u}" for u in usps)

    sector_ids = cached.get("matched_sector_ids_bedrock", [])
    if not sector_ids:
        print(f"[WARN] {company_name}: no matched_sector_ids_bedrock found — skipping or falling back may give wrong results.")

    structured    = structure_company_overview(company_name, rich_input)
    overview_text = structured_overview_to_text(structured)

    result = run_topic_agent(company_name, overview_text, sector_ids, client)

    cached["structured_overview_agent_weighted_gemini"]      = structured
    cached["matched_sector_ids_agent_weighted_gemini"]       = sector_ids
    cached["matched_topics_flat_agent_weighted_gemini"]      = result["matched_topics_flat"]
    cached["topic_matching_reasoning_agent_weighted_gemini"] = json.dumps(
        [{"topic_id": t["topic_id"], "topic_name": t["topic_name"], "evidence": t["evidence"]}
         for t in result["matched_topics_flat"]],
        ensure_ascii=False, indent=2,
    )
    cached["agent_weighted_gemini_status"] = result["agent_status"]
    cached["agent_weighted_gemini_steps"]  = result["steps_used"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    returned_ids = {t["topic_id"] for t in result["matched_topics_flat"]}
    gt_ids       = GROUND_TRUTH[company_name]
    hits         = returned_ids & gt_ids
    recall       = len(hits) / len(gt_ids) if gt_ids else 0

    print(f"\n{'='*60}")
    print(f"COMPANY: {company_name}  |  status={result['agent_status']}  steps={result['steps_used']}")
    print(f"{'='*60}")
    for t in result["matched_topics_flat"]:
        hit = "✅" if t["topic_id"] in gt_ids else "  "
        print(f"  {hit} {t['topic_id']} | {t['topic_name']}")
        print(f"       evidence: {t['evidence']}")
    print(f"\n  Recall: {recall:.2f}  |  Hits: {len(hits)}/{len(gt_ids)}")
    print(f"  Returned: {len(returned_ids)} topics total")