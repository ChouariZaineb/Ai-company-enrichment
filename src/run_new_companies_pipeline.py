# src/run_new_companies_pipeline.py
"""
Full pipeline for the 7 new companies:
  1. Sector matching via Claude Bedrock (sector_matcher_bedrock.py)
  2. PE-format structuring (structuring_agent.py)
  3. Topic matching via Claude agent (topic_agent_weighted.py)
  4. Topic matching via Gemini Pro agent (topic_agent_weighted_gemini.py)
  5. Taxonomy gap suggestions (taxonomy_suggester.py)

Results saved to each company's cache file under new keys.
Run from project root: python src/run_new_companies_pipeline.py
"""
import json
import os
import sys
_SRC  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader import build_taxonomy
from sector_matcher_bedrock import match_sectors_to_taxonomy_bedrock
from structuring_agent import structure_company_overview, structured_overview_to_text
from topic_agent_weighted import run_topic_agent as run_claude_agent
from topic_agent_weighted_gemini import run_topic_agent as run_gemini_agent
from claude_bedrock_adapter import ClaudeBedrockLLMClient
from gemini_pro_adapter import GeminiProLLMClient
from taxonomy_suggester import suggest_missing_topics
from cache_utils import save_cached_result

_BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR  = os.path.join(_BASE, "data", "overviews_cache")
TAXONOMY   = build_taxonomy(
    sectors_path=os.path.join(_BASE, "data", "sectors_formatted.txt"),
    topics_path=os.path.join(_BASE, "data", "topics_formatted.txt"),
)

# Ground truth for scoring
GROUND_TRUTH = {
    14362: {"sector_ids": {7, 16, 21}, "topic_ids": {2282, 2984, 4204, 4541, 5751, 6897, 8640, 8791, 8793}},
    30336: {"sector_ids": {7},         "topic_ids": {722, 1859, 7980}},
    30344: {"sector_ids": {7, 16},     "topic_ids": {776, 1719, 1783, 1858, 1864}},
    30345: {"sector_ids": {20},        "topic_ids": {2828, 8399, 9241}},
    30346: {"sector_ids": {7, 19},     "topic_ids": {1956, 2282, 9025}},
    30350: {"sector_ids": {7, 17},     "topic_ids": {1719, 2931, 5477}},
    30352: {"sector_ids": {18, 20},    "topic_ids": {4777, 8136, 8206}},
}

COMPANIES = [
    ("14362_mckenzie_intelligence_services.json", "McKenzie Intelligence Services", 14362),
    ("30336_operations1.json",                    "Operations1",                    30336),
    ("30344_oxane_partners.json",                 "Oxane Partners",                 30344),
    ("30345_centor_rx.json",                      "Centor Rx",                      30345),
    ("30346_axel_health.json",                    "Axel Health",                    30346),
    ("30350_talent_funnel.json",                  "Talent Funnel",                  30350),
    ("30352_teknoice.json",                       "Teknoice",                       30352),
]

claude_client = ClaudeBedrockLLMClient()
gemini_client = GeminiProLLMClient()

MAX_SECTORS = 3


def _score_recall(predicted_ids: set, gt_ids: set) -> dict:
    if not gt_ids:
        return {"recall": 0.0, "hits": 0, "total": 0}
    hits   = len(predicted_ids & gt_ids)
    recall = hits / len(gt_ids)
    return {"recall": round(recall, 2), "hits": hits, "total": len(gt_ids)}


def _rich_input(cached: dict) -> str:
    business_summary = cached.get("business_summary", "")
    industry_tags    = cached.get("industry", [])
    products         = cached.get("products_offered", [])
    services         = cached.get("services_offered", [])
    usps             = cached.get("unique_selling_propositions", [])

    rich = business_summary
    if industry_tags:
        rich += f"\n\nIndustry tags: {', '.join(industry_tags)}"
    if products:
        rich += "\n\nProducts offered:\n" + "\n".join(f"- {p}" for p in products)
    if services:
        rich += "\n\nServices offered:\n" + "\n".join(f"- {s}" for s in services)
    if usps:
        rich += "\n\nKey differentiators:\n" + "\n".join(f"- {u}" for u in usps)
    return rich


for filename, company_name, company_id in COMPANIES:
    path = os.path.join(CACHE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        cached = json.load(f)

    gt = GROUND_TRUTH.get(company_id, {})
    gt_sector_ids = gt.get("sector_ids", set())
    gt_topic_ids  = gt.get("topic_ids", set())

    print(f"\n{'='*65}")
    print(f"COMPANY: {company_name}  (id={company_id})")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # STEP 1: Sector matching (skip if already done)
    # ------------------------------------------------------------------
    if not cached.get("matched_sector_ids_bedrock"):
        print("  [1/4] Running sector matching...")
        sector_result = match_sectors_to_taxonomy_bedrock(
            company_name=company_name,
            business_summary=cached.get("business_summary", ""),
            generated_sectors=cached.get("sectors", []),
            sectors_prompt_text=TAXONOMY["sectors_prompt_text"],
            valid_sector_ids=TAXONOMY["valid_sector_ids"],
        )
        # cap at MAX_SECTORS
        sector_ids = sector_result.get("matched_sector_ids", [])[:MAX_SECTORS]
        cached["matched_sector_ids_bedrock"]          = sector_ids
        cached["sector_matching_reasoning_bedrock"]   = sector_result.get("reasoning", "")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cached, f, ensure_ascii=False, indent=2)
        print(f"     → matched sectors: {sector_ids}")
    else:
        sector_ids = cached["matched_sector_ids_bedrock"][:MAX_SECTORS]
        print(f"  [1/4] Sectors already matched: {sector_ids}")

    sector_recall = _score_recall(set(sector_ids), gt_sector_ids)
    print(f"     sector recall: {sector_recall['recall']:.2f} ({sector_recall['hits']}/{sector_recall['total']})")

    # ------------------------------------------------------------------
    # STEP 2: PE structuring
    # ------------------------------------------------------------------
    print("  [2/4] Structuring PE overview...")
    rich_input    = _rich_input(cached)
    structured    = structure_company_overview(company_name, rich_input)
    overview_text = structured_overview_to_text(structured)
    cached["structured_overview_new"] = structured

    # ------------------------------------------------------------------
    # STEP 3: Claude agent topic matching
    # ------------------------------------------------------------------
    print("  [3/4] Claude agent topic matching...")
    claude_result = run_claude_agent(company_name, overview_text, sector_ids, claude_client)

    cached["matched_sector_ids_agent_new_claude"]      = sector_ids
    cached["matched_topics_flat_agent_new_claude"]     = claude_result["matched_topics_flat"]
    cached["topic_reasoning_agent_new_claude"]         = json.dumps(
        [{"topic_id": t["topic_id"], "topic_name": t["topic_name"], "evidence": t["evidence"]}
         for t in claude_result["matched_topics_flat"]],
        ensure_ascii=False, indent=2,
    )
    cached["agent_new_claude_status"] = claude_result["agent_status"]

    claude_returned = {t["topic_id"] for t in claude_result["matched_topics_flat"]}
    claude_recall   = _score_recall(claude_returned, gt_topic_ids)
    print(f"     Claude recall: {claude_recall['recall']:.2f} ({claude_recall['hits']}/{claude_recall['total']})")
    for t in claude_result["matched_topics_flat"]:
        hit = "✅" if t["topic_id"] in gt_topic_ids else "  "
        print(f"       {hit} {t['topic_id']} | {t['topic_name']}")

    # ------------------------------------------------------------------
    # STEP 4: Gemini agent topic matching
    # ------------------------------------------------------------------
    print("  [4/4] Gemini Pro agent topic matching...")
    gemini_result = run_gemini_agent(company_name, overview_text, sector_ids, gemini_client)

    cached["matched_sector_ids_agent_new_gemini"]      = sector_ids
    cached["matched_topics_flat_agent_new_gemini"]     = gemini_result["matched_topics_flat"]
    cached["topic_reasoning_agent_new_gemini"]         = json.dumps(
        [{"topic_id": t["topic_id"], "topic_name": t["topic_name"], "evidence": t["evidence"]}
         for t in gemini_result["matched_topics_flat"]],
        ensure_ascii=False, indent=2,
    )
    cached["agent_new_gemini_status"] = gemini_result["agent_status"]

    gemini_returned = {t["topic_id"] for t in gemini_result["matched_topics_flat"]}
    gemini_recall   = _score_recall(gemini_returned, gt_topic_ids)
    print(f"     Gemini recall: {gemini_recall['recall']:.2f} ({gemini_recall['hits']}/{gemini_recall['total']})")
    for t in gemini_result["matched_topics_flat"]:
        hit = "✅" if t["topic_id"] in gt_topic_ids else "  "
        print(f"       {hit} {t['topic_id']} | {t['topic_name']}")



    # ------------------------------------------------------------------
    # STEP 5: Taxonomy gap suggestions (using Claude result as base)
    # ------------------------------------------------------------------
    print("  [5/5] Checking for missing taxonomy terms...")
    suggestions = suggest_missing_topics(
        company_id=company_id,
        company_name=company_name,
        structured_overview_text=overview_text,
        matched_topics=claude_result["matched_topics_flat"],
    )
    cached["taxonomy_suggestions"] = suggestions
    if suggestions:
        print(f"     → {len(suggestions)} suggestion(s):")
        for s in suggestions:
            print(f"       [{s['confidence']}] {s['suggested_term']} — {s['evidence'][:80]}")
    else:
        print("     → no gaps found")

    # save final cache
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    print(f"\n  SUMMARY — {company_name}:")
    print(f"    Sector recall : {sector_recall['recall']:.2f}")
    print(f"    Claude recall : {claude_recall['recall']:.2f}")
    print(f"    Gemini recall : {gemini_recall['recall']:.2f}")

print(f"\n\n{'='*65}")
print("ALL COMPANIES DONE")
print(f"Taxonomy suggestions logged to: data/taxonomy_suggestions.json")
print(f"{'='*65}")

