# src/run_multi_agent_pipeline.py
"""
Multi-agent pipeline orchestrator.

Agent 1 (Researcher)        — Claude Sonnet 5, tool-loop, browses + searches
Agent 2 (Sector Classifier) — Claude Sonnet 5, single call
Agent 3 (Topic Specialist)  — Claude Sonnet 5, theme-based two-stage matching

Input:  company_name + source URL from cache files
Output: profile + sectors + topics + taxonomy suggestions → saved to cache

Run from project root: python src/run_multi_agent_pipeline.py
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent1_researcher     import run_researcher_agent
from agent2_sector_classifier import run_sector_classifier
from agent3_topic_specialist  import run_topic_specialist
from claude_sonnet5_adapter   import ClaudeSonnet5LLMClient

_BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_BASE, "data", "overviews_cache")

FORCE_RESCRAPE = False   # re-run Agent 1 (research) even if a cached profile exists
FORCE_RESECTOR = False   # re-run Agent 2 (sectors) even if cached sector_ids exist

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


def _recall(predicted: set, gt: set) -> dict:
    if not gt:
        return {"recall": 0.0, "hits": 0, "total": 0}
    hits = len(predicted & gt)
    return {"recall": round(hits / len(gt), 2), "hits": hits, "total": len(gt)}


client = ClaudeSonnet5LLMClient()

total_sector_recall = []
total_topic_recall  = []

for company_id, company_name, filename in COMPANIES:
    path = os.path.join(CACHE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        cached = json.load(f)

    source_url = cached.get("source", "")
    if not source_url:
        print(f"\n[SKIP] {company_name} — no source URL in cache.")
        continue

    gt            = GROUND_TRUTH.get(company_id, {})
    gt_sector_ids = gt.get("sector_ids", set())
    gt_topic_ids  = gt.get("topic_ids", set())

    print(f"\n{'='*65}")
    print(f"COMPANY: {company_name}  (id={company_id})")
    print(f"URL: {source_url}")
    print(f"{'='*65}")

    # ── Agent 1: Research (reuse cache unless forced) ──────────────────
    cached_research = cached.get("agent1_research_profile")
    if cached_research and cached_research.get("full_profile") and not FORCE_RESCRAPE:
        print(f"\n  [Agent 1] Using cached profile for {company_name} (skipped re-scraping)")
        research = cached_research
    else:
        print(f"\n  [Agent 1] Researching {company_name}...")
        research = run_researcher_agent(company_name, source_url, client)
        print(f"  → Status: {research['agent_status']} | Steps: {research['steps_used']}")
        print(f"  → Sources visited: {research.get('sources_visited', [])}")

    company_profile = research.get("full_profile", "")
    if not company_profile:
        print(f"  [WARN] Empty profile — skipping sector + topic matching.")
        continue

    # ── Agent 2: Sector Classification (reuse cache unless forced) ─────
    cached_sector_ids       = cached.get("matched_sector_ids_multi_agent")
    cached_sector_reasoning = cached.get("sector_reasoning_multi_agent", "")
    if cached_sector_ids and not FORCE_RESECTOR:
        print(f"\n  [Agent 2] Using cached sectors for {company_name} (skipped re-classification)")
        sector_ids            = cached_sector_ids
        sector_reasoning_text = cached_sector_reasoning
    else:
        print(f"\n  [Agent 2] Classifying sectors...")
        sector_result          = run_sector_classifier(company_name, company_profile)
        sector_ids             = sector_result["matched_sector_ids"]
        sector_reasoning_text  = sector_result["sector_reasoning"]
        print(f"  → Sectors: {sector_ids}")
        print(f"  → Reasoning: {sector_reasoning_text[:150]}...")

    sector_recall = _recall(set(sector_ids), gt_sector_ids)
    print(f"  → Sector recall: {sector_recall['recall']:.2f} ({sector_recall['hits']}/{sector_recall['total']})")

    # ── Agent 3: Topic Matching (always runs) ───────────────────────────
    print(f"\n  [Agent 3] Matching topics...")
    topic_result = run_topic_specialist(
        company_id=company_id,
        company_name=company_name,
        company_profile=company_profile,
        sector_ids=sector_ids,
    )

    returned_topic_ids = {t["topic_id"] for t in topic_result["matched_topics_flat"]}
    topic_recall        = _recall(returned_topic_ids, gt_topic_ids)
    print(f"  → Topic recall: {topic_recall['recall']:.2f} ({topic_recall['hits']}/{topic_recall['total']})")

    # ── Diagnostic: was a miss never found, or found then cut in narrowing? ─
    pre_finalize_ids = set(topic_result.get("pre_finalize_topic_ids", []))
    pre_finalize_recall = _recall(pre_finalize_ids, gt_topic_ids)
    lost_in_narrowing = (pre_finalize_ids & gt_topic_ids) - returned_topic_ids
    never_found       = gt_topic_ids - pre_finalize_ids
    print(f"  → DIAGNOSTIC: {pre_finalize_recall['hits']}/{pre_finalize_recall['total']} ground-truth topics "
          f"were found during scanning (before narrowing to {len(returned_topic_ids)} final)")
    if lost_in_narrowing:
        print(f"    -> LOST DURING NARROWING (found, then cut): {sorted(lost_in_narrowing)}")
    if never_found:
        print(f"    -> NEVER FOUND (missed during theme scanning): {sorted(never_found)}")

    for t in topic_result["matched_topics_flat"]:
        hit = "✅" if t["topic_id"] in gt_topic_ids else "  "
        print(f"    {hit} {t['topic_id']} | {t['topic_name']}")
        ev = t.get("evidence", "")[:80]
        print(f"         {ev}")

    if topic_result["taxonomy_suggestions"]:
        print(f"\n  → {len(topic_result['taxonomy_suggestions'])} taxonomy suggestion(s):")
        for s in topic_result["taxonomy_suggestions"]:
            print(f"    [{s['confidence']}] {s['suggested_term']}")

    # ── Save to cache ──────────────────────────────────────────────────
    cached["agent1_research_profile"]           = research
    cached["matched_sector_ids_multi_agent"]    = sector_ids
    cached["sector_reasoning_multi_agent"]      = sector_reasoning_text
    cached["matched_topics_flat_multi_agent"]   = topic_result["matched_topics_flat"]
    cached["topic_reasoning_multi_agent"]       = json.dumps(
        topic_result["topic_reasoning"], ensure_ascii=False, indent=2
    )
    cached["taxonomy_suggestions_multi_agent"]  = topic_result["taxonomy_suggestions"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    total_sector_recall.append(sector_recall["recall"])
    total_topic_recall.append(topic_recall["recall"])

    print(f"\n  SUMMARY — {company_name}:")
    print(f"    Sector recall: {sector_recall['recall']:.2f}")
    print(f"    Topic recall:  {topic_recall['recall']:.2f}")

# ── Overall summary ────────────────────────────────────────────────────
if total_sector_recall:
    avg_s = round(sum(total_sector_recall) / len(total_sector_recall), 2)
    avg_t = round(sum(total_topic_recall)  / len(total_topic_recall),  2)
    avg_o = round((avg_s + avg_t) / 2, 2)
    print(f"\n{'='*65}")
    print(f"OVERALL AVERAGES ({len(total_sector_recall)} companies):")
    print(f"  Sector recall: {avg_s}")
    print(f"  Topic recall:  {avg_t}")
    print(f"  Overall recall:{avg_o}")
    print(f"{'='*65}")