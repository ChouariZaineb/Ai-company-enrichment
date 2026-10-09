"""
check_topic_reachability.py

Diagnostic: for each of the 7 new companies, checks whether the ground-truth
topic IDs are actually present in the sector-scoped candidate pool that
Agent 3 sees (i.e. reachable via topic_sectors.json under the sector_ids
that Agent 2 actually selected for that company).

This tells us whether the topic-recall misses are a RETRIEVAL problem
(the correct topic was never shown to Agent 3 at all) or a SELECTION
problem (the topic was in the pool but the model didn't pick it) —
which determines whether pool restructuring / prompt changes can help.

Run from the project root:
    python check_topic_reachability.py
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

TOPIC_SECTORS_PATH = os.path.join(BASE, "data", "topic_sectors.json")
CACHE_DIR = os.path.join(BASE, "data", "overviews_cache")

GROUND_TRUTH_BY_COMPANY_ID = {
    14362: {"sector_ids": {7, 16, 21}, "topic_ids_flat": {2282, 2984, 4204, 4541, 5751, 6897, 8640, 8791, 8793}},
    30336: {"sector_ids": {7},          "topic_ids_flat": {722, 1859, 7980}},
    30344: {"sector_ids": {7, 16},      "topic_ids_flat": {776, 1719, 1783, 1858, 1864}},
    30345: {"sector_ids": {20},         "topic_ids_flat": {2828, 8399, 9241}},
    30346: {"sector_ids": {7, 19},      "topic_ids_flat": {1956, 2282, 9025}},
    30350: {"sector_ids": {7, 17},      "topic_ids_flat": {1719, 2931, 5477}},
    30352: {"sector_ids": {18, 20},     "topic_ids_flat": {4777, 8136, 8206}},
}

COMPANY_FILES = {
    14362: "14362_mckenzie_intelligence_services.json",
    30336: "30336_operations1.json",
    30344: "30344_oxane_partners.json",
    30345: "30345_centor_rx.json",
    30346: "30346_axel_health.json",
    30350: "30350_talent_funnel.json",
    30352: "30352_teknoice.json",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    topic_sectors = load_json(TOPIC_SECTORS_PATH)  # topic_id(str) -> [sector_ids]

    print(f"{'Company':30} {'Agent2 sectors':18} {'GT topics':>9} {'Reachable':>9} {'Selected':>8} {'Hit':>5}")
    print("-" * 95)

    for company_id, filename in COMPANY_FILES.items():
        cache_path = os.path.join(CACHE_DIR, filename)
        if not os.path.exists(cache_path):
            print(f"{company_id:<30} MISSING CACHE FILE {filename}")
            continue

        cache = load_json(cache_path)
        company_name = cache.get("company_name", str(company_id))
        agent2_sectors = set(cache.get("matched_sector_ids_multi_agent", []))
        agent3_selected = {
            t["topic_id"] if isinstance(t, dict) else t
            for t in cache.get("matched_topics_flat_multi_agent", [])
        }

        gt = GROUND_TRUTH_BY_COMPANY_ID.get(company_id, {})
        gt_topics = gt.get("topic_ids_flat", set())

        reachable, unreachable = set(), set()
        for tid in gt_topics:
            sids = set(topic_sectors.get(str(tid), []))
            (reachable if sids & agent2_sectors else unreachable).add(tid)

        hit = gt_topics & agent3_selected

        print(f"{company_name:<30} {str(sorted(agent2_sectors)):18} "
              f"{len(gt_topics):>9} {len(reachable):>9} {len(agent3_selected):>8} {len(hit):>5}")

        if unreachable:
            print(f"    UNREACHABLE (not in scoped pool at all): {sorted(unreachable)}")
            for tid in sorted(unreachable):
                print(f"      topic {tid}: actually mapped to sectors {topic_sectors.get(str(tid), [])} "
                      f"(agent2 picked {sorted(agent2_sectors)})")

        missed_but_reachable = (gt_topics - hit) & reachable
        if missed_but_reachable:
            print(f"    MISSED despite being reachable (selection problem): {sorted(missed_but_reachable)}")

    print()
    print("Interpretation:")
    print("- UNREACHABLE topics were lost before Agent 3 ever ran. No prompt or pool")
    print("  restructuring fixes this — it means topic_sectors.json's mapping for that")
    print("  topic doesn't include a sector Agent 2 picked, or Agent 2 missed a sector.")
    print("- MISSED-but-reachable topics were shown to Agent 3 and it still didn't pick")
    print("  them. That's where prompt changes (evidence requirement, pool structure,")
    print("  ensembling) can actually move the needle.")


if __name__ == "__main__":
    main()