"""
check_pool_size_and_position.py

For each of the 7 companies: how many candidate topics are in Agent 3's pool
(given the sectors Agent 2 picked), and where do the ground-truth topics sit
within that pool once it's sorted the same way Agent 3 sees it (category-type
first, then by specificity score)?

If a missed topic sits near the top of a small pool, pool size isn't the
problem. If it's buried deep in a large pool, that supports splitting pools
into smaller groups.

Run from the project root:
    python check_pool_size_and_position.py
"""
import json
import os
import sys
import math

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))

from loader import load_topics  # reuse the project's own topic-file parser

TOPIC_SECTORS_PATH  = os.path.join(BASE, "data", "topic_sectors.json")
TOPIC_TYPES_PATH    = os.path.join(BASE, "data", "topic_types.json")
TOPIC_METADATA_PATH = os.path.join(BASE, "data", "topic_metadata.json")
TOPICS_PATH         = os.path.join(BASE, "data", "topics_formatted.txt")
CACHE_DIR           = os.path.join(BASE, "data", "overviews_cache")

GROUND_TRUTH = {
    14362: {2282, 2984, 4204, 4541, 5751, 6897, 8640, 8791, 8793},
    30336: {722, 1859, 7980},
    30344: {776, 1719, 1783, 1858, 1864},
    30345: {2828, 8399, 9241},
    30346: {1956, 2282, 9025},
    30350: {1719, 2931, 5477},
    30352: {4777, 8136, 8206},
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


def specificity_score(name, count):
    return len(name.strip().split()) * math.log(1 + count)


def main():
    topic_sectors  = load_json(TOPIC_SECTORS_PATH)
    topic_types    = load_json(TOPIC_TYPES_PATH)
    topic_metadata = load_json(TOPIC_METADATA_PATH) if os.path.exists(TOPIC_METADATA_PATH) else {}
    topic_names    = {t.topic_id: t.topic_name for t in load_topics(TOPICS_PATH)}

    print(f"[DEBUG] Loaded {len(topic_names)} topic names, {len(topic_sectors)} topic-sector mappings\n")

    for company_id, filename in COMPANY_FILES.items():
        cache_path = os.path.join(CACHE_DIR, filename)
        if not os.path.exists(cache_path):
            print(f"[SKIP] {filename} not found")
            continue
        cache = load_json(cache_path)
        company_name   = cache.get("company_name", str(company_id))
        agent2_sectors = set(cache.get("matched_sector_ids_multi_agent", []))

        # Rebuild the exact pool Agent 3 sees, same sort as _get_scoped_pool()
        pool = []
        for tid_str, sids in topic_sectors.items():
            if not (set(sids) & agent2_sectors):
                continue
            tid = int(tid_str)
            if tid not in topic_names:
                continue
            ttype = topic_types.get(tid_str, "feature")
            meta  = topic_metadata.get(tid_str, {})
            count = meta.get("target_company_count") or meta.get("all_company_count", 0)
            score = specificity_score(topic_names[tid], count)
            pool.append((tid, topic_names[tid], ttype, score))
        pool.sort(key=lambda x: (0 if x[2] == "category" else 1, -x[3]))

        print(f"\n{company_name} — sectors {sorted(agent2_sectors)} — pool size: {len(pool)} topics")

        gt = GROUND_TRUTH.get(company_id, set())
        for tid in sorted(gt):
            position = next((i for i, p in enumerate(pool) if p[0] == tid), None)
            if position is None:
                print(f"  topic {tid}: NOT IN POOL")
            else:
                name = pool[position][1]
                pct = round(100 * (position + 1) / len(pool))
                print(f"  topic {tid} ({name}): position {position + 1} of {len(pool)}  (~{pct}% of the way down the list)")


if __name__ == "__main__":
    main()