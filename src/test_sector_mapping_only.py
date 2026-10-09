# src/test_sector_mapping_only.py
"""
Shows all topics mapped to each company's matched sectors — before any
agent selection. This is the raw candidate pool the agent chooses from.
Saves results to data/sector_mapping_results.json for demo purposes.
Run from project root: python src/test_sector_mapping_only.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from topic_agent import _TOPIC_SECTORS, _TOPIC_TYPES, _TOPIC_DESCRIPTIONS, _ALL_TOPICS

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR   = os.path.join(_BASE, "data", "overviews_cache")
OUTPUT_PATH = os.path.join(_BASE, "data", "sector_mapping_results.json")

COMPANIES = [
    ("1002_monzo.json",          "Monzo"),
    ("1004_darktrace.json",      "Darktrace"),
    ("1006_octopus_energy.json", "Octopus Energy"),
]

GROUND_TRUTH = {
    "Monzo":         {1870, 8619, 8747, 1465, 722},
    "Darktrace":     {5358, 8640, 2777, 1719, 1661},
    "Octopus Energy":{1619, 7268, 4110, 8364, 3137, 2475, 7335, 5163, 2474, 2054},
}

output = {}

for filename, company_name in COMPANIES:
    path = os.path.join(CACHE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        cached = json.load(f)

    sector_ids = cached.get(
        "matched_sector_ids_bedrock",
        cached.get("matched_sector_ids", [])
    )

    sector_set = set(sector_ids)

    all_mapped = []
    for tid_str, sids in _TOPIC_SECTORS.items():
        if not (sector_set & set(sids)):
            continue
        tid_int = int(tid_str)
        if tid_int not in _ALL_TOPICS:
            continue
        ttype = _TOPIC_TYPES.get(tid_str, "feature")
        desc  = _TOPIC_DESCRIPTIONS.get(tid_str, "")
        all_mapped.append({
            "topic_id":        tid_int,
            "topic_name":      _ALL_TOPICS[tid_int],
            "type":            ttype,
            "description":     desc,
            "is_ground_truth": tid_int in GROUND_TRUTH[company_name],
        })

    all_mapped.sort(key=lambda x: (0 if x["type"] == "category" else 1, x["topic_id"]))

    gt_ids     = GROUND_TRUTH[company_name]
    mapped_ids = {t["topic_id"] for t in all_mapped}
    gt_in_pool = gt_ids & mapped_ids
    gt_missing = gt_ids - mapped_ids

    categories = [t for t in all_mapped if t["type"] == "category"]
    features   = [t for t in all_mapped if t["type"] == "feature"]

    print(f"\n{'='*60}")
    print(f"COMPANY: {company_name}")
    print(f"Matched sectors (bedrock): {sector_ids}")
    print(f"Total topics in pool: {len(all_mapped)}")
    print(f"  -> Categories: {len(categories)}")
    print(f"  -> Features:   {len(features)}")
    print(f"Ground truth in pool: {len(gt_in_pool)}/{len(gt_ids)}")
    if gt_missing:
        print(f"Ground truth NOT in pool: {gt_missing}")
    print(f"\nGround truth topics:")
    for tid in sorted(gt_ids):
        in_pool = tid in mapped_ids
        name    = _ALL_TOPICS.get(tid, "???")
        ttype   = _TOPIC_TYPES.get(str(tid), "???")
        print(f"  {'IN POOL' if in_pool else 'MISSING'} | {tid} | {name} [{ttype}]")

    output[company_name] = {
        "matched_sector_ids":       sector_ids,
        "total_topics_in_pool":     len(all_mapped),
        "total_categories":         len(categories),
        "total_features":           len(features),
        "ground_truth_in_pool":     len(gt_in_pool),
        "ground_truth_missing":     list(gt_missing),
        "all_mapped_topics":        all_mapped,
    }

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nFull results saved to data/sector_mapping_results.json")