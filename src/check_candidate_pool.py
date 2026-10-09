# src/check_candidate_pool.py
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from loader import load_topics

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE, "data", "topic_sectors.json")) as f:
    ts = json.load(f)

all_topics = {
    t.topic_id: t.topic_name
    for t in load_topics(os.path.join(BASE, "data", "topics_formatted.txt"))
}

ground_truth = {
    "Monzo": {
        "sector_ids": [7, 16],
        "topic_ids": [1870, 8619, 8747, 1465, 722],
    },
    "Darktrace": {
        "sector_ids": [4, 7],
        "topic_ids": [5358, 8640, 2777, 1719, 1661],
    },
    "Octopus Energy": {
        "sector_ids": [7, 11, 30],
        "topic_ids": [1619, 7268, 4110, 8364, 3137, 2475, 7335, 5163, 2474, 2054],
    },
}

for company, data in ground_truth.items():
    sector_set = set(data["sector_ids"])
    print(f"\n=== {company} (sectors {data['sector_ids']}) ===")
    for tid in data["topic_ids"]:
        tagged_sectors = ts.get(str(tid), [])
        in_pool = bool(sector_set & set(tagged_sectors))
        name = all_topics.get(tid, "???")
        status = "IN POOL  " if in_pool else "MISSING  "
        print(f"  {status} | {tid} | {name} | tagged sectors: {tagged_sectors}")