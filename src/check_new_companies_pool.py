# src/check_new_companies_pool.py
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topic_agent_weighted import _TOPIC_SECTORS, _TOPIC_TYPES, _ALL_TOPICS

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROUND_TRUTH = {
    14362: {"sectors": [7, 17, 21], "topics": {2282, 2984, 4204, 4541, 5751, 6897, 8640, 8791, 8793}},
    30336: {"sectors": [7, 20, 41], "topics": {722, 1859, 7980}},
    30344: {"sectors": [7, 16],     "topics": {776, 1719, 1783, 1858, 1864}},
    30345: {"sectors": [6, 7, 20],  "topics": {2828, 8399, 9241}},
    30346: {"sectors": [7, 19],     "topics": {1956, 2282, 9025}},
    30350: {"sectors": [7, 17],     "topics": {1719, 2931, 5477}},
    30352: {"sectors": [13, 20, 41],"topics": {4777, 8136, 8206}},
}

NAMES = {
    14362: "McKenzie", 30336: "Operations1", 30344: "Oxane",
    30345: "Centor Rx", 30346: "Axel Health", 30350: "Talent Funnel", 30352: "Teknoice"
}

for cid, data in GROUND_TRUTH.items():
    sector_set = set(data["sectors"])
    print(f"\n=== {NAMES[cid]} (sectors={data['sectors']}) ===")
    for tid in sorted(data["topics"]):
        tagged = _TOPIC_SECTORS.get(str(tid), [])
        in_pool = bool(sector_set & set(tagged))
        name = _ALL_TOPICS.get(tid, "???")
        ttype = _TOPIC_TYPES.get(str(tid), "???")
        print(f"  {'IN POOL' if in_pool else 'MISSING '} | {tid} | {name} [{ttype}] | tagged sectors: {tagged}")