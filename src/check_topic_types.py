# src/check_topic_types.py
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from loader import load_topics

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(_BASE, "data", "topic_types.json")) as f:
    types_map = json.load(f)

all_topics = {t.topic_id: t.topic_name for t in load_topics(os.path.join(_BASE, "data", "topics_formatted.txt"))}

# topics we care about most
check_ids = [
    # Darktrace ground truth
    5358, 8640, 2777, 1719, 1661,
    # Darktrace agent picks (sub-topics we want eliminated)
    3496, 4622, 8936, 2908, 2973, 2600,
    # Octopus ground truth
    1619, 7268, 4110, 8364, 3137, 2475, 7335, 5163, 2474, 2054,
    # Octopus agent picks (we want eliminated)
    1692, 4797, 1737, 490,
    # Monzo ground truth
    1870, 8619, 8747, 1465, 722,
    # Monzo agent picks (we want eliminated)
    6027, 6120, 2193, 8797,
]

print(f"{'ID':<8} {'Type':<10} {'Name'}")
print("-" * 50)
for tid in check_ids:
    t = types_map.get(str(tid), "NOT FOUND")
    name = all_topics.get(tid, "???")
    print(f"{tid:<8} {t:<10} {name}")