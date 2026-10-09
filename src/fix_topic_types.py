# src/fix_topic_types.py
import json
import os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(_BASE, "data", "topic_types.json")

with open(PATH) as f:
    types_map = json.load(f)

fixes = {
    "5358": "category",   # Security Solutions — broad security category label
    "8364": "category",   # Solar Solutions — broader than Solar Panels
    "8747": "category",   # Banking Services — category label for banking identity
}

for tid, new_type in fixes.items():
    old = types_map.get(tid, "NOT FOUND")
    types_map[tid] = new_type
    print(f"Fixed {tid}: {old} -> {new_type}")

with open(PATH, "w") as f:
    json.dump(types_map, f, ensure_ascii=False, indent=2)

print("Done.")