# src/fix_topic_9241.py
import json
import os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- topic_sectors.json ---
# Container Solutions → Manufacturing (20) + Chemicals and Materials (6)
ts_path = os.path.join(_BASE, "data", "topic_sectors.json")
with open(ts_path) as f:
    ts = json.load(f)
ts["9241"] = [6, 20]
with open(ts_path, "w") as f:
    json.dump(ts, f, ensure_ascii=False, indent=2)
print("topic_sectors.json: added 9241 -> [6, 20]")

# --- topic_types.json ---
tt_path = os.path.join(_BASE, "data", "topic_types.json")
with open(tt_path) as f:
    tt = json.load(f)
tt["9241"] = "feature"
with open(tt_path, "w") as f:
    json.dump(tt, f, ensure_ascii=False, indent=2)
print("topic_types.json: added 9241 -> feature")

# --- topic_descriptions.json ---
td_path = os.path.join(_BASE, "data", "topic_descriptions.json")
with open(td_path) as f:
    td = json.load(f)
td["9241"] = "packaging containers for pharmaceutical products"
with open(td_path, "w") as f:
    json.dump(td, f, ensure_ascii=False, indent=2)
print("topic_descriptions.json: added 9241 description")

# --- topic_metadata.json ---
# No real count yet since it was just added — set to 0
tm_path = os.path.join(_BASE, "data", "topic_metadata.json")
with open(tm_path) as f:
    tm = json.load(f)
tm["9241"] = {
    "topic_name":           "Container Solutions",
    "target_company_count": 0,
    "target_intel_count":   0,
    "all_company_count":    0,
}
with open(tm_path, "w") as f:
    json.dump(tm, f, ensure_ascii=False, indent=2)
print("topic_metadata.json: added 9241 with count=0")

print("\nDone. Topic 9241 Container Solutions is now fully registered.")