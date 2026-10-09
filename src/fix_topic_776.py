# src/fix_topic_776.py
import json, os
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path  = os.path.join(_BASE, "data", "topic_sectors.json")

with open(path) as f:
    ts = json.load(f)

# Servicing (776) applies to financial servicing too — add sector 16 and 17
ts["776"] = [3, 16, 17, 33]
print(f"Updated 776 | Servicing -> {ts['776']}")

with open(path, "w") as f:
    json.dump(ts, f, ensure_ascii=False, indent=2)
print("Done.")