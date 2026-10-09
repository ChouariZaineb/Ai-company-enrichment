# src/test_structuring_agent.py
"""
Smoke test: runs structuring_agent on the 3 ground-truth companies
using their existing cached business_summary.
Run from project root: python src/test_structuring_agent.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from structuring_agent import structure_company_overview, structured_overview_to_text

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "overviews_cache")

COMPANIES = [
    ("1002_monzo.json",         "Monzo"),
    ("1004_darktrace.json",     "Darktrace"),
    ("1006_octopus_energy.json","Octopus Energy"),
]

for filename, company_name in COMPANIES:
    path = os.path.join(CACHE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        cached = json.load(f)

    business_summary = cached.get("business_summary", "")
    if not business_summary:
        print(f"\n[ERROR] No business_summary found for {company_name}")
        continue

    print(f"\n{'='*60}")
    print(f"COMPANY: {company_name}")
    print(f"MATCHED SECTOR IDS (bedrock): {cached.get('matched_sector_ids_bedrock', cached.get('matched_sector_ids', []))}")
    print(f"{'='*60}")

    structured = structure_company_overview(company_name, business_summary)
    print(structured_overview_to_text(structured))