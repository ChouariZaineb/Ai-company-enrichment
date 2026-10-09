# src/build_topic_metadata.py
"""
Parses the two supervisor-provided topic-popularity CSVs into a single JSON
lookup file: data/topic_metadata.json

Input files (place directly in data/, unmodified):
  - data/topics attached to target companies.csv
      columns: actum_industry_meta_id | actum_industry_meta_name |
               attached_company_count | attached_intel_count
  - data/topics attached to companies (all companies and advisor companies).csv
      columns: actum_industry_meta_id | actum_industry_meta_name |
               attached_company_count

NOTE: the header row in these files is comma-separated but the data rows
are pipe-separated. Column names are hardcoded below instead of relying
on the header, and the header line is simply skipped.

Output: data/topic_metadata.json
  { "<topic_id>": {
        "topic_name": str,
        "target_company_count": int,
        "target_intel_count": int,
        "all_company_count": int,
    }, ... }

Run from project root: python src/build_topic_metadata.py
"""
import json
import os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_BASE, "data")

TARGET_CSV  = os.path.join(DATA_DIR, "topics attached to target companies.csv")
ALL_CSV     = os.path.join(DATA_DIR, "topics attached to companies (all companies and advisor companies).csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "topic_metadata.json")

TARGET_FIELDS = ["actum_industry_meta_id", "actum_industry_meta_name",
                  "attached_company_count", "attached_intel_count"]
ALL_FIELDS    = ["actum_industry_meta_id", "actum_industry_meta_name",
                  "attached_company_count"]


def _read_pipe_csv(path: str, fieldnames: list[str]) -> list[dict]:
    """
    Reads a file whose header row may not match the data delimiter.
    Skips line 1 unconditionally and splits every subsequent line on '|',
    zipping values against the known fieldnames.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    rows = []
    for raw_line in lines[1:]:  # skip header line
        line = raw_line.strip()
        if not line:
            continue
        values = line.split("|")
        if len(values) != len(fieldnames):
            print(f"[WARN] skipping malformed line in {os.path.basename(path)}: {line!r}")
            continue
        rows.append(dict(zip(fieldnames, values)))
    return rows


def build_topic_metadata() -> dict:
    metadata: dict[str, dict] = {}

    # --- target companies file (primary weighting signal) ---
    for row in _read_pipe_csv(TARGET_CSV, TARGET_FIELDS):
        tid = str(int(row["actum_industry_meta_id"].strip()))
        metadata[tid] = {
            "topic_name":           row["actum_industry_meta_name"].strip(),
            "target_company_count": int(row["attached_company_count"].strip()),
            "target_intel_count":   int(row["attached_intel_count"].strip()),
            "all_company_count":    0,
        }

    # --- all companies + advisors file (fallback for topics missing above) ---
    for row in _read_pipe_csv(ALL_CSV, ALL_FIELDS):
        tid = str(int(row["actum_industry_meta_id"].strip()))
        all_count = int(row["attached_company_count"].strip())
        if tid in metadata:
            metadata[tid]["all_company_count"] = all_count
        else:
            metadata[tid] = {
                "topic_name":           row["actum_industry_meta_name"].strip(),
                "target_company_count": 0,
                "target_intel_count":   0,
                "all_company_count":    all_count,
            }

    return metadata


if __name__ == "__main__":
    metadata = build_topic_metadata()
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(metadata)} topics -> {OUTPUT_JSON}")

    for sample_id in ("2282", "8640", "1465"):
        if sample_id in metadata:
            m = metadata[sample_id]
            print(f"  {sample_id} | {m['topic_name']} | "
                  f"target={m['target_company_count']} "
                  f"intel={m['target_intel_count']} "
                  f"all={m['all_company_count']}")