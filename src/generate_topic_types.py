# src/generate_topic_types.py
"""
One-time batch job: classifies every taxonomy topic as either 'category'
or 'feature'. Output feeds get_topics_for_sectors() in topic_agent.py,
which annotates each topic in the candidate pool so the agent can reliably
apply "category label beats feature topic" during elimination.

category = describes an industry, sector, business model, or technology domain
           (e.g. "FinTech", "Cyber Security", "GreenTech", "SaaS", "Renewable Energy")
feature  = describes a specific product, sub-domain, tool, or capability
           (e.g. "Email Security", "Solar Panels", "Fraud Detection", "Heat Pumps")

Resumable: saves every batch, skips already-typed topic_ids on restart.
Run from project root: python src/generate_topic_types.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from loader import load_topics
from bedrock_client import call_claude_bedrock

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPICS_PATH = os.path.join(_BASE, "data", "topics_formatted.txt")
OUTPUT_PATH = os.path.join(_BASE, "data", "topic_types.json")
BATCH_SIZE = 60


def load_existing() -> dict:
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save(types_map: dict):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(types_map, f, ensure_ascii=False, indent=2)


def build_prompt(batch: list) -> str:
    topics_block = "\n".join(f"{t.topic_id} | {t.topic_name}" for t in batch)
    return f"""Classify each topic below as either 'category' or 'feature'.

DEFINITIONS:
- category: a broad industry sector, business model, technology domain, or market category.
  Examples: "FinTech", "Cyber Security", "GreenTech", "SaaS", "Renewable Energy",
  "Artificial Intelligence", "Digital Banking", "CleanTech", "EnergyTech", "Neobanking",
  "Software Development", "Machine Learning", "BusTech", "Energy Transition".

- feature: a specific product, sub-domain, tool, capability, or narrow technology.
  Examples: "Email Security", "Solar Panels", "Fraud Detection", "Heat Pumps",
  "EV Charging", "Endpoint Security", "Cloud Security", "Savings Solutions",
  "Pension Management", "Threat Detection", "Identity Management".

RULE: when in doubt, ask — is this a broad label someone would use to describe
an entire company's identity, or a specific thing a company offers/does?
If it describes a whole company identity → category. If it describes one thing a company does → feature.

Topics (id | name):
{topics_block}

Respond with ONLY a JSON object, no markdown fences, no commentary:
{{"topic_id": "category" or "feature", ...}}"""


def main():
    all_topics = load_topics(TOPICS_PATH)
    types_map = load_existing()
    remaining = [t for t in all_topics if str(t.topic_id) not in types_map]

    print(f"{len(all_topics)} topics total, {len(remaining)} still to classify.")

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i:i + BATCH_SIZE]
        prompt = build_prompt(batch)

        try:
            raw = call_claude_bedrock(prompt, max_tokens=2000, temperature=0)
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
            parsed = json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            print(f"[WARN] batch at index {i} failed ({e}) — skipping, retry on next run")
            continue

        valid_batch_ids = {str(t.topic_id) for t in batch}
        valid_values = {"category", "feature"}
        rejected = 0

        for topic_id, type_val in parsed.items():
            if topic_id not in valid_batch_ids:
                continue
            if type_val not in valid_values:
                rejected += 1
                continue
            types_map[topic_id] = type_val

        if rejected:
            print(f"[WARN] batch {i // BATCH_SIZE + 1}: {rejected} invalid values dropped")

        save(types_map)
        print(f"  batch {i // BATCH_SIZE + 1}: {len(types_map)}/{len(all_topics)} classified so far")

    print("Done.")


if __name__ == "__main__":
    main()