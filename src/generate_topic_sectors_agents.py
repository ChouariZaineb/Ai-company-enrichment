
# src/generate_topic_sectors_agents.py
"""
One-time batch job: tags every taxonomy topic with 1-3 sector_ids from the
official 38-sector taxonomy (sectors_formatted.txt). Output feeds the topic
agent's taxonomy scoping later - it filters topic_sectors.json by whatever
matched_sector_ids sector_matcher_bedrock.py already produced for a company.
Resumable: saves after every batch, skips already-tagged topic_ids on restart.
Run manually: python src/generate_topic_sectors_agents.py
"""
import json
import os
from loader import load_sectors, load_topics
from bedrock_client import call_claude_bedrock


_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTORS_PATH = os.path.join(_BASE, "data", "sectors_formatted.txt")
TOPICS_PATH  = os.path.join(_BASE, "data", "topics_formatted.txt")
OUTPUT_PATH  = os.path.join(_BASE, "data", "topic_sectors.json")
BATCH_SIZE = 40


def load_existing() -> dict:
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save(sectors_map: dict):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sectors_map, f, ensure_ascii=False, indent=2)


def build_prompt(batch: list, sectors_prompt_text: str, valid_sector_ids: set) -> str:
    topics_block = "\n".join(f"{t.topic_id} | {t.topic_name}" for t in batch)
    return f"""Classify each topic below into 1 to 3 sectors from this official taxonomy ONLY - use the sector ids exactly as listed, never invent one:
{sectors_prompt_text}

Topics to classify (id | name):
{topics_block}

Every topic must get at least 1 sector - pick the closest fit even if imperfect, never leave one out.
Only use ids from this exact set: {sorted(valid_sector_ids)}

Respond with ONLY a JSON object, no markdown fences, no commentary:
{{"topic_id": [sector_id, sector_id (optional 2nd/3rd)], ...}}"""


def main():
    all_topics = load_topics(TOPICS_PATH)
    all_sectors = load_sectors(SECTORS_PATH)
    valid_sector_ids = {s.sector_id for s in all_sectors}
    sectors_prompt_text = "\n".join(f"{s.sector_id} | {s.sector_name}" for s in all_sectors)

    sectors_map = load_existing()
    remaining = [t for t in all_topics if str(t.topic_id) not in sectors_map]

    print(f"{len(all_topics)} topics total, {len(remaining)} still to tag.")

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i:i + BATCH_SIZE]
        prompt = build_prompt(batch, sectors_prompt_text, valid_sector_ids)

        try:
            raw = call_claude_bedrock(prompt, max_tokens=2000, temperature=0)
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
            parsed = json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            print(f"[WARN] batch at index {i} failed ({e}) - skipping, will retry on next run")
            continue

        valid_batch_ids = {str(t.topic_id) for t in batch}
        for topic_id, sector_ids in parsed.items():
            if topic_id not in valid_batch_ids:
                continue  # model invented an id not in this batch - ignore

            clean_ids = sorted({int(s) for s in sector_ids if int(s) in valid_sector_ids})[:3]
            hallucinated = sorted({int(s) for s in sector_ids} - valid_sector_ids)
            if hallucinated:
                print(f"[WARN] topic {topic_id}: dropped non-existent sector id(s) {hallucinated}")

            if clean_ids:
                sectors_map[topic_id] = clean_ids

        save(sectors_map)  # incremental save - survives interruption
        print(f"  batch {i // BATCH_SIZE + 1}: {len(sectors_map)}/{len(all_topics)} tagged so far")

    print("Done.")


if __name__ == "__main__":
    main()