import json
import time
from src.loader import build_taxonomy
from src.gemini_key_manager import build_gemini_pro_url
import httpx

taxonomy = build_taxonomy(
    sectors_path="data/sectors_formatted.txt",
    topics_path="data/topics_formatted.txt",
)
topics = taxonomy["topics"]

OUTPUT_PATH = "data/topic_descriptions.json"
BATCH_SIZE = 40

# Resume support — don't regenerate what we already have
try:
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        descriptions = json.load(f)
except FileNotFoundError:
    descriptions = {}

remaining = [t for t in topics if str(t.topic_id) not in descriptions]
print(f"{len(descriptions)} already done, {len(remaining)} remaining.")

batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]

for idx, batch in enumerate(batches, start=1):
    topics_text = "\n".join(f"{t.topic_id} | {t.topic_name}" for t in batch)
    prompt = f"""For each business/technology topic below, write a SHORT description of EXACTLY 3-5 words that clarifies its specific meaning — enough to disambiguate it from similar-sounding topics, without being vague or generic.

Topics (format: id | name):
{topics_text}

Rules:
- 3-5 words per description, no more, no less if possible.
- Be specific and concrete, not generic (avoid words like "solutions", "services" alone with no other content).
- Do not just repeat the topic name — add clarifying context.
- Example: "SaaS" -> "cloud-hosted subscription software delivery"
- Example: "Threat Detection" -> "identifying active cyberattacks in progress"
- Example: "Neobanking" -> "app-only bank, no branches"

Respond ONLY in this exact JSON format, with no extra text before or after:
{{
  "descriptions": [
    {{"topic_id": <int>, "description": "3-5 word description"}}
  ]
}}
"""
    try:
        response = httpx.post(
            build_gemini_pro_url(),
            json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}},
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)

        for item in parsed.get("descriptions", []):
            descriptions[str(item["topic_id"])] = item["description"]

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)

        print(f"Batch {idx}/{len(batches)} done. Total so far: {len(descriptions)}")

    except Exception as e:
        print(f"[ERROR] Batch {idx} failed: {e} — will retry on next run (progress saved).")
        time.sleep(2)

print(f"Done. {len(descriptions)}/{len(topics)} topics have descriptions.")