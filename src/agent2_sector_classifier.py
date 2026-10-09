# src/agent2_sector_classifier.py
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bedrock_client_sonnet5 import call_sonnet5
from loader import build_taxonomy

_BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAXONOMY = build_taxonomy(
    sectors_path=os.path.join(_BASE, "data", "sectors_formatted.txt"),
    topics_path=os.path.join(_BASE, "data", "topics_formatted.txt"),
)
MAX_SECTORS = 3


def run_sector_classifier(
    company_name: str,
    company_profile: str,
    max_retries: int = 3,
) -> dict:
    sectors_block    = TAXONOMY["sectors_prompt_text"]
    valid_sector_ids = TAXONOMY["valid_sector_ids"]

    prompt = f"""You are a senior private equity sector classification specialist.

Classify this company into the official sector taxonomy.

COMPANY: {company_name}

RESEARCH PROFILE:
{company_profile}

OFFICIAL SECTOR TAXONOMY (id | name):
{sectors_block}

RULES:
1. Select at most {MAX_SECTORS} sectors.
2. ONLY include sectors where the company's OWN products/technology/operations belong.
   Ask: "Does this company BUILD this, or just SELL TO people in this field?"
   - INCLUDE: cybersecurity software company → sector 7 (Computing/Tech) + sector 4 (Defence/Security)
   - EXCLUDE: same company serving banks → do NOT add sector 16 (banks are customers, not the company's identity)
   - INCLUDE: recruitment SaaS → sector 7 + sector 17 (Professional Services)
   - INCLUDE: ice cream machinery → sector 20 (Manufacturing) + sector 18 (Food & Beverage)
3. Only use sector ids that appear in the taxonomy.
4. State specific evidence from the profile for each sector.

Respond ONLY in this exact JSON format, no markdown fences:
{{
  "matched_sector_ids": [list of integer ids, max {MAX_SECTORS}],
  "sector_reasoning": "For sector X: [evidence]. For sector Y: [evidence]."
}}"""

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_sonnet5(prompt, max_tokens=800)
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found")
            parsed     = json.loads(raw[start:end])
            sector_ids = sorted(
                {int(i) for i in parsed.get("matched_sector_ids", [])} & valid_sector_ids
            )[:MAX_SECTORS]
            return {
                "matched_sector_ids": sector_ids,
                "sector_reasoning":   parsed.get("sector_reasoning", ""),
            }
        except Exception as e:
            if attempt < max_retries:
                print(f"  [WARN] Sector classifier attempt {attempt} failed: {e}, retrying...")
                continue
            print(f"  [ERROR] Sector classifier failed for {company_name}: {e}")
            return {"matched_sector_ids": [], "sector_reasoning": ""}

    return {"matched_sector_ids": [], "sector_reasoning": ""}