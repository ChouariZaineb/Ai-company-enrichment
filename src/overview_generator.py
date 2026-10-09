import os
import json
import httpx
import time

from src.scraper_direct import search_company_fallback
from src.gemini_key_manager import build_gemini_pro_url


MAX_CONTENT_CHARS = 30000
COMPLETENESS_CHECK_KEYS = ["sectors", "products_offered", "services_offered"]


def _generate_overview_once(company_name: str, content: str, max_retries: int = 3):
    """
    One full attempt at generating the overview, retrying on transient 503s
    AND on malformed JSON (a rare Gemini generation glitch, not a real failure —
    a fresh attempt usually comes back clean). Returns the parsed dict, or None
    if it ultimately failed after all retries.
    """
    truncated_content = content[:MAX_CONTENT_CHARS]

    prompt = f"""You are analyzing raw research data about the company "{company_name}" to produce a categorized business overview.

Raw research data:
{truncated_content}

Based only on this data, produce a categorized overview with the following sections:

1. Industry: short tags describing the specific industry niches this company operates in.

2. Sectors: broader sector categories this company touches. Distinguish between industries the company SERVES AS CUSTOMERS versus sectors that reflect the company's OWN product, technology, or core business activity:
   - Do not include sectors that only describe customer types or buyer industries.
   - INCLUDE a sector if the company's own products, technology, or core operations genuinely belong to that sector — even if it also happens to serve that industry as a customer.
   - When in doubt, ask: "Is this what the company builds/does, or just who buys it?" Only exclude for the latter.
   
3. Products Offered: specific named tools, products, software, platforms, hardware, or tools, materials the company provides.

4. Services Offered: ongoing activities, operations, or support the company performs FOR the customer — things done as a service, not a standalone tool (e.g. "enforcement field services", "infringement management", "consulting", "outsourced operations", "reservations management", "implementation", "support", "maintenance", "outsourcing".....ect). If a named product is used to deliver an ongoing operational service, list the operational activity here, not just the underlying product again.

5. Unique Selling Propositions: what makes this company distinct from competitors, based only on claims actually supported by the research data.

6. Business Summary: 2 short paragraphs summarizing what the company does, who it serves, and any notable recent developments.

Rules:
- Base everything strictly on the provided research data. Do not invent details not supported by the text.
- Keep each Industry/Sector/Product/Service/USP entry short (a few words to one sentence), not full paragraphs.
- For Products Offered and Services Offered: consolidate closely related or overlapping items into a single, more general entry rather than listing every specific variant separately. Aim for roughly 6-10 well-chosen entries per section, not an exhaustive enumeration of every phrase mentioned in the text.
- If a section genuinely has no supporting information, return an empty list for it rather than guessing.

Respond ONLY in this exact JSON format, with no extra text before or after:
{{
  "industry": ["...", "..."],
  "sectors": ["...", "..."],
  "products_offered": ["...", "..."],
  "services_offered": ["...", "..."],
  "unique_selling_propositions": ["...", "..."],
  "business_summary": "paragraph 1\\n\\nparagraph 2"
}}
"""

    for attempt in range(1, max_retries + 1):
        try:
            response = httpx.post(
                build_gemini_pro_url(),
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0},
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            raw_output = data["candidates"][0]["content"]["parts"][0]["text"].strip()

            if raw_output.startswith("```"):
                raw_output = raw_output.strip("`")
                if raw_output.lower().startswith("json"):
                    raw_output = raw_output[4:].strip()

            try:
                return json.loads(raw_output)
            except json.JSONDecodeError:
                if attempt < max_retries:
                    print(f"[WARN] {company_name}: malformed JSON on attempt {attempt}/{max_retries}, retrying...")
                    continue
                print(f"[ERROR] Could not parse overview JSON for {company_name} after {max_retries} attempts:\n{raw_output}")
                return None

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (503, 429) and attempt < max_retries:
                wait = 20 if e.response.status_code == 429 else 5  # 429 = rate limit, needs a longer cooldown
                print(f"[WARN] Gemini {e.response.status_code}, waiting {wait}s and retrying ({attempt}/{max_retries})...")
                time.sleep(wait)
                continue
            print(f"[ERROR] Overview generation failed for {company_name}: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Overview generation failed for {company_name}: {e}")
            return None

    return None


def generate_categorized_overview(company_name: str, source_url: str, combined_content: str) -> dict:
    """
    Generate a categorized company overview. If sectors/products_offered/services_offered
    come back empty, run ONE completeness fallback search and regenerate once —
    these three feed downstream matching steps, so an empty one has real consequences.
    """
    if not combined_content or not combined_content.strip():
        return {}

    parsed = _generate_overview_once(company_name, combined_content)
    if parsed is None:
        return {}

    missing_sections = [key for key in COMPLETENESS_CHECK_KEYS if not parsed.get(key)]
    completeness_fallback_triggered = False

    if missing_sections:
        print(f"[INFO] {company_name}: {', '.join(missing_sections)} came back empty — running completeness fallback search.")
        fallback = search_company_fallback(company_name)
        if fallback["content"]:
            enriched_content = combined_content + f"\n\n=== COMPLETENESS FALLBACK SEARCH ===\n{fallback['content']}"
            retry_parsed = _generate_overview_once(company_name, enriched_content)
            if retry_parsed is not None:
                still_missing = [key for key in missing_sections if not retry_parsed.get(key)]
                if len(still_missing) < len(missing_sections):
                    parsed = retry_parsed
            completeness_fallback_triggered = True

    parsed["company_name"] = company_name
    parsed["source"] = source_url
    parsed["completeness_fallback_triggered"] = completeness_fallback_triggered
    if missing_sections:
        parsed["completeness_fallback_sections"] = missing_sections

    return parsed