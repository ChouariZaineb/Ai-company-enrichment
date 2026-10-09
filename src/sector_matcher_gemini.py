#src/sector_matcher_gemini.py
"""
Cloud (Gemini Pro) version of sector matching — comparison sibling to
src/sector_matcher.py, NOT a replacement. Same inputs, same output shape
({"reasoning": str, "matched_sector_ids": [int, ...]}).

Uses the dedicated pro API key (no rotation needed — paid tier, no 429s).
"""
import json
import httpx

from src.gemini_key_manager import build_gemini_pro_url


def match_sectors_to_taxonomy_gemini(company_name: str, business_summary: str, generated_sectors: list,
                                      sectors_prompt_text: str, valid_sector_ids: set,
                                      max_retries: int = 3) -> dict:
    generated_sectors_str = "\n".join(f"- {s}" for s in generated_sectors)
    prompt = f"""You are matching a company to an official taxonomy of sectors.

Company: {company_name}

Business summary:
{business_summary}

Informally-generated sector tags for this company (a partial, possibly incomplete summary — do NOT treat this as the only source of truth):
{generated_sectors_str}

Official taxonomy sectors (format: id | name):
{sectors_prompt_text}

Instructions:
1.Use both the informal tags and the business summary as evidence. The informal tags provide the primary indication of the company's business domains, while the business summary provides additional context and clarification.

When reading the business summary, extract only high-level business domains and industry-specific concepts that help interpret or disambiguate the informal tags (e.g., cybersecurity, renewable energy, industrial automation, medical imaging, cloud computing). Ignore customer types, target audiences, products, features, account types, service descriptions, company scale, marketing language, technologies used internally, certifications, locations, or operational details unless they clearly describe an industry domain.

Do not infer new taxonomy sectors solely from descriptive text in the business summary. The business summary should only strengthen, refine, or disambiguate mappings suggested by the informal tags. It must not introduce sectors based on customer groups (e.g., consumers, businesses, students), product variants (e.g., personal accounts, business accounts), or other contextual details.
2. The informal tags (and the summary's wording) may be WORDED DIFFERENTLY from the official taxonomy names for the same underlying idea. Match based on MEANING, not on exact wording overlap.
3. Distinguish between industries the company SERVES AS CUSTOMERS versus sectors that reflect the company's OWN product, technology, or core business activity:
   - EXCLUDE a sector if it only describes who buys from the company.
   - INCLUDE a sector if the company's own products, technology, or core operations genuinely belong to that sector — even if it also happens to serve that industry as a customer.
4. Avoid duplicate sector ids.
5. Do not invent sector ids. Only use ids that appear in the official taxonomy list above.
6. Only include a sector if the business summary gives explicit, concrete evidence. No speculation.
7. Generic tags/summary language describing the company's OWN operating method or core technology (e.g. "Technology", "Software", "Digital", "AI-powered") should be INCLUDED under the matching technology-related sector unless the business summary gives a specific reason not to.
8. Be thorough: a company can genuinely belong to 2-3 sectors if its core business spans them (e.g. a company can be both a technology company AND a specific-industry company like energy, healthcare, or security, if its core product IS that industry's technology). Do not under-select out of excessive caution — if the summary gives clear evidence for a sector, include it, even if it wasn't in the informal tags.

Respond ONLY in this exact JSON format, with no extra text before or after:
{{
  "reasoning": "your step by step reasoning here, citing specific evidence from the business summary",
  "matched_sector_ids": [list of integer ids]
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
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()
            raw_output = data["candidates"][0]["content"]["parts"][0]["text"].strip()

            if raw_output.startswith("```"):
                raw_output = raw_output.strip("`")
                if raw_output.lower().startswith("json"):
                    raw_output = raw_output[4:].strip()

            parsed = json.loads(raw_output)
            reasoning = parsed.get("reasoning", "")
            raw_ids = parsed.get("matched_sector_ids", [])

            confirmed_ids = sorted(set(int(i) for i in raw_ids) & valid_sector_ids)
            hallucinated_ids = sorted(set(int(i) for i in raw_ids) - valid_sector_ids)
            if hallucinated_ids:
                print(f"[WARN] {company_name}: Gemini Pro returned non-existent sector id(s) {hallucinated_ids} — dropped.")

            return {"reasoning": reasoning, "matched_sector_ids": confirmed_ids}

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 503) and attempt < max_retries:
                print(f"[WARN] {company_name}: Gemini Pro {e.response.status_code}, retrying ({attempt}/{max_retries})...")
                continue
            print(f"[ERROR] Gemini Pro sector matching failed for {company_name}: {e}")
            return {"reasoning": "", "matched_sector_ids": []}
        except json.JSONDecodeError:
            if attempt < max_retries:
                continue
            print(f"[ERROR] {company_name}: could not parse Gemini Pro JSON after {max_retries} attempts.")
            return {"reasoning": "", "matched_sector_ids": []}
        except Exception as e:
            print(f"[ERROR] Gemini Pro sector matching failed for {company_name}: {e}")
            return {"reasoning": "", "matched_sector_ids": []}

    return {"reasoning": "", "matched_sector_ids": []}