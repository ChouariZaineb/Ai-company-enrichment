import ollama
import json
import re

OLLAMA_MODEL = "mistral:7b"


def match_sectors_to_taxonomy(company_name: str, business_summary: str, generated_sectors: list,
                               sectors_prompt_text: str, valid_sector_ids: set,
                               max_retries: int = 3) -> dict:
    """
    Match a company's informally-generated sector tags to the official taxonomy.

    valid_sector_ids: the set of real sector IDs from loader.build_taxonomy() —
    used as a safety guard so a hallucinated ID can never reach the final result.
    """
    generated_sectors_str = "\n".join(f"- {s}" for s in generated_sectors)
    prompt = f"""You are matching a company's informally-generated sector tags to an official taxonomy of sectors.

Company: {company_name}

Business summary:
{business_summary}

Informally-generated sector tags for this company:
{generated_sectors_str}

Official taxonomy sectors (format: id | name):
{sectors_prompt_text}

Instructions:
1. For each informally-generated tag listed above (and ONLY those tags — do not evaluate other sectors from the taxonomy that have no corresponding tag), decide which official taxonomy sector(s) genuinely match, based on what the company actually does (per the business summary) — not based on similar-sounding words.
2. The informal tags may be WORDED DIFFERENTLY from the official taxonomy names for the same underlying idea. Match based on MEANING, not on exact wording overlap — do not reject a match just because the phrasing differs. Example: "Payment Processing" and "Financial Technology (Payments)" refer to the same sector; "Urban Mobility" or "Smart Cities" can map to a sector like "Transportation and Logistics" if the taxonomy has no exact "Smart Cities" entry. If genuinely unsure which taxonomy sector an informal tag's meaning maps to, re-read the business summary for concrete evidence before deciding — don't discard a tag just because no taxonomy entry shares its exact words.
3. Distinguish between industries the company SERVES AS CUSTOMERS versus sectors that reflect the company's OWN product, technology, or core business activity:
   - EXCLUDE a sector if it only describes who buys from the company.
   - INCLUDE a sector if the company's own products, technology, or core operations genuinely belong to that sector — even if it also happens to serve that industry as a customer.
   - When in doubt, ask: "Is this what the company builds/does, or just who buys it?" Only exclude for the latter.

Worked example: A company sells parking management software to hospitals, governments, and airports, and separately sells its own payment-processing app.
- "Healthcare (Hospitals)" → EXCLUDE. Hospitals are just a customer.
- "Government" → EXCLUDE. Same reasoning.
- "Financial Technology (Payments)" → INCLUDE. The company built and sells its own payment product — this is its own activity, not just serving a financial customer.
- "Transportation and Logistics" → INCLUDE, if the company's core technology itself operates within transportation.

4. Multiple informal tags may map to the same official sector — avoid duplicate sector ids.
5. Do not invent sector ids. Only use ids that appear in the official taxonomy list above.
6. Only include a sector if the business summary gives explicit, concrete evidence. No speculation — if evidence is weak or requires "possibly," leave it out.
7. Generic tags describing the company's OWN operating method or core technology — such as "Technology", "Software", "Digital", "AI-powered", "Tech-enabled", "Smart [X]" — describe what the company itself is built on, NOT a customer industry. Do not apply the customer-exclusion rule (instruction 3) to these; they should be INCLUDED under the matching technology-related sector unless the business summary gives a specific reason not to. If an informal tag shares an obvious word with an official taxonomy sector name (e.g. "Technology and Software" vs. "Computing, Technology, Robotics & AI"), treat that as strong evidence of a match on its own — do not require additional justification for something already explicit.

Respond ONLY in this exact JSON format, with no extra text before or after:
{{
  "reasoning": "your step by step reasoning here",
  "matched_sector_ids": [list of integer ids]
}}
"""

    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"num_ctx": 4096, "num_predict": 500, "temperature": 0},
            )
        except Exception as e:
            print(f"[ERROR] Ollama call failed for {company_name} (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                continue
            return {"reasoning": "", "matched_sector_ids": []}

        raw_output = response["message"]["content"].strip()
        json_blocks = re.findall(r"\{.*?\}(?=\s*[,{]|\s*$)", raw_output, re.DOTALL)

        all_reasoning = []
        all_sector_ids = set()
        parsed_any = False

        for block in json_blocks:
            try:
                parsed = json.loads(block, strict=False)
                parsed_any = True
                reasoning = parsed.get("reasoning", "")
                ids = parsed.get("matched_sector_ids", [])
                if reasoning:
                    all_reasoning.append(reasoning)
                all_sector_ids.update(ids)
            except json.JSONDecodeError:
                continue

        if not parsed_any:
            if attempt < max_retries:
                print(f"[WARN] {company_name}: could not parse any JSON (attempt {attempt}/{max_retries}), retrying...")
                continue
            print(f"[ERROR] {company_name}: giving up after {max_retries} attempts — no parseable JSON from Ollama.")
            return {"reasoning": "", "matched_sector_ids": []}

        confirmed_ids = sorted(all_sector_ids & valid_sector_ids)
        hallucinated_ids = sorted(all_sector_ids - valid_sector_ids)
        if hallucinated_ids:
            print(f"[WARN] {company_name}: model returned non-existent sector id(s) {hallucinated_ids} — dropped.")

        return {"reasoning": "\n".join(all_reasoning), "matched_sector_ids": confirmed_ids}

    return {"reasoning": "", "matched_sector_ids": []}


