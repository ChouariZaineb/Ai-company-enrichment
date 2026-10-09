#sector_matcher_bedrock.py
"""
Cloud (Claude Sonnet 4.6 via Bedrock) version of sector matching — comparison
sibling to sector_matcher.py / sector_matcher_gemini.py. Same inputs, same
output shape ({"reasoning": str, "matched_sector_ids": [int, ...]}).
"""
import json

from src.bedrock_client import call_claude_bedrock


def match_sectors_to_taxonomy_bedrock(company_name: str, business_summary: str, generated_sectors: list,
                                       sectors_prompt_text: str, valid_sector_ids: set,
                                       max_retries: int = 3) -> dict:
    generated_sectors_str = "\n".join(f"- {s}" for s in generated_sectors)
    prompt = f"""<role>
You are a taxonomy classification specialist matching a company to its core business sectors in an official taxonomy.
</role>

<company_data>
<company_name>{company_name}</company_name>
<business_summary>{business_summary}</business_summary>
<informally_generated_sector_tags>{generated_sectors_str}</informally_generated_sector_tags>
</company_data>

<official_taxonomy_sectors>
{sectors_prompt_text}
</official_taxonomy_sectors>

<task>
Evaluate every official taxonomy sector above as an independent binary decision: INCLUDE or EXCLUDE. Judge each one on its own — do not compare sectors to each other, do not assume the informal tags are complete or authoritative, and do not stop once you've found one or two plausible sectors. The informal tags are a partial, sometimes-incomplete hint, not a checklist to fill or a ceiling to respect — the business_summary is the primary source of truth.
</task>

<evidence_standard>
INCLUDE a sector only if the business_summary contains a specific, concrete statement that the company's OWN products, technology, or core operations belong to that sector — not that it sounds plausible given the industry, not that a competitor in that space would fit, not that the informal tags loosely gesture at it.
Use semantic reasoning, not just exact keyword matching. If the description expresses concepts equivalent or closely related to technology, digital innovation, Innovative, smart,  software, intelligent systems, automation, scalable platforms, engineering, or other technology-oriented capabilities—even if different wording is used—consider this evidence that the company operates in the Technology sector.
Distinguish company identity from customer base:
- EXCLUDE a sector if the business_summary only shows the company SERVING that industry as a customer base (e.g. "used by hospitals and airports" does NOT make the company a Healthcare or Transportation company).
- INCLUDE a sector if the company's own product/technology/operations concretely belong to it, even if it also serves that industry as a customer.

Worked examples of the standard:
- business_summary states the company "provides proactive threat detection through its AI Security Platform" → INCLUDE the cybersecurity/defence sector AND the technology/AI sector, since both are directly, concretely stated as the company's own activity.
- business_summary mentions the company's clients include hospitals and governments, with no further detail on what the company itself builds for them → EXCLUDE any healthcare or government sector; those are customers, not the company's identity.
- business_summary describes the company as a "digital bank" delivered via "mobile application" → INCLUDE both the financial-services sector (the core offering) and the technology sector (the delivery mechanism is explicitly the product's core nature, not incidental).
- business_summary describes sustainability/decarbonization goals or environmentally-friendly products as part of a company whose core identity is a different specific industry (e.g. an energy supplier, a manufacturer) → EXCLUDE a general "Environment" sector unless the business summary shows environmental services/consulting/impact-measurement as a distinct offering in itself, not just a quality or motivation behind the company's actual products.
- business_summary mentions the company installs, sells, or distributes physical hardware/equipment (e.g. heat pumps, chargers, panels) as part of delivering its core service → do NOT separately include a general electronics/hardware-manufacturing sector unless the company itself designs or manufactures that hardware; installing or reselling equipment as part of a service is not the same as being an electronics/hardware company.
- business_summary describes the company as operating through multiple internal "divisions" or business units (e.g. a company with an "Energy Services division," "Financial Services division," and "Home Services division" as three named parts of ONE integrated offering) → do NOT treat each named division as its own separate sector match. Identify what the company's core, unified identity actually is (e.g. it installs, finances, and maintains home equipment as one bundled service), and only include the sector(s) that describe that real overall identity. A division that exists purely to support or fund the core offering (e.g. an in-house financing arm that lets customers pay for the company's own installations) is a supporting function, not evidence the company itself belongs to that division's sector.
When genuinely unsure whether the evidence is concrete enough, EXCLUDE. A precise 1-2 sector match is far better than a padded 4-5 sector match with weak justification for the extras.
</evidence_standard>

<output_instructions>
Before listing matched_sector_ids, use the reasoning field to briefly state, for each sector you are including, the specific fact/phrase from business_summary that justifies it. Do not include reasoning for excluded sectors — only justify what you keep.

Then set matched_sector_ids to only the ids you decided INCLUDE for, using ids that appear verbatim in the official taxonomy list above. Do not invent ids.

Respond ONLY in this exact JSON format, with no extra text before or after:
{{
  "reasoning": "brief evidence justification for each included sector only",
  "matched_sector_ids": [list of integer ids]
}}
"""

    for attempt in range(1, max_retries + 1):
        try:
            raw_output = call_claude_bedrock(prompt)

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
                print(f"[WARN] {company_name}: Bedrock returned non-existent sector id(s) {hallucinated_ids} — dropped.")

            return {"reasoning": reasoning, "matched_sector_ids": confirmed_ids}

        except json.JSONDecodeError:
            if attempt < max_retries:
                print(f"[WARN] {company_name}: malformed JSON on attempt {attempt}/{max_retries}, retrying...")
                continue
            print(f"[ERROR] {company_name}: could not parse Bedrock JSON after {max_retries} attempts.")
            return {"reasoning": "", "matched_sector_ids": []}
        except Exception as e:
            print(f"[ERROR] Bedrock sector matching failed for {company_name}: {e}")
            if attempt < max_retries:
                continue
            return {"reasoning": "", "matched_sector_ids": []}

    return {"reasoning": "", "matched_sector_ids": []}