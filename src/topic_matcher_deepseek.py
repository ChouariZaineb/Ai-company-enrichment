"""
Cloud (DeepSeek 3.2 via Bedrock) topic matching — same two-stage design
and prompts as topic_matcher_bedrock_v2.py (Claude + enriched embeddings),
calling DeepSeek instead. Reuses the SAME enriched embeddings (embedding
choice is independent of which model reasons over the candidates).
"""
import json

from src.bedrock_deepseek_client import call_deepseek_bedrock
from src.topic_embeddings_enriched import (
    load_or_build_topic_embeddings_enriched,
    get_candidate_topics_flat_enriched,
)

CHUNK_SIZE = 25
CANDIDATE_POOL_SIZE_PER_SECTOR = 50


def _chunk_list(items: list, chunk_size: int) -> list:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _format_list(items: list) -> str:
    items = [str(i) for i in items if i]
    return "(none)" if not items else "; ".join(items)


def _format_candidates(candidates: list) -> str:
    if not candidates:
        return "(none)"
    return "\n".join(f"{i}. {t.topic_id} | {t.topic_name}" for i, t in enumerate(candidates, start=1))


def _build_selection_prompt(company_data: dict, sector_names_str: str, candidates: list) -> str:
    return f"""<role>
You are a taxonomy classification specialist mapping a company to candidate topics.
</role>

<sector_context>
Matched sectors: {sector_names_str}
</sector_context>

<company_data>
<company_name>{company_data.get('company_name', '')}</company_name>
<business_summary>{company_data.get('business_summary') or "(none)"}</business_summary>
<industry>{_format_list(company_data.get('industry', []))}</industry>
<generated_sectors>{_format_list(company_data.get('sectors', []))}</generated_sectors>
<products_offered>{_format_list(company_data.get('products_offered', []))}</products_offered>
<services_offered>{_format_list(company_data.get('services_offered', []))}</services_offered>
<unique_selling_propositions>{_format_list(company_data.get('unique_selling_propositions', []))}</unique_selling_propositions>
</company_data>

<candidate_topics>
{_format_candidates(candidates)}
</candidate_topics>

<task>
Evaluate every candidate topic above as an independent binary decision: INCLUDE or EXCLUDE.
Judge each one in isolation from the others — do not compare candidates to each other, do not rank them, and do not stop once you've found a few good matches. Every single candidate must be judged, one at a time.
</task>

<evidence_standard>
INCLUDE a topic only if you can point to a specific field above that states the company actually does this — not implies it, not is commonly associated with it in this sector, not sounds plausible given its industry.

Worked examples of the standard:
- services_offered explicitly lists "payroll processing" alongside a core accounting product. Candidate: "Payroll Software" → INCLUDE. There is a specific, quotable fact.
- industry says "Financial Services" and a candidate is "Wealth Management" → EXCLUDE, unless another field specifically states wealth management activity. Sharing a broad industry label is not evidence of a specific activity within it.
- business_summary describes "retail banking" and a candidate is literally named "Retail" → EXCLUDE. Word overlap is not evidence; judge actual business meaning, never surface text matching.

When genuinely unsure whether the evidence is specific enough, EXCLUDE. Returning fewer, well-evidenced topics is always preferable to including a plausible-sounding one you cannot point to concrete evidence for.

At this stage, do not worry about whether a topic describes the company's core identity versus a secondary feature — that distinction is handled in a later refinement pass. Here, only judge whether the topic is factually supported at all.
</evidence_standard>

<output_instructions>
Before listing matched_topic_ids, use the reasoning field to work through every candidate topic in order: for each one, name the specific field and fact that supports it, or state plainly that no such field/fact exists, then note your INCLUDE/EXCLUDE decision. Cover every candidate in the list, in order, inside reasoning.

Then set matched_topic_ids to only the ids you decided INCLUDE for, using ids that appear verbatim in the candidate list above. If no candidates genuinely pass the evidence bar, return an empty matched_topic_ids list.

Respond ONLY in this exact JSON format, with no extra text before or after:
{{
  "reasoning": "your per-candidate evidence walkthrough here",
  "matched_topic_ids": [list of integer ids]
}}
"""


def _build_refinement_prompt(company_data: dict, sector_names_str: str, shortlisted: list) -> str:
    items_text = "\n".join(f"{i}. {t['topic_id']} | {t['topic_name']}" for i, t in enumerate(shortlisted, start=1))
    return f"""<role>
You are a taxonomy classification specialist refining an already evidence-checked shortlist of topics down to the ones that define what kind of company this is.
</role>

<matched_sectors>
{sector_names_str}
</matched_sectors>

<company_data>
<company_name>{company_data.get('company_name', '')}</company_name>
<business_summary>{company_data.get('business_summary') or "(none)"}</business_summary>
<industry>{_format_list(company_data.get('industry', []))}</industry>
<products_offered>{_format_list(company_data.get('products_offered', []))}</products_offered>
<services_offered>{_format_list(company_data.get('services_offered', []))}</services_offered>
</company_data>

<shortlisted_topics>
{items_text}
</shortlisted_topics>

<context>
Every topic in shortlisted_topics has already passed an evidence-based selection stage. Your job here is NOT to re-check evidence. Your job is to keep only topics that (a) clearly belong under one of the matched_sectors above, (b) describe what kind of company this is — its core category/identity, not a specific feature or secondary offering — and (c) are not redundant near-synonyms of another kept topic.
</context>

<procedure>
STEP 0 — Sector alignment: drop any topic that does not clearly and specifically belong under one of the matched_sectors listed above, even if it's a true fact about the company. A topic must fit within the scope of a matched sector to survive.

STEP 1 — Identity vs. feature test: for topics that pass Step 0, ask "if someone described this company in one sentence to explain what kind of company it fundamentally is, would this topic be part of that sentence?"
- IDENTITY topics (category, core business model, core technology type) survive.
- FEATURE topics (a specific capability, product, delivery channel, secondary offering — even if true and well-evidenced) get dropped.
Example: a cyber security AI company keeps "Cyber Security" and "Artificial Intelligence" (identity) but drops "Threat Detection", "Cloud Security" (a feature of the cyber security identity, not a separate identity).
Example: a renewable energy company keeps "Renewable Energy", "sustainable energy", "Solar Solutions", "CleanTech", "clean energy" (identity/category + generation types it operates in) but drops "Energy Storage," "Heat Pumps," "EV Charging" (specific installed products/services, not the company's identity).

STEP 2 — Near-synonym cluster check (apply aggressively): group the surviving topics into clusters where multiple topics express essentially the same underlying business identity using different words or different levels of genericness (e.g. "Banking," "Financial Services," "Digital Finance," "Consumer Finance," "Transaction Banking" all describing the same core banking activity). Within each cluster, keep ONLY the 1-2 topics that are most specific and most precisely descriptive of the company's actual identity — drop every other topic in that cluster, even though each one individually would pass Step 1. Do not keep multiple topics from the same cluster just because each is separately true; the point of this step is to eliminate redundant near-synonyms, not to verify each one's truth again.
Example: a digital bank's shortlist includes "Banking," "Financial Services," "Digital Banking," "FinTech," "Neobanking," "Banking Services," "Digital Finance," "Business Banking," "Consumer Finance," "Transaction Banking." These are ONE cluster (they all describe "this company is a bank/financial technology company"). Keep only the most precise: "Digital Banking," "Neobanking," "FinTech", "Banking Services". Drop the rest: "Banking" (too generic vs. Digital Banking), "Financial Services" (too generic), "Digital Finance" (redundant with Digital Banking/FinTech), "Business Banking," "Consumer Finance," "Transaction Banking" (redundant sub-slices of the same core identity already captured).

STEP 3 — Final count check: after Steps 0-2, if more than 5-6 topics remain, that is a signal you were not aggressive enough in Step 2 — re-examine for missed near-synonym clusters. A precise identity is usually captured by a small number of topics, not a long list.
</procedure>

<output_instructions>
Do NOT write a line-by-line justification for every dropped topic — that is not required and wastes space. Just apply the steps above internally, then output the result.

Respond ONLY in this exact JSON format, with no extra text before or after:
{{
  "reasoning": "one short paragraph (2-4 sentences) summarizing your overall approach, which clusters you collapsed, and any notable judgment calls",
  "refined_topic_ids": [list of integer ids that survived all steps]
}}
"""


def _call_and_parse(prompt: str, id_key: str, valid_ids: set, max_retries: int = 3, max_tokens: int = 3000) -> tuple:
    for attempt in range(1, max_retries + 1):
        raw_output = ""
        try:
            raw_output = call_deepseek_bedrock(prompt, max_tokens=max_tokens)

            if raw_output.startswith("```"):
                raw_output = raw_output.strip("`")
                if raw_output.lower().startswith("json"):
                    raw_output = raw_output[4:].strip()

            parsed = json.loads(raw_output)
            raw_ids = parsed.get(id_key, [])
            valid_returned = [int(i) for i in raw_ids if int(i) in valid_ids]
            invalid_returned = [int(i) for i in raw_ids if int(i) not in valid_ids]

            if invalid_returned:
                print(f"[WARN] DeepSeek returned invalid id(s), filtered out: {invalid_returned}")

            return valid_returned, parsed.get("reasoning", "")

        except json.JSONDecodeError:
            if attempt < max_retries:
                print(f"[WARN] JSON parse failed on attempt {attempt}/{max_retries}, retrying...")
                print(f"[DEBUG] Raw output was: {raw_output[:500]}")
                continue
            print(f"[ERROR] Could not parse DeepSeek JSON after {max_retries} attempts.")
            print(f"[DEBUG] Final raw output was: {raw_output[:500]}")
            return [], ""
        except Exception as e:
            print(f"[ERROR] DeepSeek call failed: {e}")
            if attempt < max_retries:
                continue
            return [], ""

    return [], ""


def match_topics_flat_deepseek(company_name: str, business_summary: str, matched_sectors: list,
                                topics: list, company_data_extra: dict = None,
                                top_n_per_sector: int = CANDIDATE_POOL_SIZE_PER_SECTOR) -> dict:
    """Two-stage topic matching using DeepSeek, same design as the Claude v2 pipeline."""
    company_data_extra = company_data_extra or {}
    company_data = {
        "company_name": company_name,
        "business_summary": business_summary,
        **company_data_extra,
    }

    topic_embeddings = load_or_build_topic_embeddings_enriched(topics)
    candidate_topics = get_candidate_topics_flat_enriched(
        company_name, business_summary, matched_sectors, topics, topic_embeddings, top_n_per_sector,
    )

    if not candidate_topics:
        return {"topics": [], "reasoning": "No candidates retrieved."}

    sector_names_str = ", ".join(s["sector_name"] for s in matched_sectors)
    topic_by_id = {t.topic_id: t for t in candidate_topics}

    chunks = _chunk_list(candidate_topics, CHUNK_SIZE)
    merged_ids = []
    seen = set()
    stage1_reasoning_parts = []

    for idx, chunk in enumerate(chunks, start=1):
        valid_ids_in_chunk = {t.topic_id for t in chunk}
        prompt = _build_selection_prompt(company_data, sector_names_str, chunk)
        ids, reasoning = _call_and_parse(prompt, "matched_topic_ids", valid_ids_in_chunk, max_tokens=3000)
        print(f"  [Stage 1] Chunk {idx}/{len(chunks)}: {len(ids)} included.")
        for tid in ids:
            if tid not in seen:
                seen.add(tid)
                merged_ids.append(tid)
        if reasoning:
            stage1_reasoning_parts.append(f"Chunk {idx}: {reasoning}")

    if not merged_ids:
        return {"topics": [], "reasoning": " ".join(stage1_reasoning_parts)}

    shortlisted = [{"topic_id": tid, "topic_name": topic_by_id[tid].topic_name} for tid in merged_ids]
    print(f"  [Stage 1 complete] {len(shortlisted)} topics shortlisted, moving to refinement...")

    valid_shortlist_ids = {t["topic_id"] for t in shortlisted}
    refine_prompt = _build_refinement_prompt(company_data, sector_names_str, shortlisted)
    refined_ids, refine_reasoning = _call_and_parse(
        refine_prompt, "refined_topic_ids", valid_shortlist_ids, max_tokens=2000
    )

    if not refined_ids and not refine_reasoning:
        print("[WARN] Refinement failed after retries — falling back to unrefined shortlist.")
        return {"topics": shortlisted, "reasoning": " ".join(stage1_reasoning_parts)}

    final_topics = [{"topic_id": tid, "topic_name": topic_by_id[tid].topic_name} for tid in refined_ids]
    return {"topics": final_topics, "reasoning": refine_reasoning}