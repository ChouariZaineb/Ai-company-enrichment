# src/topic_agent_weighted.py
"""
EXPERIMENTAL variant of topic_agent.py — adds topic popularity weighting
(data/topic_metadata.json, built by src/build_topic_metadata.py) as a
tiebreaker signal. topic_agent.py is left completely untouched so the
original behaviour stays available to fall back to.

Flow (same as topic_agent.py):
  Pass 1 — tool-loop agent: get_topics_for_sectors → submit_topics
  Pass 2 — category verification: reviews unselected category topics
  Pass 3 — taxonomy filter: removes known generic/sub-feature topics
             when enough specific topics remain

Recursion limit: MAX_STEPS = 8 (safety net — agent should use 2 steps).

STEP 2 (this file, current version):
  get_topics_for_sectors now annotates each candidate topic with
  "(attached to N companies)", sourced from topic_metadata.json.
  Sorting within each category/feature group is now by count descending.

STEP 3 (not yet applied): system prompt tiebreaker instruction.
STEP 4 (not yet applied): verification pass prompt tiebreaker instruction.
"""
import json
import os
from agent_core import ToolSpec, run_tool_agent, AgentResult
from topic_ranker import rank_topics
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPIC_SECTORS_PATH  = os.path.join(_BASE, "data", "topic_sectors.json")
TOPIC_TYPES_PATH    = os.path.join(_BASE, "data", "topic_types.json")
TOPICS_PATH         = os.path.join(_BASE, "data", "topics_formatted.txt")
TOPIC_DESC_PATH     = os.path.join(_BASE, "data", "topic_descriptions.json")
TOPIC_METADATA_PATH = os.path.join(_BASE, "data", "topic_metadata.json")

MAX_STEPS  = 10
MAX_TOPICS = 10
MIN_TOPICS = 3


# ---------------------------------------------------------------------------
# Load taxonomy data once at module level
# ---------------------------------------------------------------------------

def _load_topic_sectors() -> dict[str, list[int]]:
    with open(TOPIC_SECTORS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_topic_types() -> dict[str, str]:
    with open(TOPIC_TYPES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_all_topics() -> dict[int, str]:
    from loader import load_topics
    topics = load_topics(TOPICS_PATH)
    return {t.topic_id: t.topic_name for t in topics}


def _load_topic_descriptions() -> dict[str, str]:
    with open(TOPIC_DESC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_topic_metadata() -> dict[str, dict]:
    """
    Loads topic_id -> {topic_name, target_company_count, target_intel_count,
    all_company_count}, built by src/build_topic_metadata.py.
    Returns {} (with a warning) if the file hasn't been generated yet, so the
    agent still runs — just without popularity annotations.
    """
    if not os.path.exists(TOPIC_METADATA_PATH):
        print(
            f"[WARN] {TOPIC_METADATA_PATH} not found — topic counts will show "
            f"as 0 for everything. Run `python src/build_topic_metadata.py` first."
        )
        return {}
    with open(TOPIC_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _topic_count(tid_str: str) -> int:
    """Primary signal = target_company_count; falls back to all_company_count
    for topics with no target-company attachments on record."""
    meta = _TOPIC_METADATA.get(tid_str, {})
    return meta.get("target_company_count") or meta.get("all_company_count", 0)


_TOPIC_SECTORS:      dict[str, list[int]] = _load_topic_sectors()
_TOPIC_TYPES:        dict[str, str]       = _load_topic_types()
_ALL_TOPICS:         dict[int, str]       = _load_all_topics()
_TOPIC_DESCRIPTIONS: dict[str, str]       = _load_topic_descriptions()
_TOPIC_METADATA:     dict[str, dict]      = _load_topic_metadata()


# ---------------------------------------------------------------------------
# Taxonomy-level filter rules
# ---------------------------------------------------------------------------

def _apply_taxonomy_filter(topics: list[dict]) -> list[dict]:
    # TAXONOMY RULES TEMPORARILY DISABLED — comment back in _TAXONOMY_RULES to re-enable
    return topics


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _get_topics_for_sectors(sector_ids: list[int]) -> str:
    """
    Returns all taxonomy topics tagged with at least one of the given
    sector_ids, annotated as [category] or [feature] with description
    AND a real-world usage count, categories listed first (then by
    count descending within each group).
    """
    sector_set = set(sector_ids)
    matched = []
    for tid_str, sids in _TOPIC_SECTORS.items():
        if not (sector_set & set(sids)):
            continue
        tid_int = int(tid_str)
        if tid_int not in _ALL_TOPICS:
            continue
        ttype = _TOPIC_TYPES.get(tid_str, "feature")
        desc  = _TOPIC_DESCRIPTIONS.get(tid_str, "")
        count = _topic_count(tid_str)
        matched.append((tid_int, _ALL_TOPICS[tid_int], ttype, desc, count))

    matched = rank_topics(matched)

    if not matched:
        return "No topics found for those sector ids."

    lines = []
    for tid, tname, ttype, desc, count in matched:
        if desc:
            lines.append(f"{tid} | {tname} [{ttype}] (attached to {count} companies) — {desc}")
        else:
            lines.append(f"{tid} | {tname} [{ttype}] (attached to {count} companies)")

    n_cats = sum(1 for _, _, t, _, _ in matched if t == "category")
    return (
        f"{len(matched)} topics ({n_cats} categories listed first):\n"
        + "\n".join(lines)
    )


def _validate_and_submit(topics: list[dict]) -> list[dict]:
    """
    Enforces MAX_TOPICS cap, validates topic_ids exist in taxonomy,
    and requires a non-empty evidence string for every topic.
    """
    valid    = []
    seen_ids = set()

    for t in topics:
        if len(valid) >= MAX_TOPICS:
            break
        try:
            tid = int(t["topic_id"])
        except (KeyError, ValueError, TypeError):
            continue
        if tid not in _ALL_TOPICS:
            continue
        if tid in seen_ids:
            continue
        evidence = t.get("evidence", "").strip()
        if not evidence:
            continue
        seen_ids.add(tid)
        valid.append({
            "topic_id":   tid,
            "topic_name": _ALL_TOPICS[tid],
            "evidence":   evidence,
        })

    return valid


# ---------------------------------------------------------------------------
# Tool specs
# ---------------------------------------------------------------------------

def _build_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="get_topics_for_sectors",
            description=(
                "Retrieves all taxonomy topics for the given sector ids, "
                "annotated as [category] or [feature] with a short description. "
                "Call this ONCE, then immediately call submit_topics."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sector_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "The sector ids already matched for this company.",
                    }
                },
                "required": ["sector_ids"],
            },
            handler=lambda sector_ids: _get_topics_for_sectors(sector_ids),
        ),
        ToolSpec(
            name="submit_topics",
            description=(
                "Submit your final topic selections. Call this immediately after "
                "get_topics_for_sectors. Submit between 3 and 8 topics. "
                "Every topic MUST include evidence from the overview."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic_id":   {"type": "integer"},
                                "topic_name": {"type": "string"},
                                "evidence":   {"type": "string"},
                            },
                            "required": ["topic_id", "topic_name", "evidence"],
                        },
                    }
                },
                "required": ["topics"],
            },
            handler=lambda topics: _validate_and_submit(topics),
        ),
    ]


# ---------------------------------------------------------------------------
# System prompt — UNCHANGED from topic_agent.py for now (Step 3 pending)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior private equity analyst specialising in taxonomy classification.

Your task: select the most accurate topics from a taxonomy to describe a company from a PE investment perspective.

STRICT WORKFLOW — exactly 2 tool calls:
STEP 1: Call get_topics_for_sectors with the matched sector ids.
STEP 2: Call submit_topics immediately with your final selections.
Do NOT call any tool more than once. Do NOT do additional lookups.

CRITICAL SELECTION PRINCIPLE:
Select topics based on what the company IS at the business model level, not based on which topic names appear as keywords in the overview text.

Each topic is annotated as [category] or [feature] with a short description and a real-world usage count:
- [category] = industry sector, business model, or technology domain. ALWAYS prioritise these.
- [feature] = specific product, tool, or sub-domain capability. Only include if no [category] covers that area.
- (attached to N companies) = how many real PE deals in the Actum database use this topic.

USING THE COUNT AS A TIEBREAKER:
When two or more topics are equally supported by the company overview, prefer the one attached to more companies in the database. This reflects what experienced PE analysts actually apply in practice.
Example: if both "Cyber Security" (attached to 800 companies) and "Cyber Security Services" (attached to 50 companies) are both valid, pick "Cyber Security" — it is the standard label PE analysts use.
Example: if "BusTech" (attached to 1063 companies) and "Software Development" (attached to 200 companies) both apply, pick "BusTech" — it is more commonly applied to B2B technology companies.
Do NOT pick a topic purely because it has a high count if it is not genuinely supported by the company overview.
The count is a tiebreaker, not a replacement for evidence.

PE ANALYST MINDSET:
A PE analyst reading about a digital bank immediately tags: FinTech, Neobanking, Digital Banking, SaaS (if subscription-based), Banking Services.
A PE analyst reading about a B2B AI cybersecurity platform immediately tags: Cyber Security, Artificial Intelligence, BusTech, SaaS, Software Development, Security Solutions.
A PE analyst reading about a green energy supplier immediately tags: CleanTech, GreenTech, Renewable Energy, Clean Energy, EnergyTech, Energy Transition, Sustainable Energy.
A PE analyst reading about a digital health company offering AI-powered medical consultations, virtual primary care, and technology-enabled healthcare immediately tags: Digital Health, Telehealth, HealthTech, Preventive Care
These business-model-level labels must be selected EVEN IF those exact words never appear in the overview.

SELECTION RULES:
- [category] topics: valid to infer from business model even if exact words absent.
- [feature] topics: include when there is DIRECT, SPECIFIC evidence in the overview.
  Do NOT skip feature topics just because a broader category covers the same area —
  if the overview explicitly mentions the specific activity, include the specific topic.
  Example: if overview mentions "recruitment services", include "Recruitment Services" (feature)
  even if you already selected "HRTech" (category).
  Example: if overview mentions "disaster recovery", include "Disaster Recovery Services" (feature)
  even if you already selected "Software Solutions" (category).
- Near-synonym collapse: only collapse if two topics describe EXACTLY the same thing at the
  same level of specificity. Do not collapse a specific feature into a generic category.
- When in doubt between a specific and generic topic, prefer the specific one.
- Confidence over completeness: only submit what you are certain about.
STRICT COUNT: submit between {min_topics} and {max_topics} topics.
""".format(min_topics=MIN_TOPICS, max_topics=MAX_TOPICS)

# ---------------------------------------------------------------------------
# Pass 2: category verification — UNCHANGED from topic_agent.py for now
# (Step 4 pending)
# ---------------------------------------------------------------------------

def _run_category_verification_pass(
    company_name: str,
    structured_overview_text: str,
    matched_sector_ids: list[int],
    already_selected_ids: set[int],
) -> list[dict]:
    """
    Shows only unselected [category] topics to a plain LLM call and asks
    if any were missed. Returns additional topics to merge into pass 1.
    """
    from bedrock_client import call_claude_bedrock

    sector_set = set(matched_sector_ids)
    missed = []
    for tid_str, sids in _TOPIC_SECTORS.items():
        if not (sector_set & set(sids)):
            continue
        tid_int = int(tid_str)
        if tid_int not in _ALL_TOPICS:
            continue
        ttype = _TOPIC_TYPES.get(tid_str, "feature")
        count = _topic_count(tid_str)
        if ttype == "feature" and count < 50:
            continue  # skip low-count features, keep high-value ones
        if tid_int in already_selected_ids:
            continue
        desc  = _TOPIC_DESCRIPTIONS.get(tid_str, "")
        count = _topic_count(tid_str)
        missed.append((tid_int, _ALL_TOPICS[tid_int], desc, count))

    if not missed:
        return []

    # sort by count descending so high-frequency topics appear first
    missed.sort(key=lambda x: -x[3])

    topics_text = "\n".join(
        f"{tid} | {tname} (attached to {count} companies) — {desc}" if desc
        else f"{tid} | {tname} (attached to {count} companies)"
        for tid, tname, desc, count in missed
    )

    prompt = f"""You are a senior private equity analyst reviewing topic classification for {company_name}.

A first pass already selected some topics. Review the topics NOT yet selected below.
They include both CATEGORY-level and high-value FEATURE-level topics, sorted by real-world usage.

Company overview:
{structured_overview_text}

Unselected topics (sorted by usage count, highest first):
{topics_text}

For each topic: does it accurately describe {company_name}?
- For [category] topics: apply PE analyst reasoning, select if business model clearly implies it
- For [feature] topics: select ONLY if there is direct, specific evidence in the overview
  (e.g. "Recruitment Services" requires evidence the company provides recruitment services,
   not just that it operates in HR)
When two topics are equally supported, prefer the one with higher attached_company_count.
Be strict but thorough — the ground truth often includes specific feature-level topics.

Respond with ONLY a JSON array, no markdown fences:
[{{"topic_id": int, "topic_name": str, "evidence": str}}, ...]

If none should be added, respond with: []"""
    try:
        raw = call_claude_bedrock(prompt, max_tokens=1500, temperature=0)
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        additions = json.loads(raw[start:end])
        result = []
        for t in additions:
            try:
                tid = int(t["topic_id"])
            except (KeyError, ValueError):
                continue
            if tid not in _ALL_TOPICS:
                continue
            if tid in already_selected_ids:
                continue
            evidence = t.get("evidence", "").strip()
            if not evidence:
                continue
            result.append({
                "topic_id":   tid,
                "topic_name": _ALL_TOPICS[tid],
                "evidence":   evidence,
            })
        return result
    except Exception as e:
        print(f"[WARN] verification pass failed for {company_name}: {e}")
        return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_topic_agent(
    company_name: str,
    structured_overview_text: str,
    matched_sector_ids: list[int],
    llm_client,
) -> dict:
    """
    Runs the 3-pass topic agent for one company.
    Returns:
      {
        "matched_topics_flat": [{"topic_id": int, "topic_name": str, "evidence": str}, ...],
        "agent_status": "success" | "max_steps_exceeded",
        "steps_used": int,
      }
    """
    user_prompt = (
        f"Company: {company_name}\n"
        f"Matched sector ids: {matched_sector_ids}\n\n"
        f"Structured company overview:\n{structured_overview_text}\n\n"
        f"Now follow the workflow: call get_topics_for_sectors first, "
        f"then immediately call submit_topics with your final selections."
    )

    tools  = _build_tools()
    result: AgentResult = run_tool_agent(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tools=tools,
        llm_client=llm_client,
        final_tool_name="submit_topics",
        max_steps=MAX_STEPS,
    )

    if result.status != "success" or not result.payload:
        print(f"[WARN] {company_name}: topic agent ended with "
              f"status={result.status}, steps={result.steps_used}")
        return {
            "matched_topics_flat": [],
            "agent_status":        result.status,
            "steps_used":          result.steps_used,
        }

    # --- pass 1 results ---
    pass1_topics = result.payload
    pass1_ids    = {t["topic_id"] for t in pass1_topics}

    # --- pass 2: category verification ---
    additions = _run_category_verification_pass(
        company_name=company_name,
        structured_overview_text=structured_overview_text,
        matched_sector_ids=matched_sector_ids,
        already_selected_ids=pass1_ids,
    )

    # merge pass 1 + pass 2
    all_topics = pass1_topics[:]
    for t in additions:
        if len(all_topics) >= MAX_TOPICS:
            break
        if t["topic_id"] not in pass1_ids:
            all_topics.append(t)
            pass1_ids.add(t["topic_id"])

    # --- pass 3: taxonomy filter ---
    all_topics = _apply_taxonomy_filter(all_topics)

    return {
        "matched_topics_flat": all_topics,
        "agent_status":        "success",
        "steps_used":          result.steps_used,
    }