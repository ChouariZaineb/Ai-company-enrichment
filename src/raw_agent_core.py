# src/raw_agent_core.py
"""
Shared tool-loop agent for the raw-text pipeline.
Takes pure scraped text, matches sectors, gets scoped topic pool,
selects topics, generates description — all in one agent loop.

Tools:
  get_sectors(scraped_text)          → sector_ids from the 38-sector taxonomy
  get_topics_for_sectors(sector_ids) → scoped topic pool from topic_sectors.json
  submit_final(...)                  → final answer, exits the loop

MAX_STEPS = 8, expected steps = 3.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loader import build_taxonomy, load_topics
from agent_core import ToolSpec, run_tool_agent, AgentResult

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOPIC_SECTORS_PATH = os.path.join(_BASE, "data", "topic_sectors.json")
TOPICS_PATH        = os.path.join(_BASE, "data", "topics_formatted.txt")

MAX_STEPS   = 8
MAX_SECTORS = 3
MAX_TOPICS  = 10

TAXONOMY = build_taxonomy(
    sectors_path=os.path.join(_BASE, "data", "sectors_formatted.txt"),
    topics_path=TOPICS_PATH,
)

# load once
def _load_topic_sectors() -> dict[str, list[int]]:
    with open(TOPIC_SECTORS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_all_topics() -> dict[int, str]:
    return {t.topic_id: t.topic_name for t in load_topics(TOPICS_PATH)}

_TOPIC_SECTORS = _load_topic_sectors()
_ALL_TOPICS    = _load_all_topics()

VALID_SECTOR_IDS = TAXONOMY["valid_sector_ids"]
VALID_TOPIC_IDS  = {t.topic_id for t in TAXONOMY["topics"]}
SECTORS_BLOCK    = TAXONOMY["sectors_prompt_text"]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _get_sectors(scraped_text: str) -> str:
    """
    Returns the full sector taxonomy for the agent to pick from.
    The scraped_text argument is accepted but not used server-side —
    the agent already has the text in its context and uses this tool
    to access the official sector list to pick from.
    """
    return (
        f"Official sector taxonomy ({len(TAXONOMY['sectors'])} sectors).\n"
        f"Pick at most {MAX_SECTORS} that describe what this company BUILDS or DOES.\n\n"
        f"{SECTORS_BLOCK}"
    )


def _get_topics_for_sectors(sector_ids: list[int]) -> str:
    """
    Returns the topic pool scoped to the matched sectors.
    Plain id | name — no annotations, no metadata.
    """
    sector_set = set(sector_ids)
    matched = []
    for tid_str, sids in _TOPIC_SECTORS.items():
        if not (sector_set & set(sids)):
            continue
        tid_int = int(tid_str)
        if tid_int not in _ALL_TOPICS:
            continue
        matched.append((tid_int, _ALL_TOPICS[tid_int]))

    matched.sort(key=lambda x: x[0])

    if not matched:
        return "No topics found for those sector ids."

    lines = [f"{tid} | {tname}" for tid, tname in matched]
    return (
        f"{len(matched)} candidate topics for sectors {sector_ids}:\n"
        + "\n".join(lines)
    )


def _validate_submit(
    sector_ids: list[int],
    topics: list[dict],
    company_description: str,
) -> dict:
    """Validates and returns the final result."""
    clean_sectors = sorted(
        {int(s) for s in sector_ids if int(s) in VALID_SECTOR_IDS}
    )[:MAX_SECTORS]

    clean_topics = []
    seen = set()
    for t in topics:
        try:
            tid = int(t["topic_id"])
        except (KeyError, ValueError, TypeError):
            continue
        if tid not in VALID_TOPIC_IDS or tid in seen:
            continue
        evidence = t.get("evidence", "").strip()
        if not evidence:
            continue
        seen.add(tid)
        clean_topics.append({
            "topic_id":   tid,
            "topic_name": _ALL_TOPICS.get(tid, "?"),
            "evidence":   evidence,
        })
        if len(clean_topics) >= MAX_TOPICS:
            break

    return {
        "matched_sector_ids":  clean_sectors,
        "matched_topics_flat": clean_topics,
        "company_description": company_description.strip(),
    }


# ---------------------------------------------------------------------------
# Tool specs
# ---------------------------------------------------------------------------

def _build_tools(scraped_text: str) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="get_sectors",
            description=(
                "Call this FIRST. Returns the official 38-sector taxonomy. "
                "Read the scraped text in your context and select the sectors "
                f"that describe what this company builds or does (max {MAX_SECTORS})."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "scraped_text_summary": {
                        "type": "string",
                        "description": "One sentence summarising what the company does, based on the scraped text.",
                    }
                },
                "required": ["scraped_text_summary"],
            },
            handler=lambda scraped_text_summary: _get_sectors(scraped_text_summary),
        ),
        ToolSpec(
            name="get_topics_for_sectors",
            description=(
                "Call this SECOND, after get_sectors. "
                "Returns all taxonomy topics mapped to the matched sectors. "
                "Use this pool to select the most relevant topics for the company."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sector_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "The sector ids you selected in the previous step.",
                    }
                },
                "required": ["sector_ids"],
            },
            handler=lambda sector_ids: _get_topics_for_sectors(sector_ids),
        ),
        ToolSpec(
            name="submit_final",
            description=(
                f"Call this LAST. Submit your final sector ids, topic selections, "
                f"and a 2-paragraph company description. "
                f"Max {MAX_SECTORS} sectors, max {MAX_TOPICS} topics. "
                "Every topic must have an evidence quote from the scraped text."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sector_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": f"Final sector ids (max {MAX_SECTORS}).",
                    },
                    "topics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic_id":   {"type": "integer"},
                                "topic_name": {"type": "string"},
                                "evidence":   {
                                    "type": "string",
                                    "description": "Direct quote from the scraped text justifying this topic.",
                                },
                            },
                            "required": ["topic_id", "topic_name", "evidence"],
                        },
                        "description": f"Final topic selections (max {MAX_TOPICS}).",
                    },
                    "company_description": {
                        "type": "string",
                        "description": "2-paragraph description of what the company does, who it serves, and key differentiators.",
                    },
                },
                "required": ["sector_ids", "topics", "company_description"],
            },
            handler=lambda sector_ids, topics, company_description: _validate_submit(
                sector_ids, topics, company_description
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are acting simultaneously as:
- A forensic business researcher who reads website text critically and extracts only what is explicitly stated
- A data analyst who maps evidence to structured taxonomies with precision
- A private equity deal analyst who understands how PE firms classify companies for investment screening

Your task: classify a company from its raw scraped website text by selecting the correct sectors and topics from official taxonomies.

STRICT WORKFLOW — exactly 3 tool calls in this order:
STEP 1: Call get_sectors
  Read the scraped text carefully. Identify what this company BUILDS, SELLS, or OPERATES — not who buys from it.
  Select the sectors (max {MAX_SECTORS}) that best describe the company's own activities.

STEP 2: Call get_topics_for_sectors with your selected sector ids
  This returns the candidate topic pool scoped to your sectors.
  Study this pool carefully before making your selections.

STEP 3: Call submit_final
  Submit your sector ids, topic selections, and a 2-paragraph company description.

Do NOT skip steps. Do NOT call any tool more than once.

═══ SECTOR SELECTION RULES ═══
INCLUDE a sector only if the company's own products/technology/operations belong to it.
EXCLUDE sectors that only describe the company's customer base or target market.

Evidence test: ask "Does the company BUILD this, or just SELL TO people in this field?"
  ✓ INCLUDE: a cybersecurity software company → "Computing, Technology, Robotics & AI" + "Defence, Protection and Security"
  ✗ EXCLUDE: the same company serving banks → do NOT include "Financial Services" — banks are customers, not what the company does
  ✓ INCLUDE: a recruitment SaaS → "Computing, Technology, Robotics & AI" + "Professional, Business & Support Services"
  ✓ INCLUDE: an ice cream machinery manufacturer → "Manufacturing and Product Development" + "Food and Beverage"

═══ TOPIC SELECTION RULES ═══
After receiving the candidate pool in Step 2, apply these rules:

RULE 1 — SELECT BOTH IDENTITY AND ACTIVITY TOPICS:
  Identity topics describe what the company IS (e.g. "Software Solutions", "HealthTech", "SaaS")
  Activity topics describe what the company specifically DOES (e.g. "Recruitment Services", "Manufacturing Analytics", "Pharmaceutical Packaging")
  Select BOTH types when evidenced — do not only pick one or the other.

RULE 2 — SPECIFICITY OVER GENERALITY:
  When two topics describe the same concept at different levels of specificity, always pick the MORE SPECIFIC one.
  Examples:
    "Recruitment Services" > "Services" — pick Recruitment Services
    "Manufacturing Analytics" > "Analytics" — pick Manufacturing Analytics
    "Pharmaceutical Packaging" > "Packaging" — pick Pharmaceutical Packaging
    "Geospatial Solutions" > "Software Solutions" — pick Geospatial Solutions if directly evidenced
  Exception: keep a generic topic only if it adds a distinct dimension not covered by the specific one.

RULE 3 — DIRECT EVIDENCE REQUIRED:
  Every topic must be justified by a direct quote or close paraphrase from the scraped text.
  Do not infer topics from general industry knowledge — only from what the text actually says.

RULE 4 — NEAR-SYNONYM COLLAPSE:
  If two topics in the pool mean essentially the same thing, keep only the more specific one.
  Example: "Recruitment Solutions" and "Recruitment Services" — if both apply, keep both (they are distinct: one is the product, one is the service). But "Analytics" and "Data Analytics" — keep "Data Analytics".

RULE 5 — DO NOT PAD:
  Only submit topics you are fully confident about. 5 precise topics beat 10 uncertain ones.
  Max {MAX_TOPICS} topics — but fewer is fine if not all are evidenced.

═══ DESCRIPTION RULES ═══
2 short paragraphs. Cover: what the company does, who it serves, key products/services, notable differentiators.
Based strictly on the scraped text — do not add information not present in the source material."""

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_raw_agent(
    company_name: str,
    raw_scraped_text: str,
    llm_client,
    max_content_chars: int = 30000,
) -> dict:
    """
    Runs the 3-step raw-text tool-loop agent.
    Returns the validated final result dict.
    """
    truncated_text = raw_scraped_text[:max_content_chars]

    user_prompt = (
        f"Company: {company_name}\n\n"
        f"Raw scraped website text:\n{truncated_text}\n\n"
        f"Now follow the 3-step workflow: "
        f"call get_sectors first, then get_topics_for_sectors, then submit_final."
    )

    tools  = _build_tools(truncated_text)
    result: AgentResult = run_tool_agent(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tools=tools,
        llm_client=llm_client,
        final_tool_name="submit_final",
        max_steps=MAX_STEPS,
    )

    if result.status != "success" or not result.payload:
        print(f"  [WARN] {company_name}: agent ended with status={result.status}, steps={result.steps_used}")
        return {
            "matched_sector_ids":  [],
            "matched_topics_flat": [],
            "company_description": "",
            "agent_status":        result.status,
            "steps_used":          result.steps_used,
        }

    payload = result.payload
    payload["agent_status"] = "success"
    payload["steps_used"]   = result.steps_used
    return payload