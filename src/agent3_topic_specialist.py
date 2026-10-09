import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bedrock_client_sonnet5 import call_sonnet5_with_tools
from loader import load_topics
from taxonomy_suggester import suggest_missing_topics

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOPIC_TYPES_PATH         = os.path.join(_BASE, "data", "topic_types.json")
TOPIC_METADATA_PATH      = os.path.join(_BASE, "data", "topic_metadata.json")
TOPICS_PATH               = os.path.join(_BASE, "data", "topics_formatted.txt")
TOPIC_SUBCATEGORIES_PATH  = os.path.join(_BASE, "data", "topic_subcategories.json")

MAX_TOPICS = 15
STRENGTH_FILTER_THRESHOLD  = 40   # only run the extra narrowing round if there are more candidates than this
STRENGTH_FILTER_BATCH_SIZE = 25   # how many candidates the AI reviews at once during narrowing

SELECT_THEMES_TOOL = {
    "name": "select_themes",
    "description": "Select every theme name that could plausibly apply to this company.",
    "input_schema": {
        "type": "object",
        "properties": {
            "selected_themes": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["selected_themes"],
    },
}

SELECT_TOPICS_TOOL = {
    "name": "select_topics",
    "description": "Submit the topics you are selecting, with evidence and reasoning for each.",
    "input_schema": {
        "type": "object",
        "properties": {
            "matched_topic_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "topic_reasoning": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic_id":   {"type": "integer"},
                        "topic_name": {"type": "string"},
                        "evidence":   {"type": "string"},
                        "reason":     {"type": "string"},
                    },
                    "required": ["topic_id", "topic_name", "evidence", "reason"],
                },
            },
        },
        "required": ["matched_topic_ids", "topic_reasoning"],
    },
}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_TOPIC_TYPES         = _load_json(TOPIC_TYPES_PATH)
_TOPIC_METADATA       = _load_json(TOPIC_METADATA_PATH) if os.path.exists(TOPIC_METADATA_PATH) else {}
_ALL_TOPICS           = {t.topic_id: t.topic_name for t in load_topics(TOPICS_PATH)}
_VALID_TOPIC_IDS      = set(_ALL_TOPICS.keys())
_TOPIC_SUBCATEGORIES  = _load_json(TOPIC_SUBCATEGORIES_PATH) if os.path.exists(TOPIC_SUBCATEGORIES_PATH) else {}

_TYPE_LOOKUP  = {int(tid): ttype for tid, ttype in _TOPIC_TYPES.items()}
_COUNT_LOOKUP = {
    int(tid): (meta.get("target_company_count") or meta.get("all_company_count", 0))
    for tid, meta in _TOPIC_METADATA.items()
}


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _get_theme_pool(sector_ids):
    themes = {}
    for sid in sector_ids:
        sector_groups = _TOPIC_SUBCATEGORIES.get(str(sid), {})
        for group_name, topic_ids in sector_groups.items():
            bucket = themes.setdefault(group_name, set())
            for tid in topic_ids:
                if tid in _VALID_TOPIC_IDS:
                    bucket.add(tid)
    return {name: sorted(ids) for name, ids in themes.items() if ids}


def _call_tool(prompt, tool, max_retries=3, max_tokens=3000):
    for attempt in range(1, max_retries + 1):
        try:
            response = call_sonnet5_with_tools(
                messages=[{"role": "user", "content": prompt}],
                tools=[tool],
                system="",
                max_tokens=max_tokens,
            )
            for block in response.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    return block.get("input", {})
            raise ValueError("No tool_use block returned by model")
        except Exception as e:
            if attempt < max_retries:
                print(f"      [WARN] call attempt {attempt} failed: {e}, retrying...")
                continue
            print(f"      [ERROR] call failed after {max_retries} attempts: {e}")
            return None
    return None


def _parse_topic_reasoning(parsed):
    results = []
    for item in _as_list(parsed.get("topic_reasoning", [])):
        if not isinstance(item, dict):
            continue
        try:
            tid = int(item["topic_id"])
        except (KeyError, ValueError, TypeError):
            continue
        if tid not in _VALID_TOPIC_IDS:
            continue
        evidence = str(item.get("evidence", "")).strip()
        if not evidence:
            continue
        results.append({
            "topic_id":   tid,
            "topic_name": _ALL_TOPICS[tid],
            "evidence":   evidence,
            "reason":     item.get("reason", ""),
        })
    return results


def _select_themes(company_name, company_profile, theme_pool):
    lines = [f"{name} ({len(ids)} topics)" for name, ids in theme_pool.items()]
    theme_text = "\n".join(lines)

    prompt = f"""You are a private equity taxonomy specialist choosing which THEMES to look at closely
for a company, before examining individual topics.

COMPANY: {company_name}

COMPANY PROFILE:
{company_profile}

CANDIDATE THEMES (name and how many topics each contains):
{theme_text}

Select every theme that could plausibly relate to this company's identity or activities — this is
a coarse first pass, not a final answer. Be generous: a theme you don't select here is never looked
at again, but selecting a theme that turns out irrelevant just costs one extra (cheap) look. When
in doubt, select it.

Call select_themes with every theme name (copied EXACTLY as shown above) that applies."""

    parsed = _call_tool(prompt, SELECT_THEMES_TOOL, max_tokens=2000)
    if not parsed:
        return list(theme_pool.keys())

    raw_selected = _as_list(parsed.get("selected_themes", []))
    selected = [name for name in raw_selected if isinstance(name, str) and name in theme_pool]
    return selected if selected else list(theme_pool.keys())


def _scan_theme(company_name, company_profile, theme_name, topic_ids):
    lines = [f"{tid} | {_ALL_TOPICS[tid]}" for tid in topic_ids if tid in _ALL_TOPICS]
    topics_text = "\n".join(lines)

    prompt = f"""You are a private equity taxonomy specialist reviewing ONE THEME of topics for {company_name}.

COMPANY PROFILE:
{company_profile}

THEME: {theme_name}
TOPICS IN THIS THEME (id | name):
{topics_text}

Flag every topic in this theme that plausibly applies to the company, based on direct evidence or a
reasonable inference from the profile. Extra flags cost nothing, but a topic you skip here is gone
for good. If several topics here describe overlapping or near-synonymous aspects of the company —
a broad term and a more specific one that both fit — flag ALL of them, don't pick just one. If
nothing in this theme applies, submit an empty list.

Call select_topics with everything from this theme that applies."""

    parsed = _call_tool(prompt, SELECT_TOPICS_TOOL, max_tokens=2000)
    if not parsed:
        return []
    return _parse_topic_reasoning(parsed)


def _strength_filter_batch(company_name, company_profile, batch):
    lines = []
    for c in batch:
        lines.append(
            f"{c['topic_id']} | {c['topic_name']} [{c.get('ttype', 'feature')}] "
            f"(used by {c.get('count', 0)} companies) — evidence: {c['evidence']}"
        )
    cand_text = "\n".join(lines)

    prompt = f"""You are a private equity taxonomy specialist reviewing a batch of already-flagged
candidate topics for {company_name}, deciding which are genuinely STRONG matches worth carrying
forward to the final shortlist.

COMPANY PROFILE:
{company_profile}

CANDIDATE TOPICS IN THIS BATCH, each with the evidence already found for it:
{cand_text}

Unlike the earlier scanning pass, this step is about QUALITY, not just plausibility. Keep only
topics that are clearly, confidently supported by the evidence — the company's core identity or
what it specifically does. Discard topics that are only loosely, weakly, or speculatively related.
When two topics in this batch describe the same aspect of the company, keep the one with stronger,
more specific evidence AND the one with a clearly distinct meaning — don't discard a broad,
obviously-correct term just because a narrower one is also present.

Call select_topics with only the topics from this batch that are strong, well-evidenced matches."""

    parsed = _call_tool(prompt, SELECT_TOPICS_TOOL, max_tokens=2000)
    if not parsed:
        return batch  # fail-safe: keep the batch as-is rather than losing it
    survivors = _parse_topic_reasoning(parsed)
    if not survivors:
        return batch  # model returned nothing usable — don't let a bad call erase real candidates
    survivor_ids = {s["topic_id"] for s in survivors}
    # keep full candidate dicts (with ttype/count) for anything that survived
    return [c for c in batch if c["topic_id"] in survivor_ids]


def _finalize_pick(company_name, company_profile, candidates):
    cand_lines = []
    for c in candidates:
        cand_lines.append(
            f"{c['topic_id']} | {c['topic_name']} [{c.get('ttype', 'feature')}] "
            f"(used by {c.get('count', 0)} companies) — evidence: {c['evidence']}"
        )
    cand_text = "\n".join(cand_lines)

    prompt = f"""You are a private equity taxonomy specialist finalizing the topic selection for {company_name}.

COMPANY PROFILE:
{company_profile}

SHORTLISTED TOPICS, each with the evidence already found for it:
{cand_text}

From this shortlist, select the final set of up to {MAX_TOPICS} topics. Rules:

RULE 1 — BOTH IDENTITY AND ACTIVITY:
Pick topics that describe what the company IS (identity) AND what it specifically DOES (activity).

RULE 2 — KEEP OVERLAPPING/NEAR-SYNONYM TOPICS WHEN SLOTS ALLOW:
If two topics describe the same or overlapping aspect of the company, prefer keeping both over
picking only one — but if you must cut to fit {MAX_TOPICS}, prefer the topic with stronger evidence
and higher usage count over a weaker near-duplicate.

RULE 3 — USAGE COUNT IS A REAL SIGNAL, NOT JUST A TIEBREAKER:
Topics used by many companies are more likely to be the standard label PE analysts actually apply.
When multiple topics are similarly well-evidenced, prefer the more commonly-used one.

RULE 4 — MATCH THE COMPANY'S OWN WORDING:
When two or more candidate topics describe the same underlying concept, prefer whichever topic's
NAME most closely echoes the exact words or phrases the company profile itself uses. For example,
if the profile says "SaaS platform", prefer a topic literally named "SaaS" over a broader
alternative like "Cloud Solutions", even if both are reasonably evidenced — the company's own
language is the strongest signal for which specific label is the intended fit.

RULE 5 — SELECT UP TO {MAX_TOPICS} TOPICS.
Use the full budget if that many genuinely apply.

Call select_topics with your final selection (reuse the evidence already given, don't rewrite it)."""

    parsed = _call_tool(prompt, SELECT_TOPICS_TOOL, max_tokens=2000)
    if not parsed:
        return candidates[:MAX_TOPICS]

    final = []
    seen = set()
    for item in _parse_topic_reasoning(parsed):
        tid = item["topic_id"]
        if tid in seen:
            continue
        seen.add(tid)
        final.append({"topic_id": tid, "topic_name": item["topic_name"], "evidence": item["evidence"]})
        if len(final) >= MAX_TOPICS:
            break

    return final if final else candidates[:MAX_TOPICS]


def _finalize_twice(company_name, company_profile, candidates):
    pick1 = _finalize_pick(company_name, company_profile, candidates)
    pick2 = _finalize_pick(company_name, company_profile, candidates)
    merged = {}
    for t in pick1 + pick2:
        merged.setdefault(t["topic_id"], t)
    return list(merged.values())


def _expand_synonyms(company_name, company_profile, selected, candidates):
    selected_ids = {t["topic_id"] for t in selected}
    remaining = [c for c in candidates if c["topic_id"] not in selected_ids]
    if not remaining:
        return selected

    selected_text  = "\n".join(f"{t['topic_id']} | {t['topic_name']}" for t in selected)
    remaining_text = "\n".join(f"{c['topic_id']} | {c['topic_name']} — evidence: {c['evidence']}" for c in remaining)

    prompt = f"""Review this final topic selection for {company_name} for missed synonyms.

COMPANY PROFILE:
{company_profile}

ALREADY SELECTED:
{selected_text}

OTHER FLAGGED CANDIDATES (not yet selected):
{remaining_text}

For each already-selected topic, check the other candidates: is there a close synonym or
near-duplicate concept that was flagged but left out? If so, add it — extra topics cost nothing
here. Only add genuine synonyms/near-duplicates of an already-selected topic, nothing unrelated.

Call select_topics with the FULL final list: everything already selected PLUS any synonyms added."""

    parsed = _call_tool(prompt, SELECT_TOPICS_TOOL, max_tokens=2000)
    if not parsed:
        return selected

    expanded = _parse_topic_reasoning(parsed)
    merged = {t["topic_id"]: t for t in selected}
    for e in expanded:
        merged.setdefault(e["topic_id"], {"topic_id": e["topic_id"], "topic_name": e["topic_name"], "evidence": e["evidence"]})
    return list(merged.values())


def run_topic_specialist(
    company_id: int,
    company_name: str,
    company_profile: str,
    sector_ids: list,
    max_retries: int = 3,
) -> dict:
    theme_pool = _get_theme_pool(sector_ids)

    if not theme_pool:
        print(f"    [Agent 3] No themes found for sectors {sector_ids}")
        selected_topics, topic_reasoning, pre_finalize_ids = [], [], []
    else:
        print(f"    [Agent 3] {len(theme_pool)} candidate themes available across sectors {sector_ids}")
        selected_theme_names = _select_themes(company_name, company_profile, theme_pool)
        print(f"    [Agent 3] Selected {len(selected_theme_names)}/{len(theme_pool)} themes to scan closely")

        flagged = {}
        for i, theme_name in enumerate(selected_theme_names, 1):
            topic_ids = theme_pool[theme_name]
            results = _scan_theme(company_name, company_profile, theme_name, topic_ids)
            for r in results:
                if r["topic_id"] not in flagged:
                    flagged[r["topic_id"]] = r
            if results:
                print(f"      theme {i}/{len(selected_theme_names)} '{theme_name}': flagged {len(results)}")

        candidates = []
        for tid, r in flagged.items():
            candidates.append({
                **r,
                "ttype": _TYPE_LOOKUP.get(tid, "feature"),
                "count": _COUNT_LOOKUP.get(tid, 0),
            })

        pre_finalize_ids = [c["topic_id"] for c in candidates]
        print(f"    [Agent 3] {len(candidates)} unique topics flagged across all selected themes.")

        if len(candidates) <= MAX_TOPICS:
            selected_topics = [
                {"topic_id": c["topic_id"], "topic_name": c["topic_name"], "evidence": c["evidence"]}
                for c in candidates
            ]
        else:
            selected_topics = _finalize_twice(company_name, company_profile, candidates)
            selected_topics = _expand_synonyms(company_name, company_profile, selected_topics, candidates)

        reason_lookup   = {c["topic_id"]: c.get("reason", "") for c in candidates}
        topic_reasoning = [
            {**t, "reason": reason_lookup.get(t["topic_id"], "")}
            for t in selected_topics
        ]

    suggestions = suggest_missing_topics(
        company_id=company_id,
        company_name=company_name,
        structured_overview_text=company_profile,
        matched_topics=selected_topics,
    )

    return {
        "matched_topics_flat":   selected_topics,
        "topic_reasoning":       topic_reasoning,
        "taxonomy_suggestions":  suggestions,
        "pre_finalize_topic_ids": pre_finalize_ids,  # diagnostic: everything flagged before narrowing
    }