"""
build_topic_subcategories.py

ONE-TIME batch job: organizes every sector's topics into small, named
thematic groups (e.g. under "Agriculture": "Agricultural Inputs",
"AgTech & Agribusiness", "Crop & Farm Production"...), automatically,
using the AI instead of by hand.

Saves the result to data/topic_subcategories.json:
    { "<sector_id>": { "<group name>": [topic_id, topic_id, ...], ... }, ... }

Safe to interrupt and re-run: already-completed sectors are skipped unless
listed in FORCE_SECTOR_IDS below (use that to redo just the sectors that
came out badly, without re-doing the ones that already worked).

Run from the project root:
    python build_topic_subcategories.py
"""
import os
import sys
import json
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))

from loader import load_topics
from bedrock_client_sonnet5 import call_sonnet5_with_tools

TOPIC_SECTORS_PATH = os.path.join(BASE, "data", "topic_sectors.json")
SECTORS_PATH        = os.path.join(BASE, "data", "sectors_formatted.txt")
TOPICS_PATH          = os.path.join(BASE, "data", "topics_formatted.txt")
OUTPUT_PATH          = os.path.join(BASE, "data", "topic_subcategories.json")
SUMMARY_PATH         = os.path.join(BASE, "data", "topic_subcategories_summary.txt")

BATCH_SIZE       = 80    # topics per grouping call (reduced so answers never get cut off)
PROPOSE_MAX_TOKENS = 6000
MERGE_MAX_TOKENS   = 6000
OTHER_WARNING_RATIO = 0.15  # flag a sector if more than 15% of its topics end up "Other"

# Redo just these sectors (known to have come out badly last time). Leave empty
# once you've re-run and they look good, or add more ids if new problems show up.
FORCE_SECTOR_IDS = {10, 14, 19}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_sector_names(path):
    names = {}
    if not os.path.exists(path):
        return names
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            tid_str, name = line.split("|", 1)
            try:
                names[int(tid_str.strip())] = name.strip()
            except ValueError:
                continue
    print(f"[DEBUG] Loaded {len(names)} sector names from {os.path.basename(path)}")
    return names


def _to_int(value):
    """Coerce a topic id to int whether the model wrote it as a number or as text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


PROPOSE_GROUPS_TOOL = {
    "name": "propose_groups",
    "description": "Propose named thematic groups covering every topic given.",
    "input_schema": {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group_name": {"type": "string"},
                        "topic_ids": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["group_name", "topic_ids"],
                },
            },
        },
        "required": ["groups"],
    },
}

MERGE_GROUPS_TOOL = {
    "name": "merge_groups",
    "description": "Map each raw group name to its final consolidated group name.",
    "input_schema": {
        "type": "object",
        "properties": {
            "mapping": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "raw_group_name":   {"type": "string"},
                        "final_group_name": {"type": "string"},
                    },
                    "required": ["raw_group_name", "final_group_name"],
                },
            },
        },
        "required": ["mapping"],
    },
}


def _call_tool(prompt, tool, max_retries=3, max_tokens=6000):
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
                wait = 2 * attempt
                print(f"    [WARN] attempt {attempt} failed: {e}, waiting {wait}s and retrying...")
                time.sleep(wait)
                continue
            print(f"    [ERROR] failed after {max_retries} attempts: {e}")
            return None
    return None


def _propose_groups_for_batch(sector_name, batch, valid_ids):
    lines = [f"{tid} | {name}" for tid, name in batch]
    topics_text = "\n".join(lines)

    prompt = f"""You are organizing a taxonomy of business topics into clear, named thematic groups.

SECTOR: {sector_name}

TOPICS TO ORGANIZE (id | name):
{topics_text}

Group these topics into named thematic sub-categories, each covering a coherent theme within this
sector (for example, under "Agriculture" you might use groups like "Agricultural Inputs", "AgTech
& Agribusiness", "Crop & Farm Production", "Forestry & Horticulture", "Livestock").

Rules:
- Every topic listed above must be assigned to exactly one group.
- Aim for roughly 8-15 topics per group — fewer or more is fine if a theme genuinely needs it.
- Give each group a short, clear, human-readable name.
- Group by theme/subject matter, not by how specific or generic a topic sounds.
- Keep your answer efficient: short group names, no extra commentary outside the tool call.

Call propose_groups with your full set of groups covering every topic listed above."""

    parsed = _call_tool(prompt, PROPOSE_GROUPS_TOOL, max_tokens=PROPOSE_MAX_TOKENS)
    if not parsed:
        return []

    raw_list = parsed.get("groups", [])
    if isinstance(raw_list, str):
        # the model sometimes returns this field as a JSON-encoded string instead of
        # an actual array — parse it properly instead of iterating it char by char
        try:
            raw_list = json.loads(raw_list)
        except (json.JSONDecodeError, TypeError):
            print(f"    [WARN] 'groups' came back as an unparsable string, skipping this batch")
            raw_list = []
        if not isinstance(raw_list, list):
            raw_list = []

    groups = []
    for g in raw_list:
        if not isinstance(g, dict):
            continue
        name = str(g.get("group_name", "")).strip()
        ids = []
        for tid in g.get("topic_ids", []):
            n = _to_int(tid)
            if n is not None and n in valid_ids:
                ids.append(n)
        if name and ids:
            groups.append((name, ids))

    if raw_list and not groups:
        print(f"    [WARN] model returned {len(raw_list)} raw group(s) but none survived validation — sample: {raw_list[0]}")

    return groups


def _consolidate_groups(sector_name, raw_groups):
    if len(raw_groups) <= 1:
        return {name: ids for name, ids in raw_groups}

    summary_lines = [f"{name} ({len(ids)} topics)" for name, ids in raw_groups]
    summary_text  = "\n".join(summary_lines)

    prompt = f"""You previously organized topics for the "{sector_name}" sector into named groups,
created in separate independent batches. Here are ALL the group names proposed, with how many
topics each has:

{summary_text}

Many of these represent the SAME theme (near-duplicate or overlapping names) since they were
created independently in different batches — for example "HR & Workforce Management", "Employee
Experience & HR Operations", and "Workforce & HR Management" are almost certainly the same theme
and should become ONE final group. MERGE AGGRESSIVELY: if in doubt whether two names are the same
theme, merge them. Only keep names separate when they are genuinely different subjects. Aim to
reduce the total number of distinct final names substantially — a sector like this should end up
with well under half as many final groups as raw names listed above.

Call merge_groups: for EVERY raw group name above, state which final group name it should map to
(reuse one of the existing names as the final name — don't invent a brand new one)."""

    parsed = _call_tool(prompt, MERGE_GROUPS_TOOL, max_tokens=MERGE_MAX_TOKENS)
    if not parsed:
        return {name: ids for name, ids in raw_groups}

    mapping_list = parsed.get("mapping", [])
    if isinstance(mapping_list, str):
        try:
            mapping_list = json.loads(mapping_list)
        except (json.JSONDecodeError, TypeError):
            mapping_list = []
        if not isinstance(mapping_list, list):
            mapping_list = []

    name_to_final = {}
    for m in mapping_list:
        if not isinstance(m, dict):
            continue
        raw   = str(m.get("raw_group_name", "")).strip()
        final = str(m.get("final_group_name", "")).strip()
        if raw and final:
            name_to_final[raw] = final

    merged = {}
    for name, ids in raw_groups:
        final_name = name_to_final.get(name, name)
        merged.setdefault(final_name, [])
        for tid in ids:
            if tid not in merged[final_name]:
                merged[final_name].append(tid)
    return merged


def main():
    topic_sectors = _load_json(TOPIC_SECTORS_PATH)
    sector_names  = _load_sector_names(SECTORS_PATH)
    all_topics    = {t.topic_id: t.topic_name for t in load_topics(TOPICS_PATH)}
    valid_ids     = set(all_topics.keys())
    print(f"[DEBUG] Loaded {len(all_topics)} topic names, {len(topic_sectors)} topic-sector mappings")

    sector_to_topics = {}
    for tid_str, sids in topic_sectors.items():
        tid = int(tid_str)
        if tid not in valid_ids:
            continue
        for sid in sids:
            sector_to_topics.setdefault(sid, []).append((tid, all_topics[tid]))

    output = _load_json(OUTPUT_PATH) if os.path.exists(OUTPUT_PATH) else {}

    for sector_id in sorted(sector_to_topics.keys()):
        already_done = str(sector_id) in output
        if already_done and sector_id not in FORCE_SECTOR_IDS:
            print(f"[SKIP] Sector {sector_id} already done ({len(output[str(sector_id)])} groups)")
            continue

        sector_name = sector_names.get(sector_id, f"Sector {sector_id}")
        topics      = sorted(sector_to_topics[sector_id], key=lambda x: x[1])
        batches     = [topics[i:i + BATCH_SIZE] for i in range(0, len(topics), BATCH_SIZE)]

        tag = "[REDO]" if already_done else ""
        print(f"\n{tag}[{sector_id}] {sector_name} — {len(topics)} topics, {len(batches)} batch(es)")

        raw_groups = []
        for i, batch in enumerate(batches, 1):
            print(f"  batch {i}/{len(batches)} ({len(batch)} topics)...")
            groups = _propose_groups_for_batch(sector_name, batch, valid_ids)
            raw_groups.extend(groups)
            print(f"    -> {len(groups)} groups proposed")

        final_groups = _consolidate_groups(sector_name, raw_groups)

        covered = {tid for ids in final_groups.values() for tid in ids}
        all_ids = {tid for tid, _ in topics}
        missing = sorted(all_ids - covered)
        if missing:
            final_groups.setdefault("Other / Uncategorized", [])
            final_groups["Other / Uncategorized"].extend(missing)
            print(f"  [WARN] {len(missing)} topic(s) not placed by the model — added to 'Other / Uncategorized'")

        print(f"  FINAL: {len(final_groups)} groups for {sector_name}")

        output[str(sector_id)] = final_groups

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    # readable summary + a flagged list of sectors that still need attention
    needs_attention = []
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        for sector_id_str, groups in output.items():
            sector_name = sector_names.get(int(sector_id_str), f"Sector {sector_id_str}")
            total = sum(len(ids) for ids in groups.values())
            other = len(groups.get("Other / Uncategorized", []))
            ratio = (other / total) if total else 0
            f.write(f"\n=== [{sector_id_str}] {sector_name} — {len(groups)} groups ===\n")
            for group_name, ids in groups.items():
                f.write(f"  {group_name} ({len(ids)} topics)\n")
            if ratio > OTHER_WARNING_RATIO:
                needs_attention.append((sector_id_str, sector_name, other, total, round(ratio * 100)))

    print(f"\nDone. Saved to {OUTPUT_PATH}")
    print(f"Readable summary saved to {SUMMARY_PATH}")

    if needs_attention:
        print(f"\n[ATTENTION] {len(needs_attention)} sector(s) still have a lot in 'Other / Uncategorized':")
        for sid, name, other, total, pct in needs_attention:
            print(f"  [{sid}] {name}: {other}/{total} topics ({pct}%) uncategorized")
    else:
        print("\nAll sectors look clean — no large 'Other / Uncategorized' buckets left.")


if __name__ == "__main__":
    main()