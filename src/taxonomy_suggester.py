# src/taxonomy_suggester.py
"""
After topic matching, suggests new taxonomy terms for concepts clearly present
in the company overview but missing from the existing taxonomy.
Logs to data/taxonomy_suggestions.json — does NOT modify the live taxonomy.
The supervisor reviews the log and decides which suggestions to promote.
"""
import json
import os
from datetime import datetime
from bedrock_client import call_claude_bedrock

_BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUGGESTIONS_PATH = os.path.join(_BASE, "data", "taxonomy_suggestions.json")


def _load_suggestions() -> list:
    if os.path.exists(SUGGESTIONS_PATH):
        with open(SUGGESTIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_suggestions(suggestions: list):
    with open(SUGGESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, ensure_ascii=False, indent=2)


def suggest_missing_topics(
    company_id: int,
    company_name: str,
    structured_overview_text: str,
    matched_topics: list[dict],
    max_retries: int = 3,
) -> list[dict]:
    """
    Asks the LLM: given what we already selected, are there important
    concepts in the overview that have NO good match in the taxonomy?
    Returns a list of suggestions and logs them to taxonomy_suggestions.json.
    Each suggestion: {suggested_term, description, evidence, confidence}
    """
    already_selected = "\n".join(
        f"- {t['topic_name']}" for t in matched_topics
    ) or "None"

    prompt = f"""You are a private equity taxonomy analyst reviewing a company classification.

Company: {company_name}

Structured overview:
{structured_overview_text}

Topics already selected from the existing taxonomy:
{already_selected}

Your task: identify concepts that are CLEARLY AND SPECIFICALLY relevant to this company
but are NOT adequately covered by the already-selected topics above.

These are gaps in the taxonomy — important business concepts, technologies, or market
categories that a PE analyst would want to tag this company with, but that don't exist
in the current topic list.

Rules:
- Only suggest terms that are genuinely missing — not near-synonyms of already-selected topics
- Each suggestion must have a direct evidence quote from the overview above
- Be specific: "Ice Cream Equipment Manufacturing" is better than "Manufacturing"
- Suggest between 0 and 5 terms. If the existing topics already cover everything well, suggest 0.
- Rate your confidence: HIGH (clearly missing and important), MEDIUM (useful but borderline), LOW (marginal)

Respond with ONLY a JSON array, no markdown fences:
[
  {{
    "suggested_term": "short topic name (2-5 words)",
    "description": "one sentence describing what this topic covers",
    "evidence": "direct quote or close paraphrase from the overview justifying this",
    "confidence": "HIGH|MEDIUM|LOW"
  }}
]

If nothing is missing, respond with: []"""

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_claude_bedrock(prompt, max_tokens=1000, temperature=0)
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return []

            suggestions = json.loads(raw[start:end])

            # validate and clean
            valid = []
            for s in suggestions:
                term = s.get("suggested_term", "").strip()
                desc = s.get("description", "").strip()
                evid = s.get("evidence", "").strip()
                conf = s.get("confidence", "MEDIUM").strip().upper()
                if not term or not evid:
                    continue
                if conf not in ("HIGH", "MEDIUM", "LOW"):
                    conf = "MEDIUM"
                valid.append({
                    "suggested_term": term,
                    "description":    desc,
                    "evidence":       evid,
                    "confidence":     conf,
                })

            if valid:
                _log_suggestions(company_id, company_name, valid)

            return valid

        except Exception as e:
            if attempt < max_retries:
                print(f"[WARN] taxonomy suggester attempt {attempt} failed for {company_name}: {e}")
                continue
            print(f"[ERROR] taxonomy suggester failed for {company_name}: {e}")
            return []

    return []


def _log_suggestions(company_id: int, company_name: str, suggestions: list[dict]):
    """Appends new suggestions to the log file, skipping exact duplicates."""
    existing   = _load_suggestions()
    existing_terms = {
        (e["company_id"], e["suggested_term"].lower())
        for e in existing
    }

    new_entries = []
    for s in suggestions:
        key = (company_id, s["suggested_term"].lower())
        if key in existing_terms:
            continue
        new_entries.append({
            "company_id":     company_id,
            "company_name":   company_name,
            "suggested_term": s["suggested_term"],
            "description":    s["description"],
            "evidence":       s["evidence"],
            "confidence":     s["confidence"],
            "logged_at":      datetime.utcnow().isoformat(),
            "status":         "pending",  # pending | approved | rejected
        })
        existing_terms.add(key)

    if new_entries:
        existing.extend(new_entries)
        _save_suggestions(existing)
        print(f"  [taxonomy] logged {len(new_entries)} new suggestion(s) for {company_name}")