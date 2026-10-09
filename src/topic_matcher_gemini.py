#src/topic_matcher_gemini.py
"""
Cloud-only (Gemini Pro) topic matching — replaces the local mistral+Ollama
version entirely for the cloud comparison path. Key differences from
topic_matcher.py:
  - Embeddings are computed via Gemini's cloud embedding model instead of
    local Ollama (nomic-embed-text). Cached separately at
    data/topic_embeddings_cache_gemini.json so it never collides with the
    local Ollama embedding cache.
  - Candidate narrowing still runs once per matched sector (same query
    pattern as the local version: company + sector name + summary), but
    all sectors' candidates are merged and deduped into ONE flat pool.
  - Verification is a SINGLE Gemini Pro call over that flat pool — no
    batching/chunking — returning one flat, deduped topic list for the
    whole company (not nested per sector).
"""
import os
import json
import math
import httpx

from src.gemini_key_manager import build_gemini_pro_url, build_gemini_embedding_url

EMBEDDING_CACHE_PATH_GEMINI = "data/topic_embeddings_cache_gemini.json"
CANDIDATE_POOL_SIZE_PER_SECTOR_GEMINI = 100  # merged across sectors -> ~100 for 2 sectors
MAX_TOTAL_MATCHES = 15  # safety cap on the final flat list


def _embed_text_gemini(text: str) -> list:
    response = httpx.post(
        build_gemini_embedding_url(),
        json={"content": {"parts": [{"text": text}]}},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]["values"]


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_or_build_topic_embeddings_gemini(topics: list) -> dict:
    """Same caching pattern as the local version, separate file, cloud-computed vectors."""
    current_ids = {t.topic_id for t in topics}
    if os.path.exists(EMBEDDING_CACHE_PATH_GEMINI):
        with open(EMBEDDING_CACHE_PATH_GEMINI, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_ids = {int(k) for k in cached.keys()}
        if cached_ids == current_ids:
            return {int(k): v for k, v in cached.items()}
        print("[INFO] Topic list changed since last Gemini embedding cache — rebuilding.")

    print(f"[INFO] Building Gemini embeddings for {len(topics)} topics (one-time cost, cached after this)...")
    embeddings = {}
    for i, t in enumerate(topics, start=1):
        if i % 200 == 0:
            print(f"  [Embeddings/Gemini] {i}/{len(topics)}...")
        embeddings[t.topic_id] = _embed_text_gemini(t.topic_name)

    os.makedirs(os.path.dirname(EMBEDDING_CACHE_PATH_GEMINI), exist_ok=True)
    with open(EMBEDDING_CACHE_PATH_GEMINI, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)
    return embeddings


def _get_candidate_topics_flat(company_name: str, business_summary: str, matched_sectors: list,
                                topics: list, topic_embeddings: dict,
                                top_n_per_sector: int) -> list:
    """
    Runs one embedding query per matched sector (same pattern as the local
    version), then merges + dedupes all sectors' candidates into one flat list.
    """
    topic_by_id = {t.topic_id: t for t in topics}
    merged_ids = set()
    merged_topics = []

    for sector in matched_sectors:
        sector_name = sector["sector_name"]
        query_text = f"{company_name} operating in the '{sector_name}' sector. {business_summary}"
        query_embedding = _embed_text_gemini(query_text)

        scored = []
        for t in topics:
            emb = topic_embeddings.get(t.topic_id)
            if emb is None:
                continue
            scored.append((_cosine_similarity(query_embedding, emb), t))
        scored.sort(key=lambda x: x[0], reverse=True)

        for _, t in scored[:top_n_per_sector]:
            if t.topic_id not in merged_ids:
                merged_ids.add(t.topic_id)
                merged_topics.append(t)

    return merged_topics


def match_topics_flat_gemini(company_name: str, business_summary: str, matched_sectors: list,
                              topics: list, top_n_per_sector: int = CANDIDATE_POOL_SIZE_PER_SECTOR_GEMINI,
                              max_retries: int = 3) -> list:
    """
    Returns a FLAT list of matched topics for the whole company:
    [{"topic_id": int, "topic_name": str}, ...]
    Not nested per sector, unlike the local mistral version.
    """
    topic_embeddings = load_or_build_topic_embeddings_gemini(topics)

    candidate_topics = _get_candidate_topics_flat(
        company_name, business_summary, matched_sectors, topics, topic_embeddings, top_n_per_sector,
    )

    sector_names_str = ", ".join(s["sector_name"] for s in matched_sectors)
    candidates_text = "\n".join(f"{t.topic_id} | {t.topic_name}" for t in candidate_topics)

    prompt = f"""You are identifying the specific topics that genuinely apply to a company, across all of its matched business sectors.

Company: {company_name}
Matched sectors: {sector_names_str}

Business summary:
{business_summary}

Here are candidate topics (format: id | name), pre-filtered as plausibly relevant to this company's sectors:
{candidates_text}

Instructions:
Instructions:
1. Select a topic ONLY if it is a core, defining part of what this company actually does — not just loosely related.
2. EXCLUDE generic business functions that almost any company has (e.g. Marketing, Customer Support, Certification Services, Legal Services, general Analytics) UNLESS that function is literally the company's core specific business activity.
3. Require CONCRETE, STATED evidence from the business summary — a specific product, service, or activity the summary actually describes. Do not select a topic just because it shares an adjective or general theme with the summary.
4. BEWARE OF NEAR-SYNONYMS AND VAGUE UMBRELLA TERMS. The candidate list often contains several topics that all loosely relate to the same broad industry word . Do NOT select all of them just because the company is in that industry. Select ONLY the topic(s) that most precisely and specifically describe what the company actually does. When two candidate topics both plausibly apply but one is clearly more specific and accurate than the other, select only the more specific one.
5. Ask yourself for each candidate: "Is this topic name so generic that it would also apply to most other companies in this company's broad industry?" If yes, exclude it unless the summary gives a specific reason THIS company (not the industry in general) is defined by it.
6. Select AT MOST {MAX_TOTAL_MATCHES} topics total. Prefer selecting fewer, more precise topics over more, weaker ones — a shorter, highly accurate list is much better than a longer list padded with plausible-sounding generic terms.
7. Do not invent topic ids. Only use ids that appear in the candidate list above.
8. Return ONE flat list — do not group or label topics by sector.
9. PREFER MID-LEVEL INDUSTRY-CATEGORY TOPICS over narrow specific product/feature topics, when both describe the same underlying activity. For example, if the candidates include both a broad category term and a narrower specific feature of that same activity, select the broader category term as the primary match — only include the narrower one too if it represents a genuinely distinct, additional part of the business the broader term doesn't already cover.
Respond ONLY in this exact JSON format, with no extra text before or after:
{{
  "reasoning": "a few sentences explaining the selections",
  "matched_topic_ids": [list of integer ids, at most {MAX_TOTAL_MATCHES}]
}}
"""

    topic_by_id = {t.topic_id: t for t in candidate_topics}
    valid_candidate_ids = set(topic_by_id.keys())

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
            raw_ids = parsed.get("matched_topic_ids", [])

            confirmed_ids = [int(i) for i in raw_ids if int(i) in valid_candidate_ids][:MAX_TOTAL_MATCHES]
            hallucinated = [int(i) for i in raw_ids if int(i) not in valid_candidate_ids]
            if hallucinated:
                print(f"[WARN] {company_name}: Gemini Pro returned topic id(s) not in candidate list {hallucinated} — dropped.")

            return {
                "topics": [
                    {"topic_id": tid, "topic_name": topic_by_id[tid].topic_name}
                    for tid in confirmed_ids
                ],
                "reasoning": parsed.get("reasoning", ""),
            }

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 503) and attempt < max_retries:
                print(f"[WARN] {company_name}: Gemini Pro {e.response.status_code}, retrying ({attempt}/{max_retries})...")
                continue
            print(f"[ERROR] Gemini Pro topic matching failed for {company_name}: {e}")
            return {"topics": [], "reasoning": ""}
        except json.JSONDecodeError:
            if attempt < max_retries:
                continue
            print(f"[ERROR] {company_name}: could not parse Gemini Pro JSON after {max_retries} attempts.")
            return {"topics": [], "reasoning": ""}
        except Exception as e:
            print(f"[ERROR] Gemini Pro topic matching failed for {company_name}: {e}")
            return {"topics": [], "reasoning": ""}

    return {"topics": [], "reasoning": ""}

def refine_matched_topics_gemini(company_name: str, business_summary: str, matched_sectors: list,
                                  initial_topics: list, max_retries: int = 3) -> dict:
    """
    Second-pass refinement: reviews the initially matched topics and drops any
    that aren't strongly, concretely justified. This trades some recall for
    higher precision — mirrors a manual "keep only what I'm confident about"
    review step, but automated and applied consistently across companies.
    """
    if not initial_topics:
        return {"topics": [], "reasoning": "No initial topics to refine."}

    sector_names_str = ", ".join(s["sector_name"] for s in matched_sectors)
    topics_text = "\n".join(f"{t['topic_id']} | {t['topic_name']}" for t in initial_topics)

    prompt = f"""You previously identified candidate topics for a company. Now review them critically and keep ONLY the ones you are highly confident genuinely belong.

Company: {company_name}
Matched sectors: {sector_names_str}

Business summary:
{business_summary}

Previously selected candidate topics (format: id | name):
{topics_text}

Instructions:
1. For EACH topic above, ask: "Is this a precise, core, defining part of what this company does, with clear concrete evidence in the business summary?" If you have ANY meaningful doubt, DROP it.
2. A wrong or overly-generic topic is worse than a missing one — bias strongly toward precision. It is completely acceptable, and expected, to end up with fewer topics than you started with.
3. Among topics that are near-synonyms or overlapping in meaning, keep only the single most accurate/specific one, unless multiple genuinely represent distinct parts of the business.
4. Do not add any NEW topic ids that weren't in the list above — this is a filtering pass only, not a chance to add more.
5. It's fine to keep all of them if every single one is genuinely well-justified, and fine to drop most of them if few are.

Respond ONLY in this exact JSON format, with no extra text before or after:
{{
  "reasoning": "brief explanation of what was kept and what was dropped, and why",
  "kept_topic_ids": [list of integer ids, a subset of the ids above]
}}
"""

    valid_ids = {t["topic_id"] for t in initial_topics}
    topic_by_id = {t["topic_id"]: t for t in initial_topics}

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
            raw_ids = parsed.get("kept_topic_ids", [])
            kept_ids = [int(i) for i in raw_ids if int(i) in valid_ids]

            return {
                "topics": [topic_by_id[tid] for tid in kept_ids],
                "reasoning": parsed.get("reasoning", ""),
            }

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 503) and attempt < max_retries:
                print(f"[WARN] {company_name}: refinement {e.response.status_code}, retrying ({attempt}/{max_retries})...")
                continue
            print(f"[ERROR] Refinement failed for {company_name}: {e}")
            return {"topics": initial_topics, "reasoning": "Refinement call failed — kept unrefined list."}
        except json.JSONDecodeError:
            if attempt < max_retries:
                continue
            print(f"[ERROR] {company_name}: could not parse refinement JSON.")
            return {"topics": initial_topics, "reasoning": "Refinement JSON parse failed — kept unrefined list."}
        except Exception as e:
            print(f"[ERROR] Refinement failed for {company_name}: {e}")
            return {"topics": initial_topics, "reasoning": "Refinement call failed — kept unrefined list."}

    return {"topics": initial_topics, "reasoning": "Refinement retries exhausted — kept unrefined list."}