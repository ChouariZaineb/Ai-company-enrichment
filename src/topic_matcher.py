import ollama
import json
import re
import os
import numpy as np

OLLAMA_MODEL = "mistral:7b"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 60
MAX_MATCHES_PER_BATCH = 3
CANDIDATE_POOL_SIZE_PER_SECTOR = 150  # narrowed per sector, not globally
PROGRESS_DIR = "data/topic_matching_progress"
EMBEDDING_CACHE_PATH = "data/topic_embeddings_cache.json"


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def parse_json_blocks(raw_output: str) -> list:
    json_blocks = re.findall(r"\{.*?\}(?=\s*[,{]|\s*$)", raw_output, re.DOTALL)
    parsed_blocks = []
    for block in json_blocks:
        try:
            parsed_blocks.append(json.loads(block, strict=False))
        except json.JSONDecodeError:
            continue
    return parsed_blocks


def _progress_path(progress_key: str) -> str:
    safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", progress_key).strip("_").lower()
    return os.path.join(PROGRESS_DIR, f"{safe_key}_progress.json")


def _load_progress(progress_key: str) -> dict:
    path = _progress_path(progress_key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_completed_batch": 0, "matched_ids": [], "reasoning": []}


def _save_progress(progress_key: str, last_completed_batch: int, matched_ids: set, reasoning: list) -> None:
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    path = _progress_path(progress_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "last_completed_batch": last_completed_batch,
            "matched_ids": sorted(matched_ids),
            "reasoning": reasoning,
        }, f, indent=2, ensure_ascii=False)


def _clear_progress(progress_key: str) -> None:
    path = _progress_path(progress_key)
    if os.path.exists(path):
        os.remove(path)


def _embed_text(text: str) -> list:
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


def load_or_build_topic_embeddings(topics: list) -> dict:
    """
    Returns {topic_id: embedding_vector}. Computed once and cached to disk —
    the taxonomy is fixed, so this only runs in full the very first time.
    """
    current_ids = {t.topic_id for t in topics}

    if os.path.exists(EMBEDDING_CACHE_PATH):
        with open(EMBEDDING_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_ids = {int(k) for k in cached.keys()}
        if cached_ids == current_ids:
            return {int(k): v for k, v in cached.items()}
        print("[INFO] Topic list changed since last embedding cache — rebuilding.")

    print(f"[INFO] Building embeddings for {len(topics)} topics (one-time cost, cached after this)...")
    embeddings = {}
    for i, t in enumerate(topics, start=1):
        if i % 200 == 0:
            print(f"  [Embeddings] {i}/{len(topics)}...")
        embeddings[t.topic_id] = _embed_text(t.topic_name)

    os.makedirs(os.path.dirname(EMBEDDING_CACHE_PATH), exist_ok=True)
    with open(EMBEDDING_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    return embeddings


def _cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _get_candidate_topics_for_sector(company_name: str, business_summary: str, sector_name: str,
                                      topics: list, topic_embeddings: dict, top_n: int) -> list:
    """
    Narrows the full topic list down to the top_n topics most semantically
    relevant to THIS company WITHIN this specific sector — scoping the query
    to one sector at a time is what prevents cross-contamination between
    sectors (e.g. a tech topic bleeding into the financial-services bucket).
    """
    query_text = f"{company_name} operating in the '{sector_name}' sector. {business_summary}"
    query_embedding = _embed_text(query_text)

    scored = []
    for t in topics:
        emb = topic_embeddings.get(t.topic_id)
        if emb is None:
            continue
        scored.append((_cosine_similarity(query_embedding, emb), t))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:top_n]]


def _run_batches_for_sector(company_name: str, sector_name: str, business_summary: str,
                             candidate_topics: list, progress_key: str,
                             chunk_size: int, resume: bool) -> tuple:
    chunks = list(chunk_list(candidate_topics, chunk_size))
    total_chunks = len(chunks)

    progress = _load_progress(progress_key) if resume else {"last_completed_batch": 0, "matched_ids": [], "reasoning": []}
    all_matched_ids = set(progress["matched_ids"])
    all_reasoning = progress["reasoning"]
    start_batch = progress["last_completed_batch"]

    for chunk_index, topic_chunk in enumerate(chunks, start=1):
        if chunk_index <= start_batch:
            continue

        print(f"    [Topics] Sector '{sector_name}' — batch {chunk_index}/{total_chunks}...")
        topics_chunk_text = "\n".join(f"{t.topic_id} | {t.topic_name}" for t in topic_chunk)

        prompt = f"""You are identifying specific topics that genuinely belong under ONE particular sector for a company.

Company: {company_name}
Sector being evaluated: {sector_name}

Business summary:
{business_summary}

Here is ONE BATCH of possible topics (format: id | name), pre-filtered as plausibly relevant to this sector:
{topics_chunk_text}

Instructions:
1. Look ONLY at the topics listed above in this batch.
2. Select a topic ONLY if it is a core, defining part of what this company actually does WITHIN the "{sector_name}" sector — not just loosely related.
3. EXCLUDE generic business functions that almost any company has (e.g. Marketing, Customer Support, Certification Services, Legal Services, general Analytics) UNLESS that function is literally the company's core specific business activity.
4. Require CONCRETE, STATED evidence from the business summary — a specific product, service, or activity the summary actually describes. Do not select a topic just because it shares an adjective or general theme with the summary (e.g. the summary saying "mobile application" does NOT by itself justify "Mobile Marketing", "Smart Devices", or "E-Commerce Solutions" — none of those are concretely described; a mobile banking app is not evidence of e-commerce, marketing, or smart-device technology).
5. Select AT MOST {MAX_MATCHES_PER_BATCH} topics from this batch. Prefer selecting fewer, more precise topics over more, weaker ones — do not fill the cap if fewer topics genuinely qualify.
6. If none of the topics in this batch genuinely apply, return an empty list — do not force a match.
7. Do not invent topic ids. Only use ids that appear in this batch.

Do NOT list, name, or comment on individual topics in your reasoning. Your reasoning must be a single short sentence, never mentioning specific topic names or ids.

Respond ONLY in this exact JSON format, with no extra text before or after:
{{
  "reasoning": "one short general sentence, no topic names or ids mentioned",
  "matched_topic_ids": [list of integer ids, at most {MAX_MATCHES_PER_BATCH}]
}}
"""
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"num_ctx": 4096, "num_predict": 200, "temperature": 0},
            )
            raw_output = response["message"]["content"].strip()
        except Exception as e:
            print(f"    [ERROR] Batch {chunk_index}: Ollama call failed — {e}. Progress saved, re-run to resume.")
            _save_progress(progress_key, chunk_index - 1, all_matched_ids, all_reasoning)
            return all_matched_ids, all_reasoning

        parsed_blocks = parse_json_blocks(raw_output)

        if not parsed_blocks:
            print(f"    [WARN] Batch {chunk_index}: could not parse output, skipping.")
        else:
            valid_ids_in_batch = {t.topic_id for t in topic_chunk}
            for parsed in parsed_blocks:
                ids = parsed.get("matched_topic_ids", [])
                reasoning = parsed.get("reasoning", "")
                accepted_ids = [i for i in ids if i in valid_ids_in_batch][:MAX_MATCHES_PER_BATCH]
                if accepted_ids:
                    all_matched_ids.update(accepted_ids)
                    if reasoning and reasoning.lower() != "none relevant":
                        all_reasoning.append(f"Batch {chunk_index}: {reasoning}")

        _save_progress(progress_key, chunk_index, all_matched_ids, all_reasoning)

    _clear_progress(progress_key)
    return all_matched_ids, all_reasoning


def match_topics_by_sector(company_name: str, business_summary: str, matched_sectors: list,
                            topics: list, chunk_size: int = CHUNK_SIZE, resume: bool = True,
                            candidate_pool_size: int = CANDIDATE_POOL_SIZE_PER_SECTOR) -> list:
    """
    Matches topics to a company SEPARATELY for each matched sector, so every
    topic ends up nested under the sector it belongs to.

    matched_sectors: list of dicts, e.g.
        [{"sector_id": 16, "sector_name": "Financial, Investment and Insurance Services"},
         {"sector_id": 7, "sector_name": "Computing, Technology, Robotics & AI"}]
    (build this from taxonomy["sectors"] filtered down to the ids returned by
    match_sectors_to_taxonomy in sector_matcher.py)

    Returns:
        [{"sector_id": 16, "sector_name": "...", "topics": [{"topic_id":.., "topic_name":..}, ...], "reasoning": "..."}, ...]
    """
    topic_embeddings = load_or_build_topic_embeddings(topics)
    topic_by_id = {t.topic_id: t for t in topics}

    output = []

    for sector in matched_sectors:
        sector_id = sector["sector_id"]
        sector_name = sector["sector_name"]
        print(f"  [Topics] Matching topics for sector: {sector_name}")

        progress_key = f"{company_name}_{sector_id}"

        candidate_topics = _get_candidate_topics_for_sector(
            company_name, business_summary, sector_name, topics, topic_embeddings,
            top_n=candidate_pool_size,
        )

        matched_ids, reasoning_list = _run_batches_for_sector(
            company_name, sector_name, business_summary, candidate_topics,
            progress_key, chunk_size, resume,
        )

        output.append({
            "sector_id": sector_id,
            "sector_name": sector_name,
            "topics": [
                {"topic_id": tid, "topic_name": topic_by_id[tid].topic_name}
                for tid in sorted(matched_ids)
            ],
            "reasoning": "\n".join(reasoning_list) if reasoning_list else "No topics matched.",
        })

    return output