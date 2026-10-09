"""
Standalone enriched-embedding module — embeds 'topic_name: short description'
instead of the bare topic name, for richer/more disambiguated vectors.
Fully separate from topic_matcher_gemini.py's original embeddings, so this
can be tested and reverted independently without touching anything working.
"""
import os
import json
import time

from src.topic_matcher_gemini import _embed_text_gemini, _cosine_similarity

EMBEDDING_CACHE_PATH_ENRICHED = "data/topic_embeddings_cache_gemini_enriched.json"
TOPIC_DESCRIPTIONS_PATH = "data/topic_descriptions.json"


def _load_topic_descriptions() -> dict:
    with open(TOPIC_DESCRIPTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)




def load_or_build_topic_embeddings_enriched(topics: list) -> dict:
    """
    Returns {topic_id: embedding_vector}, computed from 'name: description'
    text instead of bare name. Cached separately from the original.
    Saves progress every 100 topics so a network blip never loses more
    than ~100 embeddings of work.
    """
    descriptions = _load_topic_descriptions()
    current_ids = {t.topic_id for t in topics}

    embeddings = {}
    if os.path.exists(EMBEDDING_CACHE_PATH_ENRICHED):
        with open(EMBEDDING_CACHE_PATH_ENRICHED, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_ids = {int(k) for k in cached.keys()}
        if cached_ids == current_ids:
            return {int(k): v for k, v in cached.items()}
        # Partial cache from an earlier interrupted run — reuse what we can
        embeddings = {int(k): v for k, v in cached.items() if int(k) in current_ids}
        print(f"[INFO] Resuming enriched embeddings — {len(embeddings)} already done, {len(current_ids) - len(embeddings)} remaining.")

    remaining = [t for t in topics if t.topic_id not in embeddings]
    if not remaining:
        return embeddings

    print(f"[INFO] Building ENRICHED embeddings for {len(remaining)} topics...")

    for i, t in enumerate(remaining, start=1):
        description = descriptions.get(str(t.topic_id), "")
        text_to_embed = f"{t.topic_name}: {description}" if description else t.topic_name

        for attempt in range(1, 4):
            try:
                embeddings[t.topic_id] = _embed_text_gemini(text_to_embed)
                break
            except Exception as e:
                print(f"[WARN] Embed failed for topic {t.topic_id} (attempt {attempt}/3): {e}")
                if attempt < 3:
                    time.sleep(2 * attempt)
                else:
                    print(f"[ERROR] Giving up on topic {t.topic_id} after 3 attempts — will retry on next run.")

        if i % 100 == 0:
            print(f"  [Embeddings/Enriched] {i}/{len(remaining)} this run ({len(embeddings)}/{len(current_ids)} total)...")
            os.makedirs(os.path.dirname(EMBEDDING_CACHE_PATH_ENRICHED), exist_ok=True)
            with open(EMBEDDING_CACHE_PATH_ENRICHED, "w", encoding="utf-8") as f:
                json.dump(embeddings, f)

    os.makedirs(os.path.dirname(EMBEDDING_CACHE_PATH_ENRICHED), exist_ok=True)
    with open(EMBEDDING_CACHE_PATH_ENRICHED, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    return embeddings


def get_candidate_topics_flat_enriched(company_name: str, business_summary: str, matched_sectors: list,
                                        topics: list, topic_embeddings: dict,
                                        top_n_per_sector: int) -> list:
    """
    Same retrieval logic as _get_candidate_topics_flat in topic_matcher_gemini.py,
    duplicated here so this module has zero dependency on the original
    function's internals — fully self-contained, fully revertible.
    """
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