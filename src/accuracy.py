# src/accuracy.py
"""
Accuracy scoring against known official answers, across all pipeline
variants. Topic scoring uses recall-only per supervisor instruction:
extra/plausible topics are not penalized as long as they are grounded
in the submitted source material.
"""
import json
import glob
import os

GROUND_TRUTH_BY_COMPANY_ID = {
    1002: {  # Monzo
        "sector_ids": {16, 7},
        "topic_ids_flat": {1870, 8619, 8747, 1465, 722},
    },
    1004: {  # Darktrace
        "sector_ids": {4, 7},
        "topic_ids_flat": {5358, 8640, 2777, 1719, 1661},
    },
    1006: {  # Octopus Energy
        "sector_ids": {11, 7, 30},
        "topic_ids_flat": {1619, 7268, 4110, 8364, 3137, 2475, 7335, 5163, 2474, 2054},
    },
    
    1010: {  # babylon holdings
        "sector_ids": {7, 19},
        "topic_ids_flat": {5787, 6099, 6261, 2978},
    },
    14362: {
        "sector_ids": {7, 16, 21},
        "topic_ids_flat": {2282, 2984, 4204, 4541, 5751, 6897, 8640, 8791, 8793},
    },
    30336: {
        "sector_ids": {7},
        "topic_ids_flat": {722, 1859, 7980},
    },
    30344: {
        "sector_ids": {7, 16},
        "topic_ids_flat": {776, 1719, 1783, 1858, 1864},
    },
    30345: {
        "sector_ids": {20},
        "topic_ids_flat": {2828, 8399, 9241},
    },
    30346: {
        "sector_ids": {7, 19},
        "topic_ids_flat": {1956, 2282, 9025},
    },
    30350: {
        "sector_ids": {7, 17},
        "topic_ids_flat": {1719, 2931, 5477},
    },
    30352: {
        "sector_ids": {18, 20},
        "topic_ids_flat": {4777, 8136, 8206},
    },
}

MODEL_CONFIG = {
    "mistral": {
        "label": "Mistral (local)",
        "sector_key": "matched_sector_ids",
        "topics_key": "topics_by_sector",
        "reasoning_sector_key": "sector_matching_reasoning",
        "reasoning_topics_key": None,
        "shape": "nested",
    },
    "gemini_pro": {
        "label": "Gemini Pro",
        "sector_key": "matched_sector_ids_gemini_pro",
        "topics_key": "matched_topics_flat_gemini_pro",
        "reasoning_sector_key": "sector_matching_reasoning_gemini_pro",
        "reasoning_topics_key": "topic_matching_reasoning_gemini_pro",
        "shape": "flat",
    },
    "bedrock": {
        "label": "Claude (Bedrock)",
        "sector_key": "matched_sector_ids_bedrock",
        "topics_key": "matched_topics_flat_bedrock",
        "reasoning_sector_key": "sector_matching_reasoning_bedrock",
        "reasoning_topics_key": "topic_matching_reasoning_bedrock",
        "shape": "flat",
    },
    "bedrock_v2": {
        "label": "Claude (Bedrock) — Enriched Embeddings",
        "sector_key": "matched_sector_ids_bedrock_v2",
        "topics_key": "matched_topics_flat_bedrock_v2",
        "reasoning_sector_key": None,
        "reasoning_topics_key": "topic_matching_reasoning_bedrock_v2",
        "shape": "flat",
    },
    "deepseek": {
        "label": "DeepSeek 3.2 (Bedrock)",
        "sector_key": "matched_sector_ids_deepseek",
        "topics_key": "matched_topics_flat_deepseek",
        "reasoning_sector_key": "sector_matching_reasoning_deepseek",
        "reasoning_topics_key": "topic_matching_reasoning_deepseek",
        "shape": "flat",
    },
    "deepseek_v2": {
        "label": "DeepSeek 3.2 (Bedrock) — Enriched Embeddings",
        "sector_key": "matched_sector_ids_deepseek_v2",
        "topics_key": "matched_topics_flat_deepseek_v2",
        "reasoning_sector_key": "sector_matching_reasoning_deepseek_v2",
        "reasoning_topics_key": "topic_matching_reasoning_deepseek_v2",
        "shape": "flat",
    },
    "agent": {
        "label": "Claude Agent (Bedrock)",
        "sector_key": "matched_sector_ids_agent",
        "topics_key": "matched_topics_flat_agent",
        "reasoning_sector_key": None,
        "reasoning_topics_key": "topic_matching_reasoning_agent",
        "shape": "flat",
    },
    "agent_deepseek": {
        "label": "DeepSeek Agent (Bedrock)",
        "sector_key": "matched_sector_ids_agent_deepseek",
        "topics_key": "matched_topics_flat_agent_deepseek",
        "reasoning_sector_key": None,
        "reasoning_topics_key": "topic_matching_reasoning_agent_deepseek",
        "shape": "flat",
    },
    "agent_weighted": {
        "label": "Claude Agent — Weighted (Bedrock) [experimental]",
        "sector_key": "matched_sector_ids_agent_weighted",
        "topics_key": "matched_topics_flat_agent_weighted",
        "reasoning_sector_key": None,
        "reasoning_topics_key": "topic_matching_reasoning_agent_weighted",
        "shape": "flat",
    },
    "agent_weighted_gemini": {
        "label": "Gemini Pro Agent — Weighted",
        "sector_key": "matched_sector_ids_agent_weighted_gemini",
        "topics_key": "matched_topics_flat_agent_weighted_gemini",
        "reasoning_sector_key": None,
        "reasoning_topics_key": "topic_matching_reasoning_agent_weighted_gemini",
        "shape": "flat",
    },
    "agent_new_claude": {
        "label": "Claude Agent — New Companies",
        "sector_key": "matched_sector_ids_agent_new_claude",
        "topics_key": "matched_topics_flat_agent_new_claude",
        "reasoning_sector_key": None,
        "reasoning_topics_key": "topic_reasoning_agent_new_claude",
        "shape": "flat",
    },
    "agent_new_gemini": {
        "label": "Gemini Pro Agent — New Companies",
        "sector_key": "matched_sector_ids_agent_new_gemini",
        "topics_key": "matched_topics_flat_agent_new_gemini",
        "reasoning_sector_key": None,
        "reasoning_topics_key": "topic_reasoning_agent_new_gemini",
        "shape": "flat",
    },
    "raw_agent_claude": {
        "label": "Claude Raw Agent (tool-loop)",
        "sector_key": "matched_sector_ids_raw_agent_claude",
        "topics_key": "matched_topics_flat_raw_agent_claude",
        "reasoning_sector_key": None,
        "reasoning_topics_key": "topic_reasoning_raw_agent_claude",
        "shape": "flat",
    },
    "raw_agent_gemini": {
        "label": "Gemini Pro Raw Agent (tool-loop)",
        "sector_key": "matched_sector_ids_raw_agent_gemini",
        "topics_key": "matched_topics_flat_raw_agent_gemini",
        "reasoning_sector_key": None,
        "reasoning_topics_key": "topic_reasoning_raw_agent_gemini",
        "shape": "flat",
    },
    "multi_agent": {
        "label": "Multi-Agent Pipeline (Sonnet 5)",
        "sector_key": "matched_sector_ids_multi_agent",
        "topics_key": "matched_topics_flat_multi_agent",
        "reasoning_sector_key": "sector_reasoning_multi_agent",
        "reasoning_topics_key": "topic_reasoning_multi_agent",
        "shape": "flat",
    },
}


def _precision_recall_f1(predicted: set, actual: set) -> dict:
    if not predicted and not actual:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not actual:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(predicted & actual)
    precision = tp / len(predicted)
    recall    = tp / len(actual)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
    }


def _recall_only_score(predicted: set, actual: set) -> dict:
    """
    Per supervisor: extra/plausible topics should not be penalized, as long
    as they are grounded in the source material submitted for matching.
    Topic scoring measures ONLY whether the true topics were found — recall —
    with no precision penalty for additional topics.
    """
    if not actual:
        return {"recall": 1.0, "f1": 1.0}
    if not predicted:
        return {"recall": 0.0, "f1": 0.0}

    tp     = len(predicted & actual)
    recall = tp / len(actual)
    return {"recall": round(recall, 3), "f1": round(recall, 3)}


def _flatten_topic_ids(topics_field, shape: str) -> set:
    if not topics_field:
        return set()
    if shape == "flat":
        return {t["topic_id"] for t in topics_field}
    if shape == "nested":
        ids = set()
        for sector_block in topics_field:
            for t in sector_block.get("topics", []):
                ids.add(t["topic_id"])
        return ids
    return set()


def score_against_ground_truth(
    company_id: int,
    matched_sector_ids: list,
    topics_field,
    shape: str,
) -> dict:
    ground_truth = GROUND_TRUTH_BY_COMPANY_ID.get(company_id)
    if not ground_truth:
        return {"available": False}

    predicted_sector_ids = set(matched_sector_ids or [])
    true_sector_ids      = ground_truth["sector_ids"]
    sector_scores        = _precision_recall_f1(predicted_sector_ids, true_sector_ids)

    predicted_topic_ids = _flatten_topic_ids(topics_field, shape)
    true_topic_ids      = ground_truth["topic_ids_flat"]
    topic_scores        = _recall_only_score(predicted_topic_ids, true_topic_ids)

    # correct / missed / extra for display in app
    correct_sector_ids  = predicted_sector_ids & true_sector_ids
    missed_sector_ids   = true_sector_ids - predicted_sector_ids
    extra_sector_ids    = predicted_sector_ids - true_sector_ids

    correct_topic_ids   = predicted_topic_ids & true_topic_ids
    missed_topic_ids    = true_topic_ids - predicted_topic_ids
    extra_topic_ids     = predicted_topic_ids - true_topic_ids

    overall_recall = round((sector_scores["recall"] + topic_scores["recall"]) / 2, 3)

    return {
        "available":            True,
        "sector_scores":        sector_scores,
        "topic_scores":         topic_scores,
        "overall":              {"recall": overall_recall},
        "correct_sector_ids":   sorted(correct_sector_ids),
        "missed_sector_ids":    sorted(missed_sector_ids),
        "extra_sector_ids":     sorted(extra_sector_ids),
        "correct_topic_ids":    sorted(correct_topic_ids),
        "missed_topic_ids":     sorted(missed_topic_ids),
        "extra_topic_ids":      sorted(extra_topic_ids),
    }


def score_all_companies(model: str) -> dict:
    config = MODEL_CONFIG.get(model)
    if not config:
        raise ValueError(f"Unknown model: {model}. Choose from {list(MODEL_CONFIG.keys())}")

    sector_key = config["sector_key"]
    topics_key = config["topics_key"]
    shape      = config["shape"]

    results = {}
    for path in glob.glob(os.path.join("data", "overviews_cache", "*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        company_id = data.get("company_id")
        if company_id not in GROUND_TRUTH_BY_COMPANY_ID:
            continue
        if sector_key not in data:
            continue

        score = score_against_ground_truth(
            company_id=company_id,
            matched_sector_ids=data.get(sector_key),
            topics_field=data.get(topics_key),
            shape=shape,
        )
        results[data.get("company_name", str(company_id))] = score

    if results:
        avg_recall = round(
            sum(r["overall"]["recall"] for r in results.values()) / len(results), 3
        )
        results["_AVERAGE_"] = {"overall_recall": avg_recall}

    return results

def get_model_detail(model: str, company_name: str) -> dict:
    config = MODEL_CONFIG.get(model)
    if not config:
        return {}

    for path in glob.glob(os.path.join("data", "overviews_cache", "*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("company_name") != company_name:
            continue

        sector_ids   = data.get(config["sector_key"], [])
        topics_field = data.get(config["topics_key"])
        shape        = config["shape"]

        if shape == "flat":
            topics_flat = topics_field or []
        else:
            topics_flat = []
            for sector_block in (topics_field or []):
                topics_flat.extend(sector_block.get("topics", []))

        sector_reasoning = (
            data.get(config["reasoning_sector_key"], "")
            if config["reasoning_sector_key"] else ""
        )
        topic_reasoning = (
            data.get(config["reasoning_topics_key"], "")
            if config["reasoning_topics_key"] else ""
        )

        return {
            "sector_ids":      sector_ids,
            "topics_flat":     topics_flat,
            "sector_reasoning": sector_reasoning,
            "topic_reasoning": topic_reasoning,
        }

    return {}