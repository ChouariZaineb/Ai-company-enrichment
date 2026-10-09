# src/topic_ranker.py
"""
Specificity-weighted topic ranking.

Score formula: word_count(topic_name) * log(1 + target_company_count)

This naturally makes multi-word specific terms ("Banking Services", 2 words)
rank above single-word generic terms ("Banking", 1 word) even when the generic
term has more raw attachments — while keeping high-value single-word niche terms
("Neobanking", moderate count) competitive via the log-count component.

Used in get_topics_for_sectors() to sort the candidate pool before
presenting it to the agent.
"""
import math


def specificity_score(topic_name: str, count: int) -> float:
    """
    Higher = more specific AND more commonly used by PE analysts.
    word_count rewards multi-word phrases over single generic words.
    log(1 + count) rewards real-world usage without letting raw count dominate.
    """
    word_count = len(topic_name.strip().split())
    return word_count * math.log(1 + count)


def rank_topics(
    topics: list[tuple],
    name_idx: int = 1,
    count_idx: int = 4,
) -> list[tuple]:
    """
    Sorts a list of topic tuples by specificity_score descending.
    Within each category/feature group, higher-scoring topics appear first.
    topics: list of (tid_int, tname, ttype, desc, count, ...)
    name_idx: index of topic_name in the tuple
    count_idx: index of count in the tuple
    """
    return sorted(
        topics,
        key=lambda x: (
            0 if x[2] == "category" else 1,   # categories first
            -specificity_score(x[name_idx], x[count_idx]),  # then by score desc
            x[0],                               # then by id for stability
        ),
    )