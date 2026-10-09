import re
from pydantic import BaseModel


LINE_PATTERN = re.compile(r"^\s*(\d+)\s*\|?\s*(.+?)\s*$")


def parse_line(line: str):
    """Retourne (id, nom) ou None si la ligne est invalide."""
    match = LINE_PATTERN.match(line)
    if not match:
        return None
    id_str, name = match.groups()
    return int(id_str), name


class Sector(BaseModel):
    sector_id: int
    sector_name: str


class Topic(BaseModel):
    topic_id: int
    topic_name: str


def load_sectors(path: str) -> list[Sector]:
    sectors = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            parsed = parse_line(line)
            if parsed is None:
                print(f"[WARN] sectors: ligne {line_num} ignorée: {line!r}")
                continue
            sector_id, sector_name = parsed
            sectors.append(Sector(sector_id=sector_id, sector_name=sector_name))
    return sectors


def load_topics(path: str) -> list[Topic]:
    topics = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            parsed = parse_line(line)
            if parsed is None:
                print(f"[WARN] topics: ligne {line_num} ignorée: {line!r}")
                continue
            topic_id, topic_name = parsed
            topics.append(Topic(topic_id=topic_id, topic_name=topic_name))
    return topics


def build_taxonomy(sectors_path: str, topics_path: str) -> dict:
    sectors = load_sectors(sectors_path)
    topics = load_topics(topics_path)

    valid_sector_ids = {s.sector_id for s in sectors}
    valid_topic_ids = {t.topic_id for t in topics}

    sectors_prompt_text = "\n".join(
        f"{s.sector_id} | {s.sector_name}" for s in sectors
    )
    topics_prompt_text = "\n".join(
        f"{t.topic_id} | {t.topic_name}" for t in topics
    )

    return {
        "sectors": sectors,
        "topics": topics,
        "valid_sector_ids": valid_sector_ids,
        "valid_topic_ids": valid_topic_ids,
        "sectors_prompt_text": sectors_prompt_text,
        "topics_prompt_text": topics_prompt_text,
    }