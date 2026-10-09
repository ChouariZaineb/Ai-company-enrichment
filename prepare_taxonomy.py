import json
import os

with open("data/taxonomy 1.json", "r", encoding="utf-8") as f:
    taxonomy = json.load(f)

sectors = taxonomy["sectors"]
topics = taxonomy["topics"]

os.makedirs("data", exist_ok=True)

with open("data/sectors_formatted.txt", "w", encoding="utf-8") as f:
    for sector in sectors:
        line = f"{sector['sector_id']} | {sector['sector_name']}\n"
        f.write(line)

print(f"sectors_formatted.txt created — {len(sectors)} sectors")

max_topic_id_len = max(len(str(t["topic_id"])) for t in topics)

with open("data/topics_formatted.txt", "w", encoding="utf-8") as f:
    for topic in topics:
        topic_id_str = str(topic["topic_id"]).ljust(max_topic_id_len)
        line = f"{topic_id_str}   {topic['topic_name']}\n"
        f.write(line)

print(f"topics_formatted.txt created — {len(topics)} topics")