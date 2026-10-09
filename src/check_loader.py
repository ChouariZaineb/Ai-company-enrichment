from loader import build_taxonomy

taxonomy = build_taxonomy(
    sectors_path="../data/sectors_formatted.txt",
    topics_path="../data/topics_formatted.txt",
)

print("Nombre de secteurs:", len(taxonomy["sectors"]))
print("Nombre de topics:", len(taxonomy["topics"]))

print("\nPremiers secteurs:")
for s in taxonomy["sectors"][:3]:
    print(" -", s.sector_id, s.sector_name)

print("\nPremiers topics:")
for t in taxonomy["topics"][:3]:
    print(" -", t.topic_id, t.topic_name)

print("\n16 dans valid_sector_ids ?", 16 in taxonomy["valid_sector_ids"])
print("1870 dans valid_topic_ids ?", 1870 in taxonomy["valid_topic_ids"])

print("\nTaille sectors_prompt_text:", len(taxonomy["sectors_prompt_text"]), "caractères")
print("Taille topics_prompt_text:", len(taxonomy["topics_prompt_text"]), "caractères")