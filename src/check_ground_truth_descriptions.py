# src/fix_topic_descriptions.py
import json
import os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(_BASE, "data", "topic_descriptions.json")

with open(PATH) as f:
    desc = json.load(f)

fixes = {
    # BusTech — PE context means B2B enterprise technology, NOT public transit buses
    "2777": "B2B technology solutions sold to businesses and enterprises",

    # SaaS — make it clearer it applies to subscription software platforms
    "722": "cloud-hosted software delivered via subscription to businesses or consumers",

    # Security Solutions — broader than just physical/digital, clarify PE context
    "5358": "integrated security products and services protecting organisations",

    # Software Development — clarify it's a company category not just an activity
    "1719": "companies building and selling proprietary software products",

    # BusTech already fixed above, also fix GreenTech to be clearer vs CleanTech
    "7268": "technology companies focused on environmental sustainability solutions",

    # CleanTech vs GreenTech distinction
    "4110": "clean technology reducing pollution and ecological impact across sectors",

    # EnergyTech — clarify it's tech-driven energy companies
    "3137": "technology-driven companies optimising energy generation, storage or distribution",

    # Solar Solutions vs Solar Panels distinction
    "8364": "integrated solar energy systems and photovoltaic solutions",

    # Energy Transition — clarify the shift context
    "2475": "companies enabling the shift from fossil fuels to renewable energy sources",

    # Sustainable Energy vs Renewable Energy distinction
    "2474": "long-term environmentally responsible energy supply and management",

    # Clean Energy vs Renewable Energy distinction
    "1619": "zero or low emission energy production from non-polluting sources",
}

for tid, new_desc in fixes.items():
    old = desc.get(tid, "NOT FOUND")
    desc[tid] = new_desc
    print(f"Fixed {tid}: '{old}' -> '{new_desc}'")

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(desc, f, ensure_ascii=False, indent=2)

print("Done.")