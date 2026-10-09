# src/structuring_agent.py
"""
Stage 1 of the agent pipeline: transforms raw scraped website text into a
structured PE-format company overview. Single-shot LLM call - no tool loop.
Output feeds directly into the topic agent (stage 3) as its primary input.
One shared function, called by all three model-specific topic matchers.
"""
import json
from bedrock_client import call_claude_bedrock

PE_STRUCTURE_PROMPT = """You are a private equity research analyst. Your job is to read raw scraped company website text and rewrite it into a clean, structured company overview in standard PE due-diligence format.

Output a JSON object with EXACTLY these fields — no extras, no omissions:

{
  "business_model": "How the company makes money. Revenue streams, pricing model, recurring vs one-off.",
  "products_and_services": "What the company actually sells or delivers. Be specific — name the products/platforms/services.",
  "target_customers": "Who buys from them. Be specific: enterprise, SME, consumer, specific verticals, geographies.",
  "technology_and_ip": "Core technology, proprietary systems, patents, data assets, platform infrastructure. Write 'None identified' if absent.",
  "market_and_geography": "Markets served and geographic footprint. Specific countries/regions if mentioned.",
  "competitive_positioning": "How they differentiate. Key claims made about their edge vs competitors.",
  "ownership_and_stage": "Funding stage, known investors, ownership structure, listed/private. Write 'Not disclosed' if absent.",
  "esg_and_regulatory": "Any sustainability, ESG, compliance, or regulatory context mentioned. Write 'None identified' if absent."
}

Rules:
- Write only what is directly supported by the scraped text — do not infer or embellish.
- Each field should be 1-3 concise sentences. No bullet points inside fields.
- If a field has genuinely no information in the scraped text, write "Not identified in source material."
- Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


def structure_company_overview(
    company_name: str,
    scraped_text: str,
    max_retries: int = 3,
) -> dict:
    """
    Takes raw scraped website text and returns a structured PE-format dict.
    Falls back to a minimal dict on failure so downstream steps never crash.
    """
    prompt = f"{PE_STRUCTURE_PROMPT}\n\nCompany: {company_name}\n\nScraped text:\n{scraped_text}"

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_claude_bedrock(prompt, max_tokens=1500, temperature=0)

            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

            parsed = json.loads(raw)

            required_fields = {
                "business_model", "products_and_services", "target_customers",
                "technology_and_ip", "market_and_geography",
                "competitive_positioning", "ownership_and_stage", "esg_and_regulatory",
            }
            missing = required_fields - parsed.keys()
            if missing:
                print(f"[WARN] {company_name}: structured overview missing fields {missing} on attempt {attempt}")
                if attempt < max_retries:
                    continue

            return parsed

        except json.JSONDecodeError:
            if attempt < max_retries:
                print(f"[WARN] {company_name}: malformed JSON on attempt {attempt}/{max_retries}, retrying...")
                continue
            print(f"[ERROR] {company_name}: structuring failed after {max_retries} attempts.")

        except Exception as e:
            print(f"[ERROR] {company_name}: structuring agent exception: {e}")
            if attempt < max_retries:
                continue

    # fallback - never return None, downstream steps check for empty strings
    return {
        "business_model": "Not identified in source material.",
        "products_and_services": "Not identified in source material.",
        "target_customers": "Not identified in source material.",
        "technology_and_ip": "Not identified in source material.",
        "market_and_geography": "Not identified in source material.",
        "competitive_positioning": "Not identified in source material.",
        "ownership_and_stage": "Not identified in source material.",
        "esg_and_regulatory": "Not identified in source material.",
    }


def structured_overview_to_text(structured: dict) -> str:
    """
    Flattens the structured dict into a single labelled text block.
    This is what the topic agent receives as its primary input — easier
    to quote evidence from than raw JSON.
    """
    labels = {
        "business_model": "Business Model",
        "products_and_services": "Products & Services",
        "target_customers": "Target Customers",
        "technology_and_ip": "Technology & IP",
        "market_and_geography": "Market & Geography",
        "competitive_positioning": "Competitive Positioning",
        "ownership_and_stage": "Ownership & Stage",
        "esg_and_regulatory": "ESG & Regulatory",
    }
    lines = []
    for key, label in labels.items():
        value = structured.get(key, "Not identified in source material.")
        lines.append(f"{label}: {value}")
    return "\n".join(lines)