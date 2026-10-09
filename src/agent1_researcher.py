# src/agent1_researcher.py
"""
Agent 1 — The Researcher.
Browses company website and searches for information, then submits a structured profile.
Uses Claude Sonnet 5 via tool-loop (MAX_STEPS=10).
"""
import os
import sys
import asyncio
import concurrent.futures
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_core import ToolSpec, ToolCall, run_tool_agent, AgentResult
from scraper_direct import scrape_page, search_company_fallback

MAX_STEPS = 10

SYSTEM_PROMPT = """You are a senior private equity research analyst.

Your mission: research a company by browsing its website and searching for information, then submit a structured profile.

TOOLS:
- browse_page(url): fetches a web page. Use this to read company website pages.
- search_company(query): runs a Google search. Use when website content is unclear or thin.
- submit_profile(...): submits your final profile. YOU MUST CALL THIS after gathering enough information.

STRICT WORKFLOW:
1. Browse the homepage first
2. Browse 2-4 relevant subpages (products, about, services, solutions)
3. If content is unclear, run 1-2 searches
4. CALL submit_profile — DO NOT keep browsing indefinitely

IMPORTANT: You have a maximum of 10 tool calls total. After browsing 5-7 pages, you MUST call submit_profile with whatever information you have gathered. Do not spend all steps browsing — save at least 1 step for submit_profile.

PROFILE FIELDS (all required in submit_profile):
- business_description: 2-3 sentences on what the company fundamentally does
- products_and_services: specific named products/services with brief descriptions  
- target_customers: who buys from them, which industries/verticals
- technology_and_approach: how they deliver value (SaaS, hardware, AI, services, etc.)
- key_differentiators: what makes them distinct from competitors
- sources_visited: list of URLs browsed and searches run"""


def _browse_sync(url: str) -> str:
    """Runs async scrape_page in a thread pool to avoid event loop conflicts."""
    def _run():
        return asyncio.run(scrape_page(url))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            result = future.result(timeout=90)

        content = result.get("content", "").strip()
        if not content:
            return f"[No content from {url} — page may be blocked or empty]"
        if len(content) > 6000:
            content = content[:6000] + "\n[content truncated]"
        return f"=== {url} ===\n{content}"
    except Exception as e:
        return f"[Error browsing {url}: {e}]"


def _search_sync(query: str) -> str:
    """Runs search_company_fallback synchronously."""
    try:
        result = search_company_fallback(query)
        content = result.get("content", "").strip()
        if not content:
            return f"[No search results for: {query}]"
        if len(content) > 4000:
            content = content[:4000] + "\n[truncated]"
        return f"=== SEARCH: {query} ===\n{content}"
    except Exception as e:
        return f"[Search error: {e}]"


def _validate_profile(
    business_description: str,
    products_and_services: str,
    target_customers: str,
    technology_and_approach: str,
    key_differentiators: str,
    sources_visited: list,
) -> dict:
    return {
        "business_description":    business_description.strip(),
        "products_and_services":   products_and_services.strip(),
        "target_customers":        target_customers.strip(),
        "technology_and_approach": technology_and_approach.strip(),
        "key_differentiators":     key_differentiators.strip(),
        "sources_visited":         sources_visited,
        "full_profile": "\n\n".join([
            f"Business: {business_description.strip()}",
            f"Products & Services: {products_and_services.strip()}",
            f"Target Customers: {target_customers.strip()}",
            f"Technology & Approach: {technology_and_approach.strip()}",
            f"Key Differentiators: {key_differentiators.strip()}",
        ]),
    }


def _build_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="browse_page",
            description="Fetches a web page and returns its text content. Use to read company website pages.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL including https://"}
                },
                "required": ["url"],
            },
            handler=lambda url: _browse_sync(url),
        ),
        ToolSpec(
            name="search_company",
            description="Runs a Google search. Use when website is thin or to clarify product names.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"}
                },
                "required": ["query"],
            },
            handler=lambda query: _search_sync(query),
        ),
        ToolSpec(
            name="submit_profile",
            description=(
                "Submit your final company profile. CALL THIS after browsing 5-7 pages. "
                "All fields required. This ends the research phase."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "business_description":    {"type": "string"},
                    "products_and_services":   {"type": "string"},
                    "target_customers":        {"type": "string"},
                    "technology_and_approach": {"type": "string"},
                    "key_differentiators":     {"type": "string"},
                    "sources_visited": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "business_description", "products_and_services",
                    "target_customers", "technology_and_approach",
                    "key_differentiators", "sources_visited",
                ],
            },
            handler=lambda business_description, products_and_services,
                           target_customers, technology_and_approach,
                           key_differentiators, sources_visited: _validate_profile(
                business_description, products_and_services,
                target_customers, technology_and_approach,
                key_differentiators, sources_visited,
            ),
        ),
    ]


def run_researcher_agent(
    company_name: str,
    website_url: str,
    llm_client,
) -> dict:
    user_prompt = (
        f"Research this company and build a profile.\n\n"
        f"Company: {company_name}\n"
        f"Website: {website_url}\n\n"
        f"Start by browsing {website_url}. Then browse 3-5 subpages. "
        f"Then call submit_profile. You have 10 tool calls — use at most 8 for browsing, "
        f"then MUST call submit_profile."
    )

    result: AgentResult = run_tool_agent(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tools=_build_tools(),
        llm_client=llm_client,
        final_tool_name="submit_profile",
        max_steps=MAX_STEPS,
    )

    if result.status != "success" or not result.payload:
        # if agent ran out of steps, try to recover by running a search fallback
        print(f"  [WARN] Researcher hit max steps — falling back to search")
        search_result = _search_sync(f"{company_name} company products services overview")
        fallback_profile = _validate_profile(
            business_description=f"Research compiled from web search for {company_name}.",
            products_and_services=search_result[:2000],
            target_customers="See search results above.",
            technology_and_approach="See search results above.",
            key_differentiators="See search results above.",
            sources_visited=[f"search: {company_name}"],
        )
        fallback_profile["agent_status"] = "fallback"
        fallback_profile["steps_used"]   = result.steps_used
        return fallback_profile

    payload = result.payload
    payload["agent_status"] = "success"
    payload["steps_used"]   = result.steps_used
    return payload