from urllib.parse import urljoin, urlparse
from collections import Counter
import os
import httpx

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from src.gemini_key_manager import build_gemini_pro_url


RELEVANT_KEYWORDS = [
    "about", "product", "service", "solution", "platform", "sectors",
    "what-we-do", "who-we-are", "company", "offering",
]

MAX_PAGES = 7
MAX_PER_KEYWORD = 2  # no single keyword can fill the whole page budget

BROWSER_CONFIG = BrowserConfig(    #hides automation fingerprints (things that reveal "this is a script, not a human") that anti-bot systems look for.
    headless=True,
    enable_stealth=True,
)

PAGE_CONFIG_KWARGS = dict(
    remove_consent_popups=True,
    remove_overlay_elements=True,
    excluded_tags=["nav", "footer", "header", "form"],
    word_count_threshold=10,
    cache_mode=CacheMode.BYPASS,
    magic=True,
    wait_until="load",
    max_retries=0,  # Monzo-style hard blocks never succeed on retry — don't pay for it
)


async def scrape_page(url: str) -> dict:
    """
    Scrape a single page. Each call opens its OWN isolated AsyncWebCrawler/browser
    context, deliberately not shared across pages. Deep crawling shares one browser
    session across every page in the crawl, and combined with anti-bot retries that
    session can get torn down mid-crawl — a known open Crawl4AI issue (e.g.
    github.com/unclecode/crawl4ai/issues/1379, /issues/842). Isolating each page
    fetch avoids it entirely.
    """
    config = CrawlerRunConfig(**PAGE_CONFIG_KWARGS)
    try:
        async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
            result = await crawler.arun(url=url, config=config)
            if result.success:
                links = result.links.get("internal", []) if hasattr(result, "links") else []
                return {"content": result.markdown, "links": links}
            else:
                print(f"[WARN] Scraping failed for {url}: {result.error_message}")
                return {"content": "", "links": []}
    except Exception as e:
        print(f"[ERROR] Exception while scraping {url}: {e}")
        return {"content": "", "links": []}


def select_relevant_links(links: list, base_url: str,
                           max_per_keyword: int = MAX_PER_KEYWORD,
                           max_total: int = MAX_PAGES) -> list:
    """
    Bucket same-domain links by which keyword matched, cap each bucket at
    max_per_keyword, then round-robin across buckets until max_total is reached.
    Fixes one keyword (e.g. "sectors") silently filling the whole page budget
    with lookalike pages while others (solutions, about) get none.
    """
    base_domain = urlparse(base_url).netloc
    buckets: dict[str, list[str]] = {kw: [] for kw in RELEVANT_KEYWORDS}
    seen = set()

    for link in links:
        href = link.get("href") if isinstance(link, dict) else link
        if not href:
            continue
        full_url = urljoin(base_url, href)
        if full_url in seen or urlparse(full_url).netloc != base_domain:
            continue

        lower_url = full_url.lower()
        matched_kw = next((kw for kw in RELEVANT_KEYWORDS if kw.lower() in lower_url), None)
        if matched_kw and len(buckets[matched_kw]) < max_per_keyword:
            buckets[matched_kw].append(full_url)
            seen.add(full_url)

    selected = []
    active_buckets = [urls for urls in buckets.values() if urls]
    idx = 0
    while active_buckets and len(selected) < max_total:
        bucket = active_buckets[idx % len(active_buckets)]
        selected.append(bucket.pop(0))
        active_buckets = [b for b in active_buckets if b]
        idx += 1

    return selected[:max_total]


def remove_repeated_boilerplate(pages_content: list[str], min_line_length: int = 3) -> list[str]:
    """Defensive second layer on top of excluded_tags, for boilerplate outside semantic tags."""
    pages_lines = [content.split("\n") for content in pages_content]

    line_counter = Counter()
    for lines in pages_lines:
        unique_lines_in_page = set(line.strip() for line in lines if len(line.strip()) >= min_line_length)
        for line in unique_lines_in_page:
            line_counter[line] += 1

    threshold = max(2, len(pages_content) // 2)
    boilerplate_lines = {line for line, count in line_counter.items() if count >= threshold}

    cleaned_pages = []
    for lines in pages_lines:
        cleaned = [line for line in lines if line.strip() not in boilerplate_lines]
        cleaned_pages.append("\n".join(cleaned))

    return cleaned_pages


COOKIE_KEYWORDS = ["cookie", "cookies", "consent", "accept all", "reject all"]
MIN_USEFUL_CONTENT_LENGTH = 500


def needs_fallback(scraped_content: str) -> bool:
    """Decide if the scraped website content is too thin OR too cookie-dominated to be useful."""
    content = scraped_content.strip()
    if len(content) < MIN_USEFUL_CONTENT_LENGTH:
        return True
    lower = content.lower()
    cookie_hits = sum(lower.count(kw) for kw in COOKIE_KEYWORDS)
    return cookie_hits >= 5 and len(content) < 3000


async def scrape_company_website(base_url: str, company_name: str = "") -> dict:
    """
    Scrape the homepage, pick a diverse set of relevant sub-pages, scrape those,
    remove boilerplate, and fall back to Google Search grounding if needed.
    """
    if not base_url:
        print(f"[INFO] No website provided for {company_name}, using search fallback.")
        fallback = search_company_fallback(company_name)
        return {
            "combined_content": fallback["content"],
            "pages_scraped": [],
            "used_fallback": True,
            "fallback_sources": fallback["sources"],
        }

    homepage_result = await scrape_page(base_url)
    homepage_content = homepage_result["content"]

    if not homepage_content:
        print(f"[WARN] Homepage scrape failed for {base_url}, using search fallback.")
        fallback = search_company_fallback(company_name)
        return {
            "combined_content": fallback["content"],
            "pages_scraped": [],
            "used_fallback": True,
            "fallback_sources": fallback["sources"],
        }

    relevant_links = select_relevant_links(homepage_result["links"], base_url)

    raw_contents = [homepage_content]
    pages_scraped = [base_url]

    for link in relevant_links:
        page_result = await scrape_page(link)
        if page_result["content"]:
            raw_contents.append(page_result["content"])
            pages_scraped.append(link)

    cleaned_contents = remove_repeated_boilerplate(raw_contents)

    all_content = [
        f"=== PAGE: {url} ===\n{content}"
        for url, content in zip(pages_scraped, cleaned_contents)
    ]
    combined_content = "\n\n".join(all_content)

    if needs_fallback(combined_content):
        print(f"[INFO] Scraped content too thin for {company_name}, adding search fallback.")
        fallback = search_company_fallback(company_name)
        combined_content += f"\n\n=== GOOGLE SEARCH FALLBACK ===\n{fallback['content']}"
        return {
            "combined_content": combined_content,
            "pages_scraped": pages_scraped,
            "used_fallback": True,
            "fallback_sources": fallback["sources"],
        }

    return {
        "combined_content": combined_content,
        "pages_scraped": pages_scraped,
        "used_fallback": False,
        "fallback_sources": [],
    }


def search_company_fallback(company_name: str) -> dict:
    """Use Gemini's Google Search grounding when direct scraping is missing or too thin.
    Calls the dedicated Pro key directly — no need to rotate through free-tier
    keys first now that the Pro key has real quota."""
    prompt = f"""Search for up-to-date, factual information about the company "{company_name}": what it does, its products/services, its industry, and any notable recent news.

Do not use Wikipedia as a source. Prefer the company's own official website, official press releases, reputable news outlets, or business databases (e.g. Crunchbase, LinkedIn, Bloomberg).

Summarize what you find in a few clear paragraphs."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
    }

    try:
        response = httpx.post(build_gemini_pro_url(), json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        candidate = data["candidates"][0]
        text = candidate["content"]["parts"][0]["text"].strip()

        sources = []
        grounding = candidate.get("groundingMetadata", {})
        for chunk in grounding.get("groundingChunks", []):
            uri = chunk.get("web", {}).get("uri", "")
            if uri:
                sources.append(uri)

        used_wikipedia = any("wikipedia.org" in s for s in sources)
        if used_wikipedia:
            print(f"[WARN] Gemini used Wikipedia as a source for {company_name} despite instructions.")

        return {"content": text, "sources": sources, "used_wikipedia": used_wikipedia}

    except httpx.HTTPStatusError as e:
        print(f"[ERROR] Google Search fallback (Pro key) failed for {company_name}: {e}")
        return {"content": "", "sources": [], "used_wikipedia": False}
    except Exception as e:
        print(f"[ERROR] Google Search fallback failed for {company_name}: {e}")
        return {"content": "", "sources": [], "used_wikipedia": False}