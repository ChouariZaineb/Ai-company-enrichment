"""
Batch runner: runs the full pipeline (scrape -> overview -> sector matching)
for every company in data/sample_input 1.json, caching each result via
cache_utils.save_cached_result (e.g. data/overviews_cache/1002_monzo.json).

Already-cached companies are skipped by default, so re-running this script
after an interruption (or when you just want to add new companies) picks up
where it left off without re-scraping / re-calling Gemini or Ollama for
companies you've already processed.

This intentionally reuses the exact same run_pipeline_for_company() logic
that app.py uses, so cached results stay consistent with what the Streamlit
interface would produce for the same company.

Usage:
    python -m src.run_all_companies                # run everything not yet cached
    python -m src.run_all_companies --force         # ignore cache, regenerate all
    python -m src.run_all_companies --only 1002 1005  # only these company_ids
"""

import argparse
import asyncio
import json
import sys
import time

from src.scraper_direct import scrape_company_website
from src.overview_generator import generate_categorized_overview
from src.sector_matcher import match_sectors_to_taxonomy
from src.loader import build_taxonomy
from src.cache_utils import load_cached_result, save_cached_result

COMPANIES_PATH = "data/sample_input 1.json"
SECTORS_PATH = "data/sectors_formatted.txt"
TOPICS_PATH = "data/topics_formatted.txt"
FAILURES_LOG_PATH = "data/overviews_cache/_failures.json"
DELAY_BETWEEN_COMPANIES_SECONDS = 2  # courtesy pause to avoid Gemini free-tier 429s


def load_companies() -> list:
    with open(COMPANIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_for_company(company: dict, taxonomy: dict) -> dict:
    """
    Scrape -> generate overview -> match sectors, for one company. Raises on failure.
    Identical logic to app.py's run_pipeline_for_company, kept in sync deliberately —
    if you change one, change the other (or factor this out into a shared module
    later, e.g. src/pipeline.py, once topic matching is wired in too).
    """
    display_name = company.get("company_display_name") or company.get("company_name")
    website = company.get("website")

    scrape_result = asyncio.run(scrape_company_website(website, display_name))

    if not scrape_result["combined_content"] or not scrape_result["combined_content"].strip():
        raise RuntimeError("No content came back from scraping or the search fallback.")

    overview = generate_categorized_overview(
        company_name=display_name,
        source_url=website or "",
        combined_content=scrape_result["combined_content"],
    )

    if not overview or not overview.get("business_summary"):
        raise RuntimeError("Overview generation failed — check the terminal for the Gemini error.")

    generated_sectors = overview.get("sectors", []) or []
    business_summary = overview.get("business_summary", "") or ""

    sector_match_result = match_sectors_to_taxonomy(
        company_name=display_name,
        business_summary=business_summary,
        generated_sectors=generated_sectors,
        sectors_prompt_text=taxonomy["sectors_prompt_text"],
        valid_sector_ids=taxonomy["valid_sector_ids"],
    )

    sector_id_to_name = {s.sector_id: s.sector_name for s in taxonomy["sectors"]}
    matched_sector_names = [
        f"{sid} | {sector_id_to_name.get(sid, 'UNKNOWN ID')}"
        for sid in sector_match_result.get("matched_sector_ids", [])
    ]

    result = dict(overview)  # industry, sectors, products_offered, services_offered, usps, business_summary, etc.
    result["company_id"] = company.get("company_id")
    result["matched_sector_ids"] = sector_match_result.get("matched_sector_ids", [])
    result["matched_sector_names"] = matched_sector_names
    result["sector_matching_reasoning"] = sector_match_result.get("reasoning", "")
    result["matched_topic_ids"] = None  # placeholder — filled in once topic_matcher.py is wired in
    result["_scrape_meta"] = {
        "pages_scraped": scrape_result["pages_scraped"],
        "used_fallback": scrape_result["used_fallback"],
    }

    return result


def load_failures_log() -> dict:
    try:
        with open(FAILURES_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_failures_log(failures: dict) -> None:
    import os
    os.makedirs("data/overviews_cache", exist_ok=True)
    with open(FAILURES_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Batch-run the enrichment pipeline for all companies.")
    parser.add_argument("--force", action="store_true",
                         help="Ignore existing cache and regenerate every company from scratch.")
    parser.add_argument("--only", nargs="+", type=int, default=None,
                         help="Only run these specific company_ids (space-separated), e.g. --only 1002 1005")
    args = parser.parse_args()

    taxonomy = build_taxonomy(sectors_path=SECTORS_PATH, topics_path=TOPICS_PATH)
    companies = load_companies()

    if args.only:
        companies = [c for c in companies if c.get("company_id") in args.only]
        if not companies:
            print(f"[ERROR] None of the requested ids {args.only} were found in {COMPANIES_PATH}.")
            sys.exit(1)

    total = len(companies)
    succeeded, skipped, failed = [], [], []
    failures_log = load_failures_log()

    print(f"[INFO] Starting batch run for {total} companies (force={args.force})\n")

    for idx, company in enumerate(companies, start=1):
        company_id = company.get("company_id")
        display_name = company.get("company_display_name") or company.get("company_name")
        label = f"[{idx}/{total}] {company_id} — {display_name}"

        if not args.force:
            cached = load_cached_result(company_id, display_name)
            if cached:
                print(f"{label}: already cached, skipping.")
                skipped.append(company_id)
                continue

        print(f"{label}: running pipeline...")
        start = time.time()
        try:
            result = run_pipeline_for_company(company, taxonomy)
            saved_path = save_cached_result(company_id, display_name, result)
            elapsed = time.time() - start
            print(f"{label}: done in {elapsed:.1f}s -> saved to {saved_path}\n")
            succeeded.append(company_id)
            failures_log.pop(str(company_id), None)  # clear a previous failure record, if any
        except Exception as e:
            elapsed = time.time() - start
            print(f"{label}: FAILED after {elapsed:.1f}s — {e}\n")
            failed.append((company_id, str(e)))
            failures_log[str(company_id)] = {"company_name": display_name, "error": str(e)}

        save_failures_log(failures_log)  # persist after every company, not just at the end

        if idx < total:
            time.sleep(DELAY_BETWEEN_COMPANIES_SECONDS)

    print("=" * 60)
    print(f"[SUMMARY] {len(succeeded)} succeeded, {len(skipped)} skipped (cached), {len(failed)} failed out of {total}")
    if failed:
        print("\nFailed companies:")
        for cid, err in failed:
            print(f"  - {cid}: {err}")
        print(f"\nDetails saved to {FAILURES_LOG_PATH}. Retry just these with:")
        print(f"  python -m src.run_all_companies --only {' '.join(str(cid) for cid, _ in failed)}")


if __name__ == "__main__":
    main()