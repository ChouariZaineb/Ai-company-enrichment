import json
import asyncio
import glob
import os
import streamlit as st

from src.scraper_direct import scrape_company_website
from src.overview_generator import generate_categorized_overview
from src.sector_matcher import match_sectors_to_taxonomy
from src.topic_matcher import match_topics_by_sector
from src.loader import build_taxonomy
from src.cache_utils import load_cached_result, save_cached_result
from src.accuracy import score_all_companies, get_model_detail, MODEL_CONFIG

COMPANIES_PATH = "data/sample_input 1.json"

st.set_page_config(page_title="Company Research Demo", page_icon="🔎", layout="wide")
st.title("🔎 Company Research Pipeline Demo")


@st.cache_resource
def load_taxonomy_once():
    return build_taxonomy(
        sectors_path="data/sectors_formatted.txt",
        topics_path="data/topics_formatted.txt",
    )


@st.cache_data
def load_companies():
    with open(COMPANIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def render_list_section(title: str, items):
    st.subheader(title)
    if items and isinstance(items, list):
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.markdown("_No data found for this section._")


def run_pipeline_for_company(company: dict, taxonomy: dict) -> dict:
    display_name = company.get("company_display_name") or company.get("company_name")
    website      = company.get("website")

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
    business_summary  = overview.get("business_summary", "") or ""

    sector_match_result = match_sectors_to_taxonomy(
        company_name=display_name,
        business_summary=business_summary,
        generated_sectors=generated_sectors,
        sectors_prompt_text=taxonomy["sectors_prompt_text"],
        valid_sector_ids=taxonomy["valid_sector_ids"],
    )

    sector_id_to_name   = {s.sector_id: s.sector_name for s in taxonomy["sectors"]}
    matched_sector_names = [
        f"{sid} | {sector_id_to_name.get(sid, 'UNKNOWN ID')}"
        for sid in sector_match_result.get("matched_sector_ids", [])
    ]

    result = dict(overview)
    result["company_id"]               = company.get("company_id")
    result["matched_sector_ids"]       = sector_match_result.get("matched_sector_ids", [])
    result["matched_sector_names"]     = matched_sector_names
    result["sector_matching_reasoning"] = sector_match_result.get("reasoning", "")
    result["topics_by_sector"]         = None
    result["_scrape_meta"] = {
        "pages_scraped": scrape_result["pages_scraped"],
        "used_fallback": scrape_result["used_fallback"],
    }

    return result


def ensure_topics_matched(result: dict, taxonomy: dict, company_id, display_name: str) -> tuple:
    if result.get("topics_by_sector"):
        return result, False

    sector_id_to_name    = {s.sector_id: s.sector_name for s in taxonomy["sectors"]}
    matched_sectors_input = [
        {"sector_id": sid, "sector_name": sector_id_to_name.get(sid, "UNKNOWN ID")}
        for sid in result.get("matched_sector_ids", [])
    ]

    if not matched_sectors_input:
        result["topics_by_sector"] = []
        save_cached_result(company_id, display_name, result)
        return result, True

    topics_result = match_topics_by_sector(
        company_name=display_name,
        business_summary=result.get("business_summary", "") or "",
        matched_sectors=matched_sectors_input,
        topics=taxonomy["topics"],
    )

    result["topics_by_sector"] = topics_result
    save_cached_result(company_id, display_name, result)
    return result, True


def _render_structured_overview(selected_company: str, overview_key: str):
    """Renders the PE-format structured overview stored in the cache."""
    for path in glob.glob(os.path.join("data", "overviews_cache", "*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("company_name") == selected_company:
            structured = data.get(overview_key)
            if structured:
                st.divider()
                st.markdown("**PE-Format Structured Overview (Agent Input)**")
                labels = {
                    "business_model":          "Business Model",
                    "products_and_services":   "Products & Services",
                    "target_customers":        "Target Customers",
                    "technology_and_ip":       "Technology & IP",
                    "market_and_geography":    "Market & Geography",
                    "competitive_positioning": "Competitive Positioning",
                    "ownership_and_stage":     "Ownership & Stage",
                    "esg_and_regulatory":      "ESG & Regulatory",
                }
                for key, label in labels.items():
                    value = structured.get(key, "Not identified in source material.")
                    st.markdown(f"**{label}:** {value}")
            break


def render_model_page(model_key: str):
    config = MODEL_CONFIG[model_key]
    st.header(f"📊 {config['label']}")
    st.caption("Accuracy against supervisor-provided ground truth (Monzo, Darktrace, Octopus Energy).")
    st.caption("Topic scoring: recall only (extra grounded topics are not penalized). Sector scoring: full precision/recall/F1.")

    scores = score_all_companies(model_key)
    if not scores:
        st.warning(f"No scored companies found yet for {config['label']}.")
        return

    average_f1 = scores.pop("_AVERAGE_", {}).get("overall_f1")

    st.subheader("Summary")
    summary_rows = []
    for company_name, s in scores.items():
        row = {
            "Company":    company_name,
            "Sector F1":  s["sector_scores"]["f1"],
            "Topic Recall": s["topic_scores"]["recall"],
            "Overall F1": s["overall"]["f1"],
        }
        # sector precision/recall only for non-agent models (they have full p/r/f1)
        if "precision" in s["sector_scores"]:
            row["Sector Precision"] = s["sector_scores"]["precision"]
            row["Sector Recall"]    = s["sector_scores"]["recall"]
        summary_rows.append(row)
    st.table(summary_rows)

    if average_f1 is not None:
        st.metric("Average Overall F1 (recall-based)", average_f1)

    st.divider()

    st.subheader("Per-Company Detail")
    selected_company = st.selectbox(
        "Select a company", list(scores.keys()), key=f"select_{model_key}"
    )

    if selected_company:
        s = scores[selected_company]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Sector Match**")
            st.json(s["sector_scores"])
        with col2:
            st.markdown("**Topic Match (recall only)**")
            st.json(s["topic_scores"])
        with col3:
            st.markdown("**Overall**")
            st.json(s["overall"])

        detail = get_model_detail(model_key, selected_company)
        if detail:
            st.divider()
            st.markdown("**Matched Sectors**")
            for sid in detail["sector_ids"]:
                st.markdown(f"- {sid}")
            if detail["sector_reasoning"]:
                with st.expander("Sector reasoning"):
                    st.markdown(detail["sector_reasoning"])

            st.markdown("**Matched Topics**")
            for t in detail["topics_flat"]:
                st.markdown(f"- {t['topic_id']} | {t['topic_name']}")

            if detail["topic_reasoning"]:
                with st.expander("Topic evidence (per topic)"):
                    try:
                        evidence_list = json.loads(detail["topic_reasoning"])
                        for item in evidence_list:
                            st.markdown(f"**{item['topic_id']} | {item['topic_name']}**")
                            st.markdown(f"> {item['evidence']}")
                            st.markdown("---")
                    except Exception:
                        st.markdown(detail["topic_reasoning"])

        # show PE structured overview for agent models
        if model_key == "agent":
            _render_structured_overview(selected_company, "structured_overview_agent")
        elif model_key == "agent_deepseek":
            _render_structured_overview(selected_company, "structured_overview_agent_deepseek")
        elif model_key == "agent_weighted_gemini":
            _render_structured_overview(selected_company, "structured_overview_agent_weighted_gemini")
        elif model_key == "agent_new_claude":
            _render_structured_overview(selected_company, "structured_overview_new")
        elif model_key == "agent_new_gemini":
            _render_structured_overview(selected_company, "structured_overview_new")
# --- Sidebar routing ---
page = st.sidebar.radio(
    "View",
    [
        "Company Pipeline (raw data)",
        "Mistral (local)",
        "Gemini Pro",
        "Claude (Bedrock)",
        "Claude (Bedrock) — Enriched Embeddings",
        "DeepSeek 3.2 (Bedrock)",
        "DeepSeek 3.2 (Bedrock) — Enriched Embeddings",
        "Claude Agent (Bedrock)",
        "DeepSeek Agent (Bedrock)",
        "Claude Agent — Weighted (Bedrock) ",
        "Gemini Pro Agent — Weighted",
        "Claude Agent — New Companies",
        "Gemini Pro Agent — New Companies",
        "Taxonomy Suggestions",
    ],
)

PAGE_TO_MODEL_KEY = {
    "Mistral (local)":                              "mistral",
    "Gemini Pro":                                   "gemini_pro",
    "Claude (Bedrock)":                             "bedrock",
    "Claude (Bedrock) — Enriched Embeddings":       "bedrock_v2",
    "DeepSeek 3.2 (Bedrock)":                       "deepseek",
    "DeepSeek 3.2 (Bedrock) — Enriched Embeddings": "deepseek_v2",
    "Claude Agent (Bedrock)":                       "agent",
    "DeepSeek Agent (Bedrock)":                     "agent_deepseek",
    "Claude Agent — Weighted (Bedrock) ":           "agent_weighted",
    "Gemini Pro Agent — Weighted":                  "agent_weighted_gemini",
    "Claude Agent — New Companies":                 "agent_new_claude",
    "Gemini Pro Agent — New Companies":              "agent_new_gemini",
}

if page in PAGE_TO_MODEL_KEY:
    render_model_page(PAGE_TO_MODEL_KEY[page])
    st.stop()

if page == "Taxonomy Suggestions":
    st.header("🧪 Taxonomy Gap Suggestions")
    st.caption("Terms the agent identified as missing from the taxonomy. Review and approve/reject.")

    suggestions_path = os.path.join("data", "taxonomy_suggestions.json")
    if not os.path.exists(suggestions_path):
        st.info("No suggestions logged yet. Run the new companies pipeline first.")
        st.stop()

    with open(suggestions_path, encoding="utf-8") as f:
        suggestions = json.load(f)

    if not suggestions:
        st.info("No suggestions found.")
        st.stop()

    # filters
    col1, col2 = st.columns(2)
    with col1:
        conf_filter = st.multiselect(
            "Filter by confidence",
            ["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM"],
        )
    with col2:
        status_filter = st.multiselect(
            "Filter by status",
            ["pending", "approved", "rejected"],
            default=["pending"],
        )

    filtered = [
        s for s in suggestions
        if s.get("confidence") in conf_filter
        and s.get("status", "pending") in status_filter
    ]

    st.markdown(f"**{len(filtered)} suggestion(s)** matching filters ({len(suggestions)} total)")
    st.divider()

    for s in filtered:
        with st.expander(
            f"[{s.get('confidence','?')}] {s['suggested_term']} — {s['company_name']} (id {s['company_id']})"
        ):
            st.markdown(f"**Description:** {s.get('description', 'N/A')}")
            st.markdown(f"**Evidence:** {s.get('evidence', 'N/A')}")
            st.markdown(f"**Status:** `{s.get('status', 'pending')}`")
            st.markdown(f"**Logged:** {s.get('logged_at', 'N/A')}")

    st.stop()
# --- Company Pipeline page ---
taxonomy  = load_taxonomy_once()
companies = load_companies()
companies_by_id = {c["company_id"]: c for c in companies}

with st.expander("Available company IDs", expanded=False):
    for c in companies:
        st.markdown(f"- **{c['company_id']}** — {c.get('company_display_name') or c.get('company_name')}")

with st.form("company_id_form"):
    company_id_input = st.number_input("Company ID", min_value=0, step=1, format="%d")
    force_refresh    = st.checkbox("Force re-run (ignore cache)", value=False)
    run_clicked      = st.form_submit_button("Load / Generate")

if run_clicked:
    company_id = int(company_id_input)
    company    = companies_by_id.get(company_id)

    if not company:
        st.error(f"No company found with ID {company_id} in {COMPANIES_PATH}.")
    else:
        display_name = company.get("company_display_name") or company.get("company_name")
        cached = None if force_refresh else load_cached_result(company_id, display_name)

        if cached:
            st.info("Loaded from cache — no scraping or Gemini calls made.")
            result = cached
        else:
            try:
                with st.spinner(f"Running full pipeline for {display_name}..."):
                    result = run_pipeline_for_company(company, taxonomy)
                    save_cached_result(company_id, display_name, result)
                st.success("Done — saved to cache.")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                result = None

        if result:
            with st.spinner("Matching topics to sectors (Mistral)..."):
                result, topics_newly_computed = ensure_topics_matched(
                    result, taxonomy, company_id, display_name
                )
            if topics_newly_computed:
                st.success("Mistral topic matching complete — saved to cache.")

            st.session_state["current_result"]       = result
            st.session_state["current_company_id"]   = company_id
            st.session_state["current_display_name"] = display_name
            st.session_state["current_company"]      = company

if "current_result" in st.session_state:
    result       = st.session_state["current_result"]
    company      = st.session_state["current_company"]
    display_name = st.session_state["current_display_name"]

    source_display = (
        result.get("source")
        or company.get("website")
        or "(no website — search fallback used)"
    )

    st.header(f"{result.get('company_name', display_name)} – Company Overview")
    st.caption(f"Source: {source_display}")

    st.subheader("Categorised Overview")
    render_list_section("Industry",                    result.get("industry", []))
    render_list_section("Sectors",                     result.get("sectors", []))
    render_list_section("Products Offered",            result.get("products_offered", []))
    render_list_section("Services Offered",            result.get("services_offered", []))
    render_list_section("Unique Selling Propositions", result.get("unique_selling_propositions", []))

    st.subheader("Business Summary")
    summary = (result.get("business_summary") or "").strip()
    st.markdown(summary if summary else "_No summary generated._")

    st.info("Switch to a model page in the sidebar to see its accuracy scores and matched sectors/topics.")

    with st.expander("Debug / Trace Details", expanded=False):
        meta = result.get("_scrape_meta", {})
        st.markdown("**Pages scraped:**")
        pages_scraped = meta.get("pages_scraped", [])
        if pages_scraped:
            for p in pages_scraped:
                st.markdown(f"- {p}")
        else:
            st.markdown("_No pages scraped._")
        st.markdown(f"**Google Search fallback used:** `{meta.get('used_fallback', False)}`")