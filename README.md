# Actum AI Company Enrichment

An AI-powered system for automated company research and classification, designed for private equity analysis. The system extracts structured information from company websites, matches them to predefined sector and topic taxonomies, and evaluates accuracy across multiple LLM models.

## Features

- **Automated Web Scraping**: Intelligent website content extraction with anti-bot detection and Google Search fallback
- **Structured Overview Generation**: Extracts industry, sectors, products, services, USPs, and business summaries using Gemini Pro
- **Taxonomy Classification**: Matches companies to official sector and topic taxonomies using local Mistral (Ollama) with semantic embeddings
- **Multi-Agent Pipeline**: Three-agent system (Researcher, Sector Classifier, Topic Specialist) using Claude Sonnet 5 via AWS Bedrock
- **Model Comparison**: Evaluates 15+ model variants across different LLMs with accuracy scoring against ground truth
- **Taxonomy Gap Detection**: Identifies missing taxonomy terms and logs suggestions for review
- **Caching System**: Persists results to avoid redundant API calls with resumable progress tracking

## Technologies

### Large Language Models
- **Google Gemini Pro**: Overview generation, Google Search grounding fallback, completeness checks
- **AWS Bedrock Claude Sonnet 3.5**: Sector and topic matching in single-call variants
- **AWS Bedrock Claude Sonnet 5**: Multi-agent pipeline (researcher, sector classifier, topic specialist)
- **AWS Bedrock DeepSeek 3.2**: Alternative sector and topic matching variants
- **Local Ollama Mistral 7B**: Sector matching, topic classification (core pipeline)
- **Local Ollama nomic-embed-text**: Semantic embeddings for topic candidate narrowing

### Web Scraping & Search
- **crawl4ai**: AsyncWebCrawler with BrowserConfig (stealth mode, headless), CrawlerRunConfig (consent popup removal, overlay removal)
- **httpx**: HTTP client for Gemini API calls and search fallback
- **Google Search API**: Grounding via Gemini Pro for fallback when scraping fails

### Data Processing
- **Pydantic**: Data validation for Sector and Topic models
- **NumPy**: Cosine similarity calculations for embeddings
- **Python JSON**: Persistence for cache, taxonomy data, and results

### User Interface
- **Streamlit**: Web interface for pipeline demo, model accuracy visualization, taxonomy suggestions review

### Cloud Infrastructure
- **AWS SDK (boto3)**: Bedrock runtime client for Claude and DeepSeek models
- **Environment Variables**: python-dotenv for API key and credential management

## Architecture

### Core Pipeline

1. **Scraper** (`scraper_direct.py`):
   - **Technology**: crawl4ai with AsyncWebCrawler, BrowserConfig with stealth mode, httpx for fallback
   - Scrapes homepage and selects relevant subpages using keyword matching (about, products, services, solutions, sectors)
   - Removes boilerplate content (repeated lines across pages, cookie consent overlays)
   - Falls back to Gemini Pro with Google Search grounding if scraping fails or content is insufficient
   - Isolates each page fetch to avoid session issues with anti-bot systems

2. **Overview Generator** (`overview_generator.py`):
   - **Technology**: Google Gemini Pro API with key rotation via gemini_key_manager.py
   - Generates structured JSON with: industry, sectors, products_offered, services_offered, unique_selling_propositions, business_summary
   - Distinguishes between customer industries vs. company's own sector identity
   - Completeness fallback: triggers additional Google Search if key sections (sectors, products, services) are empty
   - Retries on 503/429 errors with exponential backoff

3. **Sector Matcher** (`sector_matcher.py`):
   - **Technology**: Local Ollama Mistral 7B with ollama.chat
   - Matches informally-generated sector tags to official taxonomy IDs
   - Validates against valid_sector_ids to prevent hallucinations
   - Uses detailed prompts with worked examples to guide the model
   - Returns matched_sector_ids and reasoning

4. **Topic Matcher** (`topic_matcher.py`):
   - **Technology**: Local Ollama nomic-embed-text for embeddings, Mistral 7B for classification
   - **Embedding stage**: Computes cosine similarity between company profile and all topics, narrows to top 150 per sector
   - **Batching**: Processes topics in chunks of 60 with progress tracking (resumable if interrupted)
   - **Per-sector matching**: Runs separately for each matched sector to prevent cross-contamination
   - Returns nested structure: sector_id, sector_name, topics (with topic_id, topic_name), reasoning

### Multi-Agent Pipeline

The multi-agent pipeline uses Claude Sonnet 5 via AWS Bedrock with a three-agent architecture:

#### Agent 1: Researcher
- **Technology**: Claude Sonnet 5 with tool-loop (max 10 steps), browse_page and search_company tools
- **Process**:
  1. Browses company homepage
  2. Browses 2-4 relevant subpages (products, about, services)
  3. Runs Google Search if content is unclear
  4. Submits structured profile with: business_description, products_and_services, target_customers, technology_and_approach, key_differentiators, sources_visited
- **Fallback**: If agent hits max steps, uses search fallback to compile profile
- **Output**: Full research profile saved to cache

#### Agent 2: Sector Classifier
- **Technology**: Claude Sonnet 5 single-call inference
- **Process**:
  - Takes research profile as input
  - Classifies into at most 3 sectors from official taxonomy
  - Distinguishes between customer industries vs. company's own sector identity
  - Provides evidence-based reasoning for each sector
- **Output**: matched_sector_ids (max 3), sector_reasoning

#### Agent 3: Topic Specialist
- **Technology**: Claude Sonnet 5 with tool-use (select_themes, select_topics tools)
- **Process**:
  1. **Theme Selection**: Selects relevant theme groups from topic_subcategories.json (grouped by sector)
  2. **Scanning**: For each selected theme, flags all plausibly relevant topics
  3. **Strength Filtering**: If >40 candidates, runs AI review in batches of 25 to remove weak matches
  4. **Final Selection**: Picks up to 15 topics using usage counts, company language matching, and evidence strength
  5. **Synonym Expansion**: Adds near-duplicate topics of already-selected topics
- **Output**: matched_topics_flat (max 15), topic_reasoning (with evidence per topic), taxonomy_suggestions

### Topic Data Structures

The system uses multiple data files to enrich topic matching:

- **topic_types.json**: Classifies each topic as "category" (broad industry category) or "feature" (specific capability/technology)
- **topic_descriptions.json**: Provides concise descriptions for each topic (e.g., "electrical energy generation and supply")
- **topic_sectors.json**: Maps each topic ID to its valid sector IDs (a topic can belong to multiple sectors)
- **topic_subcategories.json**: Groups topics into themes within each sector (e.g., "Digital Advertising & AdTech", "Branding & Brand Strategy")

Note: Usage statistics (`topic_metadata.json`) and summary files are excluded from the public repository as they contain proprietary company data.

### Category vs Feature Distinction

When matching topics, the system uses the category/feature classification to guide selection:

- **Category topics**: Broad industry or market segments (e.g., "Financial Services", "Healthcare"). Used to describe the company's primary business domain.
- **Feature topics**: Specific capabilities, technologies, or methods (e.g., "Artificial Intelligence", "Cloud Computing"). Used to describe how the company delivers value.

The multi-agent pipeline prioritizes selecting a balanced mix of category and feature topics to capture both what the company does (activity) and what the company is (identity).

### Topic-to-Sector Matching

Topics are matched to sectors through a multi-stage process:

1. **Pre-computed mappings**: topic_sectors.json defines which sectors each topic can belong to
2. **Sector-based scoping**: When Agent 3 runs, it only considers topics that belong to the sectors already matched by Agent 2
3. **Theme grouping**: topic_subcategories.json organizes topics into thematic groups within each sector (e.g., within sector 7, themes include "AI & Machine Learning", "Cloud Infrastructure", "Enterprise Software")
4. **Cross-sector topics**: Topics mapped to multiple sectors can be selected under any of their valid sectors

### Company Metadata Usage

The system leverages company metadata from the input JSON:

- **company_id**: Unique identifier for caching and ground truth comparison
- **company_name / company_display_name**: Used for search queries and context
- **website**: Primary URL for scraping; if missing, skips directly to search fallback
- **hq_city_town, country, founded_in**: Used for context in search queries (e.g., "Monzo London UK fintech 2015")
- **ch_company_profile**: Existing Companies House description used as supplementary context
- **is_advisor**: Flag for advisor vs. operating companies (affects classification approach)

### Key Modules

- `app.py`: Streamlit interface for pipeline demo and model accuracy visualization
- `accuracy.py`: Ground truth scoring for 12 test companies with precision/recall/F1 metrics
- `loader.py`: Taxonomy loading from text files with Pydantic validation
- `cache_utils.py`: Result persistence and progress tracking with resumable batches
- `taxonomy_suggester.py`: Gap detection and suggestion logging to taxonomy_suggestions.json
- `gemini_key_manager.py`: API key rotation across multiple free-tier keys
- `bedrock_client_sonnet5.py`: AWS Bedrock adapter for Claude Sonnet 5
- `agent_core.py`: Core tool-loop infrastructure for multi-agent pipeline

## Installation

### Prerequisites

- Python 3.x
- Ollama installed locally
- Google Gemini Pro API key(s)
- AWS credentials (if using Bedrock models)

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install and start Ollama, then pull required models:
```bash
ollama pull mistral:7b
ollama pull nomic-embed-text
```

3. Configure environment variables:
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual API keys
```

Required credentials in `.env`:
- `GEMINI_API_KEY`: Gemini API keys (comma-separated for rotation)
- `GEMINI_PRO_API_KEY`: Dedicated Gemini Pro key
- `AWS_ACCESS_KEY_ID`: AWS access key for Bedrock
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for Bedrock
- `AWS_REGION`: AWS region (e.g., us-east-1)
- `BEDROCK_INFERENCE_PROFILE_ARN_sonnet5`: ARN for Claude Sonnet 5 inference profile

## Usage

### Streamlit Interface

Run the web interface:
```bash
streamlit run app.py
```

- **Company Pipeline (raw data)**: Run individual companies through the pipeline
- **Model Pages**: View accuracy scores and matched sectors/topics for each model variant
- **Taxonomy Suggestions**: Review and manage taxonomy gap suggestions

### Multi-Agent Pipeline

Process companies with the multi-agent system:
```bash
python src/run_multi_agent_pipeline.py
```

This runs the three-agent pipeline on 7 companies with ground truth, outputting sector/topic recall and diagnostic information.

### Data Format

Company input file (`data/sample_input_demo.json`) format:
```json
{
  "company_id": 1002,
  "company_name": "Monzo Bank Limited",
  "company_display_name": "Monzo",
  "website": "https://monzo.com",
  ...
}
```

Taxonomy files use `id | name` format:
```
7 | Computing, Technology, Robotics & AI
16 | Financial, Investment and Insurance Services
```

**Included taxonomy files:**
- `sectors_formatted.txt` - Official sector taxonomy
- `topics_formatted.txt` - Official topic taxonomy
- `topic_types.json` - Category/feature classifications
- `topic_descriptions.json` - Topic descriptions
- `topic_sectors.json` - Topic-to-sector mappings
- `topic_subcategories.json` - Theme groupings

**Excluded (proprietary):**
- `topic_metadata.json` - Usage statistics with company counts (used for topic weighting)
- `topic_subcategories_summary.txt` - Internal taxonomy summaries
- CSV files with company-specific topic assignments

**Note:** The code includes fallback logic to handle missing proprietary files. Without `topic_metadata.json`, the system will run without usage-based topic weighting, but core functionality remains intact.

## Configuration Notes

- The system distinguishes between customer industries and a company's own sector identity when classifying
- Topic embeddings are computed once and cached (~180MB file)
- Ground truth data is hardcoded in `accuracy.py` for 12 companies (Monzo, Darktrace, Octopus Energy, etc.)
- Caching is enabled by default to avoid redundant API calls; use "Force re-run" in the UI to bypass

## Model Variants Tested

The system evaluates accuracy across 15+ model variants:
- Mistral (local)
- Gemini Pro
- Claude (Bedrock)
- DeepSeek (Bedrock)
- Multiple agent-based variants with different LLM backends
- Enriched embedding variants

Sector scoring uses precision/recall/F1. Topic scoring uses recall-only (extra grounded topics not penalized). Overall score averages sector and topic recall.
