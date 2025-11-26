# ArxivCode

Machine learning project for paper-code understanding and retrieval.

## Project Structure

```
arxivcode/
├── src/                        # Source code
│   ├── data_collection/        # Data collection pipeline
│   │   ├── arxiv_github_collector.py  # Main collector
│   │   └── __init__.py
│   ├── models/                 # Model training & inference
│   ├── retrieval/              # Retrieval system
│   └── api/                    # Backend API
├── data/                       # Data storage
│   ├── raw/papers/             # Paper-code pairs
│   ├── processed/              # Cleaned data
│   └── metadata/               # Logs, statistics
├── tests/                      # Tests
│   └── test_arxiv_github.py
├── docs/                       # Documentation
│   └── setup/
│       └── DATA_COLLECTION_GUIDE.md
├── .env.example
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment (Python 3.11)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure GitHub Token

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your GitHub token
# Get token at: https://github.com/settings/tokens
# Required scopes: public_repo, read:org
```

### 3. Collect Paper-Code Pairs

We use a **two-stage collection approach**:

#### Stage 1: Initial Curated Collection (Starting Point)
```bash
# Collect from manually curated list of 163 high-impact papers
python src/data_collection/pwc_hf_collector.py
```
- **Purpose**: Provides high-quality baseline with landmark papers
- **Papers**: BERT, GPT-3, LLaMA, CLIP, Mistral, Mamba, FlashAttention, etc.
- **Result**: 153 validated papers (some repos no longer exist or have <50 stars)
- **Maintenance**: Update [curated_papers_list.py](src/data_collection/curated_papers_list.py) with new landmark papers as they emerge

#### Stage 2: Automated Periodic Expansion ⭐
```bash
# Automatically scrape community-curated "Awesome" lists on GitHub
python src/data_collection/awesome_papers_collector.py
```
- **Purpose**: Periodically discover new papers from the community
- **Sources**: 10+ "Awesome" lists (Awesome-LLM, ML-Papers-of-the-Week, etc.)
- **Fully Automated**: Scrapes markdown files, extracts ArXiv IDs and GitHub URLs
- **Sustainable**: Lists maintained by the community, no manual intervention
- **Result**: ~100 new papers per run

#### Merge Collections
```bash
# Combine both sources and remove duplicates
python src/data_collection/merge_collections.py
```
- Deduplicates by ArXiv ID
- Generates unified dataset and statistics

**Complete Workflow**

Quick start (runs all stages):
```bash
./scripts/collect_papers.sh
```

Or run stages individually:
```bash
# Initial collection
python src/data_collection/pwc_hf_collector.py

# Expand with latest papers (run periodically - weekly/monthly)
python src/data_collection/awesome_papers_collector.py

# Merge everything
python src/data_collection/merge_collections.py
```

To skip curated collection and only run automated scraping:
```bash
./scripts/collect_papers.sh --skip-curated
```

**Output**: `data/raw/papers/paper_code_pairs.json`

**✅ Current Collection: 249 paper-code pairs**
- **Average Stars**: 11,470
- **Year Range**: 2013-2025
- **Categories**: cs.LG (138), cs.CL (65), cs.CV (30), cs.IR (6), others (10)
- **Top Papers**: TensorFlow (192K stars), PyTorch (95K stars), DistilBERT (153K stars), LangChain (120K stars)
- **Collection Method**: Curated baseline (153) + Automated scraping (100) - 4 duplicates = 249 unique papers

## Current Status

✅ **Phase 1: Data Collection** (In Progress)
- Papers With Code integration
- ArXiv API integration
- GitHub repository search
- Filtering & metadata collection

⏳ **Phase 2: Data Processing** (Upcoming)
- Code embeddings
- Paper comprehension

⏳ **Phase 3: Retrieval System** (Upcoming)
- FAISS indexing
- Code snippet extraction

⏳ **Phase 4: API & Frontend** (Upcoming)
- Backend API
- Web interface

## Documentation

- **Data Collection Guide**: [docs/setup/DATA_COLLECTION_GUIDE.md](docs/setup/DATA_COLLECTION_GUIDE.md)

## Requirements

- Python 3.11
- GitHub Personal Access Token (recommended for data collection)

## License

MIT
