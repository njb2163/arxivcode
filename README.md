# ArxivCode

Machine learning project for paper-code understanding and retrieval.

## Project Structure

```
arxivcode/
├── src/                        # Source code
│   ├── data_collection/        # Data collection pipeline
│   │   ├── arxiv_github_collector.py
│   │   ├── collect_papers.py   # Main collection script
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

```bash
# Run the collection script
python src/data_collection/collect_papers.py
```

**Target**: 200+ paper-code pairs
- Categories: cs.CL (Computational Linguistics), cs.LG (Machine Learning)
- Date Range: 2020-2025
- Min GitHub Stars: 50+

**Output**: `data/raw/papers/paper_code_pairs.json`

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
