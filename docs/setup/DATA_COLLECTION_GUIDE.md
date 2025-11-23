# Data Collection Guide

## Quick Start

```bash
# 1. Get GitHub token at: https://github.com/settings/tokens
#    Required scopes: public_repo, read:org

# 2. Add to .env file
echo "GITHUB_TOKEN=your_token_here" > .env

# 3. Run collection
python src/data_collection/collect_papers.py
```

## Goal

Collect 200+ high-quality paper-code pairs:
- **Categories**: cs.CL (Computational Linguistics), cs.LG (Machine Learning)
- **Date Range**: 2020-2025
- **Min GitHub Stars**: 50+

**Output**: `data/raw/papers/paper_code_pairs.json`

## How It Works

Uses `arxiv_github_collector.py`:
1. Fetches papers from ArXiv API by category
2. Searches GitHub for associated repositories
3. Filters by stars and metadata
4. Saves paper-code pairs with full metadata

## Requirements

- Python 3.11
- GitHub Personal Access Token (for API rate limits)

**Rate Limits**:
- Without token: ~60 requests/hour (slow)
- With token: ~5000 requests/hour (fast)

**Estimated time with token**: 30-60 minutes
