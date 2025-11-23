# arxivcode

## Project Folder Structure

```
arxivcode/
├── .env                          # API keys (DO NOT COMMIT)
├── .gitignore                    # Ignore .env, data/, models/
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
│
├── data/                         # All data storage
│   ├── raw/                      # Raw downloaded data
│   │   ├── papers/               # ArXiv PDFs/metadata
│   │   └── repos/                # Cloned GitHub repos
│   ├── processed/                # Cleaned data
│   │   ├── paper_code_pairs.json # Main dataset
│   │   └── embeddings/           # Code embeddings (Week 1)
│   └── metadata/                 # Logs, statistics
│       └── collection_log.json
│
├── src/                          # Source code
│   ├── data_collection/          # Data collection pipeline
│   │   ├── github_collector.py   # GitHub API scripts
│   │   ├── arxiv_collector.py    # ArXiv API scripts
│   │   └── data_matcher.py       # Match papers to repos
│   ├── models/                   # Model training & inference
│   │   ├── code_encoder/         # Code understanding model
│   │   └── paper_llm/            # Paper comprehension model
│   ├── retrieval/                # Retrieval system
│   │   ├── faiss_index.py        # Dense retrieval with FAISS
│   │   ├── reranker.py           # Cross-encoder re-ranking
│   │   └── snippet_extractor.py  # Code snippet extraction
│   └── api/                      # Backend API
│       └── app.py                # FastAPI/Flask application
│
├── frontend/                     # Web interface
│   └── (Hosted on GCP or Streamlit)
│
├── notebooks/                    # Jupyter notebooks for exploration
│   └── data_exploration.ipynb
│
├── tests/                        # Unit tests
│   └── test_github_collector.py
│
└── docs/                         # Documentation
    └── architecture.md
```
