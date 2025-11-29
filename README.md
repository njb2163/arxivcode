# ArxivCode

Machine learning project for paper-code understanding and retrieval. Enables question-answering on research papers using fine-tuned LLMs.

## Quick Setup

### 1. Environment Setup

```bash
# Create virtual environment (Python 3.11)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure GitHub Token

```bash
cp .env.example .env
# Edit .env and add your GitHub token
# Get token at: https://github.com/settings/tokens
```

### 3. Collect Papers & Generate Dataset

```bash
# Run complete pipeline (all 5 stages)
python src/data_collection/run_full_pipeline.py

# Quick test (10 papers, no code download)
python src/data_collection/run_full_pipeline.py --quick-test

# Skip code download (faster, papers + QA only)
python src/data_collection/run_full_pipeline.py --skip-code-download
```

**Output**:
- `data/raw/papers/paper_code_pairs.json` - 249 papers
- `data/raw/papers/paper_code_with_files.json` - With actual code files
- `data/processed/train.json` + `eval.json` - 5,829 QA pairs

### 4. Setup Paper Comprehension Model

```bash
# Authenticate with Hugging Face
huggingface-cli login

# Test model setup
python examples/test_model_loading.py
```

See [Paper Comprehension Model Guide](docs/PAPER_COMPREHENSION_MODEL.md) for training details.

## Project Structure

```
arxivcode/
├── src/
│   ├── data_collection/        # Paper-code pair collection
│   └── models/                 # LLM training (LLaMA/Mistral + QLoRA)
├── data/
│   ├── raw/papers/             # Collected paper-code pairs
│   └── processed/              # Training data
├── docs/
│   ├── DATA_COLLECTION_GUIDE.md           # Collection system details
│   ├── COLLECTION_METHODS_EVALUATION.md   # Methods comparison
│   └── PAPER_COMPREHENSION_MODEL.md       # Model training guide
├── examples/
│   └── test_model_loading.py   # Verify model setup
└── scripts/
    └── cleanup_git_dirs.sh      # Remove nested .git dirs
```

## Current Status

**Dataset**: 249 paper-code pairs
- Average Stars: 11,470
- Year Range: 2013-2025
- Categories: cs.LG (138), cs.CL (65), cs.CV (30), others

## Documentation

See `docs/` for:
- **[DATA_COLLECTION_GUIDE.md](docs/DATA_COLLECTION_GUIDE.md)** - How collection works
- **[COLLECTION_METHODS_EVALUATION.md](docs/COLLECTION_METHODS_EVALUATION.md)** - Why this approach
- **[PAPER_COMPREHENSION_MODEL.md](docs/PAPER_COMPREHENSION_MODEL.md)** - Training the model

## Requirements

- Python 3.11
- 8GB+ RAM (16GB recommended for model training)
- GitHub Personal Access Token (for data collection)
- Hugging Face account (for model access)

## License

MIT
