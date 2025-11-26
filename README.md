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

### 3. Collect Papers

```bash
# Run complete collection pipeline
./scripts/collect_papers.sh
```

Output: `data/raw/papers/paper_code_pairs.json` (249 papers currently)

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
    └── collect_papers.sh        # Collection pipeline
```

## Current Status

**Dataset**: 249 paper-code pairs
- Average Stars: 11,470
- Year Range: 2013-2025
- Categories: cs.LG (138), cs.CL (65), cs.CV (30), others

**Progress**:
- ✅ Data Collection (2-stage: curated + automated)
- ✅ Model Setup (QLoRA with LLaMA-3/Mistral)
- ⏳ Training Pipeline (in progress)
- ⏳ Retrieval System
- ⏳ API & Frontend

## Documentation

- **[Data Collection Guide](docs/DATA_COLLECTION_GUIDE.md)** - Collection pipeline details
- **[Collection Methods Evaluation](docs/COLLECTION_METHODS_EVALUATION.md)** - Method comparison & rationale
- **[Paper Comprehension Model](docs/PAPER_COMPREHENSION_MODEL.md)** - Model training & deployment

## Requirements

- Python 3.11
- 8GB+ RAM (16GB recommended for model training)
- GitHub Personal Access Token (for data collection)
- Hugging Face account (for model access)

## License

MIT
