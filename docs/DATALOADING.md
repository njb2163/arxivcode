# DataLoader Summary

This document describes the files created during DataLoader process.

## Files Created

### 1. `data/processed/parsed_pairs.json` ⭐ **REQUIRED**

**What it contains:**
- Parsed paper-code text pairs ready for training
- Each entry has:
  - `paper_text`: Paper title + (optional abstract)
  - `code_text`: Repository name + description
  - `metadata`: ArXiv ID, paper title, repo name, repo URL, pair index

**Format:**
```json
[
  {
    "paper_text": "Attention Is All You Need Title: Attention Is All You Need Category: cs.CL Year: 2017 ArXiv ID: 1706.03762",
    "code_text": "tensorflow/tensor2tensor Library of deep learning models and datasets...",
    "metadata": {
      "arxiv_id": "1706.03762",
      "paper_title": "Attention Is All You Need",
      "repo_name": "tensorflow/tensor2tensor",
      "repo_url": "https://github.com/tensorflow/tensor2tensor",
      "pair_index": 1
    }
  },
  ...
]
```

**Why it's needed:**
- This is the main input for model training
- Contains all parsed text pairs ready for tokenization
- Can be regenerated, but saves time to reuse

**How to use:**
```python
from src.embeddings.contrastive_dataset import ContrastiveDataset

dataset = ContrastiveDataset(json_path="data/processed/parsed_pairs.json")
```

---

### 2. `data/processed/dataset_info.json` 📊 **OPTIONAL BUT USEFUL**


**What it contains:**
- Dataset statistics (train/val sizes, batch info)
- Configuration used (batch size, max length, etc.)
- Data shapes for reference

**Format:**
```json
{
  "train": {
    "num_batches": 200,
    "batch_size": 8,
    "total_samples": 1600
  },
  "val": {
    "num_batches": 50,
    "batch_size": 8,
    "total_samples": 400
  },
  "data_shapes": {
    "paper_input_ids": [8, 512],
    "code_input_ids": [8, 512],
    ...
  },
  "config": {
    "json_path": "data/processed/parsed_pairs.json",
    "model_name": "microsoft/codebert-base",
    "max_length": 512,
    "batch_size": 8,
    "train_split": 0.8,
    "seed": 42
  }
}
```

**Why it's useful:**
- Documents the exact configuration used
- Helps reproduce train/val splits (with same seed)
- Quick reference for dataset statistics

---


### Minimum Required:
1. ✅ **`data/processed/parsed_pairs.json`** - The parsed data

### Nice to Have:
2. 📊 **`data/processed/dataset_info.json`** - Statistics and config

### Not Needed (Can be Regenerated):
- Model files (CodeBERT downloads automatically)
- Tokenizer files (downloaded on first use)
- Temporary test outputs


## For Training


1. **Loading the parsed data:**
   ```python
   from src.embeddings.data_loader_setup import create_data_loaders
   
   train_loader, val_loader = create_data_loaders(
       json_path="data/processed/parsed_pairs.json",
       batch_size=8
   )
   ```

2. **Or loading the dataset directly:**
   ```python
   from src.embeddings.contrastive_dataset import ContrastiveDataset
   
   dataset = ContrastiveDataset(json_path="data/processed/parsed_pairs.json")
   ```



---

## File Locations Summary

```
arxivcode/
├── data/
│   ├── raw/
│   │   └── papers/
│   │       └── paper_code_pairs.json  (original, from data collection)
│   └── processed/
│       ├── parsed_pairs.json          ⭐ REQUIRED OUTPUT
│       └── dataset_info.json          📊 OPTIONAL OUTPUT
└── src/
    └── embeddings/
        ├── code_encoder_model.py      (Step 1: Model loading)
        ├── paper_code_parser.py       (Step 2: Parsing)
        ├── contrastive_dataset.py     (Step 3: Dataset class)
        └── data_loader_setup.py       (Step 4: DataLoader setup)
```

