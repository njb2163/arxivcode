# Embeddings Workflow

This guide covers two workflows for generating code embeddings:

1. **Training Workflow**: Fine-tune CodeBERT using contrastive learning to align papers with code
2. **Inference Workflow**: Generate embeddings using pretrained CodeBERT (Task 2 & 3)

---

## Overview

### Training Workflow

Trains two encoders (paper encoder and code encoder) using CodeBERT to create embeddings where:
- **Matching pairs** (paper + its code) are close together
- **Non-matching pairs** are far apart

These embeddings are then used for semantic code retrieval: given a paper query, find the most relevant code.

### Inference Workflow (Task 2 & 3)

Generates embeddings from code snippets using pretrained CodeBERT:
- **Task 2**: Extract function-level code snippets from Python files
- **Task 3**: Generate embeddings and save in NumPy format (`code_embeddings.npy`)

This workflow performs **inference only** - it does NOT fine-tune the model.

---

## Which Workflow Should I Use?

- **Use Training Workflow** if you want to fine-tune CodeBERT for better alignment between papers and code
- **Use Inference Workflow (Task 2 & 3)** if you want to quickly generate embeddings from code snippets using pretrained CodeBERT

Both workflows can be used independently or together.

---

## Prerequisites

1. **Data**: You need `data/raw/papers/paper_code_with_files.json` (created by data collection pipeline with abstracts and code files)
2. **Environment**: Python 3.11 with required packages (see main `requirements.txt`)
3. **GPU** (optional but recommended): Training is faster on GPU, inference works on CPU

---

# Training Workflow

## Step-by-Step Training Workflow

### Step 1: Parse Paper-Code Pairs

**Command:**
```bash
python src/embeddings/paper_code_parser.py
```

**What it does:**
- Reads `data/raw/papers/paper_code_with_files.json`
- Extracts paper abstracts and code file contents
- Creates one contrastive pair per code file: (paper_abstract, code_file_content)
- Filters out empty or invalid pairs
- Saves to `data/processed/parsed_pairs.json`

**Output:**
- `data/processed/parsed_pairs.json` - Clean text pairs ready for training

**What you'll see:**
```
Day 4 Step 2: Parsing paper_code_with_files.json
Loading JSON file: data/raw/papers/paper_code_with_files.json
Loaded 249 papers with repositories
Parsing complete!
  Total papers processed: 249
  Papers skipped: 0
  Code files processed: 1250
  Code files skipped: 5
  Text pairs created: 1245
```

---

### Step 2: Setup DataLoaders

**Command:**
```bash
python src/embeddings/data_loader_setup.py
```

**What it does:**
- Loads `parsed_pairs.json`
- Creates PyTorch Dataset that tokenizes paper and code text
- Splits data into train (80%) and validation (20%)
- Creates DataLoaders with batching, shuffling, etc.
- Saves dataset statistics to `data/processed/dataset_info.json`

**Output:**
- Train/val DataLoaders ready for training
- `data/processed/dataset_info.json` - Dataset statistics

**What you'll see:**
```
Day 4 Step 4: Setting up DataLoaders
Loading dataset from data/processed/parsed_pairs.json
Full dataset size: 249 samples
Splitting dataset: 199 train, 50 val
Creating DataLoaders (batch_size=8, num_workers=0)
✓ DataLoaders created successfully
```

---

### Step 3: Test InfoNCE Loss (Optional)

**Command:**
```bash
python src/embeddings/contrastive_loss.py
```

**What it does:**
- Tests the InfoNCE loss function with dummy data
- Shows how loss behaves with different embedding similarities
- Verifies gradients work correctly

**What you'll see:**
```
Testing InfoNCE Loss Function
Test 1: Random Embeddings (Untrained Model)
   Loss: 1.3863 (expected: ~1.39)
   Diagonal similarities: [0.12, 0.08, 0.15, 0.09]
   Off-diagonal mean: 0.02

Test 2: Partially Similar Embeddings
   Loss: 0.5234
   Gap (diagonal - off_diagonal): 0.89
```

**Why test?** Verifies the loss function works before training.

---

### Step 4: Train the Code Encoder

**Command:**
```bash
python src/embeddings/train_code_encoder.py
```

**What it does:**
1. **Loads models**: Creates two CodeBERT encoders (paper + code)
2. **Loads data**: Gets train/val DataLoaders from Step 2
3. **Training loop** (for each epoch):
   - For each batch:
     - Encodes papers → paper embeddings
     - Encodes codes → code embeddings
     - Computes InfoNCE loss (push positives together, pull negatives apart)
     - Updates model weights via backpropagation
   - Validates on validation set
   - Saves best model if validation loss improved
4. **Saves checkpoints**: Best model + periodic checkpoints

**Output:**
- `checkpoints/code_encoder/best_model.pt` - Best trained model
- `checkpoints/code_encoder/checkpoint_epoch_N.pt` - Periodic checkpoints
- `checkpoints/code_encoder/training_history.json` - Training curves

**What you'll see:**
```
Code Encoder Training
Creating DataLoaders...
Loading encoders: microsoft/codebert-base
Starting training for 3 epochs
Device: cuda
Train batches: 25
Val batches: 7

Epoch 0: Train Loss = 1.2345
Epoch 0: Val Loss = 1.1890
✓ Saved best model (val_loss: 1.1890)

Epoch 1: Train Loss = 0.8765
Epoch 1: Val Loss = 0.8234
✓ Saved best model (val_loss: 0.8234)

Epoch 2: Train Loss = 0.6543
Epoch 2: Val Loss = 0.7123
✓ Saved best model (val_loss: 0.7123)

Training complete!
Best validation loss: 0.7123
```

**Training time:** 
- CPU: ~2-4 hours for 3 epochs
- GPU: ~30-60 minutes for 3 epochs

---

### Step 5: Customize Training (Optional)

**Command with options:**
```bash
python src/embeddings/train_code_encoder.py \
    --batch_size 16 \
    --num_epochs 5 \
    --learning_rate 3e-5 \
    --temperature 0.1 \
    --checkpoint_dir checkpoints/my_experiment
```

**Parameters:**
- `--batch_size`: Larger = faster but more GPU memory (default: 8)
- `--num_epochs`: How many times to go through data (default: 3)
- `--learning_rate`: How fast to learn (default: 2e-5)
- `--temperature`: InfoNCE temperature (lower = harder negatives, default: 0.07)
- `--checkpoint_dir`: Where to save models

---

### Step 6: Resume Training (If Interrupted)

**Command:**
```bash
python src/embeddings/train_code_encoder.py \
    --resume_from checkpoints/code_encoder/checkpoint_epoch_1.pt
```

**What it does:**
- Loads model weights, optimizer state, and training history
- Continues from the saved epoch
- Useful if training was interrupted

---

## Quick Start

### For Training (Contrastive Learning)

Run everything in sequence:

```bash
# Step 1: Parse data
python src/embeddings/paper_code_parser.py

# Step 2: Setup DataLoaders
python src/embeddings/data_loader_setup.py

# Step 3: Train (this takes time!)
python src/embeddings/train_code_encoder.py
```

Or use the bash script (if available):
```bash
./scripts/run_day4.sh  # Runs steps 1-2
python src/embeddings/train_code_encoder.py  # Step 3
```

### For Embedding Generation (Task 2 & 3)

```bash
# Task 2: Extract code snippets
python src/data_collection/extract_snippets.py \
    --input data/raw/papers/paper_code_with_files.json \
    --output data/processed/code_snippets.json

# Task 3: Generate embeddings
python src/embeddings/generate_code_embeddings.py \
    --json_path data/processed/code_snippets.json \
    --output_path data/processed/snippet_embeddings.json \
    --snippets \
    --build_faiss
```

This will create:
- `data/processed/embeddings/code_embeddings.npy` (Task 3 output)
- `data/processed/embeddings/metadata.json` (Task 3 output)
- `data/processed/faiss_index.index` (for retrieval)

---

## Understanding the Training Process

### What Happens During Training

1. **Forward Pass:**
   ```
   Paper text → Tokenizer → Paper Encoder → Paper Embedding (768 dims)
   Code text  → Tokenizer → Code Encoder  → Code Embedding (768 dims)
   ```

2. **Loss Computation:**
   - For each paper in batch, compute similarity with all codes
   - Positive pair (paper[i], code[i]) should have HIGH similarity
   - Negative pairs (paper[i], code[j≠i]) should have LOW similarity
   - Loss = how well the model distinguishes positives from negatives

3. **Backward Pass:**
   - Calculate gradients (what to change)
   - Update encoder weights to reduce loss
   - Repeat for next batch

### What "Good" Training Looks Like

- **Initial loss**: ~1.0-1.5 (model can't tell pairs apart)
- **Final loss**: ~0.3-0.7 (model learned to align pairs)
- **Loss decreases**: Should steadily decrease over epochs
- **Val loss tracks train loss**: If val loss increases, you're overfitting

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```bash
# Reduce batch size
python src/embeddings/train_code_encoder.py --batch_size 4
```

### Issue: Training Too Slow

**Solution:**
- Use GPU if available (automatically detected)
- Increase batch size if you have GPU memory
- Reduce number of epochs for testing

### Issue: Loss Not Decreasing

**Possible causes:**
- Learning rate too high → try `--learning_rate 1e-5`
- Learning rate too low → try `--learning_rate 5e-5`
- Data quality issues → check `parsed_pairs.json`

### Issue: Loss is 0.0000

**This is normal in tests** (see `contrastive_loss.py` test output). In real training, loss starts around 1.0-1.5.

---

# Inference Workflow (Task 2 & 3)

## Generating Embeddings (Task 2 & 3)

This section covers generating embeddings from code snippets using pretrained CodeBERT (inference only, no fine-tuning).

### Prerequisites

1. **Data**: You need `data/raw/papers/paper_code_with_files.json` (created by data collection pipeline)
2. **Code Snippets**: Extracted function-level code snippets (Task 2)

---

### Task 2: Extract Code Snippets

**Command:**
```bash
python src/data_collection/extract_snippets.py \
    --input data/raw/papers/paper_code_with_files.json \
    --output data/processed/code_snippets.json \
    --min-lines 50 \
    --require-docstring
```

**What it does:**
- Parses Python files using AST
- Extracts individual function definitions and class methods
- Filters functions with docstrings and >50 lines (configurable)
- Saves each function as a separate JSON object with:
  - Paper information (paper_id, paper_title, paper_abstract, paper_url)
  - Repository information (repo_name, repo_url)
  - Code snippet details (file_path, function_name, code_text)
  - Metadata (line_numbers, has_docstring, num_lines)

**Output:**
- `data/processed/code_snippets.json` - Flat list of code snippets ready for embedding

**Parameters:**
- `--min-lines`: Minimum number of lines for extraction (default: 50)
- `--require-docstring`: Require docstrings for extraction (default: True)
- `--no-require-docstring`: Don't require docstrings
- `--max-papers`: Limit number of papers to process (optional)

---

### Task 3: Generate Embeddings with CodeBERT

**Command:**
```bash
python src/embeddings/generate_code_embeddings.py \
    --json_path data/processed/code_snippets.json \
    --output_path data/processed/snippet_embeddings.json \
    --snippets \
    --use-cls-token \
    --include-paper-context \
    --paper-context-weight 0.3 \
    --build_faiss
```

**What it does:**
1. **Loads CodeBERT**: Uses pretrained `microsoft/codebert-base` (inference only)
2. **Processes snippets**: For each code snippet:
   - Optionally combines paper title + abstract with code text
   - Tokenizes the text
   - Extracts CLS token embedding (768 dimensions)
3. **Saves outputs**:
   - `data/processed/embeddings/code_embeddings.npy` - NumPy array of all embeddings
   - `data/processed/embeddings/metadata.json` - Metadata in same order as embeddings
   - `data/processed/snippet_embeddings.json` - Full JSON with embeddings (backward compatibility)
4. **Builds FAISS index** (if `--build_faiss` is used):
   - Creates searchable index for fast retrieval
   - Saves to `data/processed/faiss_index.index` and `faiss_metadata.pkl`

**Output Files:**
- `data/processed/embeddings/code_embeddings.npy` - Shape: `[num_snippets, 768]`
- `data/processed/embeddings/metadata.json` - Metadata array matching embeddings order
- `data/processed/snippet_embeddings.json` - Full results with embeddings as lists
- `data/processed/faiss_index.index` - FAISS index (if built)
- `data/processed/faiss_metadata.pkl` - FAISS metadata (if built)

**Key Parameters:**
- `--snippets`: Process code snippets format (required for Task 2 output)
- `--use-cls-token`: Extract CLS token embedding (Task 3 format, default: True)
- `--no-cls-token`: Use mean pooling instead of CLS token
- `--include-paper-context`: Combine paper title + abstract with code (default: True)
- `--no-paper-context`: Code only (no paper context)
- `--paper-context-weight`: Weight for paper context (0.0-1.0, default: 0.3)
- `--build_faiss`: Build FAISS index after generating embeddings
- `--batch_size`: Batch size for processing (default: 8)
- `--max_length`: Maximum sequence length (default: 512)

**What you'll see:**
```
Generating Code Embeddings from Code Snippets
Input JSON: data/processed/code_snippets.json
Model: microsoft/codebert-base
Batch size: 8
Max length: 512
Include paper context: True
Paper context weight: 30.0%
Use CLS token (Task 3 format): True

Loading CodeBERT model: microsoft/codebert-base
NOTE: This is INFERENCE ONLY - the model will NOT be fine-tuned.
Loaded 1250 code snippets

Generating embeddings (processing 157 batches)...
Processing batches: 100%|████████| 157/157 [05:23<00:00,  2.05s/batch]

✓ Generated embeddings for 1250 code snippets

Saving Task 3 format outputs:
  Embeddings: data/processed/embeddings/code_embeddings.npy
  Metadata: data/processed/embeddings/metadata.json
✓ Saved 1250 embeddings to data/processed/embeddings/code_embeddings.npy
  Shape: (1250, 768) (snippets, embedding_dim)
✓ Saved metadata to data/processed/embeddings/metadata.json
```

---

### Processing Original Format (paper_code_with_files.json)

If you want to process the original nested format instead of snippets:

```bash
python src/embeddings/generate_code_embeddings.py \
    --json_path data/raw/papers/paper_code_with_files.json \
    --output_path data/processed/code_embeddings.json \
    --batch_size 8
```

This processes entire code files (not function-level snippets).

---

### Testing the FAISS Index

After building the index, test it:

```bash
python test_faiss_index.py \
    --index_path data/processed/faiss_index.index \
    --metadata_path data/processed/faiss_metadata.pkl \
    --embedding_dim 768
```

This will:
- Load the FAISS index
- Test retrieval with sample queries
- Show top results with scores and metadata

---

## Next Steps

After generating embeddings:

1. **Test Retrieval**:
   - Use `test_faiss_index.py` to verify index works
   - Test with semantic queries (e.g., "masked language modeling")

2. **Integration**:
   - Connect to retrieval system
   - Use in end-to-end pipeline

3. **Fine-tuning** (Optional):
   - Train custom encoder using contrastive learning (see training section above)
   - Replace pretrained embeddings with fine-tuned ones

---

## File Structure

```
src/embeddings/
├── paper_code_parser.py         # Step 1: Parse JSON → text pairs (training)
├── contrastive_dataset.py        # Step 2: Create PyTorch Dataset (training)
├── data_loader_setup.py         # Step 2: Create DataLoaders (training)
├── code_encoder_model.py        # CodeBERT encoder wrapper
├── contrastive_loss.py          # InfoNCE loss function (training)
├── train_code_encoder.py        # Step 3: Training loop
└── generate_code_embeddings.py   # Task 3: Generate embeddings (inference)

src/data_collection/
└── extract_snippets.py          # Task 2: Extract code snippets

data/processed/
├── parsed_pairs.json            # Output of Step 1 (training)
├── dataset_info.json            # Output of Step 2 (training)
├── code_snippets.json           # Output of Task 2 (snippets)
├── snippet_embeddings.json       # Output of Task 3 (full JSON)
└── embeddings/
    ├── code_embeddings.npy      # Task 3: NumPy array of embeddings
    └── metadata.json            # Task 3: Metadata matching embeddings

data/processed/
├── faiss_index.index            # FAISS index (if built)
└── faiss_metadata.pkl           # FAISS metadata (if built)

checkpoints/code_encoder/
├── best_model.pt                # Best trained model
├── checkpoint_epoch_N.pt         # Periodic checkpoints
└── training_history.json        # Training curves
```

---

## Key Concepts

- **Contrastive Learning**: Learn by comparing (positive vs negative pairs)
- **InfoNCE Loss**: Loss function that pushes positives together, pulls negatives apart
- **Embeddings**: Fixed-size vectors (768 numbers) representing text meaning
- **In-Batch Negatives**: Other pairs in the same batch serve as negatives (efficient!)

---

## Questions?

- Check training logs for detailed progress
- Review `training_history.json` for loss curves
- Test loss function separately if issues arise
- See main `README.md` for project overview

