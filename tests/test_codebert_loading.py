"""
Test script to verify CodeBERT loading (Day 4 Step 1).
This script tests loading microsoft/codebert-base and verifies it works correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embeddings.code_encoder_model import load_codebert
import torch


def test_codebert_loading():
    """Test loading CodeBERT model and tokenizer."""
    print("=" * 80)
    print("TESTING CODEBERT LOADING (Day 4 Step 1)")
    print("=" * 80)

    try:
        print("\n1. Loading CodeBERT model and tokenizer...")
        model, tokenizer, config = load_codebert()
        print("   ✓ Model and tokenizer loaded successfully")

        print("\n2. Verifying model properties...")
        # Check model is on correct device
        device = next(model.parameters()).device
        print(f"   ✓ Model device: {device}")

        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Total parameters: {total_params:,}")

        # Check config
        print(f"   ✓ Hidden size: {config.hidden_size}")
        print(f"   ✓ Number of layers: {config.num_hidden_layers}")
        print(f"   ✓ Attention heads: {config.num_attention_heads}")
        print(f"   ✓ Vocab size: {len(tokenizer)}")

        print("\n3. Testing tokenization...")
        test_text = "def hello_world(): print('Hello, World!')"
        tokens = tokenizer(
            test_text, return_tensors="pt", padding=True, truncation=True
        )
        print(f"   ✓ Tokenized text. Input shape: {tokens['input_ids'].shape}")

        print("\n4. Testing forward pass...")
        model.eval()
        with torch.no_grad():
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = model(**tokens)
            embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            print(f"   ✓ Forward pass successful. Embedding shape: {embedding.shape}")
            print(f"   ✓ Embedding norm: {torch.norm(embedding).item():.4f}")

        print("\n5. Testing batch processing...")
        test_texts = [
            "def hello_world(): print('Hello, World!')",
            "class MyClass: pass",
            "import numpy as np",
        ]
        tokens = tokenizer(
            test_texts, return_tensors="pt", padding=True, truncation=True
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]
            print(
                f"   ✓ Batch processing successful. Batch embedding shape: {embeddings.shape}"
            )

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nCodeBERT is ready to use!")
        print("Next step: Parse paper_code_pairs.json (Day 4 Step 2)")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pytorch_setup():
    """Test PyTorch and CUDA availability."""
    print("\n" + "=" * 80)
    print("TESTING PYTORCH SETUP")
    print("=" * 80)

    print(f"\n1. PyTorch version: {torch.__version__}")
    print(f"2. CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - CUDA version: {torch.version.cuda}")
        print(f"   - GPU count: {torch.cuda.device_count()}")
        print(f"   - Current device: {torch.cuda.current_device()}")
        print(f"   - Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("   - Running on CPU (MPS may be available on Apple Silicon)")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(
                f"   - MPS (Apple Silicon) available: {torch.backends.mps.is_available()}"
            )

    print("\n✓ PyTorch setup verified!")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "CODEBERT LOADING TEST" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        test_pytorch_setup()
        success = test_codebert_loading()

        if success:
            print("\n" + "=" * 80)
            print("Model Loading: COMPLETE ✓")
            print("=" * 80)
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
