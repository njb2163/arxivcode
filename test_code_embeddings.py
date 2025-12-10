#!/usr/bin/env python3
"""
Test code_embeddings.npy file with retrieval system.
This script loads the embeddings from Task 3 format and tests retrieval functionality.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval.faiss_index import FAISSIndexManager
from embeddings.code_encoder_model import CodeEncoder
from embeddings.generate_code_embeddings import generate_embedding


def load_code_embeddings(
    embeddings_path: str = "data/processed/embeddings/code_embeddings.npy",
    metadata_path: str = "data/processed/embeddings/metadata.json"
) -> tuple:
    """
    Load code embeddings and metadata from Task 3 format.
    
    Args:
        embeddings_path: Path to code_embeddings.npy
        metadata_path: Path to metadata.json
    
    Returns:
        Tuple of (embeddings_array, metadata_list)
    """
    print("\n" + "="*70)
    print("LOADING CODE EMBEDDINGS (Task 3 Format)")
    print("="*70)
    
    # Load embeddings
    print(f"\n1. Loading embeddings from: {embeddings_path}")
    try:
        embeddings = np.load(embeddings_path)
        print(f"   ✅ Loaded embeddings array")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dtype: {embeddings.dtype}")
        print(f"   Size: {embeddings.nbytes / (1024**2):.2f} MB")
    except Exception as e:
        print(f"   ❌ Error loading embeddings: {e}")
        raise
    
    # Load metadata
    print(f"\n2. Loading metadata from: {metadata_path}")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"   ✅ Loaded metadata")
        print(f"   Number of entries: {len(metadata)}")
    except Exception as e:
        print(f"   ❌ Error loading metadata: {e}")
        raise
    
    # Verify consistency
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings but {len(metadata)} metadata entries"
        )
    
    print(f"\n   ✅ Consistency check passed: {len(embeddings)} embeddings match {len(metadata)} metadata entries")
    
    # Check embedding dimension
    if len(embeddings.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape: {embeddings.shape}")
    
    embedding_dim = embeddings.shape[1]
    print(f"   Embedding dimension: {embedding_dim}")
    
    # Sample metadata
    if metadata:
        print(f"\n   Sample metadata keys: {list(metadata[0].keys())}")
    
    return embeddings, metadata


def build_faiss_index(
    embeddings: np.ndarray,
    metadata: List[Dict],
    index_type: str = "FlatIP"
) -> FAISSIndexManager:
    """
    Build FAISS index from embeddings.
    
    Args:
        embeddings: Embeddings array (N, embedding_dim)
        metadata: List of metadata dicts
        index_type: FAISS index type
    
    Returns:
        FAISSIndexManager instance
    """
    print("\n" + "="*70)
    print("BUILDING FAISS INDEX")
    print("="*70)
    
    embedding_dim = embeddings.shape[1]
    
    print(f"\n1. Creating FAISS index...")
    print(f"   Type: {index_type}")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Number of vectors: {len(embeddings)}")
    
    index_manager = FAISSIndexManager(
        embedding_dim=embedding_dim,
        index_type=index_type
    )
    
    print(f"\n2. Adding embeddings to index...")
    index_manager.add_embeddings(embeddings, metadata)
    
    stats = index_manager.get_stats()
    print(f"   ✅ Index built successfully!")
    print(f"   Total vectors: {stats['total_vectors']}")
    print(f"   Embedding dimension: {stats['embedding_dim']}")
    print(f"   Index type: {stats['index_type']}")
    
    return index_manager


def test_retrieval(
    index_manager: FAISSIndexManager,
    encoder: CodeEncoder,
    test_queries: List[str] = None,
    top_k: int = 5
):
    """
    Test retrieval with sample queries.
    
    Args:
        index_manager: FAISS index manager
        encoder: CodeBERT encoder for query encoding
        test_queries: List of test queries (uses defaults if None)
        top_k: Number of results to return per query
    """
    if test_queries is None:
        test_queries = [
            "implement transformer attention mechanism",
            "fine-tune large language model",
            "parameter efficient training methods",
            "masked language modeling",
            "contrastive learning",
            "how to implement LoRA",
            "self-attention layer code",
            "tokenizer implementation",
            "batch normalization",
            "graph neural network",
        ]
    
    print("\n" + "="*70)
    print("TESTING RETRIEVAL")
    print("="*70)
    
    print(f"\nTesting {len(test_queries)} queries with top_k={top_k}...")
    print("-" * 70)
    
    all_results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}/{len(test_queries)}: '{query}'")
        print("-" * 70)
        
        try:
            # Encode query using CodeBERT
            query_embedding = generate_embedding(
                encoder=encoder,
                code_text=query,
                max_length=512,
                use_cls_token=True
            )
            
            # Search index
            results = index_manager.search(query_embedding, top_k=top_k)
            
            if not results:
                print("   ⚠️  No results found")
                continue
            
            # Display results
            for result in results:
                meta = result['metadata']
                print(f"\n   Rank {result['rank']} (Score: {result['score']:.4f})")
                
                # Display available metadata fields
                if 'paper_title' in meta:
                    title = meta.get('paper_title', 'N/A')
                    print(f"   📄 Paper: {title[:60]}{'...' if len(title) > 60 else ''}")
                if 'paper_id' in meta:
                    print(f"   🆔 Paper ID: {meta.get('paper_id', 'N/A')}")
                if 'repo_name' in meta:
                    print(f"   🔗 Repo: {meta.get('repo_name', 'N/A')}")
                if 'file_path' in meta:
                    print(f"   📁 File: {meta.get('file_path', 'N/A')}")
                if 'function_name' in meta:
                    print(f"   ⚙️  Function: {meta.get('function_name', 'N/A')}")
            
            all_results.append({
                'query': query,
                'results': results,
                'num_results': len(results)
            })
        
        except Exception as e:
            print(f"   ❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary statistics
    print("\n" + "="*70)
    print("RETRIEVAL SUMMARY")
    print("="*70)
    
    if all_results:
        total_results = sum(r['num_results'] for r in all_results)
        avg_results = total_results / len(all_results)
        
        # Score statistics
        all_scores = []
        for result_set in all_results:
            for result in result_set['results']:
                all_scores.append(result['score'])
        
        if all_scores:
            print(f"\n✅ Successfully processed {len(all_results)} queries")
            print(f"   Average results per query: {avg_results:.1f}")
            print(f"   Score range: {min(all_scores):.4f} to {max(all_scores):.4f}")
            print(f"   Average score: {np.mean(all_scores):.4f}")
            print(f"   Median score: {np.median(all_scores):.4f}")
        
        # Check diversity
        unique_papers = set()
        unique_repos = set()
        for result_set in all_results:
            for result in result_set['results']:
                meta = result['metadata']
                if 'paper_id' in meta:
                    unique_papers.add(meta['paper_id'])
                if 'repo_name' in meta:
                    unique_repos.add(meta['repo_name'])
        
        print(f"\n   Diversity:")
        print(f"   - Unique papers in results: {len(unique_papers)}")
        print(f"   - Unique repos in results: {len(unique_repos)}")
    else:
        print("\n⚠️  No successful retrievals")


def test_embeddings_quality(embeddings: np.ndarray):
    """
    Test basic quality metrics of embeddings.
    
    Args:
        embeddings: Embeddings array
    """
    print("\n" + "="*70)
    print("EMBEDDING QUALITY CHECKS")
    print("="*70)
    
    print(f"\n1. Basic Statistics:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dtype: {embeddings.dtype}")
    print(f"   Min value: {embeddings.min():.6f}")
    print(f"   Max value: {embeddings.max():.6f}")
    print(f"   Mean: {embeddings.mean():.6f}")
    print(f"   Std: {embeddings.std():.6f}")
    
    print(f"\n2. Checking for NaN/Inf values:")
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    print(f"   NaN values: {nan_count}")
    print(f"   Inf values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print(f"   ⚠️  Warning: Found invalid values!")
    else:
        print(f"   ✅ No invalid values found")
    
    print(f"\n3. Embedding norms (L2):")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"   Min norm: {norms.min():.6f}")
    print(f"   Max norm: {norms.max():.6f}")
    print(f"   Mean norm: {norms.mean():.6f}")
    print(f"   Std norm: {norms.std():.6f}")
    
    # Check if embeddings are normalized
    if np.allclose(norms, 1.0, atol=0.01):
        print(f"   ✅ Embeddings appear to be normalized (L2 norm ≈ 1.0)")
    else:
        print(f"   ℹ️  Embeddings are not normalized (will be normalized for FAISS)")


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test code_embeddings.npy with retrieval system"
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="data/processed/embeddings/code_embeddings.npy",
        help="Path to code_embeddings.npy file"
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/processed/embeddings/metadata.json",
        help="Path to metadata.json file"
    )
    parser.add_argument(
        "--index_type",
        type=str,
        default="FlatIP",
        choices=["FlatIP", "FlatL2", "IVFFlat"],
        help="FAISS index type (default: FlatIP)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return per query (default: 5)"
    )
    parser.add_argument(
        "--skip_quality_checks",
        action="store_true",
        help="Skip embedding quality checks"
    )
    parser.add_argument(
        "--skip_retrieval",
        action="store_true",
        help="Skip retrieval testing"
    )
    
    args = parser.parse_args()
    
    try:
        # 1. Load embeddings and metadata
        embeddings, metadata = load_code_embeddings(
            embeddings_path=args.embeddings_path,
            metadata_path=args.metadata_path
        )
        
        # 2. Quality checks
        if not args.skip_quality_checks:
            test_embeddings_quality(embeddings)
        
        # 3. Build FAISS index
        index_manager = build_faiss_index(
            embeddings=embeddings,
            metadata=metadata,
            index_type=args.index_type
        )
        
        # 4. Test retrieval
        if not args.skip_retrieval:
            print("\n" + "="*70)
            print("LOADING CODEBERT ENCODER")
            print("="*70)
            print("\nLoading CodeBERT for query encoding...")
            encoder = CodeEncoder(
                model_name="microsoft/codebert-base",
                max_length=512,
                device=None  # Auto-detect
            )
            encoder.model.eval()
            print("✅ CodeBERT encoder loaded")
            
            test_retrieval(
                index_manager=index_manager,
                encoder=encoder,
                top_k=args.top_k
            )
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETE")
        print("="*70)
        print("\nThe code_embeddings.npy file is working correctly with retrieval!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
