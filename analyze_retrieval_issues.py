#!/usr/bin/env python3
"""
Analyze retrieval issues and provide diagnostics.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval.faiss_index import FAISSIndexManager
from embeddings.code_encoder_model import CodeEncoder
from embeddings.generate_code_embeddings import generate_embedding


def analyze_embeddings_distribution(embeddings_path: str):
    """Analyze embedding distribution to check for clustering issues."""
    print("\n" + "="*70)
    print("EMBEDDING DISTRIBUTION ANALYSIS")
    print("="*70)
    
    embeddings = np.load(embeddings_path)
    
    # Compute pairwise cosine similarities (sample)
    sample_size = min(1000, len(embeddings))
    sample = embeddings[:sample_size]
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    sample_norm = sample / (norms + 1e-8)
    
    # Compute similarity matrix (sample)
    similarity_matrix = np.dot(sample_norm, sample_norm.T)
    
    # Remove diagonal (self-similarity = 1.0)
    mask = ~np.eye(sample_size, dtype=bool)
    similarities = similarity_matrix[mask]
    
    print(f"\n1. Pairwise Similarity Analysis (sample of {sample_size} embeddings):")
    print(f"   Mean similarity: {similarities.mean():.4f}")
    print(f"   Median similarity: {np.median(similarities):.4f}")
    print(f"   Min similarity: {similarities.min():.4f}")
    print(f"   Max similarity: {similarities.max():.4f}")
    print(f"   Std similarity: {similarities.std():.4f}")
    
    # Check for clustering
    high_sim_threshold = 0.9
    high_sim_count = (similarities > high_sim_threshold).sum()
    print(f"\n   Similarities > {high_sim_threshold}: {high_sim_count} ({100*high_sim_count/len(similarities):.2f}%)")
    
    if similarities.mean() > 0.7:
        print(f"   ⚠️  WARNING: High average similarity suggests embeddings are too similar/clustered")
    
    # Check embedding variance
    print(f"\n2. Embedding Variance Analysis:")
    var_per_dim = np.var(embeddings, axis=0)
    print(f"   Mean variance per dimension: {var_per_dim.mean():.6f}")
    print(f"   Min variance: {var_per_dim.min():.6f}")
    print(f"   Max variance: {var_per_dim.max():.6f}")
    
    low_var_dims = (var_per_dim < 0.01).sum()
    if low_var_dims > 0:
        print(f"   ⚠️  WARNING: {low_var_dims} dimensions have very low variance (< 0.01)")


def analyze_metadata_distribution(metadata_path: str):
    """Analyze metadata to check for data quality issues."""
    print("\n" + "="*70)
    print("METADATA DISTRIBUTION ANALYSIS")
    print("="*70)
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    print(f"\n1. Basic Statistics:")
    print(f"   Total snippets: {len(metadata)}")
    
    # Count by paper
    paper_counts = Counter(m.get("paper_id", "unknown") for m in metadata)
    print(f"\n2. Paper Distribution:")
    print(f"   Unique papers: {len(paper_counts)}")
    print(f"   Top 10 papers by snippet count:")
    for paper_id, count in paper_counts.most_common(10):
        paper_title = next((m.get("paper_title", "N/A") for m in metadata if m.get("paper_id") == paper_id), "N/A")
        print(f"     {paper_id}: {count} snippets - {paper_title[:60]}...")
    
    # Count by repo
    repo_counts = Counter(m.get("repo_name", "unknown") for m in metadata)
    print(f"\n3. Repository Distribution:")
    print(f"   Unique repos: {len(repo_counts)}")
    print(f"   Top 10 repos by snippet count:")
    for repo, count in repo_counts.most_common(10):
        print(f"     {repo}: {count} snippets")
    
    # Check for malformed metadata
    print(f"\n4. Data Quality Checks:")
    malformed_titles = sum(1 for m in metadata if "arXiv Query:" in m.get("paper_title", ""))
    empty_abstracts = sum(1 for m in metadata if not m.get("paper_abstract", "").strip())
    empty_functions = sum(1 for m in metadata if not m.get("function_name", "").strip())
    
    print(f"   Malformed paper titles (arXiv Query): {malformed_titles}")
    print(f"   Empty abstracts: {empty_abstracts}")
    print(f"   Empty function names: {empty_functions}")
    
    if malformed_titles > 0:
        print(f"   ⚠️  WARNING: {malformed_titles} entries have malformed paper titles")


def test_query_encoding_mismatch():
    """Test if query encoding format matches embedding format."""
    print("\n" + "="*70)
    print("QUERY ENCODING FORMAT CHECK")
    print("="*70)
    
    encoder = CodeEncoder(model_name="microsoft/codebert-base", max_length=512)
    encoder.model.eval()
    
    # Test query as plain text
    query_plain = "implement transformer attention mechanism"
    emb_plain = generate_embedding(encoder, query_plain, use_cls_token=True)
    
    # Test query with paper context format (how embeddings were created)
    query_with_context = f"Paper: {query_plain}\n\nCode:\n{query_plain}"
    emb_with_context = generate_embedding(encoder, query_with_context, use_cls_token=True)
    
    # Compare embeddings
    similarity = np.dot(emb_plain, emb_with_context) / (
        np.linalg.norm(emb_plain) * np.linalg.norm(emb_with_context)
    )
    
    print(f"\n1. Query Encoding Comparison:")
    print(f"   Plain query: '{query_plain}'")
    print(f"   With context format: 'Paper: [query]\\n\\nCode:\\n[query]'")
    print(f"   Embedding similarity: {similarity:.4f}")
    
    if similarity < 0.8:
        print(f"   ⚠️  WARNING: Low similarity suggests format mismatch!")
        print(f"   This could cause retrieval issues if embeddings were created with paper context")
    
    print(f"\n2. Recommendation:")
    print(f"   If embeddings include paper context, queries should use the same format")
    print(f"   OR embeddings should be code-only for better semantic matching")


def analyze_retrieval_results():
    """Analyze why same results appear for different queries."""
    print("\n" + "="*70)
    print("RETRIEVAL RESULT ANALYSIS")
    print("="*70)
    
    print(f"\n1. Issues Identified:")
    print(f"   ❌ Low diversity: Same papers/repos appearing across queries")
    print(f"   ❌ Low relevance: Generic PyTorch utilities for ML concept queries")
    print(f"   ❌ Narrow score range: All scores clustered (0.65-0.69)")
    print(f"   ❌ Possible query encoding mismatch")
    
    print(f"\n2. Likely Causes:")
    print(f"   a) Query encoding format doesn't match embedding format")
    print(f"      - Embeddings: 'Paper: [title+abstract]\\n\\nCode:\\n[code]'")
    print(f"      - Queries: Plain text")
    print(f"   b) Embeddings may be too similar (clustered)")
    print(f"   c) Index dominated by common code (PyTorch utilities)")
    print(f"   d) Paper context might be diluting code semantics")
    
    print(f"\n3. Recommendations:")
    print(f"   1. Fix query encoding to match embedding format")
    print(f"   2. Consider code-only embeddings (disable paper context)")
    print(f"   3. Filter out generic utility functions")
    print(f"   4. Check if embeddings need better separation")


def main():
    """Run all analyses."""
    embeddings_path = "data/processed/embeddings/code_embeddings.npy"
    metadata_path = "data/processed/embeddings/metadata.json"
    
    print("\n" + "="*70)
    print("RETRIEVAL ISSUE DIAGNOSTICS")
    print("="*70)
    
    try:
        # 1. Analyze embeddings
        analyze_embeddings_distribution(embeddings_path)
        
        # 2. Analyze metadata
        analyze_metadata_distribution(metadata_path)
        
        # 3. Check query encoding
        test_query_encoding_mismatch()
        
        # 4. Overall analysis
        analyze_retrieval_results()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
