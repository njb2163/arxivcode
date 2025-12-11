"""
Test retrieval system with real paper-code pairs.
Run this after building the index from real data.
"""

import json
from pathlib import Path
from .dense_retrieval import DenseRetrieval


def test_retrieval_with_real_data():
    """Test retrieval system with actual queries."""
    
    print("\n" + "="*60)
    print("TESTING RETRIEVAL WITH REAL 200 PAIRS")
    print("="*60)
    
    # Load index
    print("\n1. Loading FAISS index...")
    retriever = DenseRetrieval()
    # retriever.load_index(
    #     "data/processed/FAISS/faiss_index.index",
    #     "data/processed/FAISS/faiss_metadata.pkl"
    # )
    
    # Print statistics
    stats = retriever.get_statistics()
    print(f"   ✅ Index loaded: {stats['total_vectors']} code snippets")
    
    # Test queries relevant to AI/ML papers
    test_queries = [
        # Common ML/NLP concepts
        "implement transformer attention mechanism",
        "fine-tune large language model",
        "parameter efficient training methods",
        "masked language modeling",
        "contrastive learning",
        "diffusion model implementation",
        "reinforcement learning policy gradient",
        "graph neural network",
        
        # More specific queries
        "how to implement LoRA",
        "self-attention layer code",
        "tokenizer implementation",
        "batch normalization",
    ]
    
    print("\n2. Testing retrieval quality...")
    print("-" * 60)
    
    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: '{query}'")
        print("-" * 60)
        
        results = retriever.retrieve(query, top_k=10, use_reranker=True)
        
        if not results:
            print("   ⚠️  No results found")
            continue
        
        # For LoRA query, also show pre-reranking candidates
        if "LoRA" in query:
            print("\n   🔍 Pre-reranking candidates (top 50):")
            pre_rerank_results = retriever.retrieve(query, top_k=50, use_reranker=False)
            lora_in_candidates = any("2403.17887" in r['metadata'].get('paper_id', '') for r in pre_rerank_results)
            print(f"   LoRA paper (2403.17887) in top 50 candidates: {lora_in_candidates}")
            if lora_in_candidates:
                lora_rank = next((i+1 for i, r in enumerate(pre_rerank_results) if "2403.17887" in r['metadata'].get('paper_id', '')), None)
                print(f"   LoRA paper rank before reranking: {lora_rank}")
                lora_score = next((r['score'] for r in pre_rerank_results if "2403.17887" in r['metadata'].get('paper_id', '')), None)
                print(f"   LoRA paper score before reranking: {lora_score:.4f}")
            
            # Try hybrid scoring for LoRA
            print("\n   🔍 Testing hybrid scoring for LoRA:")
            hybrid_results = retriever.retrieve(query, top_k=10, use_reranker=True, hybrid_scoring=True)
            lora_in_hybrid = any("2403.17887" in r['metadata'].get('paper_id', '') for r in hybrid_results)
            print(f"   LoRA paper in hybrid results: {lora_in_hybrid}")
            if lora_in_hybrid:
                lora_rank_hybrid = next((i+1 for i, r in enumerate(hybrid_results) if "2403.17887" in r['metadata'].get('paper_id', '')), None)
                print(f"   LoRA paper rank with hybrid scoring: {lora_rank_hybrid}")
                lora_score_hybrid = next((r['score'] for r in hybrid_results if "2403.17887" in r['metadata'].get('paper_id', '')), None)
                print(f"   LoRA paper score with hybrid scoring: {lora_score_hybrid:.4f}")
            
            # Try keyword fallback for LoRA
            print("\n   🔍 Testing keyword fallback for LoRA:")
            fallback_results = retriever.retrieve(query, top_k=10, use_reranker=True, keyword_fallback=True)
            lora_in_fallback = any("2403.17887" in r['metadata'].get('paper_id', '') for r in fallback_results)
            print(f"   LoRA paper in fallback results: {lora_in_fallback}")
            if lora_in_fallback:
                lora_rank_fallback = next((i+1 for i, r in enumerate(fallback_results) if "2403.17887" in r['metadata'].get('paper_id', '')), None)
                print(f"   LoRA paper rank with keyword fallback: {lora_rank_fallback}")
                lora_score_fallback = next((r['score'] for r in fallback_results if "2403.17887" in r['metadata'].get('paper_id', '')), None)
                print(f"   LoRA paper score with keyword fallback: {lora_score_fallback:.4f}")
        
        for result in results:
            meta = result['metadata']
            print(f"\n   Rank {result['rank']} (Score: {result['score']:.4f})")
            print(f"   📄 Paper: {meta.get('paper_title', 'N/A')[:60]}...")
            print(f"   🔗 Repo: {meta.get('repo_name', 'N/A')}")
            print(f"   ⭐ Stars: {meta.get('star_count', 0)}")
            print(f"   📁 File: {meta.get('file_path', 'N/A')}")
    
    print("\n3. Testing filters...")
    print("-" * 60)
    
    # Test with star filter
    query = "implement attention mechanism"
    results_filtered = retriever.retrieve(
        query,
        top_k=5,
        filters={'min_stars': 100},
        use_reranker=True
    )
    
    print(f"\n📝 Query: '{query}' (min_stars=100)")
    print(f"   Found {len(results_filtered)} results with ≥100 stars")
    
    if results_filtered:
        top_result = results_filtered[0]['metadata']
        print(f"   Top result: {top_result.get('repo_name')} ({top_result.get('star_count')} ⭐)")
    
    print("\n4. Validation checks...")
    print("-" * 60)
    
    # Check diversity of results
    all_repos = set()
    all_papers = set()
    
    for query in test_queries[:5]:
        results = retriever.retrieve(query, top_k=10, use_reranker=True)
        for r in results:
            meta = r['metadata']
            all_repos.add(meta.get('repo_name', ''))
            all_papers.add(meta.get('paper_id', ''))
    
    print(f"   ✅ Unique repos in results: {len(all_repos)}")
    print(f"   ✅ Unique papers in results: {len(all_papers)}")
    
    # Check score distribution
    results = retriever.retrieve(test_queries[0], top_k=20, use_reranker=True)
    scores = [r['score'] for r in results]
    
    if scores:
        print(f"   ✅ Score range: {min(scores):.4f} to {max(scores):.4f}")
        print(f"   ✅ Average score: {sum(scores)/len(scores):.4f}")
    
    print("\n" + "="*60)
    print("✅ RETRIEVAL TESTING COMPLETE")
    print("="*60)


def test_specific_paper(paper_id: str):
    """Test retrieval for a specific paper."""
    
    print(f"\n🔍 Testing retrieval for paper: {paper_id}")
    print("-" * 60)
    
    retriever = DenseRetrieval()
    retriever.load_index(
        "data/processed/FAISS/faiss_index.index",
        "data/processed/FAISS/faiss_metadata.pkl"
    )
    
    # Search with paper title or abstract keywords
    query = f"paper {paper_id}"
    results = retriever.retrieve(query, top_k=10, use_reranker=True)
    
    # Filter for specific paper
    paper_results = [
        r for r in results 
        if r['metadata'].get('paper_id') == paper_id
    ]
    
    print(f"Found {len(paper_results)} code snippets from this paper")
    
    for r in paper_results:
        meta = r['metadata']
        print(f"\n   📁 {meta.get('file_path')}")
        print(f"   🔗 {meta.get('repo_name')}")
        print(f"   Score: {r['score']:.4f}")


def main():
    """Run all tests."""
    import sys
    
    if len(sys.argv) > 1:
        # Test specific paper
        paper_id = sys.argv[1]
        test_specific_paper(paper_id)
    else:
        # Full test suite
        test_retrieval_with_real_data()


if __name__ == "__main__":
    main()