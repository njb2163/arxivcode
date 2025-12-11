"""
Dense Retrieval Pipeline for ArXivCode
End-to-end pipeline for retrieving code snippets from paper queries.
"""

print("File loaded!")

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseRetrieval:
    """End-to-end dense retrieval system for code search."""

    def __init__(
        self,
        embedding_model_name: str = "microsoft/codebert-base",  # Updated to CodeBERT
        use_gpu: bool = False
    ):
        """
        Initialize retrieval system.

        Args:
            embedding_model_name: HuggingFace model name for encoding queries
            use_gpu: Whether to use GPU
        """
        print("Using new retriever with .npy")
        self.embedding_model_name = embedding_model_name
        self.use_tfidf = False  # No longer using TF-IDF
        
        # Load pre-computed embeddings and metadata (prefer v2 if available)
        embeddings_path_v2 = Path('data/processed/embeddings_v2/code_embeddings.npy')
        metadata_path_v2 = Path('data/processed/embeddings_v2/metadata.json')
        embeddings_path_v1 = Path('data/processed/embeddings/code_embeddings.npy')
        metadata_path_v1 = Path('data/processed/embeddings/metadata.json')
        
        if embeddings_path_v2.exists() and metadata_path_v2.exists():
            embeddings_path = embeddings_path_v2
            metadata_path = metadata_path_v2
            print("Using v2 embeddings (cleaned + enhanced)")
        else:
            embeddings_path = embeddings_path_v1
            metadata_path = metadata_path_v1
            print("Using v1 embeddings")
        
        if embeddings_path.exists() and metadata_path.exists():
            self.embeddings = np.load(str(embeddings_path))
            with open(str(metadata_path), 'r') as f:
                self.metadata = json.load(f)
            self.embedding_dim = self.embeddings.shape[1]
            logger.info(f"Loaded pre-computed embeddings: {self.embeddings.shape}")
        else:
            logger.warning("Pre-computed embeddings not found, initializing empty")
            self.embeddings = np.array([])
            self.metadata = []
            self.embedding_dim = 768
        
        # Load model for query encoding
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(embedding_model_name, device='cuda' if use_gpu else 'cpu')
            logger.info(f"Loaded embedding model: {embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {embedding_model_name}: {e}")
            raise

        # Initialize cross-encoder reranker
        try:
            from .cross_encoder_reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker(device='cuda' if use_gpu else 'cpu')
            logger.info("Initialized cross-encoder reranker")
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")
            self.reranker = None

        logger.info(f"Initialized retrieval with {embedding_model_name}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def build_index_from_papers(
        self,
        paper_code_pairs_path: str,
        save_index_path: Optional[str] = None,
        save_metadata_path: Optional[str] = None
    ) -> None:
        """
        Build index from paper-code pairs (deprecated, using pre-computed embeddings).
        """
        logger.info("Using pre-computed embeddings, skipping index building")
    
    def load_index(self, index_path: str, metadata_path: str):
        """Load index (deprecated, embeddings loaded in __init__)."""
        logger.info("Embeddings already loaded in __init__, skipping load_index")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict] = None,
        use_reranker: bool = False,
        hybrid_scoring: bool = False,
        keyword_fallback: bool = False
    ) -> List[Dict]:
        """
        Retrieve code snippets for a query.
        
        Args:
            query: Natural language query (e.g., "implement transformer attention")
            top_k: Number of results to return
            filters: Optional filters (e.g., {'min_stars': 100})
            use_reranker: Whether to use cross-encoder re-ranking
            hybrid_scoring: Whether to combine semantic similarity with keyword matching
            keyword_fallback: Whether to fall back to keyword search if semantic search fails
        
        Returns:
            List of retrieval results with metadata and scores
        """
        if self.embeddings.size == 0:
            logger.warning("No embeddings loaded")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top indices (get more for filtering and reranking)
        num_candidates = top_k * 10 if use_reranker else top_k * 5
        top_indices = np.argsort(similarities)[::-1][:num_candidates]
        
        # Also include entries with exact keyword matches in code/function names
        # This ensures relevant code doesn't get missed due to lower semantic similarity
        if hybrid_scoring:
            keyword_indices = self._find_keyword_matches(query)
            # Combine with semantic candidates
            top_indices_set = set(top_indices.tolist())
            for idx in keyword_indices:
                if idx not in top_indices_set:
                    top_indices = np.append(top_indices, idx)
        
        # Build results with hybrid scoring if requested
        results = []
        for idx in top_indices:
            item = self.metadata[idx].copy()
            score = float(similarities[idx])
            
            if hybrid_scoring:
                # Add keyword matching bonus (40% keyword, 60% semantic)
                # This gives meaningful weight to exact technical term matches
                keyword_score = self._compute_keyword_score(query, item)
                score = 0.6 * score + 0.4 * keyword_score
            
            result = {'metadata': item, 'score': score}
            results.append(result)
        
        # Apply filters if provided
        if filters:
            results = self._apply_filters(results, filters)
        
        # Sort by score again after potential hybrid scoring
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Keyword fallback: if top results don't seem relevant, try keyword search
        if keyword_fallback and results:
            # Check if any top results contain query keywords in title/abstract
            query_keywords = [w.lower() for w in query.split() if len(w) > 3]
            relevant_found = any(
                any(kw in r['metadata'].get('paper_abstract', '').lower() or 
                    kw in r['metadata'].get('paper_title', '').lower()
                    for kw in query_keywords)
                for r in results[:top_k//2]  # Check top half
            )
            
            if not relevant_found:
                logger.info("Semantic search didn't find relevant results, trying keyword fallback")
                keyword_results = self._keyword_search(query, top_k=top_k)
                if keyword_results:
                    # Merge results, preferring keyword results
                    results = keyword_results + results
                    results = list({r['metadata']['paper_id']: r for r in results}.values())  # Deduplicate
                    results.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit to top_k
        results = results[:top_k]
        
        # Add rank
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        # Apply cross-encoder re-ranking if available and requested
        if use_reranker and self.reranker is not None:
            logger.info(f"Re-ranking top {len(results)} results with cross-encoder")
            results = self.reranker.rerank(query, results, top_k=top_k)
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 20
    ) -> List[List[Dict]]:
        """
        Retrieve for multiple queries at once.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
        
        Returns:
            List of result lists
        """
        # Generate embeddings for all queries
        query_embeddings = self.embedding_model.encode(
            queries,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Compute similarities for each query
        all_results = []
        for query_emb in query_embeddings:
            similarities = cosine_similarity([query_emb], self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                item = self.metadata[idx].copy()
                item['score'] = float(similarities[idx])
                results.append(item)
            all_results.append(results)
        
        return all_results
    
    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply post-retrieval filters."""
        filtered = []
        
        for result in results:
            metadata = result['metadata']
            
            # Filter by star count
            if 'min_stars' in filters:
                if metadata.get('star_count', 0) < filters['min_stars']:
                    continue
            
            # Filter by paper year
            if 'min_year' in filters:
                paper_id = metadata.get('paper_id', '')
                if paper_id:
                    try:
                        year = int('20' + paper_id[:2])
                        if year < filters['min_year']:
                            continue
                    except:
                        pass
            
            filtered.append(result)
        
        return filtered
    
    def _compute_keyword_score(self, query: str, metadata: Dict) -> float:
        """Compute keyword matching score for hybrid retrieval.
        
        Enhanced to give strong boost for exact technical term matches.
        """
        import re
        
        # Extract keywords from query (simple tokenization)
        query_lower = query.lower()
        query_words = re.findall(r'\b\w+\b', query_lower)
        
        # Remove common stop words (but keep task-specific words)
        stop_words = {'how', 'to', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'for', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'implement', 'create', 'build', 'make', 'use', 'using'}
        keywords = [word for word in query_words if word not in stop_words and len(word) > 2]
        
        # Get text to search in
        paper_abstract = metadata.get('paper_abstract', '').lower()
        code_text = metadata.get('code_text', '').lower()
        paper_title = metadata.get('paper_title', '').lower()
        function_name = metadata.get('function_name', '').lower()
        
        # Count keyword matches with different weights
        total_score = 0.0
        keywords_found_in_code = 0
        
        for keyword in keywords:
            keyword_score = 0.0
            
            # Title matches get highest weight (4x) - titles are curated
            title_exact = len(re.findall(r'\b' + re.escape(keyword) + r'\b', paper_title))
            title_partial = len(re.findall(re.escape(keyword), paper_title))
            keyword_score += (title_exact * 3 + title_partial) * 4
            
            # Function name matches get very high weight (5x) - direct relevance
            func_exact = len(re.findall(r'\b' + re.escape(keyword) + r'\b', function_name))
            func_partial = len(re.findall(re.escape(keyword), function_name))
            keyword_score += (func_exact * 3 + func_partial) * 5
            
            # Abstract matches get medium weight (2x)
            abstract_exact = len(re.findall(r'\b' + re.escape(keyword) + r'\b', paper_abstract))
            abstract_partial = len(re.findall(re.escape(keyword), paper_abstract))
            keyword_score += (abstract_exact * 2 + abstract_partial) * 2
            
            # Code matches - enhanced weight (3x) for finding implementation details
            code_exact = len(re.findall(r'\b' + re.escape(keyword) + r'\b', code_text))
            code_partial = len(re.findall(re.escape(keyword), code_text))
            code_keyword_score = (code_exact * 3 + code_partial) * 3
            keyword_score += code_keyword_score
            
            # Track if keyword found in code (for bonus)
            if code_exact > 0 or code_partial > 0:
                keywords_found_in_code += 1
            
            total_score += keyword_score
        
        # BONUS: If ALL keywords found in code, give significant boost
        # This helps find actual implementations (like LoRA in code)
        if keywords and keywords_found_in_code == len(keywords):
            total_score *= 2.0  # Double score for full keyword match in code
        elif keywords and keywords_found_in_code > 0:
            # Partial bonus based on percentage of keywords found
            bonus_multiplier = 1.0 + (keywords_found_in_code / len(keywords)) * 0.5
            total_score *= bonus_multiplier
        
        # Normalize score (0-1 range with higher ceiling for good matches)
        max_possible = len(keywords) * 20  # Adjusted for new weighting
        if max_possible == 0:
            return 0.0
        
        return min(total_score / max_possible, 1.0)
    
    def _find_keyword_matches(self, query: str, max_matches: int = 100) -> List[int]:
        """Find entries that have exact keyword matches in code or function names.
        
        This ensures we don't miss highly relevant entries just because
        their semantic embedding score is slightly lower.
        """
        import re
        
        # Extract keywords
        query_lower = query.lower()
        query_words = re.findall(r'\b\w+\b', query_lower)
        stop_words = {'how', 'to', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'for', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'can', 'shall', 'implement', 'create',
                      'build', 'make', 'use', 'using', 'example', 'code'}
        keywords = [word for word in query_words if word not in stop_words and len(word) > 2]
        
        if not keywords:
            return []
        
        # Find entries with keyword matches in code or function name
        matches = []
        for idx, item in enumerate(self.metadata):
            code_text = item.get('code_text', '').lower()
            function_name = item.get('function_name', '').lower()
            
            # Check for exact keyword matches
            found_count = 0
            for keyword in keywords:
                if keyword in function_name or keyword in code_text:
                    found_count += 1
            
            # Include if at least one keyword found
            if found_count > 0:
                matches.append((idx, found_count))
        
        # Sort by match count (descending) and return indices
        matches.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in matches[:max_matches]]
    
    def _keyword_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Perform keyword-based search as fallback."""
        import re
        
        # Extract keywords
        query_lower = query.lower()
        query_words = re.findall(r'\b\w+\b', query_lower)
        stop_words = {'how', 'to', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'for', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}
        keywords = [word for word in query_words if word not in stop_words and len(word) > 2]
        
        # Score all entries
        scored_results = []
        for idx, item in enumerate(self.metadata):
            score = self._compute_keyword_score(query, item)
            if score > 0:
                scored_results.append({
                    'metadata': item,
                    'score': score
                })
        
        # Sort by keyword score
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:top_k]
    
    def get_statistics(self) -> Dict:
        """Get retrieval system statistics."""
        return {
            'total_vectors': len(self.metadata),
            'embedding_dim': self.embedding_dim,
            'embedding_model': self.embedding_model_name
        }


def main():
    """Example usage and testing."""
    print("\n" + "="*60)
    print("Dense Retrieval Pipeline - Test Run")
    print("="*60)
    
    # Initialize retrieval system
    print("\n1. Initializing retrieval system...")
    retriever = DenseRetrieval(
        embedding_model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Create mock paper-code pairs for testing
    print("\n2. Creating mock dataset...")
    mock_data = []
    
    topics = [
        ("Transformers", "attention mechanism"),
        ("LoRA", "parameter efficient fine-tuning"),
        ("BERT", "masked language modeling"),
        ("GPT", "autoregressive generation"),
        ("Diffusion", "denoising score matching")
    ]
    
    for i, (topic, description) in enumerate(topics):
        for j in range(10):  # 10 repos per topic
            mock_data.append({
                'paper_id': f'21{i:02d}.{j:05d}',
                'title': f'{topic}: {description.title()}',
                'abstract': f'This paper presents {topic} for {description}...',
                'arxiv_url': f'https://arxiv.org/abs/21{i:02d}.{j:05d}',
                'github_repos': [{
                    'repo_name': f'{topic.lower()}-org/implementation-{j}',
                    'repo_url': f'https://github.com/{topic.lower()}/impl-{j}',
                    'star_count': 100 * (10 - j),
                    'python_files': [{
                        'file_path': f'src/{topic.lower()}_model.py',
                        'functions': [f'{topic.lower()}_function']
                    }]
                }]
            })
    
    # Save mock data
    data_dir = Path("data/processed/FAISS")
    data_dir.mkdir(parents=True, exist_ok=True)
    mock_data_path = data_dir / "mock_paper_code_pairs.json"
    
    with open(mock_data_path, 'w') as f:
        json.dump(mock_data, f, indent=2)
    
    print(f"   Created {len(mock_data)} mock paper-code pairs")
    
    # Build index
    print("\n3. Building FAISS index...")
    retriever.build_index_from_papers(
        str(mock_data_path),
        save_index_path=str(data_dir / "faiss_index.index"),
        save_metadata_path=str(data_dir / "faiss_metadata.pkl")
    )
    
    # Test queries
    print("\n4. Testing retrieval...")
    print("-" * 60)
    
    test_queries = [
        "How to implement self-attention mechanism?",
        "Parameter efficient fine-tuning methods",
        "Masked language model pre-training"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        results = retriever.retrieve(query, top_k=3)
        
        for result in results:
            meta = result['metadata']
            print(f"\nRank {result['rank']} (Score: {result['score']:.4f})")
            print(f"  📄 Paper: {meta['paper_title']}")
            print(f"  🔗 Repo: {meta['repo_name']}")
            print(f"  ⭐ Stars: {meta['star_count']}")
            print(f"  📁 File: {meta['file_path']}")
    
    # Test with filters
    print("\n5. Testing with filters (min_stars=500)...")
    print("-" * 60)
    
    filtered_results = retriever.retrieve(
        "attention mechanism implementation",
        top_k=3,
        filters={'min_stars': 500}
    )
    
    print(f"Found {len(filtered_results)} results with ≥500 stars")
    
    # Print statistics
    print("\n6. System Statistics:")
    print("-" * 60)
    stats = retriever.get_statistics()
    for key, value in stats.items():
        if key != 'embedding_model':
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ Dense Retrieval Pipeline test complete!")
    print("="*60)


if __name__ == "__main__":
    main()