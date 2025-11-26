"""
Dense Retrieval Pipeline for ArXivCode
End-to-end pipeline for retrieving code snippets from paper queries.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .faiss_index import FAISSIndexManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseRetrieval:
    """End-to-end dense retrieval system for code search."""

    def __init__(
        self,
        embedding_model_name: str = "tfidf",  # Changed default to TF-IDF
        index_manager: Optional[FAISSIndexManager] = None,
        use_gpu: bool = False
    ):
        """
        Initialize retrieval system.

        Args:
            embedding_model_name: "tfidf" for TF-IDF or HuggingFace model name
            index_manager: Pre-initialized FAISS index (or create new)
            use_gpu: Whether to use GPU (ignored for TF-IDF)
        """
        self.use_tfidf = embedding_model_name == "tfidf"
        self.embedding_dim: int = 1000  # Default dimension, will be updated if needed

        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.embedding_dim = 1000  # TF-IDF feature dimension
            logger.info("Using TF-IDF embeddings (CPU-only, stable on macOS)")
        else:
            # Fallback to sentence-transformers if specified
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(embedding_model_name, device='cpu')
                dim = self.embedding_model.get_sentence_embedding_dimension()
                self.embedding_dim = dim if dim is not None else 768  # Fallback to 768
            except Exception as e:
                logger.warning(f"Failed to load {embedding_model_name}: {e}")
                logger.info("Falling back to TF-IDF")
                self.use_tfidf = True
                self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                self.embedding_dim = 1000

        if index_manager is None:
            self.index_manager = FAISSIndexManager(
                embedding_dim=self.embedding_dim,
                index_type="FlatIP",
                use_gpu=use_gpu
            )
        else:
            self.index_manager = index_manager

        logger.info(f"Initialized retrieval with {'TF-IDF' if self.use_tfidf else embedding_model_name}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def build_index_from_papers(
        self,
        paper_code_pairs_path: str,
        save_index_path: Optional[str] = None,
        save_metadata_path: Optional[str] = None
    ) -> None:
        """
        Build FAISS index from paper-code pairs dataset.
        
        Args:
            paper_code_pairs_path: Path to JSON file with paper-code pairs
            save_index_path: Optional path to save index
            save_metadata_path: Optional path to save metadata
        """
        logger.info(f"Loading paper-code pairs from {paper_code_pairs_path}")
        
        with open(paper_code_pairs_path, 'r') as f:
            paper_code_pairs = json.load(f)
        
        logger.info(f"Loaded {len(paper_code_pairs)} paper-code pairs")
        
        # Extract code snippets and metadata
        code_texts = []
        metadata_list = []
        
        for item in paper_code_pairs:
            paper_info = item.get('paper', {})
            paper_id = paper_info.get('arxiv_id', '')
            paper_title = paper_info.get('title', '')
            paper_url = paper_info.get('url', '')
            paper_year = paper_info.get('year', '')
            
            for repo in item.get('repositories', []):
                repo_name = repo.get('name', '')
                repo_url = repo.get('url', '')
                stars = repo.get('stars', 0)
                language = repo.get('language', '')
                description = repo.get('description', '')
                
                # For now, use repo-level embeddings (no python_files yet)
                # Later (Day 6), we'll clone repos and extract functions
                code_text = f"""
                Repository: {repo_name}
                Description: {description}
                Paper: {paper_title} ({paper_year})
                Language: {language}
                Topics: {', '.join(repo.get('topics', []))}
                """
                
                code_texts.append(code_text.strip())
                metadata_list.append({
                    'paper_id': paper_id,
                    'paper_title': paper_title,
                    'paper_year': paper_year,
                    'paper_url': paper_url,
                    'repo_name': repo_name,
                    'repo_url': repo_url,
                    'star_count': stars,
                    'language': language,
                    'description': description,
                    'topics': repo.get('topics', []),
                    'file_path': 'Repository-level (files pending Day 6)',
                    'code_snippet': 'Function extraction pending (Day 6)'
                })
        
        logger.info(f"Extracted {len(code_texts)} code snippets for indexing")
        
        # Generate embeddings
        logger.info("Generating embeddings (this may take a few minutes)...")

        if self.use_tfidf:
            # Use TF-IDF vectorization (stable, no PyTorch issues)
            embeddings = self.vectorizer.fit_transform(code_texts).toarray().astype(np.float32)  # type: ignore
            logger.info(f"Generated TF-IDF embeddings with shape: {embeddings.shape}")
        else:
            # Use sentence-transformers (may have macOS issues)
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism to avoid segfaults

            embeddings = self.embedding_model.encode(
                code_texts,
                batch_size=8,  # Smaller batch size to avoid memory issues
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        # Add to index
        logger.info("Adding to FAISS index...")
        self.index_manager.add_embeddings(embeddings, metadata_list)
        
        # Save if paths provided
        if save_index_path and save_metadata_path:
            self.index_manager.save(save_index_path, save_metadata_path)
            # Also save TF-IDF vectorizer if using TF-IDF
            if self.use_tfidf:
                import pickle
                vectorizer_path = save_index_path.replace('.index', '_vectorizer.pkl')
                with open(vectorizer_path, 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                logger.info(f"TF-IDF vectorizer saved to {vectorizer_path}")
        
        logger.info("✅ Index building complete!")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve code snippets for a query.
        
        Args:
            query: Natural language query (e.g., "implement transformer attention")
            top_k: Number of results to return
            filters: Optional filters (e.g., {'min_stars': 100})
        
        Returns:
            List of retrieval results with metadata and scores
        """
        # Generate query embedding
        if self.use_tfidf:
            # Use TF-IDF vectorization for query
            query_embedding = self.vectorizer.transform([query]).toarray().astype(np.float32)[0]  # type: ignore
        else:
            # Use sentence-transformers
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_numpy=True
            )[0]
        
        # Search index
        results = self.index_manager.search(query_embedding, top_k=top_k * 2)
        
        # Apply filters if provided
        if filters:
            results = self._apply_filters(results, filters)
        
        # Limit to top_k
        results = results[:top_k]
        
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
            convert_to_numpy=True
        )
        
        # Batch search
        all_results = self.index_manager.batch_search(query_embeddings, top_k=top_k)
        
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
    
    def load_index(self, index_path: str, metadata_path: str) -> None:
        """Load pre-built index from disk."""
        self.index_manager.load(index_path, metadata_path)

        # Load TF-IDF vectorizer if using TF-IDF
        if self.use_tfidf:
            import pickle
            vectorizer_path = index_path.replace('.index', '_vectorizer.pkl')
            try:
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info(f"TF-IDF vectorizer loaded from {vectorizer_path}")
            except FileNotFoundError:
                logger.error(f"TF-IDF vectorizer not found at {vectorizer_path}")
                raise

        logger.info("Index loaded successfully")
    
    def get_statistics(self) -> Dict:
        """Get retrieval system statistics."""
        stats = self.index_manager.get_stats()
        if self.use_tfidf:
            stats['embedding_model'] = 'TF-IDF'
        else:
            stats['embedding_model'] = str(self.embedding_model)
        return stats


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