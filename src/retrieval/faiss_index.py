"""
FAISS Index Manager for ArXivCode
Handles dense retrieval of code snippets using semantic embeddings.
"""

import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """Manage FAISS index for code snippet retrieval."""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "FlatIP",  # FlatIP for Inner Product (cosine similarity)
        use_gpu: bool = False
    ):
        """
        Initialize FAISS index manager.
        
        Args:
            embedding_dim: Dimension of embeddings (768 for CodeBERT/BERT-base)
            index_type: Type of FAISS index ("FlatIP", "FlatL2", "IVFFlat")
            use_gpu: Whether to use GPU acceleration
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        
        # Initialize index
        self.index = self._create_index()
        
        # Metadata storage (maps index ID to code snippet metadata)
        self.id_to_metadata = {}
        self.next_id = 0
        
        logger.info(f"Initialized FAISS index: {index_type}, dim={embedding_dim}")
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on type."""
        if self.index_type == "FlatIP":
            # Inner Product (for cosine similarity with normalized vectors)
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "FlatL2":
            # L2 distance (Euclidean)
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IVFFlat":
            # Inverted File Index (faster for large datasets)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
            # Note: IVF requires training before adding vectors
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
            logger.info("FAISS index moved to GPU")
        
        return index
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict]
    ) -> None:
        """
        Add embeddings and their metadata to the index.
        
        Args:
            embeddings: numpy array of shape (N, embedding_dim)
            metadata_list: List of metadata dicts for each embedding
                Expected keys: paper_id, repo_name, file_path, code_snippet, etc.
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        if len(embeddings) != len(metadata_list):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings but "
                f"{len(metadata_list)} metadata entries"
            )
        
        # Normalize embeddings for cosine similarity (if using Inner Product)
        if self.index_type == "FlatIP":
            faiss.normalize_L2(embeddings)
        
        # Train index if IVF type (only needed once)
        if self.index_type == "IVFFlat" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            logger.info("Training complete")
        
        # Add to index
        start_id = self.next_id
        self.index.add(embeddings)
        
        # Store metadata
        for i, metadata in enumerate(metadata_list):
            self.id_to_metadata[start_id + i] = metadata
        
        self.next_id += len(embeddings)
        
        logger.info(f"Added {len(embeddings)} embeddings. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Search for most similar code snippets.
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,) or (1, embedding_dim)
            top_k: Number of results to return
        
        Returns:
            List of dicts with keys: metadata, score, rank
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if query_embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query dim mismatch: expected {self.embedding_dim}, "
                f"got {query_embedding.shape[1]}"
            )
        
        # Normalize query for cosine similarity
        if self.index_type == "FlatIP":
            faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Convert to results with metadata
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            metadata = self.id_to_metadata.get(int(idx), {})
            results.append({
                'rank': rank + 1,
                'score': float(score),
                'metadata': metadata,
                'index_id': int(idx)
            })
        
        return results
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 20
    ) -> List[List[Dict]]:
        """
        Batch search for multiple queries.
        
        Args:
            query_embeddings: Array of shape (N, embedding_dim)
            top_k: Number of results per query
        
        Returns:
            List of result lists, one per query
        """
        if query_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch")
        
        # Normalize
        if self.index_type == "FlatIP":
            faiss.normalize_L2(query_embeddings)
        
        # Batch search
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # Convert to results
        all_results = []
        for query_idx in range(len(query_embeddings)):
            results = []
            for rank, (idx, score) in enumerate(
                zip(indices[query_idx], scores[query_idx])
            ):
                if idx == -1:
                    continue
                
                metadata = self.id_to_metadata.get(int(idx), {})
                results.append({
                    'rank': rank + 1,
                    'score': float(score),
                    'metadata': metadata,
                    'index_id': int(idx)
                })
            all_results.append(results)
        
        return all_results
    
    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Save index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index (.index)
            metadata_path: Path to save metadata (.pkl)
        """
        # Save FAISS index
        if self.use_gpu:
            # Move to CPU before saving
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'id_to_metadata': self.id_to_metadata,
                'next_id': self.next_id,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }, f)
        
        logger.info(f"Index saved to {index_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Load index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.id_to_metadata = data['id_to_metadata']
            self.next_id = data['next_id']
            self.embedding_dim = data['embedding_dim']
            self.index_type = data['index_type']
        
        logger.info(f"Index loaded from {index_path}")
        logger.info(f"Total vectors: {self.index.ntotal}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the index."""
        return {
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'use_gpu': self.use_gpu,
            'metadata_entries': len(self.id_to_metadata)
        }


def main():
    """Example usage with mock data."""
    print("\n" + "="*60)
    print("FAISS Index Manager - Test Run")
    print("="*60)
    
    # Initialize manager
    manager = FAISSIndexManager(embedding_dim=768, index_type="FlatIP")
    
    # Create mock embeddings (simulating code embeddings)
    np.random.seed(42)
    num_snippets = 200
    mock_embeddings = np.random.randn(num_snippets, 768).astype('float32')
    
    # Create mock metadata (simulating paper-code pairs)
    mock_metadata = []
    for i in range(num_snippets):
        mock_metadata.append({
            'paper_id': f'21{i//10:02d}.{i:05d}',
            'paper_title': f'Paper about Topic {i//10}',
            'repo_name': f'user/repo-{i//5}',
            'file_path': f'src/module_{i}.py',
            'function_name': f'function_{i}',
            'code_snippet': f'def function_{i}():\n    # Implementation\n    pass',
            'line_start': 10 + i,
            'line_end': 15 + i
        })
    
    # Add to index
    print("\nAdding embeddings to index...")
    manager.add_embeddings(mock_embeddings, mock_metadata)
    
    # Print stats
    stats = manager.get_stats()
    print("\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search
    print("\n" + "-"*60)
    print("Testing Search")
    print("-"*60)
    
    # Create mock query (simulating "implement transformer attention")
    query = np.random.randn(768).astype('float32')
    
    print("\nSearching for top 5 similar code snippets...")
    results = manager.search(query, top_k=5)
    
    for result in results:
        print(f"\nRank {result['rank']} (Score: {result['score']:.4f})")
        meta = result['metadata']
        print(f"  Paper: {meta['paper_title']}")
        print(f"  Repo: {meta['repo_name']}")
        print(f"  File: {meta['file_path']}")
        print(f"  Function: {meta['function_name']}")
    
    # Test save/load
    print("\n" + "-"*60)
    print("Testing Save/Load")
    print("-"*60)
    
    save_dir = Path("./data/processed/FAISS")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = save_dir / "faiss_index.index"
    metadata_path = save_dir / "faiss_metadata.pkl"
    
    print(f"\nSaving to {save_dir}/...")
    manager.save(str(index_path), str(metadata_path))
    
    # Load into new manager
    print("\nLoading into new manager...")
    new_manager = FAISSIndexManager(embedding_dim=768)
    new_manager.load(str(index_path), str(metadata_path))
    
    # Test loaded index
    print("\nTesting loaded index with same query...")
    results_loaded = new_manager.search(query, top_k=5)
    
    print(f"\nVerification: Top result score: {results_loaded[0]['score']:.4f}")
    print(f"             (Should match: {results[0]['score']:.4f})")
    
    print("\n" + "="*60)
    print("✅ FAISS Index Manager test complete!")
    print("="*60)


if __name__ == "__main__":
    main()