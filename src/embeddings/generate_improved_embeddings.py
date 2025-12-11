"""
Improved Code Embedding Generation for ArXivCode

This script generates embeddings with better relevance by:
1. Combining paper title + function name + code for embedding
2. Using a code-optimized model
3. Creating searchable text that emphasizes key identifiers
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_embedding_text(entry: Dict, strategy: str = 'enhanced') -> str:
    """
    Create text for embedding based on strategy.
    
    Strategies:
    - 'code_only': Just the code text
    - 'code_abstract': Code + abstract (original approach)
    - 'enhanced': Title + function name + docstring + code (recommended)
    - 'searchable': Optimized for search queries
    """
    paper_title = entry.get('paper_title', '')
    paper_abstract = entry.get('paper_abstract', '')
    function_name = entry.get('function_name', '')
    code_text = entry.get('code_text', '')
    file_path = entry.get('file_path', '')
    
    # Clean up generic titles
    if 'arXiv Query' in paper_title:
        paper_title = ''
    
    # Extract docstring from code if present
    docstring = ''
    if '"""' in code_text:
        try:
            start = code_text.index('"""') + 3
            end = code_text.index('"""', start)
            docstring = code_text[start:end].strip()
        except ValueError:
            pass
    elif "'''" in code_text:
        try:
            start = code_text.index("'''") + 3
            end = code_text.index("'''", start)
            docstring = code_text[start:end].strip()
        except ValueError:
            pass
    
    if strategy == 'code_only':
        return code_text
    
    elif strategy == 'code_abstract':
        # Original approach
        return f"{paper_abstract}\n\n{code_text}"
    
    elif strategy == 'enhanced':
        # Recommended: Emphasize title and function name
        parts = []
        
        # Paper title (most important for search)
        if paper_title:
            parts.append(f"Paper: {paper_title}")
        
        # Function name (very searchable)
        if function_name:
            # Convert CamelCase and underscores to spaces for better matching
            readable_name = function_name.replace('_', ' ').replace('.', ' ')
            parts.append(f"Function: {readable_name}")
        
        # File context
        if file_path:
            # Extract meaningful parts of path
            path_parts = file_path.split('/')
            meaningful_parts = [p for p in path_parts if p not in ['src', 'lib', 'core', '__pycache__']]
            if meaningful_parts:
                parts.append(f"Module: {' '.join(meaningful_parts[-2:])}")
        
        # Docstring (describes what it does)
        if docstring:
            parts.append(f"Description: {docstring[:300]}")
        
        # Abstract snippet (for context)
        if paper_abstract:
            parts.append(f"Context: {paper_abstract[:200]}")
        
        # Code (the actual implementation)
        parts.append(f"Code:\n{code_text[:1000]}")  # Limit code length
        
        return "\n\n".join(parts)
    
    elif strategy == 'searchable':
        # Optimized for matching user queries
        parts = []
        
        # Repeat title keywords for emphasis
        if paper_title:
            parts.append(paper_title)
            parts.append(paper_title)  # Double weight
        
        # Function name variations
        if function_name:
            parts.append(function_name)
            parts.append(function_name.replace('_', ' '))
            parts.append(function_name.replace('.', ' '))
        
        # Key phrases from abstract
        if paper_abstract:
            parts.append(paper_abstract[:300])
        
        # Code
        parts.append(code_text[:800])
        
        return " ".join(parts)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def generate_embeddings(
    input_path: str = 'data/processed/code_snippets_cleaned.json',
    output_dir: str = 'data/processed/embeddings_v2',
    model_name: str = 'microsoft/codebert-base',
    strategy: str = 'enhanced',
    batch_size: int = 32,
    max_length: int = 512,
    device: str = 'cpu'
) -> Dict:
    """
    Generate embeddings for the cleaned dataset.
    
    Args:
        input_path: Path to cleaned dataset
        output_dir: Directory to save embeddings
        model_name: HuggingFace model to use
        strategy: Text creation strategy ('enhanced', 'searchable', etc.)
        batch_size: Batch size for encoding
        max_length: Maximum token length
        device: 'cpu' or 'cuda'
    
    Returns:
        Statistics about the generation
    """
    print("=" * 60)
    print("GENERATING IMPROVED EMBEDDINGS")
    print("=" * 60)
    
    # Load data
    print(f"\n📂 Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    print(f"   Loaded {len(data)} entries")
    
    # Initialize model
    print(f"\n🔧 Loading model: {model_name}...")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name, device=device)
    print(f"   Model loaded on {device}")
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Create embedding texts
    print(f"\n📝 Creating embedding texts (strategy: {strategy})...")
    texts = []
    metadata = []
    
    for entry in tqdm(data, desc="Preparing texts"):
        text = create_embedding_text(entry, strategy=strategy)
        texts.append(text)
        
        # Store metadata (without code_text to save space)
        meta = {
            'paper_id': entry.get('paper_id', ''),
            'paper_title': entry.get('paper_title', ''),
            'paper_abstract': entry.get('paper_abstract', '')[:500],  # Truncate
            'paper_url': entry.get('paper_url', ''),
            'repo_name': entry.get('repo_name', ''),
            'repo_url': entry.get('repo_url', ''),
            'file_path': entry.get('file_path', ''),
            'function_name': entry.get('function_name', ''),
            'line_numbers': entry.get('line_numbers', {}),
            'has_docstring': entry.get('has_docstring', False),
            'num_lines': entry.get('num_lines', 0),
            # Add code_text for reranking
            'code_text': entry.get('code_text', '')[:2000]  # Truncate for storage
        }
        metadata.append(meta)
    
    # Generate embeddings
    print(f"\n🧮 Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    print(f"   Generated embeddings shape: {embeddings.shape}")
    
    # Save embeddings and metadata
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_file = output_path / 'code_embeddings.npy'
    metadata_file = output_path / 'metadata.json'
    config_file = output_path / 'config.json'
    
    print(f"\n💾 Saving to {output_dir}...")
    
    np.save(str(embeddings_file), embeddings)
    print(f"   Saved embeddings: {embeddings_file}")
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    print(f"   Saved metadata: {metadata_file}")
    
    # Save config for reproducibility
    config = {
        'model_name': model_name,
        'strategy': strategy,
        'embedding_dim': embeddings.shape[1],
        'num_entries': len(metadata),
        'max_length': max_length,
        'input_path': input_path
    }
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   Saved config: {config_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Entries processed:  {len(metadata):,}")
    print(f"  Embedding shape:    {embeddings.shape}")
    print(f"  Model:              {model_name}")
    print(f"  Strategy:           {strategy}")
    print(f"  Output directory:   {output_dir}")
    print("=" * 60)
    
    return {
        'num_entries': len(metadata),
        'embedding_dim': embeddings.shape[1],
        'output_dir': output_dir
    }


def test_embedding_quality(
    embeddings_dir: str = 'data/processed/embeddings_v2',
    model_name: str = 'microsoft/codebert-base'
):
    """Test the quality of generated embeddings with sample queries."""
    print("\n" + "=" * 60)
    print("TESTING EMBEDDING QUALITY")
    print("=" * 60)
    
    # Load embeddings and metadata
    embeddings = np.load(f"{embeddings_dir}/code_embeddings.npy")
    with open(f"{embeddings_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(metadata)} embeddings")
    
    # Load model for query encoding
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    model = SentenceTransformer(model_name)
    
    # Test queries
    test_queries = [
        "implement LoRA low rank adaptation",
        "transformer attention mechanism",
        "parameter efficient fine tuning",
        "reinforcement learning policy gradient",
        "diffusion model denoising"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 50)
        
        # Encode query
        query_emb = model.encode([query], normalize_embeddings=True)
        
        # Compute similarities
        similarities = cosine_similarity(query_emb, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:5]
        
        for i, idx in enumerate(top_indices, 1):
            meta = metadata[idx]
            score = similarities[idx]
            print(f"  {i}. (Score: {score:.4f})")
            print(f"     Paper: {meta['paper_title'][:60]}...")
            print(f"     Function: {meta['function_name']}")
            print(f"     File: {meta['file_path']}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate improved code embeddings')
    parser.add_argument('--input', default='data/processed/code_snippets_cleaned.json',
                        help='Input dataset path')
    parser.add_argument('--output', default='data/processed/embeddings_v2',
                        help='Output directory')
    parser.add_argument('--model', default='microsoft/codebert-base',
                        help='Model to use')
    parser.add_argument('--strategy', default='enhanced',
                        choices=['code_only', 'code_abstract', 'enhanced', 'searchable'],
                        help='Text creation strategy')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test existing embeddings')
    
    args = parser.parse_args()
    
    if args.test_only:
        test_embedding_quality(args.output, args.model)
    else:
        # Generate embeddings
        stats = generate_embeddings(
            input_path=args.input,
            output_dir=args.output,
            model_name=args.model,
            strategy=args.strategy,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Test quality
        test_embedding_quality(args.output, args.model)
