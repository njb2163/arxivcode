"""
Data Cleaning Script for ArXivCode Dataset

This script:
1. Fixes generic "arXiv Query" titles by fetching real metadata
2. Filters out irrelevant code snippets (tests, utils, configs)
3. Scores code-paper relevance and removes low-quality entries
4. Outputs a cleaned dataset ready for embedding generation
"""

import json
import re
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import xml.etree.ElementTree as ET


def fetch_arxiv_metadata(paper_id: str) -> Optional[Dict]:
    """Fetch paper metadata from arXiv API."""
    try:
        url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Parse XML response
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entry = root.find('atom:entry', ns)
            if entry is not None:
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                
                if title is not None and summary is not None:
                    return {
                        'title': ' '.join(title.text.split()),  # Clean whitespace
                        'abstract': ' '.join(summary.text.split())
                    }
    except Exception as e:
        print(f"  Error fetching {paper_id}: {e}")
    return None


def is_relevant_file(file_path: str) -> Tuple[bool, str]:
    """
    Check if a file path is likely to contain relevant implementation code.
    Returns (is_relevant, reason).
    """
    file_path_lower = file_path.lower()
    
    # Exclude patterns
    exclude_patterns = [
        ('test', 'test file'),
        ('_test.py', 'test file'),
        ('tests/', 'test directory'),
        ('testing/', 'test directory'),
        ('conftest', 'pytest config'),
        ('setup.py', 'setup file'),
        ('setup.cfg', 'config file'),
        ('__init__.py', 'init file'),
        ('__main__.py', 'main file'),
        ('config', 'config file'),
        ('utils.py', 'utility file'),
        ('helpers.py', 'helper file'),
        ('constants.py', 'constants file'),
        ('version.py', 'version file'),
        ('logging', 'logging utilities'),
        ('exceptions.py', 'exception definitions'),
        ('cli.py', 'CLI utilities'),
        ('args.py', 'argument parsing'),
        ('typing', 'type definitions'),
        ('/docs/', 'documentation'),
        ('/examples/', 'examples'),
        ('/scripts/', 'scripts'),
        ('/tools/', 'tools'),
        ('/benchmarks/', 'benchmarks'),
        ('dataset', 'data loading'),
        ('dataloader', 'data loading'),
        ('tokenization', 'tokenization utilities'),
    ]
    
    for pattern, reason in exclude_patterns:
        if pattern in file_path_lower:
            return False, reason
    
    # Include patterns (preferred files)
    include_patterns = [
        'model', 'layer', 'attention', 'transformer', 'encoder', 'decoder',
        'train', 'loss', 'optim', 'forward', 'module', 'network', 'arch'
    ]
    
    for pattern in include_patterns:
        if pattern in file_path_lower:
            return True, 'implementation file'
    
    # Default: include but with lower confidence
    return True, 'general file'


def compute_relevance_score(entry: Dict) -> float:
    """
    Compute a relevance score for a code snippet based on how well
    it matches its associated paper.
    """
    score = 0.0
    
    title = entry.get('paper_title', '').lower()
    abstract = entry.get('paper_abstract', '').lower()
    code = entry.get('code_text', '').lower()
    function_name = entry.get('function_name', '').lower()
    file_path = entry.get('file_path', '').lower()
    
    # Skip entries with generic titles
    if 'arxiv query' in title:
        score -= 0.3
    
    # Extract key terms from title (excluding common words)
    stop_words = {'a', 'an', 'the', 'of', 'for', 'and', 'or', 'in', 'on', 'to', 'with', 
                  'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'that', 'this', 'which', 'what', 'how', 'when', 'where', 'who', 'why',
                  'learning', 'neural', 'network', 'networks', 'deep', 'model', 'models',
                  'using', 'based', 'via', 'towards', 'efficient', 'scalable', 'large'}
    
    title_words = set(re.findall(r'\b[a-z]{4,}\b', title)) - stop_words
    abstract_words = set(re.findall(r'\b[a-z]{5,}\b', abstract)) - stop_words
    
    # Check title keywords in code
    title_matches = sum(1 for word in title_words if word in code or word in function_name)
    if title_words:
        score += 0.4 * (title_matches / len(title_words))
    
    # Check abstract keywords in code (less weight)
    abstract_sample = list(abstract_words)[:20]  # Sample top 20
    abstract_matches = sum(1 for word in abstract_sample if word in code)
    if abstract_sample:
        score += 0.2 * (abstract_matches / len(abstract_sample))
    
    # Bonus for relevant file paths
    is_relevant, _ = is_relevant_file(file_path)
    if is_relevant:
        score += 0.2
    
    # Bonus for having docstring
    if entry.get('has_docstring'):
        score += 0.1
    
    # Penalty for very short or very long code
    code_lines = entry.get('num_lines', 0)
    if code_lines < 5:
        score -= 0.2
    elif code_lines > 500:
        score -= 0.1
    elif 10 <= code_lines <= 100:
        score += 0.1
    
    return max(0.0, min(1.0, score))


def clean_dataset(
    input_path: str = 'data/processed/code_snippets.json',
    output_path: str = 'data/processed/code_snippets_cleaned.json',
    fix_titles: bool = True,
    filter_files: bool = True,
    min_relevance_score: float = 0.2,
    max_snippets_per_paper: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Clean the dataset by fixing titles, filtering files, and scoring relevance.
    
    Args:
        input_path: Path to input dataset
        output_path: Path to save cleaned dataset
        fix_titles: Whether to fetch real titles from arXiv
        filter_files: Whether to filter out irrelevant files
        min_relevance_score: Minimum relevance score to keep entry
        max_snippets_per_paper: Maximum snippets to keep per paper
        verbose: Whether to print progress
    
    Returns:
        Statistics about the cleaning process
    """
    print("=" * 60)
    print("DATASET CLEANING")
    print("=" * 60)
    
    # Load data
    print(f"\n📂 Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    print(f"   Loaded {len(data)} entries")
    
    stats = {
        'original_count': len(data),
        'titles_fixed': 0,
        'files_filtered': 0,
        'low_relevance_filtered': 0,
        'per_paper_limit_filtered': 0,
        'final_count': 0
    }
    
    # Step 1: Fix generic titles
    if fix_titles:
        print("\n🔧 Step 1: Fixing generic titles...")
        papers_to_fix = set()
        for entry in data:
            if 'arXiv Query' in entry.get('paper_title', ''):
                paper_id = entry.get('paper_id', '')
                if paper_id and len(paper_id) >= 5:
                    papers_to_fix.add(paper_id)
        
        print(f"   Found {len(papers_to_fix)} papers with generic titles")
        
        # Fetch metadata for papers with bad titles (with rate limiting)
        title_cache = {}
        for i, paper_id in enumerate(list(papers_to_fix)[:100]):  # Limit to 100 API calls
            if verbose and i % 10 == 0:
                print(f"   Fetching metadata... {i}/{min(100, len(papers_to_fix))}")
            
            metadata = fetch_arxiv_metadata(paper_id)
            if metadata:
                title_cache[paper_id] = metadata
                stats['titles_fixed'] += 1
            
            time.sleep(0.5)  # Rate limiting
        
        # Apply fixed titles
        for entry in data:
            paper_id = entry.get('paper_id', '')
            if paper_id in title_cache:
                entry['paper_title'] = title_cache[paper_id]['title']
                entry['paper_abstract'] = title_cache[paper_id]['abstract']
        
        print(f"   ✅ Fixed {stats['titles_fixed']} paper titles")
    
    # Step 2: Filter irrelevant files
    if filter_files:
        print("\n🔧 Step 2: Filtering irrelevant files...")
        filtered_data = []
        filter_reasons = defaultdict(int)
        
        for entry in data:
            is_relevant, reason = is_relevant_file(entry.get('file_path', ''))
            if is_relevant:
                filtered_data.append(entry)
            else:
                filter_reasons[reason] += 1
                stats['files_filtered'] += 1
        
        data = filtered_data
        
        if verbose:
            print(f"   Filter breakdown:")
            for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1])[:10]:
                print(f"     - {reason}: {count}")
        print(f"   ✅ Kept {len(data)} entries after file filtering")
    
    # Step 3: Compute relevance scores and filter
    print("\n🔧 Step 3: Computing relevance scores...")
    for entry in data:
        entry['_relevance_score'] = compute_relevance_score(entry)
    
    # Filter by relevance score
    pre_filter_count = len(data)
    data = [e for e in data if e['_relevance_score'] >= min_relevance_score]
    stats['low_relevance_filtered'] = pre_filter_count - len(data)
    print(f"   ✅ Kept {len(data)} entries with relevance >= {min_relevance_score}")
    
    # Step 4: Limit snippets per paper (keep highest relevance)
    print("\n🔧 Step 4: Limiting snippets per paper...")
    paper_entries = defaultdict(list)
    for entry in data:
        paper_id = entry.get('paper_id', 'unknown')
        paper_entries[paper_id].append(entry)
    
    final_data = []
    for paper_id, entries in paper_entries.items():
        # Sort by relevance score and keep top N
        entries.sort(key=lambda x: x['_relevance_score'], reverse=True)
        kept = entries[:max_snippets_per_paper]
        final_data.extend(kept)
        stats['per_paper_limit_filtered'] += len(entries) - len(kept)
    
    # Remove temporary score field
    for entry in final_data:
        del entry['_relevance_score']
    
    stats['final_count'] = len(final_data)
    
    # Save cleaned data
    print(f"\n💾 Saving cleaned dataset to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Original entries:        {stats['original_count']:,}")
    print(f"  Titles fixed:            {stats['titles_fixed']:,}")
    print(f"  Filtered (bad files):    {stats['files_filtered']:,}")
    print(f"  Filtered (low relevance): {stats['low_relevance_filtered']:,}")
    print(f"  Filtered (per-paper cap): {stats['per_paper_limit_filtered']:,}")
    print(f"  Final entries:           {stats['final_count']:,}")
    print(f"  Reduction:               {100*(1 - stats['final_count']/stats['original_count']):.1f}%")
    print("=" * 60)
    
    return stats


def analyze_cleaned_dataset(path: str = 'data/processed/code_snippets_cleaned.json'):
    """Analyze the cleaned dataset."""
    print("\n📊 ANALYZING CLEANED DATASET")
    print("=" * 60)
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    
    # Check title quality
    real_titles = [e for e in data if 'arXiv Query' not in e.get('paper_title', '')]
    print(f"Real titles: {len(real_titles)} ({100*len(real_titles)/len(data):.1f}%)")
    
    # Unique papers
    unique_papers = len(set(e.get('paper_id', '') for e in data))
    print(f"Unique papers: {unique_papers}")
    print(f"Avg snippets per paper: {len(data)/unique_papers:.1f}")
    
    # Sample entries
    print("\n📋 Sample cleaned entries:")
    for e in data[:3]:
        print(f"\n  Title: {e['paper_title'][:60]}...")
        print(f"  File: {e['file_path']}")
        print(f"  Function: {e['function_name']}")


if __name__ == '__main__':
    # Run cleaning
    stats = clean_dataset(
        fix_titles=False,  # Set to True to fetch from arXiv (slow)
        filter_files=True,
        min_relevance_score=0.15,
        max_snippets_per_paper=30,
        verbose=True
    )
    
    # Analyze result
    analyze_cleaned_dataset()
