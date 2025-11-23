"""
Test script for ArXiv + GitHub collector
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_collection.arxiv_github_collector import ArxivGithubCollector


def test_arxiv_search():
    """Test ArXiv paper search"""
    print("="*60)
    print("Test 1: ArXiv Paper Search")
    print("="*60)

    collector = ArxivGithubCollector()

    # Search for a small number of papers
    papers = collector.search_arxiv_papers(
        categories=['cs.CL'],
        start_year=2023,
        end_year=2024,
        max_results=5
    )

    print(f"\nFetched {len(papers)} papers from ArXiv")

    if papers:
        print("\nSample papers:")
        for i, paper in enumerate(papers[:3], 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   ArXiv ID: {paper['arxiv_id']}")
            print(f"   Published: {paper['published'][:10]}")
            print(f"   Categories: {', '.join(paper['categories'])}")

    return len(papers) > 0


def test_github_search():
    """Test GitHub repository search for a known paper"""
    print("\n" + "="*60)
    print("Test 2: GitHub Repository Search")
    print("="*60)

    collector = ArxivGithubCollector()

    # Use a well-known paper with code available
    test_paper = {
        'arxiv_id': '1706.03762',  # "Attention Is All You Need"
        'title': 'Attention Is All You Need',
    }

    print(f"Searching GitHub for: {test_paper['title']}")

    repos = collector.search_github_for_paper(
        test_paper,
        min_stars=50,
        max_repos=5
    )

    print(f"\nFound {len(repos)} repositories")

    if repos:
        print("\nRepositories:")
        for i, repo in enumerate(repos, 1):
            print(f"\n{i}. {repo['name']}")
            print(f"   URL: {repo['url']}")
            print(f"   Stars: {repo['stars']}")
            print(f"   Language: {repo['language']}")

    return len(repos) > 0


def test_small_collection():
    """Test collecting a small batch of paper-code pairs"""
    print("\n" + "="*60)
    print("Test 3: Small Collection (5 pairs)")
    print("="*60)

    collector = ArxivGithubCollector(output_dir="data/raw/papers")

    # Collect a small batch
    pairs = collector.collect_paper_code_pairs(
        categories=['cs.CL'],
        start_year=2023,
        end_year=2024,
        min_stars=50,
        target_count=5,
        max_papers_to_search=20
    )

    print(f"\nCollected {len(pairs)} paper-code pairs")

    if pairs:
        # Save results
        output_path = collector.save_results(pairs, filename="test_pairs.json")
        print(f"Results saved to: {output_path}")

        print("\nCollected pairs:")
        for i, pair in enumerate(pairs, 1):
            paper = pair['paper']
            repos = pair['repositories']
            print(f"\n{i}. {paper['title'][:60]}...")
            print(f"   ArXiv: {paper['arxiv_id']}")
            print(f"   Repositories: {len(repos)}")
            if repos:
                top_repo = max(repos, key=lambda r: r.get('stars', 0))
                print(f"   Top repo: {top_repo['name']} ({top_repo['stars']} stars)")

    return len(pairs) > 0


if __name__ == "__main__":
    print("="*60)
    print("ArXiv + GitHub Collector Test Suite")
    print("="*60)
    print("\nNote: These tests will make API calls to ArXiv and GitHub")
    print("GitHub API has rate limits. Set GITHUB_TOKEN in .env for higher limits.\n")

    try:
        # Run tests
        if not test_arxiv_search():
            print("\n❌ ArXiv search test failed!")
            sys.exit(1)

        if not test_github_search():
            print("\n⚠️  GitHub search test failed - check API rate limits or token")
            print("   Continuing with other tests...")

        if not test_small_collection():
            print("\n⚠️  Collection test failed - check API rate limits")
            print("   You may need to wait or set GITHUB_TOKEN")

        print("\n" + "="*60)
        print("✅ Tests completed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
