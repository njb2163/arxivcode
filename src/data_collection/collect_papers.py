"""
Main script to collect 200+ paper-code pairs
Uses ArXiv API + GitHub Search
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_collection.arxiv_github_collector import ArxivGithubCollector


def main():
    print("="*60)
    print("Paper-Code Pair Collection Script")
    print("="*60)

    # Check for GitHub token
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("\n⚠️  WARNING: No GITHUB_TOKEN found!")
        print("   GitHub API rate limits will be very restrictive.")
        print("   Consider setting GITHUB_TOKEN in .env file.")
        print("   Get token at: https://github.com/settings/tokens")
        print("\n   Do you want to continue anyway? (y/n): ", end='')

        response = input().strip().lower()
        if response != 'y':
            print("   Exiting. Please set GITHUB_TOKEN and try again.")
            return
        print()

    # Initialize collector
    collector = ArxivGithubCollector(github_token=github_token)

    print("\nCollection Parameters:")
    print("  Categories: cs.CL, cs.LG")
    print("  Date Range: 2020-2025")
    print("  Min Stars: 50")
    print("  Target Count: 200 pairs")
    print("  Max Papers to Search: 500")
    print()

    # Collect papers
    try:
        pairs = collector.collect_paper_code_pairs(
            categories=['cs.CL', 'cs.LG'],
            start_year=2020,
            end_year=2025,
            min_stars=50,
            target_count=200,
            max_papers_to_search=500
        )

        if pairs:
            # Save results
            output_path = collector.save_results(pairs)

            print(f"\n{'='*60}")
            print(f"✅ Collection Complete!")
            print(f"{'='*60}")
            print(f"Paper-code pairs collected: {len(pairs)}")
            print(f"Output file: {output_path}")
            print(f"{'='*60}\n")

            # Display summary statistics
            total_repos = sum(len(p.get('repositories', [])) for p in pairs)
            all_stars = [r.get('stars', 0) for p in pairs for r in p.get('repositories', [])]

            print("Summary Statistics:")
            print(f"  Total papers: {len(pairs)}")
            print(f"  Total repositories: {total_repos}")
            if all_stars:
                print(f"  Avg stars: {sum(all_stars) / len(all_stars):.0f}")
                print(f"  Max stars: {max(all_stars)}")
                print(f"  Min stars: {min(all_stars)}")

            # Display sample pairs
            print("\nSample paper-code pairs:")
            for i, pair in enumerate(pairs[:5], 1):
                paper = pair['paper']
                repos = pair['repositories']
                print(f"\n{i}. {paper['title'][:60]}...")
                print(f"   ArXiv: {paper['arxiv_id']}")
                print(f"   Published: {paper['published'][:10]}")
                print(f"   Repositories: {len(repos)}")
                if repos:
                    top_repo = max(repos, key=lambda r: r.get('stars', 0))
                    print(f"   Top repo: {top_repo['name']} ({top_repo['stars']} stars)")

        else:
            print("\n❌ No pairs collected!")
            print("   This may be due to rate limiting or filter criteria.")
            print("   Try setting GITHUB_TOKEN in .env for better rate limits.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        print("   Partial results may be available in data/raw/papers/")

    except Exception as e:
        print(f"\n❌ Collection failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
