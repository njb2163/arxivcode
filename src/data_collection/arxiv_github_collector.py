"""
ArXiv + GitHub Integration for Paper-Code Pairs
Collects papers from ArXiv and finds associated GitHub repositories
"""

import arxiv
import requests
from github import Github
import json
import time
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import logging
import re
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArxivGithubCollector:
    """
    Collector that finds paper-code pairs by:
    1. Searching ArXiv for papers in specific categories
    2. Searching GitHub for repos that reference those papers
    3. Filtering by stars and metadata
    """

    def __init__(self, github_token: Optional[str] = None, output_dir: str = "data/raw/papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GitHub API (optional token for higher rate limits)
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.github = Github(self.github_token) if self.github_token else Github()

        # Initialize ArXiv client
        self.arxiv_client = arxiv.Client()

    def search_arxiv_papers(
        self,
        categories: List[str],
        start_year: int = 2020,
        end_year: int = 2025,
        max_results: int = 500
    ) -> List[Dict]:
        """
        Search ArXiv for papers in specific categories and date range.

        Args:
            categories: List of ArXiv categories (e.g., ['cs.CL', 'cs.LG'])
            start_year: Start year for paper publication
            end_year: End year for paper publication
            max_results: Maximum number of papers to fetch

        Returns:
            List of paper dictionaries
        """
        papers = []

        for category in categories:
            logger.info(f"Searching ArXiv for category: {category}")

            # Build query for category and date range
            query = f"cat:{category} AND submittedDate:[{start_year}0101 TO {end_year}1231]"

            try:
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )

                results = self.arxiv_client.results(search)

                for result in results:
                    paper_data = {
                        'arxiv_id': result.entry_id.split('/')[-1],
                        'title': result.title,
                        'authors': [author.name for author in result.authors],
                        'abstract': result.summary,
                        'published': result.published.isoformat(),
                        'updated': result.updated.isoformat(),
                        'categories': result.categories,
                        'primary_category': result.primary_category,
                        'pdf_url': result.pdf_url,
                        'url': result.entry_id
                    }

                    papers.append(paper_data)
                    logger.info(f"Fetched: {result.title}")

            except Exception as e:
                logger.error(f"Error searching ArXiv for {category}: {e}")
                continue

        logger.info(f"Total papers fetched from ArXiv: {len(papers)}")
        return papers

    def search_github_for_paper(
        self,
        paper: Dict,
        min_stars: int = 50,
        max_repos: int = 5
    ) -> List[Dict]:
        """
        Search GitHub for repositories that reference a paper.

        Args:
            paper: Paper dictionary with arxiv_id and title
            min_stars: Minimum number of stars
            max_repos: Maximum number of repos to return per paper

        Returns:
            List of repository dictionaries
        """
        arxiv_id = paper.get('arxiv_id', '')
        title = paper.get('title', '')

        if not arxiv_id:
            return []

        repos = []

        # Clean arxiv_id (remove version)
        arxiv_id_base = arxiv_id.split('v')[0]

        # Search strategies
        search_queries = [
            f"{arxiv_id_base}",  # Direct ArXiv ID search (without version)
            f'"{title[:50]}" arxiv',  # Title (truncated) + arxiv keyword
        ]

        for query in search_queries:
            try:
                # Search GitHub repositories
                result = self.github.search_repositories(
                    query=f"{query} stars:>={min_stars}",
                    sort="stars",
                    order="desc"
                )

                # Get top repositories - use list() to get limited results
                repo_list = list(result[:max_repos])

                for repo in repo_list:
                    # Check if repo actually mentions the paper
                    if self._verify_paper_in_repo(repo, arxiv_id_base, title):
                        repo_data = {
                            'url': repo.html_url,
                            'name': repo.full_name,
                            'description': repo.description,
                            'stars': repo.stargazers_count,
                            'forks': repo.forks_count,
                            'language': repo.language,
                            'created_at': repo.created_at.isoformat() if repo.created_at else None,
                            'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
                            'topics': repo.get_topics() if hasattr(repo, 'get_topics') else []
                        }

                        # Avoid duplicates
                        if not any(r['url'] == repo_data['url'] for r in repos):
                            repos.append(repo_data)
                            logger.info(f"  Found repo: {repo.full_name} ({repo.stargazers_count} stars)")

                if repos:
                    break  # If we found repos, no need to try other queries

                time.sleep(3)  # Rate limiting

            except Exception as e:
                logger.warning(f"Error searching GitHub for '{query}': {e}")
                if "rate limit" in str(e).lower():
                    logger.error("GitHub rate limit exceeded. Set GITHUB_TOKEN in .env for higher limits.")
                    time.sleep(60)  # Wait 1 minute on rate limit
                else:
                    time.sleep(5)  # Longer sleep on other errors
                continue

        return repos

    def _verify_paper_in_repo(self, repo, arxiv_id: str, title: str) -> bool:
        """
        Verify that the repository actually references the paper.

        Args:
            repo: GitHub repository object
            arxiv_id: ArXiv ID to search for
            title: Paper title to search for

        Returns:
            True if paper is referenced in the repo
        """
        try:
            # Check README
            readme = repo.get_readme()
            readme_content = readme.decoded_content.decode('utf-8').lower()

            # Clean arxiv ID (remove version if present)
            arxiv_id_base = arxiv_id.split('v')[0]

            # Check if arxiv ID or title appears in README
            if arxiv_id_base.lower() in readme_content:
                return True

            # Check for title (simplified - just check key words)
            title_words = [w.lower() for w in title.split() if len(w) > 4]
            if len(title_words) >= 3:
                matches = sum(1 for word in title_words[:5] if word in readme_content)
                if matches >= 3:
                    return True

        except Exception as e:
            # If we can't access README, assume it might still be valid
            logger.debug(f"Could not verify paper in repo {repo.full_name}: {e}")
            return True  # Give benefit of doubt

        return False

    def collect_paper_code_pairs(
        self,
        categories: List[str] = None,
        start_year: int = 2020,
        end_year: int = 2025,
        min_stars: int = 50,
        target_count: int = 200,
        max_papers_to_search: int = 500
    ) -> List[Dict]:
        """
        Collect paper-code pairs by searching ArXiv and GitHub.

        Args:
            categories: ArXiv categories to search
            start_year: Start year for papers
            end_year: End year for papers
            min_stars: Minimum GitHub stars
            target_count: Target number of paper-code pairs
            max_papers_to_search: Maximum papers to search on ArXiv

        Returns:
            List of paper-code pair dictionaries
        """
        if categories is None:
            categories = ['cs.CL', 'cs.LG']

        logger.info(f"Starting collection with target: {target_count} pairs")
        logger.info(f"Filters - Categories: {categories}, Years: {start_year}-{end_year}, Stars: >={min_stars}")

        # Step 1: Fetch papers from ArXiv
        logger.info("\n" + "="*60)
        logger.info("Step 1: Fetching papers from ArXiv")
        logger.info("="*60)

        papers = self.search_arxiv_papers(
            categories=categories,
            start_year=start_year,
            end_year=end_year,
            max_results=max_papers_to_search
        )

        if not papers:
            logger.error("No papers found on ArXiv!")
            return []

        # Step 2: Find GitHub repositories for each paper
        logger.info("\n" + "="*60)
        logger.info("Step 2: Searching GitHub for associated repositories")
        logger.info("="*60)

        paper_code_pairs = []

        for i, paper in enumerate(papers, 1):
            if len(paper_code_pairs) >= target_count:
                logger.info(f"Reached target count of {target_count} pairs")
                break

            logger.info(f"\n[{i}/{len(papers)}] Processing: {paper['title'][:60]}...")

            repos = self.search_github_for_paper(
                paper,
                min_stars=min_stars,
                max_repos=5
            )

            if repos:
                pair = {
                    'paper': paper,
                    'repositories': repos,
                    'pair_created_at': datetime.now().isoformat()
                }
                paper_code_pairs.append(pair)
                logger.info(f"  ✓ Added pair with {len(repos)} repositories")
            else:
                logger.info(f"  ✗ No repositories found")

            # Rate limiting for GitHub API
            time.sleep(3)

        logger.info(f"\nTotal paper-code pairs collected: {len(paper_code_pairs)}")
        return paper_code_pairs

    def save_results(
        self,
        pairs: List[Dict],
        filename: str = "paper_code_pairs.json"
    ) -> Path:
        """
        Save collected paper-code pairs to JSON file.

        Args:
            pairs: List of paper-code pair dictionaries
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(pairs)} paper-code pairs to {output_path}")

        # Save statistics
        stats = {
            'total_pairs': len(pairs),
            'total_papers': len(pairs),
            'total_repositories': sum(len(p.get('repositories', [])) for p in pairs),
            'collection_date': datetime.now().isoformat(),
            'papers_by_year': {},
            'papers_by_category': {},
            'avg_stars': 0
        }

        # Calculate statistics
        all_stars = []
        for pair in pairs:
            paper = pair.get('paper', {})

            # Year distribution
            pub_date = paper.get('published', '')
            if pub_date:
                try:
                    year = datetime.fromisoformat(pub_date).year
                    stats['papers_by_year'][str(year)] = stats['papers_by_year'].get(str(year), 0) + 1
                except:
                    pass

            # Category distribution
            primary_cat = paper.get('primary_category', '')
            if primary_cat:
                stats['papers_by_category'][primary_cat] = stats['papers_by_category'].get(primary_cat, 0) + 1

            # Stars
            for repo in pair.get('repositories', []):
                stars = repo.get('stars', 0)
                if stars:
                    all_stars.append(stars)

        if all_stars:
            stats['avg_stars'] = sum(all_stars) / len(all_stars)
            stats['max_stars'] = max(all_stars)
            stats['min_stars'] = min(all_stars)

        stats_path = self.output_dir / "collection_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {stats_path}")

        return output_path


def main():
    """Main execution function"""
    # Check for GitHub token
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        logger.warning("No GITHUB_TOKEN found in environment. API rate limits will be lower.")
        logger.warning("Consider setting GITHUB_TOKEN in .env file for better performance.")

    collector = ArxivGithubCollector(github_token=github_token)

    # Collect paper-code pairs
    pairs = collector.collect_paper_code_pairs(
        categories=['cs.CL', 'cs.LG'],
        start_year=2020,
        end_year=2025,
        min_stars=50,
        target_count=200,
        max_papers_to_search=500
    )

    # Save results
    if pairs:
        output_path = collector.save_results(pairs)

        print(f"\n{'='*60}")
        print(f"Collection Complete!")
        print(f"{'='*60}")
        print(f"Paper-code pairs collected: {len(pairs)}")
        print(f"Output file: {output_path}")
        print(f"{'='*60}\n")

        # Display sample pairs
        print("Sample paper-code pairs:")
        for i, pair in enumerate(pairs[:3], 1):
            paper = pair['paper']
            repos = pair['repositories']
            print(f"\n{i}. {paper['title']}")
            print(f"   ArXiv: {paper['arxiv_id']}")
            print(f"   Published: {paper['published'][:10]}")
            print(f"   Repositories: {len(repos)}")
            if repos:
                top_repo = max(repos, key=lambda r: r.get('stars', 0))
                print(f"   Top repo: {top_repo['name']} ({top_repo['stars']} stars)")
    else:
        print("No pairs collected!")


if __name__ == "__main__":
    main()
