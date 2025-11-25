"""
Automated Papers With Code Dataset Collector
Fetches papers from paperswithcode-data GitHub repository
This dataset is automatically updated and requires no API tokens
"""

import requests
import json
import time
import os
import sys
from typing import List, Dict
from pathlib import Path
from github import Github
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PWCDatasetCollector:
    """
    Collector that pulls from the paperswithcode-data repository
    https://github.com/paperswithcode/paperswithcode-data
    """

    def __init__(self, github_token: str = None, output_dir: str = "data/raw/papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GitHub API
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.github = Github(self.github_token) if self.github_token else Github()

        # Papers With Code dataset URLs
        self.base_url = "https://raw.githubusercontent.com/paperswithcode/paperswithcode-data/main"

    def fetch_papers_from_dataset(self, min_stars: int = 50, target_count: int = 200,
                                    categories: List[str] = None,
                                    year_start: int = 2020, year_end: int = 2025) -> List[Dict]:
        """
        Fetch papers by scraping Papers With Code greatest hits and trending papers

        Args:
            min_stars: Minimum GitHub stars for repositories
            target_count: Target number of papers to collect
            categories: ArXiv categories to filter (e.g., ['cs.CL', 'cs.LG'])
            year_start: Start year for filtering papers
            year_end: End year for filtering papers

        Returns:
            List of paper dictionaries with GitHub URLs
        """
        if categories is None:
            categories = ['cs.CL', 'cs.LG', 'cs.AI', 'cs.CV']

        logger.info("Fetching papers from Papers With Code (via GitHub trending)...")
        logger.info(f"Filters: min_stars={min_stars}, years={year_start}-{year_end}, categories={categories}")

        papers = []

        # Strategy: Search GitHub for repos with "arxiv" in the README
        # that have high stars and are related to ML/NLP
        try:
            # Search for popular ML/AI repos that reference ArXiv papers
            # Using broader, more specific searches to avoid rate limits
            search_queries = [
                'arxiv neural network language:python stars:>200',
                'arxiv bert transformer language:python stars:>100',
                'arxiv gpt language model language:python stars:>100',
                'arxiv llama language:python stars:>100',
                'arxiv diffusion image language:python stars:>100',
                'arxiv vision transformer language:python stars:>100',
                'arxiv retrieval rag language:python stars:>100',
                'arxiv reinforcement learning language:python stars:>100',
            ]

            for query in search_queries:
                if len(papers) >= target_count:
                    break

                logger.info(f"Searching GitHub: {query}")

                try:
                    repos = self.github.search_repositories(query=query, sort='stars', order='desc')

                    for repo in repos[:50]:  # Limit to top 50 per query
                        if len(papers) >= target_count:
                            break

                        # Check if repo meets minimum stars requirement
                        if repo.stargazers_count < min_stars:
                            continue

                        # Try to extract ArXiv ID from README
                        try:
                            readme = repo.get_readme()
                            readme_content = readme.decoded_content.decode('utf-8', errors='ignore')

                            # Extract ArXiv IDs from README
                            import re
                            arxiv_pattern = r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)'
                            arxiv_ids = re.findall(arxiv_pattern, readme_content, re.IGNORECASE)

                            if not arxiv_ids:
                                continue

                            # Use the first ArXiv ID found
                            arxiv_id = arxiv_ids[0]

                            # Get year from ArXiv ID
                            year_prefix = arxiv_id.split('.')[0]
                            if len(year_prefix) == 4:
                                year = int('20' + year_prefix[:2])
                            else:
                                year = 2020  # Default

                            # Filter by year
                            if year < year_start or year > year_end:
                                continue

                            # Fetch paper title from ArXiv
                            paper_title = repo.description or f"Paper {arxiv_id}"
                            try:
                                arxiv_api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
                                arxiv_response = requests.get(arxiv_api_url, timeout=10)
                                if arxiv_response.status_code == 200:
                                    import xml.etree.ElementTree as ET
                                    root = ET.fromstring(arxiv_response.content)
                                    title_elem = root.find('.//{http://www.w3.org/2005/Atom}title')
                                    if title_elem is not None:
                                        paper_title = title_elem.text.strip()
                            except:
                                pass

                            # Determine category
                            category = 'cs.CL'  # Default
                            repo_topics = repo.get_topics() if hasattr(repo, 'get_topics') else []
                            if any(topic in ['computer-vision', 'cv', 'vision'] for topic in repo_topics):
                                category = 'cs.CV'
                            elif any(topic in ['machine-learning', 'ml'] for topic in repo_topics):
                                category = 'cs.LG'

                            repo_data = {
                                'url': repo.html_url,
                                'name': repo.full_name,
                                'description': repo.description,
                                'stars': repo.stargazers_count,
                                'forks': repo.forks_count,
                                'language': repo.language,
                                'created_at': repo.created_at.isoformat() if repo.created_at else None,
                                'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
                                'topics': repo_topics
                            }

                            paper = {
                                'arxiv_id': arxiv_id,
                                'title': paper_title,
                                'year': year,
                                'category': category,
                                'url': f"https://arxiv.org/abs/{arxiv_id}"
                            }

                            pair = {
                                'paper': paper,
                                'repositories': [repo_data]
                            }

                            papers.append(pair)
                            logger.info(f"[{len(papers)}/{target_count}] ✓ {paper_title[:60]}... ({repo.full_name}, {repo.stargazers_count} stars)")

                            # Rate limiting
                            time.sleep(2)

                        except Exception as e:
                            logger.debug(f"Error processing repo {repo.full_name}: {e}")
                            continue

                    time.sleep(5)  # Rate limit between queries

                except Exception as e:
                    logger.warning(f"Error with query '{query}': {e}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching paperswithcode-data: {e}")
            return []

        logger.info(f"\nTotal paper-code pairs collected: {len(papers)}")
        return papers

    def save_results(self, pairs: List[Dict], filename: str = "paper_code_pairs.json") -> Path:
        """Save collected paper-code pairs to JSON file"""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(pairs)} paper-code pairs to {output_path}")

        # Save statistics
        stats = {
            'total_pairs': len(pairs),
            'total_repositories': sum(len(p.get('repositories', [])) for p in pairs),
            'papers_by_year': {},
            'avg_stars': 0
        }

        all_stars = []
        for pair in pairs:
            paper = pair.get('paper', {})
            year = paper.get('year', '')
            if year:
                stats['papers_by_year'][str(year)] = stats['papers_by_year'].get(str(year), 0) + 1

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
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        logger.warning("No GITHUB_TOKEN found. API rate limits will be lower.")

    collector = PWCDatasetCollector(github_token=github_token)

    # Collect paper-code pairs from paperswithcode-data repository
    pairs = collector.fetch_papers_from_dataset(
        min_stars=50,
        target_count=200,
        categories=['cs.CL', 'cs.LG', 'cs.AI', 'cs.CV'],
        year_start=2020,
        year_end=2025
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
        for i, pair in enumerate(pairs[:5], 1):
            paper = pair['paper']
            repos = pair['repositories']
            print(f"\n{i}. {paper['title']}")
            print(f"   ArXiv: {paper['arxiv_id']}")
            print(f"   Year: {paper.get('year', 'N/A')}")
            print(f"   Repositories: {len(repos)}")
            if repos:
                top_repo = max(repos, key=lambda r: r.get('stars', 0))
                print(f"   Repo: {top_repo['name']} ({top_repo['stars']} stars)")
    else:
        print("No pairs collected!")


if __name__ == "__main__":
    main()
