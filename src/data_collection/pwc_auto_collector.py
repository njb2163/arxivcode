"""
Automated Papers With Code scraper
Fetches papers from Papers With Code website programmatically
"""

import requests
import json
import time
from typing import List, Dict
import re


def fetch_pwc_papers_by_category(categories: List[str] = None, min_stars: int = 50, target_count: int = 200) -> List[Dict]:
    """
    Fetch papers from Papers With Code by scraping their public pages

    Args:
        categories: List of categories (e.g., ['natural-language-processing', 'computer-vision'])
        min_stars: Minimum GitHub stars
        target_count: Target number of papers

    Returns:
        List of paper dictionaries with GitHub URLs
    """
    if categories is None:
        categories = [
            'natural-language-processing',
            'computer-vision',
            'machine-learning',
            'speech',
            'graphs',
            'reinforcement-learning'
        ]

    papers = []

    # Papers With Code has a public API endpoint
    # https://paperswithcode.com/api/v1/papers/
    base_url = "https://paperswithcode.com/api/v1/papers/"

    for category in categories:
        if len(papers) >= target_count:
            break

        try:
            # Fetch papers from the category
            # Note: This is a simplified approach - PWC API has pagination
            response = requests.get(
                base_url,
                params={
                    'ordering': '-stars',  # Order by stars descending
                    'items_per_page': 100,
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])

                for paper_data in results:
                    if len(papers) >= target_count:
                        break

                    # Extract paper info
                    arxiv_id = paper_data.get('arxiv_id', '')
                    if not arxiv_id:
                        continue

                    title = paper_data.get('title', '')
                    paper_url = paper_data.get('url_abs', '')

                    # Try to get GitHub repo from paper
                    github_url = None
                    if 'repository' in paper_data:
                        github_url = paper_data.get('repository', {}).get('url', '')

                    if github_url and 'github.com' in github_url:
                        papers.append({
                            'arxiv_id': arxiv_id,
                            'title': title,
                            'github_urls': [github_url],
                            'year': paper_data.get('published', '')[:4] if paper_data.get('published') else None,
                            'category': 'cs.CL'  # Default, can be refined
                        })

                time.sleep(2)  # Rate limiting

        except Exception as e:
            print(f"Error fetching from Papers With Code: {e}")
            continue

    return papers


def get_awesome_ml_papers() -> List[Dict]:
    """
    Fetch papers from awesome-ml lists and other curated sources
    This pulls from known high-quality GitHub repositories
    """
    # URLs to awesome lists and paper collections
    awesome_sources = [
        {
            'name': 'Awesome NLP Papers',
            'url': 'https://raw.githubusercontent.com/keon/awesome-nlp/master/README.md',
            'category': 'cs.CL'
        },
        {
            'name': 'Awesome Deep Learning Papers',
            'url': 'https://raw.githubusercontent.com/terryum/awesome-deep-learning-papers/master/README.md',
            'category': 'cs.LG'
        },
    ]

    papers = []

    for source in awesome_sources:
        try:
            response = requests.get(source['url'], timeout=30)
            if response.status_code == 200:
                content = response.text

                # Extract ArXiv IDs and GitHub URLs
                arxiv_pattern = r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)'
                github_pattern = r'https://github\.com/[\w-]+/[\w.-]+'

                arxiv_ids = re.findall(arxiv_pattern, content)
                github_urls = re.findall(github_pattern, content)

                # This is a simplified approach - would need more sophisticated parsing
                # For now, just note the source
                print(f"Found {len(arxiv_ids)} arxiv IDs and {len(github_urls)} GitHub URLs in {source['name']}")

            time.sleep(2)
        except Exception as e:
            print(f"Error fetching {source['name']}: {e}")
            continue

    return papers


if __name__ == "__main__":
    # Test the functions
    print("Fetching papers from Papers With Code...")
    pwc_papers = fetch_pwc_papers_by_category(target_count=50)
    print(f"Found {len(pwc_papers)} papers from PWC")

    if pwc_papers:
        print("\nSample papers:")
        for paper in pwc_papers[:5]:
            print(f"- {paper['title']} ({paper['arxiv_id']})")
