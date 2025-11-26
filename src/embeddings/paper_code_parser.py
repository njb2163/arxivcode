"""
Day 4 Step 2: Parse paper_code_pairs.json
Extracts paper descriptions and code descriptions from the JSON file
"""

import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import arxiv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperCodeParser:
    """
    Parser for paper_code_pairs.json that extracts paper and code text
    """

    def __init__(self, json_path: str, fetch_abstracts: bool = False):
        """
        Initialize parser

        Args:
            json_path: Path to paper_code_pairs.json
            fetch_abstracts: Whether to fetch abstracts from ArXiv API
                           (slower but more complete)
        """
        self.json_path = Path(json_path)
        self.fetch_abstracts = fetch_abstracts
        self.pairs = []

    def load_json(self) -> List[Dict]:
        """
        Load paper_code_pairs.json file

        Returns:
            List of paper-code pair dictionaries
        """
        logger.info(f"Loading JSON file: {self.json_path}")
        if not self.json_path.exists():
            raise FileNotFoundError(f"File not found: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} paper-code pairs")
        return data

    def fetch_abstract(self, arxiv_id: str) -> Optional[str]:
        """
        Fetch abstract from ArXiv API

        Args:
            arxiv_id: ArXiv paper ID (e.g., "1706.03762")

        Returns:
            Abstract text or None if not found
        """
        try:
            # Remove version suffix if present (e.g., "1706.03762v1" -> "1706.03762")
            arxiv_id_clean = arxiv_id.split("v")[0]

            # Search for paper
            search = arxiv.Search(id_list=[arxiv_id_clean])
            paper = next(search.results(), None)

            if paper:
                return paper.summary
            else:
                logger.warning(f"Could not fetch abstract for {arxiv_id}")
                return None

        except Exception as e:
            logger.warning(f"Error fetching abstract for {arxiv_id}: {e}")
            return None

    def extract_paper_text(self, paper: Dict, abstract: Optional[str] = None) -> str:
        """
        Extract paper description text

        Args:
            paper: Paper dictionary with title, arxiv_id, etc.
            abstract: Optional abstract text (fetched from ArXiv if available)

        Returns:
            Combined paper text: "title + abstract"
        """
        title = paper.get("title") or ""
        title = title.strip() if title else ""
        arxiv_id = paper.get("arxiv_id", "")

        # Build paper text
        paper_text = title

        if abstract:
            paper_text = f"{title} {abstract}"
        elif self.fetch_abstracts:
            # Try to fetch abstract
            fetched_abstract = self.fetch_abstract(arxiv_id)
            if fetched_abstract:
                paper_text = f"{title} {fetched_abstract}"
                time.sleep(0.5)  # Rate limiting for ArXiv API

        return paper_text.strip()

    def extract_code_text(self, repo: Dict) -> str:
        """
        Extract code description text

        Args:
            repo: Repository dictionary with name, description, etc.

        Returns:
            Combined code text: "repository.name + repository.description"
        """
        name = repo.get("name") or ""
        name = name.strip() if name else ""

        description = repo.get("description") or ""
        description = description.strip() if description else ""

        # Combine name and description
        code_text = name
        if description:
            code_text = f"{name} {description}"

        return code_text.strip()

    def parse(self) -> List[Tuple[str, str, Dict]]:
        """
        Parse JSON file and extract paper-code text pairs

        Returns:
            List of tuples: (paper_text, code_text, metadata)
            metadata contains: paper_id, repo_name, arxiv_id, etc.
        """
        logger.info("=" * 60)
        logger.info("Day 4 Step 2: Parsing paper_code_pairs.json")
        logger.info("=" * 60)

        # Load JSON
        data = self.load_json()

        pairs = []
        skipped = 0

        for i, pair in enumerate(data, 1):
            paper = pair.get("paper", {})
            repositories = pair.get("repositories", [])

            # Skip if paper is missing required fields
            if not paper or not paper.get("title") or paper.get("title") is None:
                logger.warning(f"Pair {i}: Skipping - missing paper title")
                skipped += 1
                continue

            # Extract paper text
            paper_text = self.extract_paper_text(paper)

            if not paper_text:
                logger.warning(f"Pair {i}: Skipping - empty paper text")
                skipped += 1
                continue

            # Process each repository
            for repo in repositories:
                # Skip if repository is missing required fields
                if not repo or not repo.get("name") or repo.get("name") is None:
                    logger.debug(f"Pair {i}: Skipping repo - missing name")
                    continue

                # Extract code text
                code_text = self.extract_code_text(repo)

                if not code_text:
                    logger.debug(f"Pair {i}: Skipping repo - empty code text")
                    continue

                # Create metadata
                metadata = {
                    "arxiv_id": paper.get("arxiv_id") or "",
                    "paper_title": paper.get("title") or "",
                    "repo_name": repo.get("name") or "",
                    "repo_url": repo.get("url") or "",
                    "pair_index": i,
                }

                pairs.append((paper_text, code_text, metadata))

                if i % 50 == 0:
                    logger.info(
                        f"Processed {i}/{len(data)} pairs... ({len(pairs)} text pairs created)"
                    )

        logger.info("=" * 60)
        logger.info(f"Parsing complete!")
        logger.info(f"  Total pairs processed: {len(data)}")
        logger.info(f"  Text pairs created: {len(pairs)}")
        logger.info(f"  Skipped: {skipped}")
        logger.info("=" * 60)

        self.pairs = pairs
        return pairs

    def filter_empty(
        self, pairs: List[Tuple[str, str, Dict]]
    ) -> List[Tuple[str, str, Dict]]:
        """
        Filter out pairs with empty text fields

        Args:
            pairs: List of (paper_text, code_text, metadata) tuples

        Returns:
            Filtered list
        """
        filtered = [
            (paper_text, code_text, metadata)
            for paper_text, code_text, metadata in pairs
            if paper_text.strip() and code_text.strip()
        ]

        logger.info(
            f"Filtered {len(pairs)} -> {len(filtered)} pairs (removed empty entries)"
        )
        return filtered

    def save_parsed_pairs(
        self, output_path: str, pairs: Optional[List[Tuple[str, str, Dict]]] = None
    ):
        """
        Save parsed pairs to JSON file

        Args:
            output_path: Path to save output JSON
            pairs: Pairs to save (uses self.pairs if None)
        """
        if pairs is None:
            pairs = self.pairs

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to list of dictionaries for JSON serialization
        output_data = [
            {
                "paper_text": paper_text,
                "code_text": code_text,
                "metadata": metadata,
            }
            for paper_text, code_text, metadata in pairs
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(output_data)} parsed pairs to {output_path}")


def parse_paper_code_pairs(
    json_path: str = "data/raw/papers/paper_code_pairs.json",
    fetch_abstracts: bool = False,
    output_path: Optional[str] = None,
) -> List[Tuple[str, str, Dict]]:
    """
    Convenience function to parse paper_code_pairs.json

    Args:
        json_path: Path to paper_code_pairs.json
        fetch_abstracts: Whether to fetch abstracts from ArXiv
        output_path: Optional path to save parsed pairs

    Returns:
        List of (paper_text, code_text, metadata) tuples
    """
    parser = PaperCodeParser(json_path, fetch_abstracts=fetch_abstracts)
    pairs = parser.parse()
    pairs = parser.filter_empty(pairs)

    if output_path:
        parser.save_parsed_pairs(output_path, pairs)

    return pairs


if __name__ == "__main__":
    # Example usage
    import sys

    # Parse without fetching abstracts (faster)
    logger.info("Parsing paper_code_pairs.json (without abstracts)...")
    pairs = parse_paper_code_pairs(
        json_path="data/raw/papers/paper_code_pairs.json",
        fetch_abstracts=False,
        output_path="data/processed/parsed_pairs.json",
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PARSING SUMMARY")
    print("=" * 60)
    print(f"Total pairs: {len(pairs)}")
    if pairs:
        print(f"\nSample pair:")
        paper_text, code_text, metadata = pairs[0]
        print(f"  Paper: {paper_text[:100]}...")
        print(f"  Code: {code_text[:100]}...")
        print(f"  ArXiv ID: {metadata['arxiv_id']}")
        print(f"  Repo: {metadata['repo_name']}")
    print("=" * 60)
