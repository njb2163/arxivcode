"""
Generate instruction fine-tuning dataset for paper comprehension.
Creates query-response pairs for the paper explanation model.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperQADatasetGenerator:
    """Generate QA pairs for paper comprehension fine-tuning."""

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        random.seed(seed)

        # Question templates for different aspects of papers
        self.question_templates = {
            'title': [
                "What is the title of the paper with ArXiv ID {arxiv_id}?",
                "What paper has ArXiv ID {arxiv_id}?",
                "Tell me the title of ArXiv paper {arxiv_id}.",
            ],
            'year': [
                "When was the {title} paper published?",
                "What year was {title} released?",
                "In what year did {title} come out?",
            ],
            'category': [
                "What category is the {title} paper in?",
                "What is the ArXiv category of {title}?",
                "Which field does {title} belong to?",
            ],
            'summary': [
                "Summarize the {title} paper.",
                "What is {title} about?",
                "Explain the {title} paper.",
                "Give me an overview of {title}.",
            ],
            'repository': [
                "What is the GitHub repository for {title}?",
                "Where can I find the code for {title}?",
                "What's the implementation repository for {title}?",
            ],
            'repo_description': [
                "Describe the {repo_name} repository that implements {title}.",
                "What does the {repo_name} repository do?",
                "Explain the {repo_name} implementation of {title}.",
            ],
            'repo_language': [
                "What programming language is {title} implemented in?",
                "What language is the {repo_name} repository written in?",
            ],
            'repo_topics': [
                "What topics does the {title} implementation cover?",
                "What are the key topics in the {repo_name} repository?",
            ],
            'popularity': [
                "How popular is the {title} implementation?",
                "How many stars does the {title} repository have?",
            ],
            'application': [
                "What is {title} used for?",
                "What are the applications of {title}?",
                "How can I use {title}?",
            ],
            'comparison': [
                "How does {title} compare to similar papers?",
                "What makes {title} unique?",
            ],
        }

    def generate_qa_pairs(self, paper_data: Dict) -> List[Dict[str, str]]:
        """
        Generate QA pairs for a single paper.

        Args:
            paper_data: Paper data dict with 'paper' and 'repositories'

        Returns:
            List of QA pairs as dicts with 'input' and 'output'
        """
        paper = paper_data['paper']
        repos = paper_data.get('repositories', [])

        qa_pairs = []

        # Title questions
        for template in self.question_templates['title']:
            qa_pairs.append({
                'input': template.format(arxiv_id=paper['arxiv_id']),
                'output': paper['title']
            })

        # Year questions
        for template in self.question_templates['year']:
            qa_pairs.append({
                'input': template.format(title=paper['title']),
                'output': f"The {paper['title']} paper was published in {paper['year']}."
            })

        # Category questions
        for template in self.question_templates['category']:
            category_name = self._get_category_name(paper['category'])
            qa_pairs.append({
                'input': template.format(title=paper['title']),
                'output': f"{paper['title']} is in the {category_name} ({paper['category']}) category."
            })

        # Summary questions (multiple variations)
        for template in self.question_templates['summary']:
            summary = self._generate_paper_summary(paper, repos)
            qa_pairs.append({
                'input': template.format(title=paper['title']),
                'output': summary
            })

        # Repository questions
        if repos:
            main_repo = repos[0]  # Use first/primary repository

            for template in self.question_templates['repository']:
                qa_pairs.append({
                    'input': template.format(title=paper['title']),
                    'output': f"The code for {paper['title']} is available at {main_repo['url']} ({main_repo['name']})."
                })

            # Repository description
            if main_repo.get('description'):
                for template in self.question_templates['repo_description']:
                    qa_pairs.append({
                        'input': template.format(
                            repo_name=main_repo['name'],
                            title=paper['title']
                        ),
                        'output': main_repo['description']
                    })

            # Programming language
            if main_repo.get('language'):
                for template in self.question_templates['repo_language']:
                    qa_pairs.append({
                        'input': template.format(
                            title=paper['title'],
                            repo_name=main_repo['name']
                        ),
                        'output': f"The {paper['title']} implementation is written in {main_repo['language']}."
                    })

            # Topics
            if main_repo.get('topics'):
                for template in self.question_templates['repo_topics']:
                    topics_str = ', '.join(main_repo['topics'])
                    qa_pairs.append({
                        'input': template.format(
                            title=paper['title'],
                            repo_name=main_repo['name']
                        ),
                        'output': f"The {main_repo['name']} repository covers: {topics_str}."
                    })

            # Popularity
            if main_repo.get('stars'):
                for template in self.question_templates['popularity']:
                    qa_pairs.append({
                        'input': template.format(title=paper['title']),
                        'output': f"The {main_repo['name']} repository has {main_repo['stars']:,} stars and {main_repo['forks']:,} forks on GitHub."
                    })

        return qa_pairs

    def _generate_paper_summary(self, paper: Dict, repos: List[Dict]) -> str:
        """Generate a comprehensive summary of the paper."""
        title = paper['title']
        year = paper['year']
        category = self._get_category_name(paper['category'])

        summary = f"{title} is a {year} paper in {category}. "

        if repos:
            main_repo = repos[0]
            summary += f"The implementation is available at {main_repo['name']} "

            if main_repo.get('description'):
                summary += f"which {main_repo['description'].lower()} "

            if main_repo.get('language'):
                summary += f"It is written in {main_repo['language']}. "

            if main_repo.get('stars'):
                summary += f"The repository has {main_repo['stars']:,} stars, indicating high community interest."

        return summary.strip()

    def _get_category_name(self, category: str) -> str:
        """Convert ArXiv category code to readable name."""
        categories = {
            'cs.CL': 'Computation and Language (Natural Language Processing)',
            'cs.CV': 'Computer Vision and Pattern Recognition',
            'cs.LG': 'Machine Learning',
            'cs.AI': 'Artificial Intelligence',
            'cs.NE': 'Neural and Evolutionary Computing',
            'cs.IR': 'Information Retrieval',
            'stat.ML': 'Machine Learning (Statistics)',
            'cs.DC': 'Distributed, Parallel, and Cluster Computing',
        }
        return categories.get(category, category)

    def generate_dataset(
        self,
        input_file: str,
        output_train: str,
        output_eval: str,
        train_ratio: float = 0.9,
    ) -> Dict:
        """
        Generate complete QA dataset from paper-code pairs.

        Args:
            input_file: Path to paper_code_pairs.json
            output_train: Path to save training data
            output_eval: Path to save evaluation data
            train_ratio: Ratio of data for training (rest for eval)

        Returns:
            Statistics dict
        """
        logger.info(f"Loading papers from {input_file}")
        with open(input_file, 'r') as f:
            papers = json.load(f)

        logger.info(f"Generating QA pairs for {len(papers)} papers...")

        all_qa_pairs = []
        for paper_data in tqdm(papers, desc="Generating QA pairs"):
            qa_pairs = self.generate_qa_pairs(paper_data)
            all_qa_pairs.extend(qa_pairs)

        # Shuffle
        random.shuffle(all_qa_pairs)

        # Split train/eval
        split_idx = int(len(all_qa_pairs) * train_ratio)
        train_data = all_qa_pairs[:split_idx]
        eval_data = all_qa_pairs[split_idx:]

        # Save training data
        logger.info(f"Saving {len(train_data)} training examples to {output_train}")
        Path(output_train).parent.mkdir(parents=True, exist_ok=True)
        with open(output_train, 'w') as f:
            json.dump(train_data, f, indent=2)

        # Save evaluation data
        logger.info(f"Saving {len(eval_data)} evaluation examples to {output_eval}")
        Path(output_eval).parent.mkdir(parents=True, exist_ok=True)
        with open(output_eval, 'w') as f:
            json.dump(eval_data, f, indent=2)

        stats = {
            'total_papers': len(papers),
            'total_qa_pairs': len(all_qa_pairs),
            'train_examples': len(train_data),
            'eval_examples': len(eval_data),
            'avg_qa_per_paper': len(all_qa_pairs) / len(papers),
        }

        # Save stats
        stats_file = output_train.replace('.json', '_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("Dataset Generation Complete!")
        logger.info("=" * 60)
        logger.info(f"Total papers: {stats['total_papers']}")
        logger.info(f"Total QA pairs: {stats['total_qa_pairs']:,}")
        logger.info(f"Training examples: {stats['train_examples']:,}")
        logger.info(f"Evaluation examples: {stats['eval_examples']:,}")
        logger.info(f"Avg QA pairs per paper: {stats['avg_qa_per_paper']:.1f}")
        logger.info(f"Training data: {output_train}")
        logger.info(f"Evaluation data: {output_eval}")
        logger.info("=" * 60)

        return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate paper comprehension QA dataset"
    )
    parser.add_argument(
        '--input',
        default='data/raw/papers/paper_code_pairs.json',
        help='Input paper-code pairs file',
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed',
        help='Output directory for train/eval data',
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Ratio of data for training (default: 0.9)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )

    args = parser.parse_args()

    generator = PaperQADatasetGenerator(seed=args.seed)

    generator.generate_dataset(
        input_file=args.input,
        output_train=f"{args.output_dir}/train.json",
        output_eval=f"{args.output_dir}/eval.json",
        train_ratio=args.train_ratio,
    )


if __name__ == '__main__':
    main()
