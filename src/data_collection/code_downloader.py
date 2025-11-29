"""
Download and extract code from GitHub repositories.
This provides the actual code files needed by the code understanding model.
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeDownloader:
    """Download code from GitHub repositories."""

    def __init__(
        self,
        output_dir: str = "data/raw/code_repos",
        max_repo_size_mb: int = 500,
        file_extensions: Optional[List[str]] = None,
    ):
        """
        Initialize code downloader.

        Args:
            output_dir: Directory to store cloned repositories
            max_repo_size_mb: Skip repos larger than this (to avoid huge repos)
            file_extensions: Code file extensions to keep (None = keep all)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_repo_size_mb = max_repo_size_mb

        # Default to common code extensions
        self.file_extensions = file_extensions or [
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
            '.m', '.mm', '.sh', '.r', '.jl', '.lua', '.dart'
        ]

    def clone_repository(
        self,
        repo_url: str,
        repo_name: str,
        shallow: bool = True,
    ) -> Optional[Path]:
        """
        Clone a repository from GitHub.

        Args:
            repo_url: GitHub repository URL
            repo_name: Repository name (owner/repo)
            shallow: Use shallow clone (faster, less storage)

        Returns:
            Path to cloned repo or None if failed
        """
        # Clean repo name for directory
        clean_name = repo_name.replace('/', '_')
        repo_path = self.output_dir / clean_name

        # Skip if already exists
        if repo_path.exists():
            logger.info(f"Repository {repo_name} already cloned, skipping")
            return repo_path

        try:
            logger.info(f"Cloning {repo_name}...")

            # Build git clone command
            cmd = ['git', 'clone']
            if shallow:
                cmd.extend(['--depth', '1'])  # Shallow clone for speed
            cmd.extend([repo_url, str(repo_path)])

            # Clone repository
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"Failed to clone {repo_name}: {result.stderr}")
                return None

            # Check repository size
            size_mb = self._get_dir_size_mb(repo_path)
            if size_mb > self.max_repo_size_mb:
                logger.warning(f"Repository {repo_name} is {size_mb:.1f}MB, "
                             f"exceeds limit of {self.max_repo_size_mb}MB. Removing.")
                shutil.rmtree(repo_path)
                return None

            # Remove .git directory to avoid nested git repos
            git_dir = repo_path / '.git'
            if git_dir.exists():
                shutil.rmtree(git_dir)
                logger.debug(f"Removed .git directory from {repo_name}")

            logger.info(f"Successfully cloned {repo_name} ({size_mb:.1f}MB)")
            return repo_path

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout cloning {repo_name}")
            if repo_path.exists():
                shutil.rmtree(repo_path)
            return None
        except Exception as e:
            logger.error(f"Error cloning {repo_name}: {e}")
            if repo_path.exists():
                shutil.rmtree(repo_path)
            return None

    def _get_dir_size_mb(self, path: Path) -> float:
        """Get directory size in MB."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)

    def extract_code_files(self, repo_path: Path) -> List[Dict]:
        """
        Extract code files from repository.

        Args:
            repo_path: Path to cloned repository

        Returns:
            List of dicts with file info {path, language, content, lines}
        """
        code_files = []

        for file_path in repo_path.rglob('*'):
            # Skip directories and hidden files
            if file_path.is_dir() or file_path.name.startswith('.'):
                continue

            # Skip if not a code file
            if file_path.suffix not in self.file_extensions:
                continue

            # Skip large files (>1MB)
            if file_path.stat().st_size > 1024 * 1024:
                continue

            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Get relative path from repo root
                rel_path = file_path.relative_to(repo_path)

                code_files.append({
                    'path': str(rel_path),
                    'extension': file_path.suffix,
                    'language': self._extension_to_language(file_path.suffix),
                    'content': content,
                    'lines': len(content.splitlines()),
                    'size_bytes': file_path.stat().st_size,
                })

            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue

        return code_files

    def _extension_to_language(self, ext: str) -> str:
        """Map file extension to programming language."""
        mapping = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C',
            '.hpp': 'C++',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.m': 'Objective-C',
            '.mm': 'Objective-C++',
            '.sh': 'Shell',
            '.r': 'R',
            '.jl': 'Julia',
            '.lua': 'Lua',
            '.dart': 'Dart',
        }
        return mapping.get(ext.lower(), 'Unknown')

    def process_paper_code_pairs(
        self,
        input_file: str = "data/raw/papers/paper_code_pairs.json",
        output_file: str = "data/raw/papers/paper_code_with_files.json",
        max_repos: Optional[int] = None,
    ):
        """
        Download code for all papers and create enhanced dataset.

        Args:
            input_file: Path to paper_code_pairs.json
            output_file: Path to save enhanced data with code files
            max_repos: Maximum number of repos to download (None = all)
        """
        logger.info(f"Loading paper-code pairs from {input_file}")
        with open(input_file, 'r') as f:
            papers = json.load(f)

        if max_repos:
            papers = papers[:max_repos]

        enhanced_papers = []
        stats = {
            'total_papers': len(papers),
            'repos_cloned': 0,
            'repos_failed': 0,
            'total_code_files': 0,
            'total_lines': 0,
        }

        logger.info(f"Processing {len(papers)} papers...")

        for paper_data in tqdm(papers, desc="Downloading repositories"):
            paper_info = paper_data['paper']
            repositories = paper_data.get('repositories', [])

            enhanced_repos = []

            for repo in repositories:
                repo_url = repo['url']
                repo_name = repo['name']

                # Clone repository
                repo_path = self.clone_repository(repo_url, repo_name)

                if repo_path is None:
                    stats['repos_failed'] += 1
                    enhanced_repos.append({
                        **repo,
                        'cloned': False,
                        'code_files': [],
                    })
                    continue

                # Extract code files
                code_files = self.extract_code_files(repo_path)
                stats['repos_cloned'] += 1
                stats['total_code_files'] += len(code_files)
                stats['total_lines'] += sum(f['lines'] for f in code_files)

                enhanced_repos.append({
                    **repo,
                    'cloned': True,
                    'clone_path': str(repo_path),
                    'code_files': code_files,
                    'num_files': len(code_files),
                    'total_lines': sum(f['lines'] for f in code_files),
                })

                logger.info(f"  {repo_name}: {len(code_files)} code files, "
                          f"{sum(f['lines'] for f in code_files)} lines")

            enhanced_papers.append({
                'paper': paper_info,
                'repositories': enhanced_repos,
            })

        # Save enhanced dataset
        logger.info(f"Saving enhanced dataset to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(enhanced_papers, f, indent=2)

        # Save statistics
        stats_file = output_file.replace('.json', '_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("Download Complete!")
        logger.info("=" * 60)
        logger.info(f"Total papers: {stats['total_papers']}")
        logger.info(f"Repositories cloned: {stats['repos_cloned']}")
        logger.info(f"Repositories failed: {stats['repos_failed']}")
        logger.info(f"Total code files: {stats['total_code_files']}")
        logger.info(f"Total lines of code: {stats['total_lines']:,}")
        logger.info(f"Enhanced dataset saved to: {output_file}")
        logger.info(f"Statistics saved to: {stats_file}")
        logger.info("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download code from GitHub repositories"
    )
    parser.add_argument(
        '--input',
        default='data/raw/papers/paper_code_pairs.json',
        help='Input paper-code pairs file',
    )
    parser.add_argument(
        '--output',
        default='data/raw/papers/paper_code_with_files.json',
        help='Output file with code files',
    )
    parser.add_argument(
        '--max-repos',
        type=int,
        help='Maximum number of repositories to download',
    )
    parser.add_argument(
        '--max-size',
        type=int,
        default=500,
        help='Maximum repository size in MB',
    )
    parser.add_argument(
        '--output-dir',
        default='data/raw/code_repos',
        help='Directory to store cloned repositories',
    )

    args = parser.parse_args()

    downloader = CodeDownloader(
        output_dir=args.output_dir,
        max_repo_size_mb=args.max_size,
    )

    downloader.process_paper_code_pairs(
        input_file=args.input,
        output_file=args.output,
        max_repos=args.max_repos,
    )


if __name__ == '__main__':
    main()
