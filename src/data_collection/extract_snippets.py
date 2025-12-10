"""
Extract code snippets (functions and methods) from Python files.
This script parses Python files using AST and extracts individual function definitions
and class methods, saving each as a separate JSON object suitable for CodeBERT embedding.
"""

import os
import json
import ast
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FunctionExtractor:
    """Extract functions and methods from Python files using AST."""

    def __init__(
        self,
        min_lines: int = 50,
        require_docstring: bool = True,
    ):
        """
        Initialize function extractor.

        Args:
            min_lines: Minimum number of lines for a function to be extracted
            require_docstring: Whether to require docstrings for extraction
        """
        self.min_lines = min_lines
        self.require_docstring = require_docstring

    def extract_functions_from_code(
        self, code_content: str, file_path: str
    ) -> List[Dict]:
        """
        Extract functions and methods from Python code using AST.

        Args:
            code_content: Python source code as string
            file_path: Path to the source file (for metadata)

        Returns:
            List of dictionaries with function metadata and code
        """
        functions = []

        try:
            tree = ast.parse(code_content, filename=file_path)
        except SyntaxError as e:
            logger.debug(f"Syntax error in {file_path}: {e}")
            return functions
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {e}")
            return functions

        # Get all lines for extracting code snippets
        lines = code_content.splitlines()

        # Traverse AST to extract functions and methods
        # We need to track class context for methods
        class Visitor(ast.NodeVisitor):
            def __init__(self, extractor, lines, code_content, file_path):
                self.extractor = extractor
                self.lines = lines
                self.code_content = code_content
                self.file_path = file_path
                self.functions = []
                self.current_class = None

            def visit_FunctionDef(self, node):
                # Determine if this is a method or top-level function
                is_method = self.current_class is not None
                class_name = self.current_class if is_method else None

                func_info = self.extractor._extract_function_info(
                    node, self.lines, self.code_content, self.file_path,
                    is_method=is_method, class_name=class_name
                )
                if func_info:
                    self.functions.append(func_info)

                # Continue visiting child nodes
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                # Track current class context
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class

        visitor = Visitor(self, lines, code_content, file_path)
        visitor.visit(tree)

        return visitor.functions

    def _extract_function_info(
        self,
        node: ast.FunctionDef,
        lines: List[str],
        code_content: str,
        file_path: str,
        is_method: bool = False,
        class_name: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Extract information about a function or method.

        Args:
            node: AST FunctionDef node
            lines: List of source code lines
            code_content: Full source code
            file_path: Path to source file
            is_method: Whether this is a class method
            class_name: Name of the parent class (if method)

        Returns:
            Dictionary with function info or None if doesn't meet criteria
        """
        # Get line numbers (AST uses 1-based indexing)
        start_line = node.lineno
        
        # Try to get end_lineno (available in Python 3.8+)
        if hasattr(node, "end_lineno") and node.end_lineno is not None:
            end_line = node.end_lineno
        else:
            # Fallback: find the last line by traversing the node
            end_line = self._find_end_line(node, lines)

        # Calculate number of lines
        num_lines = end_line - start_line + 1

        # Check docstring
        docstring = ast.get_docstring(node)
        has_docstring = docstring is not None and len(docstring.strip()) > 0

        # Apply filtering criteria
        if self.require_docstring and not has_docstring:
            return None

        if num_lines < self.min_lines:
            return None

        # Extract function code
        # Adjust for 0-based indexing (lines list is 0-indexed)
        # end_line is inclusive (1-indexed), so we need end_idx = end_line for slicing
        start_idx = start_line - 1
        end_idx = end_line  # Python slicing is exclusive on end, so this gets lines start_line to end_line
        function_lines = lines[start_idx:end_idx]
        code_text = "\n".join(function_lines)

        # Build function name
        if is_method and class_name:
            function_name = f"{class_name}.{node.name}"
        else:
            function_name = node.name

        return {
            "function_name": function_name,
            "line_numbers": {"start": start_line, "end": end_line},
            "has_docstring": has_docstring,
            "num_lines": num_lines,
            "code_text": code_text,
        }

    def _find_end_line(self, node: ast.AST, lines: List[str]) -> int:
        """
        Find the end line of an AST node by finding the maximum line number
        in all its descendants.

        Args:
            node: AST node
            lines: List of source code lines

        Returns:
            End line number
        """
        max_line = node.lineno

        for child in ast.walk(node):
            if hasattr(child, "lineno") and child.lineno is not None:
                max_line = max(max_line, child.lineno)
                # Also check end_lineno if available
                if hasattr(child, "end_lineno") and child.end_lineno is not None:
                    max_line = max(max_line, child.end_lineno)

        return max_line



class CodeSnippetExtractor:
    """Extract code snippets from paper_code_with_files.json."""

    def __init__(
        self,
        min_lines: int = 50,
        require_docstring: bool = True,
    ):
        """
        Initialize snippet extractor.

        Args:
            min_lines: Minimum number of lines for extraction
            require_docstring: Whether to require docstrings
        """
        self.extractor = FunctionExtractor(
            min_lines=min_lines, require_docstring=require_docstring
        )

    def process_paper_code_with_files(
        self,
        input_file: str = "data/raw/papers/paper_code_with_files.json",
        output_file: str = "data/processed/code_snippets.json",
        max_papers: Optional[int] = None,
    ) -> List[Dict]:
        """
        Process paper_code_with_files.json and extract function snippets.

        Args:
            input_file: Path to paper_code_with_files.json
            output_file: Path to save extracted snippets
            max_papers: Maximum number of papers to process (None = all)

        Returns:
            List of snippet dictionaries
        """
        logger.info("=" * 60)
        logger.info("Extracting Code Snippets from Python Files")
        logger.info("=" * 60)
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Filter: functions with docstrings and >{self.extractor.min_lines} lines")
        logger.info("=" * 60)

        # Load input JSON
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        logger.info(f"Loading {input_file}...")
        with open(input_path, "r", encoding="utf-8") as f:
            papers_data = json.load(f)

        if max_papers:
            papers_data = papers_data[:max_papers]
            logger.info(f"Processing first {max_papers} papers")

        all_snippets = []
        stats = {
            "total_papers": len(papers_data),
            "papers_processed": 0,
            "repos_processed": 0,
            "python_files_processed": 0,
            "functions_extracted": 0,
            "functions_skipped": 0,
        }

        logger.info(f"\nProcessing {len(papers_data)} papers...")

        for paper_entry in tqdm(papers_data, desc="Processing papers"):
            paper = paper_entry.get("paper", {})
            repositories = paper_entry.get("repositories", [])

            if not paper:
                continue

            # Extract paper information
            paper_id = paper.get("arxiv_id", "")
            paper_title = paper.get("title", "")
            paper_url = paper.get("url", "")
            paper_abstract = paper.get("abstract", "")

            if not paper_id:
                continue

            stats["papers_processed"] += 1

            # Process each repository
            for repo in repositories:
                if not repo.get("cloned", False):
                    continue

                repo_name = repo.get("name", "")
                repo_url = repo.get("url", "")
                code_files = repo.get("code_files", [])

                stats["repos_processed"] += 1

                # Process Python files only
                for code_file in code_files:
                    # Only process Python files
                    if code_file.get("extension") != ".py":
                        continue

                    file_path = code_file.get("path", "")
                    code_content = code_file.get("content", "")

                    if not code_content.strip():
                        continue

                    stats["python_files_processed"] += 1

                    # Extract functions and methods from this file
                    functions = self.extractor.extract_functions_from_code(
                        code_content, file_path
                    )

                    # Create snippet JSON for each function
                    for func_info in functions:
                        snippet = {
                            # Paper information
                            "paper_id": paper_id,
                            "paper_title": paper_title,
                            "paper_url": paper_url,
                            "paper_abstract": paper_abstract,
                            # Repository information
                            "repo_name": repo_name,
                            "repo_url": repo_url,
                            # Code snippet details
                            "file_path": file_path,
                            "function_name": func_info["function_name"],
                            "code_text": func_info["code_text"],
                            # Additional metadata
                            "line_numbers": func_info["line_numbers"],
                            "has_docstring": func_info["has_docstring"],
                            "num_lines": func_info["num_lines"],
                        }

                        all_snippets.append(snippet)
                        stats["functions_extracted"] += 1

                    # Count total functions for statistics
                    try:
                        tree = ast.parse(code_content, filename=file_path)
                        total_functions = sum(
                            1 for node in ast.walk(tree)
                            if isinstance(node, ast.FunctionDef)
                        )
                        stats["functions_skipped"] += (total_functions - len(functions))
                    except Exception:
                        pass

        # Save snippets to JSON
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nSaving {len(all_snippets)} snippets to {output_file}...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_snippets, f, indent=2, ensure_ascii=False)

        # Print statistics
        logger.info("\n" + "=" * 60)
        logger.info("Extraction Complete!")
        logger.info("=" * 60)
        logger.info(f"Total papers: {stats['total_papers']}")
        logger.info(f"Papers processed: {stats['papers_processed']}")
        logger.info(f"Repositories processed: {stats['repos_processed']}")
        logger.info(f"Python files processed: {stats['python_files_processed']}")
        logger.info(f"Functions extracted: {stats['functions_extracted']}")
        logger.info(f"Functions skipped: {stats['functions_skipped']}")
        logger.info(f"Output saved to: {output_file}")
        logger.info("=" * 60)

        return all_snippets



def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract code snippets (functions/methods) from Python files"
    )
    parser.add_argument(
        "--input",
        default="data/raw/papers/paper_code_with_files.json",
        help="Input paper_code_with_files.json file",
    )
    parser.add_argument(
        "--output",
        default="data/processed/code_snippets.json",
        help="Output file for extracted snippets",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=50,
        help="Minimum number of lines for a function to be extracted (default: 50)",
    )
    parser.add_argument(
        "--require-docstring",
        action="store_true",
        default=True,
        help="Require docstrings for extraction (default: True)",
    )
    parser.add_argument(
        "--no-require-docstring",
        action="store_false",
        dest="require_docstring",
        help="Don't require docstrings",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        help="Maximum number of papers to process (None = all)",
    )

    args = parser.parse_args()

    extractor = CodeSnippetExtractor(
        min_lines=args.min_lines, require_docstring=args.require_docstring
    )

    snippets = extractor.process_paper_code_with_files(
        input_file=args.input,
        output_file=args.output,
        max_papers=args.max_papers,
    )

    print("\n" + "=" * 60)
    print("✅ Code Snippet Extraction Complete!")
    print("=" * 60)
    print(f"Extracted {len(snippets)} code snippets")
    print(f"Saved to: {args.output}")
    if snippets:
        print("\nSample snippet:")
        sample = snippets[0]
        print(f"  Paper: {sample['paper_title'][:50]}...")
        print(f"  Repo: {sample['repo_name']}")
        print(f"  File: {sample['file_path']}")
        print(f"  Function: {sample['function_name']}")
        print(f"  Lines: {sample['num_lines']}")
        print(f"  Has docstring: {sample['has_docstring']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
