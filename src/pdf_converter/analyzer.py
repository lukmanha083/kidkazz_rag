"""Markdown quality analysis functions."""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ContentStats:
    """Statistics about markdown content."""
    characters: int
    words: int
    lines: int
    headings: int
    tables: int
    code_blocks: int
    images: int
    math_blocks: int
    links: int

    @property
    def estimated_pages(self) -> int:
        """Estimate original page count (~300 words per page)."""
        return max(1, self.words // 300)


@dataclass
class QualityReport:
    """Quality analysis report."""
    filename: str
    stats: ContentStats
    issues: list[str]

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0


def get_content_stats(content: str) -> ContentStats:
    """
    Analyze markdown content and return statistics.

    Args:
        content: Markdown content string

    Returns:
        ContentStats dataclass with analysis results
    """
    return ContentStats(
        characters=len(content),
        words=len(content.split()),
        lines=content.count("\n") + 1,
        headings=len(re.findall(r"^#{1,6}\s", content, re.MULTILINE)),
        tables=content.count("|---"),
        code_blocks=content.count("```") // 2,
        images=len(re.findall(r"!\[.*?\]\(.*?\)", content)),
        math_blocks=content.count("$$") // 2 + content.count("\\["),
        links=len(re.findall(r"\[.*?\]\(.*?\)", content)) - len(re.findall(r"!\[.*?\]\(.*?\)", content)),
    )


def detect_quality_issues(content: str, stats: ContentStats) -> list[str]:
    """
    Detect potential quality issues in the converted markdown.

    Args:
        content: Markdown content string
        stats: Pre-computed content statistics

    Returns:
        List of issue descriptions
    """
    issues = []

    # Check for garbled text (high ratio of special characters)
    if len(content) > 0:
        special_chars = len(re.findall(r"[^\w\s]", content))
        special_ratio = special_chars / len(content)
        if special_ratio > 0.15:
            issues.append("High special character ratio (possible OCR issues)")

    # Check for very short content
    if stats.words < 1000:
        issues.append("Very short output (conversion may have failed)")

    # Check for broken tables (very low pipe count relative to table markers)
    # Note: stats.tables counts |--- patterns, not actual tables
    if stats.tables > 0:
        pipe_count = content.count("|")
        # Expect at least 3 pipes per table separator cell on average
        if pipe_count / stats.tables < 3:
            issues.append("Possible broken table formatting")

    # Check for excessive empty lines (possible parsing issues)
    empty_line_ratio = content.count("\n\n\n") / max(stats.lines, 1)
    if empty_line_ratio > 0.1:
        issues.append("Excessive empty lines (possible parsing issues)")

    return issues


def analyze_quality(content: str, filename: str = "unknown") -> QualityReport:
    """
    Perform full quality analysis on markdown content.

    Args:
        content: Markdown content string
        filename: Name of the source file

    Returns:
        QualityReport with stats and issues
    """
    stats = get_content_stats(content)
    issues = detect_quality_issues(content, stats)
    return QualityReport(filename=filename, stats=stats, issues=issues)


def preview_markdown(content: str, chars: int = 3000) -> str:
    """
    Get a preview of markdown content.

    Args:
        content: Full markdown content
        chars: Maximum characters to return

    Returns:
        Truncated content with indicator if truncated
    """
    if len(content) <= chars:
        return content

    return content[:chars] + f"\n\n... [truncated, showing {chars:,}/{len(content):,} chars]"


def find_latest_markdown(output_dir: str) -> Optional[Path]:
    """
    Find the most recently modified markdown file in a directory.

    Args:
        output_dir: Directory to search

    Returns:
        Path to most recent .md file, or None if not found
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    md_files = list(output_path.rglob("*.md"))
    if not md_files:
        return None

    return max(md_files, key=lambda p: p.stat().st_mtime)
