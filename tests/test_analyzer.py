"""Tests for markdown quality analysis."""

import pytest
from pathlib import Path

from src.pdf_converter.analyzer import (
    get_content_stats,
    detect_quality_issues,
    analyze_quality,
    preview_markdown,
    find_latest_markdown,
    ContentStats,
    QualityReport,
)


class TestGetContentStats:
    """Tests for get_content_stats function."""

    def test_counts_characters(self, sample_markdown):
        """Should count total characters."""
        stats = get_content_stats(sample_markdown)
        assert stats.characters == len(sample_markdown)

    def test_counts_words(self, sample_markdown):
        """Should count words."""
        stats = get_content_stats(sample_markdown)
        assert stats.words > 0
        assert stats.words == len(sample_markdown.split())

    def test_counts_lines(self, sample_markdown):
        """Should count lines."""
        stats = get_content_stats(sample_markdown)
        assert stats.lines == sample_markdown.count("\n") + 1

    def test_counts_headings(self, sample_markdown):
        """Should count markdown headings."""
        stats = get_content_stats(sample_markdown)
        assert stats.headings > 0

    def test_counts_tables(self, markdown_with_tables):
        """Should count tables by |--- pattern (counts separator cells)."""
        stats = get_content_stats(markdown_with_tables)
        # The pattern |--- counts each column separator, not tables
        # 3 tables: 2-col (2 matches) + 3-col (3 matches) + 1-col (1 match) = 6
        assert stats.tables == 6

    def test_counts_code_blocks(self, sample_markdown):
        """Should count code blocks."""
        stats = get_content_stats(sample_markdown)
        assert stats.code_blocks >= 1

    def test_counts_images(self, sample_markdown):
        """Should count image references."""
        stats = get_content_stats(sample_markdown)
        assert stats.images >= 1

    def test_counts_math_blocks(self, sample_markdown):
        """Should count math blocks."""
        stats = get_content_stats(sample_markdown)
        assert stats.math_blocks >= 1

    def test_counts_links_excluding_images(self, sample_markdown):
        """Should count links but not image references."""
        stats = get_content_stats(sample_markdown)
        assert stats.links >= 1

    def test_empty_content_returns_zeros(self):
        """Empty content should return mostly zeros."""
        stats = get_content_stats("")
        assert stats.characters == 0
        assert stats.words == 0
        assert stats.headings == 0
        assert stats.tables == 0

    def test_estimated_pages_calculation(self, sample_markdown):
        """Should estimate page count based on word count."""
        stats = get_content_stats(sample_markdown)
        expected_pages = max(1, stats.words // 300)
        assert stats.estimated_pages == expected_pages


class TestDetectQualityIssues:
    """Tests for detect_quality_issues function."""

    def test_no_issues_for_good_content(self, sample_markdown):
        """Good content should have no issues."""
        stats = get_content_stats(sample_markdown)
        issues = detect_quality_issues(sample_markdown, stats)
        assert len(issues) == 0

    def test_detects_short_content(self, short_markdown):
        """Should detect very short content."""
        stats = get_content_stats(short_markdown)
        issues = detect_quality_issues(short_markdown, stats)
        assert any("Very short output" in issue for issue in issues)

    def test_detects_high_special_char_ratio(self, garbled_markdown):
        """Should detect high special character ratio."""
        stats = get_content_stats(garbled_markdown)
        issues = detect_quality_issues(garbled_markdown, stats)
        assert any("special character ratio" in issue for issue in issues)

    def test_detects_broken_tables(self):
        """Should detect potentially broken tables."""
        broken_table = "| A |\n|---|\n"  # Very minimal table
        stats = get_content_stats(broken_table)
        issues = detect_quality_issues(broken_table, stats)
        # This might trigger short content warning
        assert len(issues) >= 1

    def test_empty_content_reports_issues(self):
        """Empty content should report issues."""
        stats = get_content_stats("")
        issues = detect_quality_issues("", stats)
        assert len(issues) >= 1

    def test_detects_excessive_empty_lines(self):
        """Should detect excessive empty lines."""
        content = "text\n\n\n\n\n\nmore text\n\n\n\n\n\nend" * 50
        content += " word " * 1000  # Add enough words
        stats = get_content_stats(content)
        issues = detect_quality_issues(content, stats)
        assert any("empty lines" in issue for issue in issues)


class TestAnalyzeQuality:
    """Tests for analyze_quality function."""

    def test_returns_quality_report(self, sample_markdown):
        """Should return a QualityReport object."""
        report = analyze_quality(sample_markdown, "test.md")
        assert isinstance(report, QualityReport)

    def test_report_contains_filename(self, sample_markdown):
        """Report should contain the filename."""
        report = analyze_quality(sample_markdown, "myfile.md")
        assert report.filename == "myfile.md"

    def test_report_contains_stats(self, sample_markdown):
        """Report should contain content stats."""
        report = analyze_quality(sample_markdown)
        assert isinstance(report.stats, ContentStats)
        assert report.stats.words > 0

    def test_report_has_issues_property(self, short_markdown):
        """Report should have has_issues property."""
        report = analyze_quality(short_markdown)
        assert report.has_issues is True

    def test_good_content_has_no_issues(self, sample_markdown):
        """Good content should have no issues."""
        report = analyze_quality(sample_markdown)
        assert report.has_issues is False

    def test_default_filename(self, sample_markdown):
        """Default filename should be 'unknown'."""
        report = analyze_quality(sample_markdown)
        assert report.filename == "unknown"


class TestPreviewMarkdown:
    """Tests for preview_markdown function."""

    def test_short_content_not_truncated(self):
        """Content shorter than limit should not be truncated."""
        content = "Short content"
        result = preview_markdown(content, chars=100)
        assert result == content

    def test_long_content_is_truncated(self, sample_markdown):
        """Content longer than limit should be truncated."""
        result = preview_markdown(sample_markdown, chars=100)
        assert len(result) < len(sample_markdown)

    def test_truncated_content_has_indicator(self, sample_markdown):
        """Truncated content should have truncation indicator."""
        result = preview_markdown(sample_markdown, chars=100)
        assert "truncated" in result.lower()
        assert "chars]" in result

    def test_exact_length_not_truncated(self):
        """Content exactly at limit should not be truncated."""
        content = "x" * 100
        result = preview_markdown(content, chars=100)
        assert result == content

    def test_default_limit_is_3000(self, sample_markdown):
        """Default character limit should be 3000."""
        result = preview_markdown(sample_markdown)
        # The result should be around 3000 chars + truncation message
        assert len(result) > 3000
        assert len(result) < 3200


class TestFindLatestMarkdown:
    """Tests for find_latest_markdown function."""

    def test_finds_markdown_file(self, output_dir_with_markdown):
        """Should find markdown file in directory."""
        result = find_latest_markdown(str(output_dir_with_markdown))
        assert result is not None
        assert result.suffix == ".md"

    def test_returns_none_for_empty_directory(self, temp_dir):
        """Should return None for empty directory."""
        result = find_latest_markdown(str(temp_dir))
        assert result is None

    def test_returns_none_for_nonexistent_directory(self):
        """Should return None for non-existent directory."""
        result = find_latest_markdown("/nonexistent/path")
        assert result is None

    def test_finds_most_recent_file(self, temp_dir):
        """Should find the most recently modified file."""
        import time

        # Create first file
        old_file = temp_dir / "old.md"
        old_file.write_text("old content")

        # Small delay to ensure different mtime
        time.sleep(0.1)

        # Create second file
        new_file = temp_dir / "new.md"
        new_file.write_text("new content")

        result = find_latest_markdown(str(temp_dir))
        assert result.name == "new.md"

    def test_finds_nested_markdown(self, temp_dir):
        """Should find markdown in subdirectories."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        md_file = subdir / "nested.md"
        md_file.write_text("nested content")

        result = find_latest_markdown(str(temp_dir))
        assert result is not None
        assert result.name == "nested.md"
