"""Tests for Reducto.ai quality checker.

TDD: Tests written first, implementation follows.
"""

import pytest
from dataclasses import asdict


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass."""

    def test_quality_metrics_creation(self):
        """Test creating QualityMetrics with all fields."""
        from src.pdf_converter.quality_checker import QualityMetrics

        metrics = QualityMetrics(
            word_count=4500,
            line_count=300,
            estimated_pages=15,
            heading_count=12,
            table_count=3,
            code_block_count=2,
            list_count=8,
            special_char_ratio=0.02,
            empty_line_ratio=0.05,
            words_per_page=300.0,
            ocr_confidence_avg=0.92,
            low_confidence_word_count=5,
            low_confidence_word_ratio=0.001,
            broken_table_count=0,
            chunk_count=24,
            empty_chunk_count=0,
            avg_chunk_size=188.0,
        )

        assert metrics.word_count == 4500
        assert metrics.estimated_pages == 15
        assert metrics.ocr_confidence_avg == 0.92

    def test_quality_metrics_defaults(self):
        """Test QualityMetrics with default optional values."""
        from src.pdf_converter.quality_checker import QualityMetrics

        metrics = QualityMetrics(
            word_count=1000,
            line_count=50,
            estimated_pages=3,
            heading_count=5,
            table_count=0,
            code_block_count=0,
            list_count=2,
            special_char_ratio=0.01,
            empty_line_ratio=0.03,
            words_per_page=333.0,
        )

        # Optional OCR fields should default to None
        assert metrics.ocr_confidence_avg is None
        assert metrics.low_confidence_word_count is None
        assert metrics.chunk_count is None

    def test_quality_metrics_to_dict(self):
        """Test converting QualityMetrics to dictionary."""
        from src.pdf_converter.quality_checker import QualityMetrics

        metrics = QualityMetrics(
            word_count=1000,
            line_count=50,
            estimated_pages=3,
            heading_count=5,
            table_count=1,
            code_block_count=0,
            list_count=2,
            special_char_ratio=0.01,
            empty_line_ratio=0.03,
            words_per_page=333.0,
        )

        data = asdict(metrics)
        assert data["word_count"] == 1000
        assert data["table_count"] == 1


class TestQualityIssue:
    """Tests for QualityIssue dataclass."""

    def test_quality_issue_creation(self):
        """Test creating a quality issue."""
        from src.pdf_converter.quality_checker import QualityIssue, IssueSeverity

        issue = QualityIssue(
            code="LOW_OCR_CONFIDENCE",
            message="OCR confidence below threshold",
            severity=IssueSeverity.ERROR,
            metric_name="ocr_confidence_avg",
            actual_value=0.55,
            threshold_value=0.6,
        )

        assert issue.code == "LOW_OCR_CONFIDENCE"
        assert issue.severity == IssueSeverity.ERROR
        assert issue.actual_value == 0.55

    def test_quality_issue_warning(self):
        """Test creating a warning-level issue."""
        from src.pdf_converter.quality_checker import QualityIssue, IssueSeverity

        issue = QualityIssue(
            code="HIGH_SPECIAL_CHARS",
            message="Special character ratio is elevated",
            severity=IssueSeverity.WARNING,
            metric_name="special_char_ratio",
            actual_value=0.12,
            threshold_value=0.10,
        )

        assert issue.severity == IssueSeverity.WARNING


class TestQualityReport:
    """Tests for QualityReport dataclass."""

    def test_quality_report_pass(self):
        """Test creating a passing quality report."""
        from src.pdf_converter.quality_checker import (
            QualityMetrics,
            QualityReport,
            QualityStatus,
        )

        metrics = QualityMetrics(
            word_count=4500,
            line_count=300,
            estimated_pages=15,
            heading_count=12,
            table_count=3,
            code_block_count=2,
            list_count=8,
            special_char_ratio=0.02,
            empty_line_ratio=0.05,
            words_per_page=300.0,
        )

        report = QualityReport(
            status=QualityStatus.PASS,
            score=94,
            metrics=metrics,
            issues=[],
            recommendation="Safe to ingest",
        )

        assert report.status == QualityStatus.PASS
        assert report.score == 94
        assert report.passed is True
        assert len(report.issues) == 0

    def test_quality_report_fail(self):
        """Test creating a failing quality report."""
        from src.pdf_converter.quality_checker import (
            QualityMetrics,
            QualityReport,
            QualityStatus,
            QualityIssue,
            IssueSeverity,
        )

        metrics = QualityMetrics(
            word_count=500,
            line_count=40,
            estimated_pages=15,
            heading_count=2,
            table_count=0,
            code_block_count=0,
            list_count=0,
            special_char_ratio=0.25,
            empty_line_ratio=0.15,
            words_per_page=33.0,
        )

        issues = [
            QualityIssue(
                code="LOW_WORD_COUNT",
                message="Word count per page is too low",
                severity=IssueSeverity.ERROR,
                metric_name="words_per_page",
                actual_value=33.0,
                threshold_value=50.0,
            ),
            QualityIssue(
                code="HIGH_SPECIAL_CHARS",
                message="Special character ratio exceeds limit",
                severity=IssueSeverity.ERROR,
                metric_name="special_char_ratio",
                actual_value=0.25,
                threshold_value=0.20,
            ),
        ]

        report = QualityReport(
            status=QualityStatus.FAIL,
            score=35,
            metrics=metrics,
            issues=issues,
            recommendation="Do not ingest - quality issues detected",
        )

        assert report.status == QualityStatus.FAIL
        assert report.score == 35
        assert report.passed is False
        assert len(report.issues) == 2
        assert report.has_errors is True

    def test_quality_report_warning_only(self):
        """Test report with warnings but no errors passes."""
        from src.pdf_converter.quality_checker import (
            QualityMetrics,
            QualityReport,
            QualityStatus,
            QualityIssue,
            IssueSeverity,
        )

        metrics = QualityMetrics(
            word_count=3000,
            line_count=200,
            estimated_pages=10,
            heading_count=8,
            table_count=2,
            code_block_count=1,
            list_count=5,
            special_char_ratio=0.08,
            empty_line_ratio=0.06,
            words_per_page=300.0,
        )

        issues = [
            QualityIssue(
                code="ELEVATED_SPECIAL_CHARS",
                message="Special character ratio is slightly elevated",
                severity=IssueSeverity.WARNING,
                metric_name="special_char_ratio",
                actual_value=0.08,
                threshold_value=0.05,
            ),
        ]

        report = QualityReport(
            status=QualityStatus.PASS,
            score=82,
            metrics=metrics,
            issues=issues,
            recommendation="Safe to ingest with minor warnings",
        )

        assert report.status == QualityStatus.PASS
        assert report.passed is True
        assert report.has_errors is False
        assert report.has_warnings is True


class TestQualityThresholds:
    """Tests for QualityThresholds configuration."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        from src.pdf_converter.quality_checker import QualityThresholds

        thresholds = QualityThresholds()

        assert thresholds.ocr_confidence_warning == 0.8
        assert thresholds.ocr_confidence_error == 0.6
        assert thresholds.words_per_page_warning == 100
        assert thresholds.words_per_page_error == 50
        assert thresholds.special_char_ratio_warning == 0.10
        assert thresholds.special_char_ratio_error == 0.20

    def test_strict_thresholds(self):
        """Test strict threshold preset."""
        from src.pdf_converter.quality_checker import QualityThresholds

        thresholds = QualityThresholds.strict()

        assert thresholds.ocr_confidence_warning == 0.9
        assert thresholds.ocr_confidence_error == 0.75
        assert thresholds.words_per_page_warning == 150
        assert thresholds.words_per_page_error == 100
        assert thresholds.empty_line_ratio_warning == 0.45
        assert thresholds.empty_line_ratio_error == 0.60

    def test_lenient_thresholds(self):
        """Test lenient threshold preset."""
        from src.pdf_converter.quality_checker import QualityThresholds

        thresholds = QualityThresholds.lenient()

        assert thresholds.ocr_confidence_warning == 0.7
        assert thresholds.ocr_confidence_error == 0.5
        assert thresholds.words_per_page_warning == 50
        assert thresholds.words_per_page_error == 25
        assert thresholds.empty_line_ratio_warning == 0.65
        assert thresholds.empty_line_ratio_error == 0.80


class TestReductoQualityChecker:
    """Tests for ReductoQualityChecker class."""

    def test_check_markdown_content_good(self):
        """Test checking good quality markdown content."""
        from src.pdf_converter.quality_checker import ReductoQualityChecker

        checker = ReductoQualityChecker()

        markdown = """# Chapter 1: Introduction

This is a well-structured document with proper content.
It contains multiple paragraphs with meaningful text.

## Section 1.1: Overview

The overview provides context for the rest of the document.
We have tables, code blocks, and lists.

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

### Code Example

```python
def hello_world():
    print("Hello, World!")
```

- Item 1
- Item 2
- Item 3

## Section 1.2: Details

More detailed content follows here with additional paragraphs.
This ensures we have enough words for a proper quality check.
The document continues with more meaningful content.
""" * 10  # Repeat to get enough content

        report = checker.check_markdown(markdown, expected_pages=5)

        assert report.passed is True
        assert report.score >= 70
        assert report.metrics.heading_count >= 3
        assert report.metrics.table_count >= 1
        assert report.metrics.code_block_count >= 1
        assert report.metrics.list_count >= 1

    def test_check_markdown_content_poor_ocr(self):
        """Test detecting poor OCR quality (high special chars)."""
        from src.pdf_converter.quality_checker import ReductoQualityChecker

        checker = ReductoQualityChecker()

        # Simulate OCR garbage with lots of special characters
        markdown = """# Ch@pt3r 1: |ntr0duct!0n

Th!$ !$ @ p00rly $c@nn3d d0cum3nt w!th l0t$ 0f n0!$3.
M@ny ch@r@ct3r$ @r3 m!$r3c0gn!z3d by th3 0CR.
|t c0nt@!n$ m@ny $p3c!@l ch@r@ct3r$ th@t $h0uldn't b3 th3r3.
"""

        report = checker.check_markdown(markdown, expected_pages=1)

        assert report.passed is False
        assert any(issue.code == "HIGH_SPECIAL_CHARS" for issue in report.issues)

    def test_check_markdown_content_low_words(self):
        """Test detecting low word count per page."""
        from src.pdf_converter.quality_checker import ReductoQualityChecker

        checker = ReductoQualityChecker()

        # Very sparse content for 10 pages
        markdown = """# Title

Short content.
"""

        report = checker.check_markdown(markdown, expected_pages=10)

        assert report.passed is False
        assert any(issue.code == "LOW_WORD_COUNT" for issue in report.issues)

    def test_check_markdown_broken_tables(self):
        """Test detecting broken table structures."""
        from src.pdf_converter.quality_checker import ReductoQualityChecker

        checker = ReductoQualityChecker()

        # Broken table with mismatched columns
        markdown = """# Document with Tables

| Col 1 | Col 2 | Col 3 |
|-------|-------|
| Val 1 | Val 2 | Val 3 | Val 4 |
| Val 5 |

Some more content here to pad the document.
""" * 20

        report = checker.check_markdown(markdown, expected_pages=2)

        # Should detect broken tables
        assert report.metrics.broken_table_count >= 1

    def test_check_with_ocr_data(self):
        """Test checking with OCR confidence data."""
        from src.pdf_converter.quality_checker import ReductoQualityChecker

        checker = ReductoQualityChecker()

        markdown = "# Sample Document\n\nThis is sample content.\n" * 50

        ocr_data = {
            "words": [
                {"text": "Sample", "confidence": 0.95},
                {"text": "Document", "confidence": 0.92},
                {"text": "This", "confidence": 0.88},
                {"text": "is", "confidence": 0.45},  # Low confidence
                {"text": "content", "confidence": 0.91},
            ]
        }

        report = checker.check_markdown(markdown, expected_pages=3, ocr_data=ocr_data)

        assert report.metrics.ocr_confidence_avg is not None
        assert report.metrics.low_confidence_word_count == 1

    def test_check_with_low_ocr_confidence(self):
        """Test failing quality check due to low OCR confidence."""
        from src.pdf_converter.quality_checker import ReductoQualityChecker

        checker = ReductoQualityChecker()

        markdown = "# Sample Document\n\nThis is sample content.\n" * 50

        # Mostly low confidence words
        ocr_data = {
            "words": [
                {"text": "word1", "confidence": 0.45},
                {"text": "word2", "confidence": 0.52},
                {"text": "word3", "confidence": 0.48},
                {"text": "word4", "confidence": 0.55},
                {"text": "word5", "confidence": 0.50},
            ]
        }

        report = checker.check_markdown(markdown, expected_pages=3, ocr_data=ocr_data)

        assert report.passed is False
        assert any(issue.code == "LOW_OCR_CONFIDENCE" for issue in report.issues)

    def test_check_chunks_quality(self):
        """Test checking chunk quality for RAG pipeline."""
        from src.pdf_converter.quality_checker import ReductoQualityChecker

        checker = ReductoQualityChecker()

        chunks = [
            {"content": "This is a good chunk with meaningful content. " * 20, "embed": "..."},
            {"content": "Another chunk with proper length. " * 25, "embed": "..."},
            {"content": "", "embed": ""},  # Empty chunk
            {"content": "Short", "embed": "Short"},  # Too short
            {"content": "Normal chunk content here. " * 15, "embed": "..."},
        ]

        report = checker.check_chunks(chunks)

        assert report.metrics.chunk_count == 5
        assert report.metrics.empty_chunk_count == 1
        # Should have warning about empty/short chunks

    def test_check_file_path(self):
        """Test checking a markdown file by path."""
        from src.pdf_converter.quality_checker import ReductoQualityChecker
        from pathlib import Path
        import tempfile

        checker = ReductoQualityChecker()

        # Create a temporary markdown file
        content = """# Test Document

This is a test document with enough content to pass basic checks.
We need multiple paragraphs and various elements.

## Section 1

Content for section 1 goes here with multiple sentences.
The quality checker needs enough words to work with.

## Section 2

More content in section 2 to ensure we have a proper document.
Tables and lists would also be checked.

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |

- List item 1
- List item 2
- List item 3
""" * 5

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            report = checker.check_file(temp_path)

            assert report is not None
            assert report.metrics.word_count > 0
            assert report.metrics.heading_count > 0
        finally:
            temp_path.unlink()

    def test_calculate_score(self):
        """Test score calculation from metrics."""
        from src.pdf_converter.quality_checker import ReductoQualityChecker, QualityMetrics

        checker = ReductoQualityChecker()

        # Good metrics should give high score
        good_metrics = QualityMetrics(
            word_count=30000, line_count=3000, estimated_pages=100,
            heading_count=50, table_count=5, code_block_count=0, list_count=10,
            special_char_ratio=0.02, empty_line_ratio=0.05, words_per_page=300,
            words_per_line=10.0, ocr_confidence_avg=0.95, broken_table_count=0,
        )
        good_score = checker._calculate_score(metrics=good_metrics, empty_chunk_ratio=0.0)
        assert good_score >= 85

        # Poor metrics should give low score
        poor_metrics = QualityMetrics(
            word_count=3000, line_count=300, estimated_pages=100,
            heading_count=0, table_count=0, code_block_count=0, list_count=0,
            special_char_ratio=0.25, empty_line_ratio=0.20, words_per_page=30,
            words_per_line=10.0, ocr_confidence_avg=0.45, broken_table_count=5,
        )
        poor_score = checker._calculate_score(metrics=poor_metrics, empty_chunk_ratio=0.30)
        assert poor_score <= 40


class TestQualityCheckerIntegration:
    """Integration tests for quality checker with real-like data."""

    def test_full_quality_check_workflow(self):
        """Test complete quality check workflow."""
        from src.pdf_converter.quality_checker import (
            ReductoQualityChecker,
            QualityStatus,
        )

        # Use default thresholds (not strict)
        checker = ReductoQualityChecker()

        # Good quality document with dense content
        good_markdown = """# Comprehensive Guide
## Chapter 1: Introduction
This comprehensive guide provides detailed information about the subject matter. Each section is carefully structured to facilitate learning and understanding. The content is dense and meaningful.
### 1.1 Background
The background section establishes context for the topics that follow. Understanding this foundation is essential for comprehending later material. We cover the history and evolution of the concepts.
### 1.2 Objectives
By the end of this guide, readers will be able to understand core concepts, apply practical techniques, and evaluate complex scenarios effectively.
## Chapter 2: Core Concepts
### 2.1 Fundamental Principles
The fundamental principles form the basis of all advanced topics. These principles are essential knowledge.
| Principle | Description | Application |
|-----------|-------------|-------------|
| First     | Basic rule  | Common use  |
| Second    | Key insight | Specialized |
| Third     | Advanced    | Expert level|
### 2.2 Implementation
Here is a code example demonstrating the implementation:
```python
def implement_concept(data):
    \"\"\"Implement the core concept.\"\"\"
    result = process(data)
    return result
```
## Chapter 3: Advanced Topics
Advanced material builds upon the foundations established earlier. This section covers complex scenarios and edge cases that professionals encounter in real-world applications.
""" * 10  # Repeat more times for adequate word count

        report = checker.check_markdown(good_markdown, expected_pages=5)

        assert report.status == QualityStatus.PASS
        assert report.score >= 70
        assert report.metrics.heading_count >= 6
        assert report.metrics.table_count >= 1
        assert report.metrics.code_block_count >= 1

    def test_quality_report_json_serialization(self):
        """Test that quality report can be serialized to JSON."""
        from src.pdf_converter.quality_checker import ReductoQualityChecker
        import json

        checker = ReductoQualityChecker()
        markdown = "# Test\n\nContent here.\n" * 100

        report = checker.check_markdown(markdown, expected_pages=2)

        # Should be JSON serializable
        json_str = report.to_json()
        assert json_str is not None

        # Should be parseable
        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "score" in parsed
        assert "metrics" in parsed
