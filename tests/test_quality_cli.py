"""Tests for quality check CLI command.

TDD: Tests written first, implementation follows.
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import tempfile
import json


@pytest.fixture
def cli_runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_markdown_file():
    """Create a temporary markdown file for testing."""
    content = """# Test Document

This is a test document with enough content to perform quality checks.
The document contains multiple paragraphs and sections.

## Section 1: Introduction

This section introduces the main topics covered in this document.
We need sufficient content to pass basic quality thresholds.

### 1.1 Background

Background information provides context for the reader.
Understanding the background helps with comprehension.

### 1.2 Objectives

The objectives of this document include:

- Demonstrating quality checking
- Testing CLI commands
- Validating output formats

## Section 2: Main Content

The main content section contains the bulk of the information.

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

### 2.1 Details

Detailed information about the topic goes here.
Multiple sentences ensure adequate content length.

```python
def example_code():
    return "example"
```

## Section 3: Conclusion

The conclusion summarizes the key points.
""" * 3

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        path = Path(f.name)

    yield path

    # Cleanup
    if path.exists():
        path.unlink()


@pytest.fixture
def temp_markdown_dir():
    """Create a temporary directory with markdown files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create good quality file
        good_content = """# Good Document

This is a well-structured document with proper content.
Multiple paragraphs ensure adequate word count.

## Section 1

Content for section 1 with multiple sentences.
The quality should be acceptable for ingestion.

## Section 2

More content in section 2 to ensure we have enough words.
Tables and lists are properly formatted.

| Col 1 | Col 2 |
|-------|-------|
| A     | B     |
""" * 5

        (tmpdir / "good_doc.md").write_text(good_content)

        # Create poor quality file
        poor_content = """# B@d D0c

Sh0rt c0nt3nt w!th sp3c!@l ch@rs.
"""
        (tmpdir / "poor_doc.md").write_text(poor_content)

        yield tmpdir


class TestQualityCommand:
    """Tests for 'kidkazz inbox quality' command."""

    def test_quality_single_file(self, cli_runner, temp_markdown_file):
        """Test quality check on a single file."""
        from src.cli.main import app

        result = cli_runner.invoke(app, ["inbox", "quality", str(temp_markdown_file)])

        assert result.exit_code == 0
        assert "Quality Report" in result.output or "score" in result.output.lower()

    def test_quality_single_file_json(self, cli_runner, temp_markdown_file):
        """Test quality check with JSON output."""
        from src.cli.main import app

        result = cli_runner.invoke(
            app, ["inbox", "quality", str(temp_markdown_file), "--json"]
        )

        assert result.exit_code == 0

        # Output should be valid JSON
        output = result.output.strip()
        data = json.loads(output)

        assert "status" in data
        assert "score" in data
        assert "metrics" in data

    def test_quality_single_file_verbose(self, cli_runner, temp_markdown_file):
        """Test quality check with verbose output."""
        from src.cli.main import app

        result = cli_runner.invoke(
            app, ["inbox", "quality", str(temp_markdown_file), "--verbose"]
        )

        assert result.exit_code == 0
        # Verbose should show more details
        assert "word" in result.output.lower() or "metric" in result.output.lower()

    def test_quality_file_not_found(self, cli_runner):
        """Test quality check on non-existent file."""
        from src.cli.main import app

        result = cli_runner.invoke(app, ["inbox", "quality", "/nonexistent/file.md"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_quality_directory_all(self, cli_runner, temp_markdown_dir):
        """Test quality check on all files in output directory."""
        from src.cli.main import app

        result = cli_runner.invoke(
            app, ["inbox", "quality", "--all", "--dir", str(temp_markdown_dir)]
        )

        # Exit code 1 because one file (poor_doc) fails quality
        assert result.exit_code == 1
        # Should show results for multiple files
        assert "good_doc" in result.output or "poor_doc" in result.output

    def test_quality_directory_summary(self, cli_runner, temp_markdown_dir):
        """Test quality check with summary output."""
        from src.cli.main import app

        result = cli_runner.invoke(
            app,
            ["inbox", "quality", "--all", "--dir", str(temp_markdown_dir), "--summary"],
        )

        # Exit code 1 because one file (poor_doc) fails quality
        assert result.exit_code == 1
        assert "summary" in result.output.lower() or "file" in result.output.lower()

    def test_quality_pass_exit_code(self, cli_runner, temp_markdown_file):
        """Test that passing quality returns exit code 0."""
        from src.cli.main import app

        result = cli_runner.invoke(app, ["inbox", "quality", str(temp_markdown_file)])

        # Good file should pass
        assert result.exit_code == 0

    def test_quality_fail_exit_code(self, cli_runner):
        """Test that failing quality returns non-zero exit code."""
        from src.cli.main import app

        # Create a very poor quality file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("$h0rt @nd b@d")
            poor_file = Path(f.name)

        try:
            result = cli_runner.invoke(app, ["inbox", "quality", str(poor_file)])
            # Poor quality should fail (exit code 1)
            assert result.exit_code == 1
        finally:
            poor_file.unlink()


class TestQualityInParseCommand:
    """Tests for quality check integration in parse command."""

    def test_parse_with_quality_check_default(self, cli_runner):
        """Test that parse command includes quality check by default."""
        from src.cli.main import app
        from src.pdf_converter.reducto_client import ParseResult

        with patch("src.pdf_converter.reducto_client.ReductoClient") as mock_client:
            # Mock successful parse - returns markdown string directly
            mock_instance = MagicMock()
            good_content = "# Test Document\n\nThis is a test document with enough content to pass quality checks.\n" * 50
            mock_instance.parse_pdf.return_value = good_content
            # Also mock parse_pdf_with_images for when return_images is configured
            mock_instance.parse_pdf_with_images.return_value = ParseResult(
                source_path=Path("test.pdf"),
                markdown=good_content,
                success=True,
            )
            mock_client.return_value = mock_instance

            with tempfile.TemporaryDirectory() as tmpdir:
                inbox_dir = Path(tmpdir) / "inbox"
                inbox_dir.mkdir()
                (inbox_dir / "test.pdf").write_bytes(b"fake pdf")

                output_dir = Path(tmpdir) / "output"
                output_dir.mkdir()

                result = cli_runner.invoke(
                    app,
                    [
                        "inbox",
                        "parse",
                        "--inbox-path",
                        str(inbox_dir),
                        "--output-path",
                        str(output_dir),
                    ],
                )

                # Should include quality score in output
                assert "quality" in result.output.lower() or result.exit_code == 0

    def test_parse_no_quality_check(self, cli_runner):
        """Test parse command with quality check disabled."""
        from src.cli.main import app
        from src.pdf_converter.reducto_client import ParseResult

        with patch("src.pdf_converter.reducto_client.ReductoClient") as mock_client:
            # Mock returns short/poor content that would fail quality check
            mock_instance = MagicMock()
            mock_instance.parse_pdf.return_value = "Short content"
            # Also mock parse_pdf_with_images for when return_images is configured
            mock_instance.parse_pdf_with_images.return_value = ParseResult(
                source_path=Path("test.pdf"),
                markdown="Short content",
                success=True,
            )
            mock_client.return_value = mock_instance

            with tempfile.TemporaryDirectory() as tmpdir:
                inbox_dir = Path(tmpdir) / "inbox"
                inbox_dir.mkdir()
                (inbox_dir / "test.pdf").write_bytes(b"fake pdf")

                output_dir = Path(tmpdir) / "output"
                output_dir.mkdir()

                result = cli_runner.invoke(
                    app,
                    [
                        "inbox",
                        "parse",
                        "--no-quality-check",
                        "--inbox-path",
                        str(inbox_dir),
                        "--output-path",
                        str(output_dir),
                    ],
                )

                # Should save even with poor quality since quality check is disabled
                assert result.exit_code == 0

    def test_parse_quality_threshold_strict(self, cli_runner):
        """Test parse command with strict quality threshold."""
        from src.cli.main import app

        # Just verify the option is accepted
        result = cli_runner.invoke(
            app, ["inbox", "parse", "--quality-threshold", "strict", "--help"]
        )

        # Help should work, meaning the option exists
        assert result.exit_code == 0


class TestQualityOutputFormat:
    """Tests for quality report output formatting."""

    def test_human_readable_output(self, cli_runner, temp_markdown_file):
        """Test human-readable output format."""
        from src.cli.main import app

        result = cli_runner.invoke(app, ["inbox", "quality", str(temp_markdown_file)])

        assert result.exit_code == 0
        # Should have formatted output with symbols
        output = result.output
        assert any(char in output for char in ["✓", "✗", "⚠", "PASS", "FAIL"])

    def test_json_output_structure(self, cli_runner, temp_markdown_file):
        """Test JSON output has correct structure."""
        from src.cli.main import app

        result = cli_runner.invoke(
            app, ["inbox", "quality", str(temp_markdown_file), "--json"]
        )

        assert result.exit_code == 0

        data = json.loads(result.output.strip())

        # Required fields
        assert "status" in data
        assert data["status"] in ["PASS", "FAIL", "WARNING"]

        assert "score" in data
        assert isinstance(data["score"], (int, float))
        assert 0 <= data["score"] <= 100

        assert "metrics" in data
        assert isinstance(data["metrics"], dict)

        assert "issues" in data
        assert isinstance(data["issues"], list)

        assert "recommendation" in data
        assert isinstance(data["recommendation"], str)

    def test_metrics_in_output(self, cli_runner, temp_markdown_file):
        """Test that key metrics are shown in output."""
        from src.cli.main import app

        result = cli_runner.invoke(
            app, ["inbox", "quality", str(temp_markdown_file), "--verbose"]
        )

        assert result.exit_code == 0

        output = result.output.lower()
        # Should show key metrics
        assert any(
            metric in output
            for metric in ["word", "page", "heading", "table", "score"]
        )
