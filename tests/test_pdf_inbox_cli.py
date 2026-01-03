"""Tests for PDF inbox CLI commands.

This module tests the CLI commands for managing the PDF inbox.
Tests are written BEFORE implementation following TDD principles.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def temp_inbox():
    """Create a temporary inbox directory."""
    inbox_dir = tempfile.mkdtemp(prefix="kidkazz_inbox_")
    yield Path(inbox_dir)
    shutil.rmtree(inbox_dir, ignore_errors=True)


@pytest.fixture
def temp_output():
    """Create a temporary output directory."""
    output_dir = tempfile.mkdtemp(prefix="kidkazz_output_")
    yield Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def mock_embedder_env():
    """Use mock embedder for all tests."""
    import os

    with patch.dict(os.environ, {"KIDKAZZ_EMBEDDER_TYPE": "mock"}):
        yield


def create_mock_pdf(directory: Path, name: str) -> Path:
    """Create a mock PDF file for testing."""
    pdf_path = directory / name
    pdf_path.write_bytes(b"PDF")
    return pdf_path


class TestInboxStatusCommand:
    """Tests for 'kidkazz inbox status' command."""

    def test_status_empty_inbox(self, temp_inbox, temp_output):
        """Test status command with empty inbox."""
        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
            },
        ):
            result = runner.invoke(app, ["inbox", "status"])

        assert result.exit_code == 0
        assert "0" in result.stdout or "empty" in result.stdout.lower()

    def test_status_with_pending_files(self, temp_inbox, temp_output):
        """Test status command with pending files."""
        from src.cli.main import app

        create_mock_pdf(temp_inbox, "test1.pdf")
        create_mock_pdf(temp_inbox, "test2.pdf")

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
            },
        ):
            result = runner.invoke(app, ["inbox", "status"])

        assert result.exit_code == 0
        assert "2" in result.stdout  # Should show 2 pending


class TestInboxListCommand:
    """Tests for 'kidkazz inbox list' command."""

    def test_list_empty_inbox(self, temp_inbox, temp_output):
        """Test list command with empty inbox."""
        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
            },
        ):
            result = runner.invoke(app, ["inbox", "list"])

        assert result.exit_code == 0
        assert "no" in result.stdout.lower() or "empty" in result.stdout.lower()

    def test_list_shows_pdf_files(self, temp_inbox, temp_output):
        """Test list command shows PDF files."""
        from src.cli.main import app

        create_mock_pdf(temp_inbox, "document.pdf")
        create_mock_pdf(temp_inbox, "report.pdf")

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
            },
        ):
            result = runner.invoke(app, ["inbox", "list"])

        assert result.exit_code == 0
        assert "document.pdf" in result.stdout
        assert "report.pdf" in result.stdout

    def test_list_json_output(self, temp_inbox, temp_output):
        """Test list command with JSON output."""
        from src.cli.main import app

        create_mock_pdf(temp_inbox, "test.pdf")

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
            },
        ):
            result = runner.invoke(app, ["inbox", "list", "--json"])

        assert result.exit_code == 0
        # Should be valid JSON
        import json

        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "test.pdf"


class TestInboxClearCommand:
    """Tests for 'kidkazz inbox clear' command."""

    def test_clear_requires_confirmation(self, temp_inbox, temp_output):
        """Test clear command requires confirmation."""
        from src.cli.main import app

        create_mock_pdf(temp_inbox, "test.pdf")

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
            },
        ):
            # Without --force, should prompt or abort
            result = runner.invoke(app, ["inbox", "clear"], input="n\n")

        # File should still exist
        assert (temp_inbox / "test.pdf").exists()

    def test_clear_with_force(self, temp_inbox, temp_output):
        """Test clear command with --force flag."""
        from src.cli.main import app

        create_mock_pdf(temp_inbox, "test.pdf")

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
            },
        ):
            result = runner.invoke(app, ["inbox", "clear", "--force"])

        assert result.exit_code == 0
        # File should be deleted
        assert not (temp_inbox / "test.pdf").exists()


class TestInboxConfigCommand:
    """Tests for inbox configuration via 'kidkazz config' command."""

    def test_config_show_inbox_path(self):
        """Test showing inbox configuration."""
        from src.cli.main import app

        result = runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        # Should show inbox configuration section
        assert "inbox" in result.stdout.lower() or "path" in result.stdout.lower()

    def test_config_set_inbox_path(self, temp_inbox):
        """Test setting inbox path."""
        from src.cli.main import app

        result = runner.invoke(app, ["config", "set", "inbox_path", str(temp_inbox)])

        assert result.exit_code == 0

    def test_config_set_post_action(self):
        """Test setting post_action configuration."""
        from src.cli.main import app

        result = runner.invoke(app, ["config", "set", "post_action", "move"])

        assert result.exit_code == 0


class TestInboxPathExpansion:
    """Tests for path expansion in inbox configuration."""

    def test_tilde_expansion(self, temp_output):
        """Test that ~ is expanded in inbox path."""
        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": "~/kidkazz_inbox",
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
            },
        ):
            result = runner.invoke(app, ["inbox", "status"])

        # Should not fail due to path issues
        assert result.exit_code == 0


class TestInboxErrorHandling:
    """Tests for error handling in inbox commands."""

    def test_invalid_inbox_path(self, temp_output):
        """Test handling of invalid inbox path."""
        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": "/nonexistent/path/that/should/not/exist",
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
            },
        ):
            result = runner.invoke(app, ["inbox", "status"])

        # Should handle gracefully (either create dir or show error)
        # Exit code depends on implementation choice
        assert result.exit_code in [0, 1]

    def test_permission_error_handling(self, temp_inbox, temp_output):
        """Test handling of permission errors."""
        from src.cli.main import app

        # Create a read-only file
        pdf_path = create_mock_pdf(temp_inbox, "readonly.pdf")
        pdf_path.chmod(0o444)

        try:
            with patch.dict(
                "os.environ",
                {
                    "KIDKAZZ_INBOX_PATH": str(temp_inbox),
                    "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                },
            ):
                # This should handle permission error gracefully
                result = runner.invoke(app, ["inbox", "list"])

            assert result.exit_code == 0
        finally:
            pdf_path.chmod(0o644)  # Restore permissions for cleanup
