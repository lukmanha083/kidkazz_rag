"""Tests for inbox parse CLI command.

This module tests the CLI command for parsing PDFs using Reducto.ai.
Tests are written BEFORE implementation following TDD principles.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def temp_inbox(tmp_path):
    """Create a temporary inbox directory."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    return inbox


@pytest.fixture
def temp_output(tmp_path):
    """Create a temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output


@pytest.fixture
def temp_inbox_with_pdfs(temp_inbox):
    """Create inbox with test PDF files."""
    (temp_inbox / "document1.pdf").write_bytes(b"%PDF-1.4 test doc 1")
    (temp_inbox / "document2.pdf").write_bytes(b"%PDF-1.4 test doc 2")
    return temp_inbox


class TestParseCommand:
    """Tests for 'kidkazz inbox parse' command."""

    @patch("src.pdf_converter.reducto_client.ReductoClient")
    @patch("src.pdf_converter.reducto_client.ReductoConfig.from_env")
    def test_parse_success(
        self, mock_config, mock_client_class, temp_inbox_with_pdfs, temp_output
    ):
        """Test successful parsing of PDFs."""
        from src.pdf_converter.reducto_client import ParseResult

        # Setup mocks
        mock_config.return_value = MagicMock(api_key="test_key")
        mock_client = MagicMock()
        mock_client.parse_directory.return_value = [
            (temp_inbox_with_pdfs / "document1.pdf", "# Doc 1 Content"),
            (temp_inbox_with_pdfs / "document2.pdf", "# Doc 2 Content"),
        ]
        mock_client_class.return_value = mock_client

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox_with_pdfs),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                "REDUCTO_API_KEY": "test_api_key",
            },
        ):
            result = runner.invoke(app, ["inbox", "parse"])

        assert result.exit_code == 0
        assert "2" in result.stdout or "parsed" in result.stdout.lower()

    @patch("src.pdf_converter.reducto_client.ReductoConfig.from_env")
    def test_parse_no_api_key(self, mock_config, temp_inbox_with_pdfs, temp_output):
        """Test parse command when API key is missing."""
        mock_config.side_effect = ValueError("REDUCTO_API_KEY not set")

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox_with_pdfs),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
            },
            clear=True,
        ):
            # Remove REDUCTO_API_KEY if present
            import os
            os.environ.pop("REDUCTO_API_KEY", None)

            result = runner.invoke(app, ["inbox", "parse"])

        assert result.exit_code == 1
        assert "REDUCTO_API_KEY" in result.stdout or "api key" in result.stdout.lower()

    @patch("src.pdf_converter.reducto_client.ReductoClient")
    @patch("src.pdf_converter.reducto_client.ReductoConfig.from_env")
    def test_parse_empty_inbox(
        self, mock_config, mock_client_class, temp_inbox, temp_output
    ):
        """Test parse command with empty inbox."""
        mock_config.return_value = MagicMock(api_key="test_key")
        mock_client = MagicMock()
        mock_client.parse_directory.return_value = []
        mock_client_class.return_value = mock_client

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                "REDUCTO_API_KEY": "test_key",
            },
        ):
            result = runner.invoke(app, ["inbox", "parse"])

        assert result.exit_code == 0
        assert "no pdf" in result.stdout.lower() or "empty" in result.stdout.lower()

    @patch("src.pdf_converter.reducto_client.ReductoClient")
    @patch("src.pdf_converter.reducto_client.ReductoConfig.from_env")
    def test_parse_dry_run(
        self, mock_config, mock_client_class, temp_inbox_with_pdfs, temp_output
    ):
        """Test parse command with --dry-run flag."""
        mock_config.return_value = MagicMock(api_key="test_key")

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox_with_pdfs),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                "REDUCTO_API_KEY": "test_key",
            },
        ):
            result = runner.invoke(app, ["inbox", "parse", "--dry-run"])

        assert result.exit_code == 0
        # Should show files but not actually parse
        assert "document1.pdf" in result.stdout or "would" in result.stdout.lower()
        # parse_directory should NOT be called in dry-run
        mock_client_class.return_value.parse_directory.assert_not_called()

    @patch("src.pdf_converter.reducto_client.ReductoClient")
    @patch("src.pdf_converter.reducto_client.ReductoConfig.from_env")
    def test_parse_agentic_flag(
        self, mock_config, mock_client_class, temp_inbox_with_pdfs, temp_output
    ):
        """Test parse command with --agentic flag."""
        mock_reducto_config = MagicMock(api_key="test_key", agentic=False)
        mock_config.return_value = mock_reducto_config
        mock_client = MagicMock()
        mock_client.parse_directory.return_value = []
        mock_client_class.return_value = mock_client

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox_with_pdfs),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                "REDUCTO_API_KEY": "test_key",
            },
        ):
            result = runner.invoke(app, ["inbox", "parse", "--agentic"])

        assert result.exit_code == 0
        # Verify agentic flag was set on config
        assert mock_reducto_config.agentic is True

    @patch("src.pdf_converter.reducto_client.ReductoClient")
    @patch("src.pdf_converter.reducto_client.ReductoConfig.from_env")
    @patch("src.pdf_inbox.cloud_sync.CloudSync")
    def test_parse_with_sync_backup(
        self,
        mock_cloud_sync,
        mock_config,
        mock_client_class,
        temp_inbox_with_pdfs,
        temp_output,
    ):
        """Test parse command syncs to cloud backup."""
        mock_config.return_value = MagicMock(api_key="test_key")
        mock_client = MagicMock()
        mock_client.parse_directory.return_value = [
            (temp_inbox_with_pdfs / "document1.pdf", "# Content"),
        ]
        mock_client_class.return_value = mock_client

        mock_sync_instance = MagicMock()
        mock_cloud_sync.return_value = mock_sync_instance

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox_with_pdfs),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                "REDUCTO_API_KEY": "test_key",
                "KIDKAZZ_CLOUD_REMOTE": "gdrive",
                "KIDKAZZ_CLOUD_PATH": "kidkazz_inbox",
            },
        ):
            result = runner.invoke(app, ["inbox", "parse"])

        assert result.exit_code == 0
        # Should sync output to cloud
        mock_sync_instance.sync_to_remote.assert_called()

    @patch("src.pdf_converter.reducto_client.ReductoClient")
    @patch("src.pdf_converter.reducto_client.ReductoConfig.from_env")
    @patch("src.pdf_inbox.cloud_sync.CloudSync")
    def test_parse_no_sync_backup(
        self,
        mock_cloud_sync,
        mock_config,
        mock_client_class,
        temp_inbox_with_pdfs,
        temp_output,
    ):
        """Test parse command with --no-sync-backup skips cloud sync."""
        mock_config.return_value = MagicMock(api_key="test_key")
        mock_client = MagicMock()
        mock_client.parse_directory.return_value = [
            (temp_inbox_with_pdfs / "document1.pdf", "# Content"),
        ]
        mock_client_class.return_value = mock_client

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox_with_pdfs),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                "REDUCTO_API_KEY": "test_key",
                "KIDKAZZ_CLOUD_REMOTE": "gdrive",
            },
        ):
            result = runner.invoke(app, ["inbox", "parse", "--no-sync-backup"])

        assert result.exit_code == 0
        # Should NOT sync to cloud
        mock_cloud_sync.return_value.sync_to_remote.assert_not_called()

    @patch("src.pdf_converter.reducto_client.ReductoClient")
    @patch("src.pdf_converter.reducto_client.ReductoConfig.from_env")
    def test_parse_saves_output_files(
        self, mock_config, mock_client_class, temp_inbox_with_pdfs, temp_output
    ):
        """Test parse command saves markdown files to output directory."""
        mock_config.return_value = MagicMock(api_key="test_key")
        mock_client = MagicMock()
        mock_client.parse_directory.return_value = [
            (temp_inbox_with_pdfs / "document1.pdf", "# Document 1\n\nContent 1"),
            (temp_inbox_with_pdfs / "document2.pdf", "# Document 2\n\nContent 2"),
        ]
        mock_client_class.return_value = mock_client

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox_with_pdfs),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                "REDUCTO_API_KEY": "test_key",
            },
        ):
            result = runner.invoke(app, ["inbox", "parse", "--no-sync-backup"])

        assert result.exit_code == 0
        # Check output files exist
        assert (temp_output / "document1.md").exists()
        assert (temp_output / "document2.md").exists()
        assert "# Document 1" in (temp_output / "document1.md").read_text()

    @patch("src.pdf_converter.reducto_client.ReductoClient")
    @patch("src.pdf_converter.reducto_client.ReductoConfig.from_env")
    def test_parse_api_error(
        self, mock_config, mock_client_class, temp_inbox_with_pdfs, temp_output
    ):
        """Test parse command handles API errors gracefully."""
        from src.pdf_converter.reducto_client import ReductoAPIError

        mock_config.return_value = MagicMock(api_key="test_key")
        mock_client = MagicMock()
        mock_client.parse_directory.side_effect = ReductoAPIError("Rate limit exceeded")
        mock_client_class.return_value = mock_client

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox_with_pdfs),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                "REDUCTO_API_KEY": "test_key",
            },
        ):
            result = runner.invoke(app, ["inbox", "parse", "--no-sync-backup"])

        assert result.exit_code == 1
        assert "error" in result.stdout.lower() or "failed" in result.stdout.lower()


class TestParseCommandHelp:
    """Tests for parse command help and documentation."""

    def test_parse_help(self):
        """Test parse command shows help."""
        from src.cli.main import app

        result = runner.invoke(app, ["inbox", "parse", "--help"])

        assert result.exit_code == 0
        assert "reducto" in result.stdout.lower()
        assert "--agentic" in result.stdout
        assert "--dry-run" in result.stdout
