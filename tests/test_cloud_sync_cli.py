"""Tests for cloud sync CLI commands.

This module tests the CLI commands for cloud synchronization.
Tests are written BEFORE implementation following TDD principles.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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


@pytest.fixture
def temp_inbox_with_pdfs(temp_inbox):
    """Create a temporary inbox with PDF files."""
    (temp_inbox / "document1.pdf").write_bytes(b"PDF content 1")
    (temp_inbox / "document2.pdf").write_bytes(b"PDF content 2")
    return temp_inbox


@pytest.fixture(autouse=True)
def mock_embedder_env():
    """Use mock embedder for all tests."""
    import os

    with patch.dict(os.environ, {"KIDKAZZ_EMBEDDER_TYPE": "mock"}):
        yield


class TestSyncCommand:
    """Tests for 'kidkazz inbox sync' command."""

    @patch("src.pdf_inbox.cloud_sync.CloudSync.check_rclone_installed")
    @patch("src.pdf_inbox.cloud_sync.CloudSync.sync_to_remote")
    def test_sync_uploads_to_remote(
        self, mock_sync, mock_check, temp_inbox_with_pdfs, temp_output
    ):
        """Test sync command uploads files to remote."""
        from src.pdf_inbox.models import SyncResult

        mock_check.return_value = True
        mock_sync.return_value = SyncResult(
            success=True,
            files_synced=2,
            bytes_transferred=1024,
            direction="upload",
        )

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox_with_pdfs),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                "KIDKAZZ_CLOUD_REMOTE": "gdrive",
                "KIDKAZZ_CLOUD_PATH": "kidkazz_inbox",
            },
        ):
            result = runner.invoke(app, ["inbox", "sync"])

        assert result.exit_code == 0
        assert "2" in result.stdout or "synced" in result.stdout.lower()

    def test_sync_no_remote_configured(self, temp_inbox_with_pdfs, temp_output):
        """Test sync command when no remote is configured."""
        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox_with_pdfs),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                "KIDKAZZ_CLOUD_REMOTE": "",
            },
        ):
            result = runner.invoke(app, ["inbox", "sync"])

        assert result.exit_code == 1
        assert "remote" in result.stdout.lower() or "configure" in result.stdout.lower()

    @patch("src.pdf_inbox.cloud_sync.CloudSync.check_rclone_installed")
    @patch("src.pdf_inbox.cloud_sync.CloudSync.sync_to_remote")
    def test_sync_dry_run(
        self, mock_sync, mock_check, temp_inbox_with_pdfs, temp_output
    ):
        """Test sync command with --dry-run flag."""
        from src.pdf_inbox.models import SyncResult

        mock_check.return_value = True
        mock_sync.return_value = SyncResult(
            success=True,
            files_synced=0,
            bytes_transferred=0,
            direction="upload",
        )

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox_with_pdfs),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                "KIDKAZZ_CLOUD_REMOTE": "gdrive",
                "KIDKAZZ_CLOUD_PATH": "kidkazz_inbox",
            },
        ):
            result = runner.invoke(app, ["inbox", "sync", "--dry-run"])

        assert result.exit_code == 0
        # Should have called sync with dry_run=True
        mock_sync.assert_called_once()
        call_kwargs = mock_sync.call_args[1]
        assert call_kwargs.get("dry_run") is True

    @patch("src.pdf_inbox.cloud_sync.CloudSync.check_rclone_installed")
    @patch("src.pdf_inbox.cloud_sync.CloudSync.sync_from_remote")
    def test_sync_download_flag(
        self, mock_sync, mock_check, temp_inbox, temp_output
    ):
        """Test sync command with --download flag."""
        from src.pdf_inbox.models import SyncResult

        mock_check.return_value = True
        mock_sync.return_value = SyncResult(
            success=True,
            files_synced=3,
            bytes_transferred=2048,
            direction="download",
        )

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                "KIDKAZZ_CLOUD_REMOTE": "gdrive",
                "KIDKAZZ_CLOUD_PATH": "kidkazz_inbox",
            },
        ):
            result = runner.invoke(app, ["inbox", "sync", "--download"])

        assert result.exit_code == 0
        mock_sync.assert_called_once()

    @patch("src.pdf_inbox.cloud_sync.CloudSync.check_rclone_installed")
    def test_sync_check_flag(self, mock_check, temp_inbox, temp_output):
        """Test sync command with --check flag."""
        mock_check.return_value = True

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
            },
        ):
            result = runner.invoke(app, ["inbox", "sync", "--check"])

        assert result.exit_code == 0
        assert "installed" in result.stdout.lower() or "rclone" in result.stdout.lower()

    @patch("src.pdf_inbox.cloud_sync.CloudSync.list_remotes")
    def test_sync_remotes_flag(self, mock_list, temp_inbox, temp_output):
        """Test sync command with --remotes flag."""
        mock_list.return_value = ["gdrive", "dropbox"]

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
            },
        ):
            result = runner.invoke(app, ["inbox", "sync", "--remotes"])

        assert result.exit_code == 0
        assert "gdrive" in result.stdout
        assert "dropbox" in result.stdout

    @patch("src.pdf_inbox.cloud_sync.CloudSync.check_rclone_installed")
    def test_sync_rclone_not_installed(
        self, mock_check, temp_inbox_with_pdfs, temp_output
    ):
        """Test sync command when rclone is not installed."""
        mock_check.return_value = False

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox_with_pdfs),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
                "KIDKAZZ_CLOUD_REMOTE": "gdrive",
                "KIDKAZZ_CLOUD_PATH": "kidkazz_inbox",
            },
        ):
            result = runner.invoke(app, ["inbox", "sync"])

        assert result.exit_code == 1
        assert "rclone" in result.stdout.lower() or "install" in result.stdout.lower()

    @patch("src.pdf_inbox.cloud_sync.CloudSync.list_remotes")
    def test_sync_remotes_empty(self, mock_list, temp_inbox, temp_output):
        """Test sync --remotes when no remotes configured."""
        mock_list.return_value = []

        from src.cli.main import app

        with patch.dict(
            "os.environ",
            {
                "KIDKAZZ_INBOX_PATH": str(temp_inbox),
                "KIDKAZZ_OUTPUT_PATH": str(temp_output),
            },
        ):
            result = runner.invoke(app, ["inbox", "sync", "--remotes"])

        assert result.exit_code == 0
        assert "no remotes" in result.stdout.lower() or "configure" in result.stdout.lower()
