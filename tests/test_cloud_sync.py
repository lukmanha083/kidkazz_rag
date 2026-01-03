"""Tests for CloudSync class.

This module tests the CloudSync class that wraps rclone subprocess calls.
Tests are written BEFORE implementation following TDD principles.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_inbox():
    """Create a temporary inbox directory."""
    inbox_dir = tempfile.mkdtemp(prefix="kidkazz_inbox_")
    yield Path(inbox_dir)
    shutil.rmtree(inbox_dir, ignore_errors=True)


@pytest.fixture
def temp_inbox_with_pdfs(temp_inbox):
    """Create a temporary inbox with PDF files."""
    # Create mock PDF files
    (temp_inbox / "document1.pdf").write_bytes(b"PDF content 1")
    (temp_inbox / "document2.pdf").write_bytes(b"PDF content 2")
    (temp_inbox / "report.pdf").write_bytes(b"PDF content 3")
    return temp_inbox


class TestCloudSyncInit:
    """Tests for CloudSync initialization."""

    def test_create_cloud_sync_defaults(self):
        """Test creating CloudSync with defaults."""
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync()

        assert sync.remote_name == ""
        assert sync.remote_path == ""

    def test_create_cloud_sync_custom(self):
        """Test creating CloudSync with custom values."""
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync(remote_name="gdrive", remote_path="kidkazz_inbox")

        assert sync.remote_name == "gdrive"
        assert sync.remote_path == "kidkazz_inbox"


class TestCloudSyncRcloneCheck:
    """Tests for rclone installation verification."""

    @patch("subprocess.run")
    def test_check_rclone_installed_success(self, mock_run):
        """Test detecting rclone is installed."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"rclone v1.65.0\n",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync()
        result = sync.check_rclone_installed()

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "rclone" in call_args
        assert "version" in call_args

    @patch("subprocess.run")
    def test_check_rclone_not_installed(self, mock_run):
        """Test detecting rclone is not installed."""
        mock_run.side_effect = FileNotFoundError("rclone not found")

        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync()
        result = sync.check_rclone_installed()

        assert result is False

    @patch("subprocess.run")
    def test_get_rclone_version(self, mock_run):
        """Test getting rclone version string."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"rclone v1.65.0\n- os/version: linux\n",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync()
        version = sync.get_rclone_version()

        assert "1.65.0" in version


class TestCloudSyncRemotes:
    """Tests for remote management."""

    @patch("subprocess.run")
    def test_list_remotes(self, mock_run):
        """Test listing configured remotes."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"gdrive:\ndropbox:\n",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync()
        remotes = sync.list_remotes()

        assert remotes == ["gdrive", "dropbox"]

    @patch("subprocess.run")
    def test_list_remotes_empty(self, mock_run):
        """Test listing when no remotes configured."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync()
        remotes = sync.list_remotes()

        assert remotes == []

    @patch("subprocess.run")
    def test_validate_remote_exists(self, mock_run):
        """Test validating a configured remote."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"gdrive:\ndropbox:\n",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync()
        result = sync.validate_remote("gdrive")

        assert result is True

    @patch("subprocess.run")
    def test_validate_remote_not_exists(self, mock_run):
        """Test validating an unconfigured remote."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"dropbox:\n",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync()
        result = sync.validate_remote("gdrive")

        assert result is False


class TestCloudSyncUpload:
    """Tests for upload operations."""

    @patch("subprocess.run")
    def test_sync_to_remote_success(self, mock_run, temp_inbox_with_pdfs):
        """Test successful upload to remote."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"",
            stderr=b"Transferred: 3 files, 100 B\n",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync(remote_name="gdrive", remote_path="inbox")
        result = sync.sync_to_remote(local_path=temp_inbox_with_pdfs)

        assert result.success is True
        assert result.direction == "upload"
        # Verify rclone copy command was called
        call_args = mock_run.call_args[0][0]
        assert "rclone" in call_args
        assert "copy" in call_args

    @patch("subprocess.run")
    def test_sync_to_remote_failure(self, mock_run, temp_inbox_with_pdfs):
        """Test failed upload to remote."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout=b"",
            stderr=b"Error: remote not found",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync(remote_name="invalid", remote_path="inbox")
        result = sync.sync_to_remote(local_path=temp_inbox_with_pdfs)

        assert result.success is False
        assert "not found" in result.error_message.lower()

    @patch("subprocess.run")
    def test_sync_to_remote_dry_run(self, mock_run, temp_inbox_with_pdfs):
        """Test dry run upload."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"",
            stderr=b"Would copy: document1.pdf\nWould copy: document2.pdf\n",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync(remote_name="gdrive", remote_path="inbox")
        result = sync.sync_to_remote(local_path=temp_inbox_with_pdfs, dry_run=True)

        assert result.success is True
        # Verify --dry-run flag was used
        call_args = mock_run.call_args[0][0]
        assert "--dry-run" in call_args

    @patch("subprocess.run")
    def test_sync_to_remote_no_remote_configured(self, mock_run, temp_inbox_with_pdfs):
        """Test upload when no remote is configured."""
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync()  # No remote configured
        result = sync.sync_to_remote(local_path=temp_inbox_with_pdfs)

        assert result.success is False
        assert "remote" in result.error_message.lower()


class TestCloudSyncDownload:
    """Tests for download operations."""

    @patch("subprocess.run")
    def test_sync_from_remote_success(self, mock_run, temp_inbox):
        """Test successful download from remote."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"",
            stderr=b"Transferred: 3 files, 500 KB\n",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync(remote_name="gdrive", remote_path="inbox")
        result = sync.sync_from_remote(local_path=temp_inbox)

        assert result.success is True
        assert result.direction == "download"

    @patch("subprocess.run")
    def test_sync_from_remote_failure(self, mock_run, temp_inbox):
        """Test failed download from remote."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout=b"",
            stderr=b"Error: directory not found",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync(remote_name="gdrive", remote_path="nonexistent")
        result = sync.sync_from_remote(local_path=temp_inbox)

        assert result.success is False


class TestCloudSyncListRemote:
    """Tests for listing remote files."""

    @patch("subprocess.run")
    def test_list_remote_files(self, mock_run):
        """Test listing files on remote."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"  1234567 document.pdf\n  2345678 report.pdf\n",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync(remote_name="gdrive", remote_path="inbox")
        files = sync.list_remote_files()

        assert len(files) == 2
        assert files[0]["name"] == "document.pdf"
        assert files[0]["size"] == 1234567
        assert files[1]["name"] == "report.pdf"
        assert files[1]["size"] == 2345678

    @patch("subprocess.run")
    def test_list_remote_files_empty(self, mock_run):
        """Test listing empty remote directory."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync(remote_name="gdrive", remote_path="inbox")
        files = sync.list_remote_files()

        assert files == []


class TestCloudSyncErrorHandling:
    """Tests for error handling scenarios."""

    @patch("subprocess.run")
    def test_handles_timeout(self, mock_run, temp_inbox_with_pdfs):
        """Test handling sync timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("rclone", 3600)

        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync(remote_name="gdrive", remote_path="inbox")
        result = sync.sync_to_remote(local_path=temp_inbox_with_pdfs)

        assert result.success is False
        assert "timeout" in result.error_message.lower()

    @patch("subprocess.run")
    def test_handles_permission_error(self, mock_run, temp_inbox_with_pdfs):
        """Test handling permission errors."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout=b"",
            stderr=b"Error: permission denied",
        )
        from src.pdf_inbox.cloud_sync import CloudSync

        sync = CloudSync(remote_name="gdrive", remote_path="inbox")
        result = sync.sync_to_remote(local_path=temp_inbox_with_pdfs)

        assert result.success is False
        assert "permission" in result.error_message.lower()
