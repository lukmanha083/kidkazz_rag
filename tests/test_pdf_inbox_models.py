"""Tests for PDF inbox data models.

This module tests the data models used by the PDF inbox feature.
Tests are written BEFORE implementation following TDD principles.
"""

from datetime import datetime
from pathlib import Path

import pytest


class TestProcessingStatus:
    """Tests for ProcessingStatus enum."""

    def test_pending_status_exists(self):
        """Test PENDING status is available."""
        from src.pdf_inbox.models import ProcessingStatus

        assert ProcessingStatus.PENDING.value == "pending"

    def test_processing_status_exists(self):
        """Test PROCESSING status is available."""
        from src.pdf_inbox.models import ProcessingStatus

        assert ProcessingStatus.PROCESSING.value == "processing"

    def test_completed_status_exists(self):
        """Test COMPLETED status is available."""
        from src.pdf_inbox.models import ProcessingStatus

        assert ProcessingStatus.COMPLETED.value == "completed"

    def test_failed_status_exists(self):
        """Test FAILED status is available."""
        from src.pdf_inbox.models import ProcessingStatus

        assert ProcessingStatus.FAILED.value == "failed"

    def test_status_from_string(self):
        """Test creating status from string value."""
        from src.pdf_inbox.models import ProcessingStatus

        assert ProcessingStatus("pending") == ProcessingStatus.PENDING
        assert ProcessingStatus("completed") == ProcessingStatus.COMPLETED


class TestPostConversionAction:
    """Tests for PostConversionAction enum."""

    def test_delete_action_exists(self):
        """Test DELETE action is available."""
        from src.pdf_inbox.models import PostConversionAction

        assert PostConversionAction.DELETE.value == "delete"

    def test_move_action_exists(self):
        """Test MOVE action is available."""
        from src.pdf_inbox.models import PostConversionAction

        assert PostConversionAction.MOVE.value == "move"

    def test_keep_action_exists(self):
        """Test KEEP action is available."""
        from src.pdf_inbox.models import PostConversionAction

        assert PostConversionAction.KEEP.value == "keep"

    def test_action_from_string(self):
        """Test creating action from string value."""
        from src.pdf_inbox.models import PostConversionAction

        assert PostConversionAction("delete") == PostConversionAction.DELETE
        assert PostConversionAction("move") == PostConversionAction.MOVE
        assert PostConversionAction("keep") == PostConversionAction.KEEP


class TestPDFFile:
    """Tests for PDFFile dataclass."""

    def test_create_pdf_file(self):
        """Test creating a PDFFile instance."""
        from src.pdf_inbox.models import PDFFile, ProcessingStatus

        pdf = PDFFile(
            path=Path("/inbox/test.pdf"),
            name="test.pdf",
            size=1024,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            status=ProcessingStatus.PENDING,
        )

        assert pdf.path == Path("/inbox/test.pdf")
        assert pdf.name == "test.pdf"
        assert pdf.size == 1024
        assert pdf.status == ProcessingStatus.PENDING

    def test_pdf_file_optional_fields(self):
        """Test PDFFile optional fields default to None."""
        from src.pdf_inbox.models import PDFFile, ProcessingStatus

        pdf = PDFFile(
            path=Path("/inbox/test.pdf"),
            name="test.pdf",
            size=1024,
            created_at=datetime.now(),
            status=ProcessingStatus.PENDING,
        )

        assert pdf.output_path is None
        assert pdf.error_message is None

    def test_pdf_file_with_output_path(self):
        """Test PDFFile with output_path set."""
        from src.pdf_inbox.models import PDFFile, ProcessingStatus

        pdf = PDFFile(
            path=Path("/inbox/test.pdf"),
            name="test.pdf",
            size=1024,
            created_at=datetime.now(),
            status=ProcessingStatus.COMPLETED,
            output_path=Path("/output/test.md"),
        )

        assert pdf.output_path == Path("/output/test.md")

    def test_pdf_file_with_error_message(self):
        """Test PDFFile with error_message set."""
        from src.pdf_inbox.models import PDFFile, ProcessingStatus

        pdf = PDFFile(
            path=Path("/inbox/test.pdf"),
            name="test.pdf",
            size=1024,
            created_at=datetime.now(),
            status=ProcessingStatus.FAILED,
            error_message="Conversion failed: invalid PDF",
        )

        assert pdf.error_message == "Conversion failed: invalid PDF"

    def test_pdf_file_from_path(self):
        """Test creating PDFFile from a path using factory method."""
        from src.pdf_inbox.models import PDFFile

        # Create a temporary file for testing
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)

        try:
            pdf = PDFFile.from_path(temp_path)

            assert pdf.path == temp_path
            assert pdf.name == temp_path.name
            assert pdf.size == 12  # "test content" is 12 bytes
            assert pdf.status.value == "pending"
        finally:
            temp_path.unlink()

    def test_pdf_file_from_nonexistent_path(self):
        """Test creating PDFFile from nonexistent path raises error."""
        from src.pdf_inbox.models import PDFFile

        with pytest.raises(FileNotFoundError):
            PDFFile.from_path(Path("/nonexistent/file.pdf"))

    def test_pdf_file_is_pending(self):
        """Test is_pending property."""
        from src.pdf_inbox.models import PDFFile, ProcessingStatus

        pdf = PDFFile(
            path=Path("/inbox/test.pdf"),
            name="test.pdf",
            size=1024,
            created_at=datetime.now(),
            status=ProcessingStatus.PENDING,
        )

        assert pdf.is_pending is True

        pdf.status = ProcessingStatus.PROCESSING
        assert pdf.is_pending is False

    def test_pdf_file_is_completed(self):
        """Test is_completed property."""
        from src.pdf_inbox.models import PDFFile, ProcessingStatus

        pdf = PDFFile(
            path=Path("/inbox/test.pdf"),
            name="test.pdf",
            size=1024,
            created_at=datetime.now(),
            status=ProcessingStatus.COMPLETED,
        )

        assert pdf.is_completed is True

    def test_pdf_file_is_failed(self):
        """Test is_failed property."""
        from src.pdf_inbox.models import PDFFile, ProcessingStatus

        pdf = PDFFile(
            path=Path("/inbox/test.pdf"),
            name="test.pdf",
            size=1024,
            created_at=datetime.now(),
            status=ProcessingStatus.FAILED,
        )

        assert pdf.is_failed is True


class TestInboxConfig:
    """Tests for InboxConfig dataclass."""

    def test_create_inbox_config_defaults(self):
        """Test creating InboxConfig with defaults."""
        from src.pdf_inbox.models import InboxConfig, PostConversionAction

        config = InboxConfig()

        assert config.inbox_path == Path("~/.kidkazz/inbox").expanduser()
        assert config.output_path == Path("~/.kidkazz/output").expanduser()
        assert config.post_action == PostConversionAction.DELETE
        assert config.recursive is False
        assert config.processed_dir == Path("~/.kidkazz/processed").expanduser()

    def test_create_inbox_config_custom_paths(self):
        """Test creating InboxConfig with custom paths."""
        from src.pdf_inbox.models import InboxConfig, PostConversionAction

        config = InboxConfig(
            inbox_path=Path("/custom/inbox"),
            output_path=Path("/custom/output"),
            post_action=PostConversionAction.MOVE,
            recursive=True,
        )

        assert config.inbox_path == Path("/custom/inbox")
        assert config.output_path == Path("/custom/output")
        assert config.post_action == PostConversionAction.MOVE
        assert config.recursive is True

    def test_inbox_config_expand_user(self):
        """Test that ~ is expanded in paths."""
        from src.pdf_inbox.models import InboxConfig

        config = InboxConfig(inbox_path=Path("~/my_inbox"))

        # The path should be expanded
        assert "~" not in str(config.inbox_path)

    def test_inbox_config_to_dict(self):
        """Test converting config to dictionary."""
        from src.pdf_inbox.models import InboxConfig, PostConversionAction

        config = InboxConfig(
            inbox_path=Path("/inbox"),
            output_path=Path("/output"),
            post_action=PostConversionAction.DELETE,
        )

        data = config.to_dict()

        assert data["inbox_path"] == "/inbox"
        assert data["output_path"] == "/output"
        assert data["post_action"] == "delete"
        assert data["recursive"] is False

    def test_inbox_config_from_dict(self):
        """Test creating config from dictionary."""
        from src.pdf_inbox.models import InboxConfig, PostConversionAction

        data = {
            "inbox_path": "/inbox",
            "output_path": "/output",
            "post_action": "move",
            "recursive": True,
        }

        config = InboxConfig.from_dict(data)

        assert config.inbox_path == Path("/inbox")
        assert config.output_path == Path("/output")
        assert config.post_action == PostConversionAction.MOVE
        assert config.recursive is True

    def test_inbox_config_from_dict_invalid_post_action(self):
        """Test creating config with invalid post_action defaults to DELETE."""
        from src.pdf_inbox.models import InboxConfig, PostConversionAction

        data = {
            "inbox_path": "/inbox",
            "post_action": "invalid_action",
        }

        config = InboxConfig.from_dict(data)

        # Should default to DELETE for invalid values
        assert config.post_action == PostConversionAction.DELETE


class TestSyncStatus:
    """Tests for SyncStatus enum."""

    def test_pending_status_exists(self):
        """Test PENDING status is available."""
        from src.pdf_inbox.models import SyncStatus

        assert SyncStatus.PENDING.value == "pending"

    def test_syncing_status_exists(self):
        """Test SYNCING status is available."""
        from src.pdf_inbox.models import SyncStatus

        assert SyncStatus.SYNCING.value == "syncing"

    def test_synced_status_exists(self):
        """Test SYNCED status is available."""
        from src.pdf_inbox.models import SyncStatus

        assert SyncStatus.SYNCED.value == "synced"

    def test_failed_status_exists(self):
        """Test FAILED status is available."""
        from src.pdf_inbox.models import SyncStatus

        assert SyncStatus.FAILED.value == "failed"

    def test_status_from_string(self):
        """Test creating status from string value."""
        from src.pdf_inbox.models import SyncStatus

        assert SyncStatus("pending") == SyncStatus.PENDING
        assert SyncStatus("synced") == SyncStatus.SYNCED


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_create_sync_result_success(self):
        """Test creating a successful SyncResult."""
        from src.pdf_inbox.models import SyncResult

        result = SyncResult(
            success=True,
            files_synced=5,
            bytes_transferred=1024000,
            direction="upload",
        )

        assert result.success is True
        assert result.files_synced == 5
        assert result.bytes_transferred == 1024000
        assert result.direction == "upload"
        assert result.error_message is None

    def test_create_sync_result_failure(self):
        """Test creating a failed SyncResult."""
        from src.pdf_inbox.models import SyncResult

        result = SyncResult(
            success=False,
            files_synced=0,
            bytes_transferred=0,
            direction="upload",
            error_message="Remote not found",
        )

        assert result.success is False
        assert result.files_synced == 0
        assert result.error_message == "Remote not found"

    def test_sync_result_has_timestamp(self):
        """Test SyncResult has timestamp field."""
        from src.pdf_inbox.models import SyncResult

        result = SyncResult(
            success=True,
            files_synced=1,
            bytes_transferred=100,
            direction="download",
        )

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)


class TestCloudSyncConfig:
    """Tests for CloudSyncConfig dataclass."""

    def test_create_config_defaults(self):
        """Test creating CloudSyncConfig with defaults."""
        from src.pdf_inbox.models import CloudSyncConfig

        config = CloudSyncConfig()

        assert config.remote_name == ""
        assert config.remote_path == ""
        assert config.auto_sync is False

    def test_create_config_custom(self):
        """Test creating CloudSyncConfig with custom values."""
        from src.pdf_inbox.models import CloudSyncConfig

        config = CloudSyncConfig(
            remote_name="gdrive",
            remote_path="kidkazz/inbox",
            auto_sync=True,
        )

        assert config.remote_name == "gdrive"
        assert config.remote_path == "kidkazz/inbox"
        assert config.auto_sync is True

    def test_config_to_dict(self):
        """Test converting CloudSyncConfig to dictionary."""
        from src.pdf_inbox.models import CloudSyncConfig

        config = CloudSyncConfig(
            remote_name="gdrive",
            remote_path="inbox",
            auto_sync=False,
        )

        data = config.to_dict()

        assert data["remote_name"] == "gdrive"
        assert data["remote_path"] == "inbox"
        assert data["auto_sync"] is False

    def test_config_from_dict(self):
        """Test creating CloudSyncConfig from dictionary."""
        from src.pdf_inbox.models import CloudSyncConfig

        data = {
            "remote_name": "dropbox",
            "remote_path": "pdf_inbox",
            "auto_sync": True,
        }

        config = CloudSyncConfig.from_dict(data)

        assert config.remote_name == "dropbox"
        assert config.remote_path == "pdf_inbox"
        assert config.auto_sync is True

    def test_config_from_dict_with_defaults(self):
        """Test creating CloudSyncConfig from partial dictionary."""
        from src.pdf_inbox.models import CloudSyncConfig

        data = {"remote_name": "gdrive"}

        config = CloudSyncConfig.from_dict(data)

        assert config.remote_name == "gdrive"
        assert config.remote_path == ""
        assert config.auto_sync is False
