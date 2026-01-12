"""Tests for PDF inbox manager.

This module tests the PDFInboxManager class that handles PDF file
discovery, status tracking, and post-conversion cleanup.
Tests are written BEFORE implementation following TDD principles.
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest


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
def temp_processed():
    """Create a temporary processed directory."""
    processed_dir = tempfile.mkdtemp(prefix="kidkazz_processed_")
    yield Path(processed_dir)
    shutil.rmtree(processed_dir, ignore_errors=True)


def create_mock_pdf(directory: Path, name: str, content: bytes = b"PDF") -> Path:
    """Create a mock PDF file for testing."""
    pdf_path = directory / name
    pdf_path.write_bytes(content)
    return pdf_path


def create_mock_markdown(directory: Path, name: str, content: str = "# Test") -> Path:
    """Create a mock markdown file for testing."""
    md_path = directory / name
    md_path.write_text(content)
    return md_path


class TestPDFInboxManagerInit:
    """Tests for PDFInboxManager initialization."""

    def test_init_with_paths(self, temp_inbox, temp_output):
        """Test initializing manager with paths."""
        from src.pdf_inbox.manager import PDFInboxManager

        manager = PDFInboxManager(
            inbox_path=temp_inbox,
            output_path=temp_output,
        )

        assert manager.inbox_path == temp_inbox
        assert manager.output_path == temp_output

    def test_init_does_not_create_inbox_directory(self, temp_output):
        """Test manager does NOT create inbox directory automatically.

        Directory creation is deferred to commands that actually need it
        (like parse), not read-only commands (like status, list).
        """
        from src.pdf_inbox.manager import PDFInboxManager

        new_inbox = Path(tempfile.mkdtemp()) / "new_inbox"

        try:
            manager = PDFInboxManager(
                inbox_path=new_inbox,
                output_path=temp_output,
            )
            # Directory should NOT be created by init
            assert not new_inbox.exists()
        finally:
            shutil.rmtree(new_inbox.parent, ignore_errors=True)

    def test_init_with_config(self, temp_inbox, temp_output):
        """Test initializing manager from config object."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import InboxConfig, PostConversionAction

        config = InboxConfig(
            inbox_path=temp_inbox,
            output_path=temp_output,
            post_action=PostConversionAction.MOVE,
            recursive=True,
        )

        manager = PDFInboxManager.from_config(config)

        assert manager.inbox_path == temp_inbox
        assert manager.post_action == PostConversionAction.MOVE
        assert manager.recursive is True

    def test_init_default_post_action(self, temp_inbox, temp_output):
        """Test default post_action is DELETE."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import PostConversionAction

        manager = PDFInboxManager(
            inbox_path=temp_inbox,
            output_path=temp_output,
        )

        assert manager.post_action == PostConversionAction.DELETE


class TestPDFInboxManagerScan:
    """Tests for scanning inbox directory."""

    def test_scan_empty_inbox(self, temp_inbox, temp_output):
        """Test scanning empty inbox returns empty list."""
        from src.pdf_inbox.manager import PDFInboxManager

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)

        pdfs = manager.scan()

        assert pdfs == []

    def test_scan_finds_pdf_files(self, temp_inbox, temp_output):
        """Test scanning finds PDF files."""
        from src.pdf_inbox.manager import PDFInboxManager

        create_mock_pdf(temp_inbox, "test1.pdf")
        create_mock_pdf(temp_inbox, "test2.pdf")

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)

        pdfs = manager.scan()

        assert len(pdfs) == 2
        names = {pdf.name for pdf in pdfs}
        assert names == {"test1.pdf", "test2.pdf"}

    def test_scan_ignores_non_pdf_files(self, temp_inbox, temp_output):
        """Test scanning ignores non-PDF files."""
        from src.pdf_inbox.manager import PDFInboxManager

        create_mock_pdf(temp_inbox, "test.pdf")
        (temp_inbox / "readme.txt").write_text("not a pdf")
        (temp_inbox / "image.png").write_bytes(b"png data")

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)

        pdfs = manager.scan()

        assert len(pdfs) == 1
        assert pdfs[0].name == "test.pdf"

    def test_scan_case_insensitive_extension(self, temp_inbox, temp_output):
        """Test scanning handles case-insensitive .PDF extension."""
        from src.pdf_inbox.manager import PDFInboxManager

        create_mock_pdf(temp_inbox, "test1.pdf")
        create_mock_pdf(temp_inbox, "test2.PDF")
        create_mock_pdf(temp_inbox, "test3.Pdf")

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)

        pdfs = manager.scan()

        assert len(pdfs) == 3

    def test_scan_recursive_disabled(self, temp_inbox, temp_output):
        """Test scanning without recursive doesn't find nested PDFs."""
        from src.pdf_inbox.manager import PDFInboxManager

        create_mock_pdf(temp_inbox, "top.pdf")
        subdir = temp_inbox / "subdir"
        subdir.mkdir()
        create_mock_pdf(subdir, "nested.pdf")

        manager = PDFInboxManager(
            inbox_path=temp_inbox,
            output_path=temp_output,
            recursive=False,
        )

        pdfs = manager.scan()

        assert len(pdfs) == 1
        assert pdfs[0].name == "top.pdf"

    def test_scan_recursive_enabled(self, temp_inbox, temp_output):
        """Test scanning with recursive finds nested PDFs."""
        from src.pdf_inbox.manager import PDFInboxManager

        create_mock_pdf(temp_inbox, "top.pdf")
        subdir = temp_inbox / "subdir"
        subdir.mkdir()
        create_mock_pdf(subdir, "nested.pdf")

        manager = PDFInboxManager(
            inbox_path=temp_inbox,
            output_path=temp_output,
            recursive=True,
        )

        pdfs = manager.scan()

        assert len(pdfs) == 2

    def test_scan_returns_pending_status(self, temp_inbox, temp_output):
        """Test scanned PDFs have PENDING status."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import ProcessingStatus

        create_mock_pdf(temp_inbox, "test.pdf")

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)

        pdfs = manager.scan()

        assert pdfs[0].status == ProcessingStatus.PENDING


class TestPDFInboxManagerGetPending:
    """Tests for getting pending PDFs."""

    def test_get_pending_returns_only_pending(self, temp_inbox, temp_output):
        """Test get_pending returns only PENDING status PDFs."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import ProcessingStatus

        create_mock_pdf(temp_inbox, "test1.pdf")
        create_mock_pdf(temp_inbox, "test2.pdf")

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)

        # Scan first
        pdfs = manager.scan()
        # Mark one as processing
        manager.mark_processing(pdfs[0])

        pending = manager.get_pending()

        assert len(pending) == 1
        assert pending[0].status == ProcessingStatus.PENDING


class TestPDFInboxManagerMarkStatus:
    """Tests for marking PDF status."""

    def test_mark_processing(self, temp_inbox, temp_output):
        """Test marking a PDF as processing."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import ProcessingStatus

        create_mock_pdf(temp_inbox, "test.pdf")

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)
        pdfs = manager.scan()

        manager.mark_processing(pdfs[0])

        assert pdfs[0].status == ProcessingStatus.PROCESSING

    def test_mark_completed_with_output(self, temp_inbox, temp_output):
        """Test marking a PDF as completed with output path."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import ProcessingStatus

        create_mock_pdf(temp_inbox, "test.pdf")
        output_md = create_mock_markdown(temp_output, "test.md", "# Converted")

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)
        pdfs = manager.scan()

        manager.mark_completed(pdfs[0], output_md)

        assert pdfs[0].status == ProcessingStatus.COMPLETED
        assert pdfs[0].output_path == output_md

    def test_mark_failed_with_error(self, temp_inbox, temp_output):
        """Test marking a PDF as failed with error message."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import ProcessingStatus

        create_mock_pdf(temp_inbox, "test.pdf")

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)
        pdfs = manager.scan()

        manager.mark_failed(pdfs[0], "Conversion error: corrupted file")

        assert pdfs[0].status == ProcessingStatus.FAILED
        assert pdfs[0].error_message == "Conversion error: corrupted file"


class TestPDFInboxManagerVerifyOutput:
    """Tests for verifying output files."""

    def test_verify_output_valid_file(self, temp_inbox, temp_output):
        """Test verifying a valid markdown output."""
        from src.pdf_inbox.manager import PDFInboxManager

        output_md = create_mock_markdown(temp_output, "test.md", "# Valid content")

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)

        assert manager.verify_output(output_md) is True

    def test_verify_output_nonexistent_file(self, temp_inbox, temp_output):
        """Test verifying nonexistent file returns False."""
        from src.pdf_inbox.manager import PDFInboxManager

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)

        assert manager.verify_output(Path("/nonexistent/file.md")) is False

    def test_verify_output_empty_file(self, temp_inbox, temp_output):
        """Test verifying empty file returns False."""
        from src.pdf_inbox.manager import PDFInboxManager

        empty_md = create_mock_markdown(temp_output, "empty.md", "")

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)

        assert manager.verify_output(empty_md) is False

    def test_verify_output_whitespace_only(self, temp_inbox, temp_output):
        """Test verifying whitespace-only file returns False."""
        from src.pdf_inbox.manager import PDFInboxManager

        whitespace_md = create_mock_markdown(temp_output, "whitespace.md", "   \n\n  ")

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)

        assert manager.verify_output(whitespace_md) is False


class TestPDFInboxManagerCleanup:
    """Tests for post-conversion cleanup."""

    def test_cleanup_delete_action(self, temp_inbox, temp_output):
        """Test cleanup with DELETE action removes PDF."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import PostConversionAction

        pdf_path = create_mock_pdf(temp_inbox, "test.pdf")

        manager = PDFInboxManager(
            inbox_path=temp_inbox,
            output_path=temp_output,
            post_action=PostConversionAction.DELETE,
        )
        pdfs = manager.scan()

        result = manager.cleanup(pdfs[0])

        assert result is True
        assert not pdf_path.exists()

    def test_cleanup_move_action(self, temp_inbox, temp_output, temp_processed):
        """Test cleanup with MOVE action moves PDF to processed dir."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import PostConversionAction

        pdf_path = create_mock_pdf(temp_inbox, "test.pdf")

        manager = PDFInboxManager(
            inbox_path=temp_inbox,
            output_path=temp_output,
            post_action=PostConversionAction.MOVE,
            processed_dir=temp_processed,
        )
        pdfs = manager.scan()

        result = manager.cleanup(pdfs[0])

        assert result is True
        assert not pdf_path.exists()
        assert (temp_processed / "test.pdf").exists()

    def test_cleanup_keep_action(self, temp_inbox, temp_output):
        """Test cleanup with KEEP action preserves PDF."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import PostConversionAction

        pdf_path = create_mock_pdf(temp_inbox, "test.pdf")

        manager = PDFInboxManager(
            inbox_path=temp_inbox,
            output_path=temp_output,
            post_action=PostConversionAction.KEEP,
        )
        pdfs = manager.scan()

        result = manager.cleanup(pdfs[0])

        assert result is True
        assert pdf_path.exists()  # PDF should still exist

    def test_cleanup_failed_pdf_not_deleted(self, temp_inbox, temp_output):
        """Test cleanup does not delete PDF if status is FAILED."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import PostConversionAction

        pdf_path = create_mock_pdf(temp_inbox, "test.pdf")

        manager = PDFInboxManager(
            inbox_path=temp_inbox,
            output_path=temp_output,
            post_action=PostConversionAction.DELETE,
        )
        pdfs = manager.scan()
        manager.mark_failed(pdfs[0], "Conversion failed")

        result = manager.cleanup(pdfs[0])

        assert result is False  # Should not cleanup failed PDFs
        assert pdf_path.exists()  # PDF should still exist

    def test_cleanup_move_creates_processed_dir(self, temp_inbox, temp_output):
        """Test cleanup creates processed directory if it doesn't exist."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import PostConversionAction

        pdf_path = create_mock_pdf(temp_inbox, "test.pdf")
        processed_dir = temp_output / "processed_new"

        manager = PDFInboxManager(
            inbox_path=temp_inbox,
            output_path=temp_output,
            post_action=PostConversionAction.MOVE,
            processed_dir=processed_dir,
        )
        pdfs = manager.scan()
        manager.mark_completed(pdfs[0], temp_output / "test.md")

        manager.cleanup(pdfs[0])

        assert processed_dir.exists()
        assert (processed_dir / "test.pdf").exists()


class TestPDFInboxManagerWorkflow:
    """Tests for complete workflow scenarios."""

    def test_full_conversion_workflow(self, temp_inbox, temp_output):
        """Test complete workflow: scan -> process -> cleanup."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import PostConversionAction, ProcessingStatus

        # Setup
        pdf_path = create_mock_pdf(temp_inbox, "document.pdf", b"PDF content")

        manager = PDFInboxManager(
            inbox_path=temp_inbox,
            output_path=temp_output,
            post_action=PostConversionAction.DELETE,
        )

        # Scan
        pdfs = manager.scan()
        assert len(pdfs) == 1
        pdf = pdfs[0]
        assert pdf.status == ProcessingStatus.PENDING

        # Mark processing
        manager.mark_processing(pdf)
        assert pdf.status == ProcessingStatus.PROCESSING

        # Simulate conversion - create output markdown
        output_md = create_mock_markdown(temp_output, "document.md", "# Converted Document")

        # Verify output
        assert manager.verify_output(output_md) is True

        # Mark completed
        manager.mark_completed(pdf, output_md)
        assert pdf.status == ProcessingStatus.COMPLETED
        assert pdf.output_path == output_md

        # Cleanup
        result = manager.cleanup(pdf)
        assert result is True
        assert not pdf_path.exists()  # PDF should be deleted

    def test_failed_conversion_workflow(self, temp_inbox, temp_output):
        """Test workflow with failed conversion."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import PostConversionAction, ProcessingStatus

        # Setup
        pdf_path = create_mock_pdf(temp_inbox, "corrupted.pdf", b"bad data")

        manager = PDFInboxManager(
            inbox_path=temp_inbox,
            output_path=temp_output,
            post_action=PostConversionAction.DELETE,
        )

        # Scan
        pdfs = manager.scan()
        pdf = pdfs[0]

        # Mark processing
        manager.mark_processing(pdf)

        # Simulate conversion failure
        manager.mark_failed(pdf, "Conversion failed: corrupted PDF")

        # Cleanup should not delete
        result = manager.cleanup(pdf)
        assert result is False
        assert pdf_path.exists()  # PDF should be preserved
        assert pdf.error_message == "Conversion failed: corrupted PDF"

    def test_batch_processing(self, temp_inbox, temp_output):
        """Test processing multiple PDFs in batch."""
        from src.pdf_inbox.manager import PDFInboxManager
        from src.pdf_inbox.models import PostConversionAction

        # Setup multiple PDFs
        create_mock_pdf(temp_inbox, "doc1.pdf")
        create_mock_pdf(temp_inbox, "doc2.pdf")
        create_mock_pdf(temp_inbox, "doc3.pdf")

        manager = PDFInboxManager(
            inbox_path=temp_inbox,
            output_path=temp_output,
            post_action=PostConversionAction.DELETE,
        )

        # Scan all
        pdfs = manager.scan()
        assert len(pdfs) == 3

        # Process each
        for pdf in pdfs:
            manager.mark_processing(pdf)
            output_md = create_mock_markdown(temp_output, pdf.name.replace(".pdf", ".md"))
            manager.mark_completed(pdf, output_md)
            manager.cleanup(pdf)

        # All PDFs should be deleted
        remaining = list(temp_inbox.glob("*.pdf"))
        assert len(remaining) == 0


class TestPDFInboxManagerGetStats:
    """Tests for getting inbox statistics."""

    def test_get_stats_empty_inbox(self, temp_inbox, temp_output):
        """Test stats for empty inbox."""
        from src.pdf_inbox.manager import PDFInboxManager

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)

        stats = manager.get_stats()

        assert stats["total"] == 0
        assert stats["pending"] == 0
        assert stats["processing"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 0

    def test_get_stats_with_files(self, temp_inbox, temp_output):
        """Test stats with various file states."""
        from src.pdf_inbox.manager import PDFInboxManager

        create_mock_pdf(temp_inbox, "pending1.pdf")
        create_mock_pdf(temp_inbox, "pending2.pdf")
        create_mock_pdf(temp_inbox, "processing.pdf")
        create_mock_pdf(temp_inbox, "completed.pdf")
        create_mock_pdf(temp_inbox, "failed.pdf")

        manager = PDFInboxManager(inbox_path=temp_inbox, output_path=temp_output)
        pdfs = manager.scan()

        # Set various states
        manager.mark_processing(pdfs[2])
        output_md = create_mock_markdown(temp_output, "completed.md")
        manager.mark_completed(pdfs[3], output_md)
        manager.mark_failed(pdfs[4], "Error")

        stats = manager.get_stats()

        assert stats["total"] == 5
        assert stats["pending"] == 2
        assert stats["processing"] == 1
        assert stats["completed"] == 1
        assert stats["failed"] == 1
