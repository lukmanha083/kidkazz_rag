"""PDF Inbox Manager for handling PDF file lifecycle.

This module provides the PDFInboxManager class which handles:
- Scanning inbox directory for PDF files
- Tracking processing status
- Executing post-conversion cleanup (delete/move/keep)
- Verifying conversion output
"""

import shutil
from pathlib import Path
from typing import Optional

from src.pdf_inbox.models import (
    InboxConfig,
    PDFFile,
    PostConversionAction,
    ProcessingStatus,
)


class PDFInboxManager:
    """Manages PDF files in an inbox directory.

    This class handles the lifecycle of PDF files from discovery
    through conversion to cleanup.

    Attributes:
        inbox_path: Directory to watch for PDF files.
        output_path: Directory for converted markdown files.
        post_action: Action to take after successful conversion.
        recursive: Whether to scan subdirectories.
        processed_dir: Directory for moved PDFs (if post_action is MOVE).
    """

    def __init__(
        self,
        inbox_path: Path,
        output_path: Path,
        post_action: PostConversionAction = PostConversionAction.DELETE,
        recursive: bool = False,
        processed_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the inbox manager.

        Args:
            inbox_path: Directory to watch for PDF files.
            output_path: Directory for converted markdown files.
            post_action: Action after successful conversion.
            recursive: Whether to scan subdirectories.
            processed_dir: Directory for moved PDFs (MOVE action only).
        """
        self.inbox_path = Path(inbox_path).expanduser()
        self.output_path = Path(output_path).expanduser()
        self.post_action = post_action
        self.recursive = recursive
        self.processed_dir = (
            Path(processed_dir).expanduser()
            if processed_dir
            else self.inbox_path / "processed"
        )

        # Create inbox directory if it doesn't exist
        self.inbox_path.mkdir(parents=True, exist_ok=True)

        # Internal tracking of scanned files
        self._files: dict[Path, PDFFile] = {}

    @classmethod
    def from_config(cls, config: InboxConfig) -> "PDFInboxManager":
        """Create manager from configuration object.

        Args:
            config: InboxConfig with settings.

        Returns:
            Configured PDFInboxManager instance.
        """
        return cls(
            inbox_path=config.inbox_path,
            output_path=config.output_path,
            post_action=config.post_action,
            recursive=config.recursive,
            processed_dir=config.processed_dir,
        )

    def scan(self) -> list[PDFFile]:
        """Scan inbox directory for PDF files.

        Returns:
            List of PDFFile objects found in the inbox.
        """
        pdf_files: list[PDFFile] = []

        # Use iterdir for case-insensitive matching
        def scan_directory(directory: Path, recurse: bool) -> None:
            try:
                for item in directory.iterdir():
                    if item.is_dir():
                        # Skip processed directory
                        if item == self.processed_dir:
                            continue
                        if recurse:
                            scan_directory(item, recurse)
                    elif item.is_file() and item.suffix.lower() == ".pdf":
                        # Check if already tracked
                        if item in self._files:
                            pdf_files.append(self._files[item])
                        else:
                            try:
                                pdf_file = PDFFile.from_path(item)
                                self._files[item] = pdf_file
                                pdf_files.append(pdf_file)
                            except FileNotFoundError:
                                continue
            except PermissionError:
                pass

        scan_directory(self.inbox_path, self.recursive)
        return pdf_files

    def get_pending(self) -> list[PDFFile]:
        """Get all PDFs with PENDING status.

        Returns:
            List of PDFFile objects that are pending processing.
        """
        return [
            pdf for pdf in self._files.values() if pdf.status == ProcessingStatus.PENDING
        ]

    def mark_processing(self, pdf_file: PDFFile) -> None:
        """Mark a PDF as currently being processed.

        Args:
            pdf_file: The PDF file to mark.
        """
        pdf_file.status = ProcessingStatus.PROCESSING

    def mark_completed(self, pdf_file: PDFFile, output_path: Path) -> None:
        """Mark a PDF conversion as completed.

        Args:
            pdf_file: The PDF file that was converted.
            output_path: Path to the generated markdown file.
        """
        pdf_file.status = ProcessingStatus.COMPLETED
        pdf_file.output_path = output_path

    def mark_failed(self, pdf_file: PDFFile, error: str) -> None:
        """Mark a PDF conversion as failed.

        Args:
            pdf_file: The PDF file that failed conversion.
            error: Error message describing the failure.
        """
        pdf_file.status = ProcessingStatus.FAILED
        pdf_file.error_message = error

    def verify_output(self, output_path: Path) -> bool:
        """Verify that a markdown output file is valid.

        A valid output file:
        - Exists
        - Is not empty
        - Contains non-whitespace content

        Args:
            output_path: Path to the markdown file to verify.

        Returns:
            True if the output is valid, False otherwise.
        """
        if not output_path.exists():
            return False

        try:
            content = output_path.read_text()
            return bool(content.strip())
        except Exception:
            return False

    def cleanup(self, pdf_file: PDFFile) -> bool:
        """Execute post-conversion cleanup for a PDF.

        The action depends on the configured post_action:
        - DELETE: Remove the PDF file
        - MOVE: Move to processed directory
        - KEEP: Do nothing

        For FAILED PDFs, cleanup always returns False and
        preserves the original file.

        Args:
            pdf_file: The PDF file to clean up.

        Returns:
            True if cleanup was successful, False otherwise.
        """
        # Don't clean up failed conversions
        if pdf_file.status == ProcessingStatus.FAILED:
            return False

        try:
            if self.post_action == PostConversionAction.DELETE:
                pdf_file.path.unlink()
                return True

            elif self.post_action == PostConversionAction.MOVE:
                # Create processed directory if needed
                self.processed_dir.mkdir(parents=True, exist_ok=True)
                dest = self.processed_dir / pdf_file.name
                shutil.move(str(pdf_file.path), str(dest))
                return True

            elif self.post_action == PostConversionAction.KEEP:
                # Nothing to do
                return True

        except Exception:
            return False

        return False

    def get_stats(self) -> dict[str, int]:
        """Get statistics about the inbox.

        Returns:
            Dictionary with counts for each status.
        """
        stats = {
            "total": 0,
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
        }

        for pdf in self._files.values():
            stats["total"] += 1
            if pdf.status == ProcessingStatus.PENDING:
                stats["pending"] += 1
            elif pdf.status == ProcessingStatus.PROCESSING:
                stats["processing"] += 1
            elif pdf.status == ProcessingStatus.COMPLETED:
                stats["completed"] += 1
            elif pdf.status == ProcessingStatus.FAILED:
                stats["failed"] += 1

        return stats
