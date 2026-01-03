"""Data models for PDF inbox management.

This module defines the data structures used by the PDF inbox feature:
- ProcessingStatus: Enum for PDF processing states
- PostConversionAction: Enum for actions after conversion
- PDFFile: Dataclass representing a PDF file
- InboxConfig: Configuration settings for the inbox
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ProcessingStatus(Enum):
    """Status of a PDF file in the processing pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PostConversionAction(Enum):
    """Action to take after successful PDF conversion."""

    DELETE = "delete"
    MOVE = "move"
    KEEP = "keep"


@dataclass
class PDFFile:
    """Represents a PDF file in the inbox.

    Attributes:
        path: Full path to the PDF file.
        name: Filename of the PDF.
        size: File size in bytes.
        created_at: When the file was added/discovered.
        status: Current processing status.
        output_path: Path to converted markdown (if completed).
        error_message: Error details (if failed).
    """

    path: Path
    name: str
    size: int
    created_at: datetime
    status: ProcessingStatus
    output_path: Optional[Path] = None
    error_message: Optional[str] = None

    @classmethod
    def from_path(cls, path: Path) -> "PDFFile":
        """Create a PDFFile instance from a file path.

        Args:
            path: Path to the PDF file.

        Returns:
            PDFFile instance with metadata populated.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        stat = path.stat()
        return cls(
            path=path,
            name=path.name,
            size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            status=ProcessingStatus.PENDING,
        )

    @property
    def is_pending(self) -> bool:
        """Check if PDF is pending processing."""
        return self.status == ProcessingStatus.PENDING

    @property
    def is_completed(self) -> bool:
        """Check if PDF conversion is completed."""
        return self.status == ProcessingStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if PDF conversion failed."""
        return self.status == ProcessingStatus.FAILED


@dataclass
class InboxConfig:
    """Configuration for the PDF inbox.

    Attributes:
        inbox_path: Directory to watch for PDF files.
        output_path: Directory for converted markdown files.
        post_action: Action to take after successful conversion.
        recursive: Whether to scan subdirectories.
        processed_dir: Directory for moved PDFs (if post_action is MOVE).
    """

    inbox_path: Path = field(default_factory=lambda: Path("~/.kidkazz/inbox"))
    output_path: Path = field(default_factory=lambda: Path("~/.kidkazz/output"))
    post_action: PostConversionAction = PostConversionAction.DELETE
    recursive: bool = False
    processed_dir: Path = field(default_factory=lambda: Path("~/.kidkazz/processed"))

    def __post_init__(self) -> None:
        """Expand user paths after initialization."""
        self.inbox_path = Path(self.inbox_path).expanduser()
        self.output_path = Path(self.output_path).expanduser()
        self.processed_dir = Path(self.processed_dir).expanduser()

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "inbox_path": str(self.inbox_path),
            "output_path": str(self.output_path),
            "post_action": self.post_action.value,
            "recursive": self.recursive,
            "processed_dir": str(self.processed_dir),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InboxConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary with config values.

        Returns:
            InboxConfig instance.
        """
        post_action_str = data.get("post_action", "delete")
        try:
            post_action = PostConversionAction(post_action_str)
        except ValueError:
            # Default to DELETE for invalid values
            post_action = PostConversionAction.DELETE

        return cls(
            inbox_path=Path(data.get("inbox_path", "~/.kidkazz/inbox")),
            output_path=Path(data.get("output_path", "~/.kidkazz/output")),
            post_action=post_action,
            recursive=data.get("recursive", False),
            processed_dir=Path(data.get("processed_dir", "~/.kidkazz/processed")),
        )
