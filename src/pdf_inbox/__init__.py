"""PDF Inbox module for managing PDF files before conversion.

This module provides:
- PDFFile: Data model for PDF file metadata
- ProcessingStatus: Enum for tracking PDF processing state
- PostConversionAction: Enum for post-conversion cleanup actions
- InboxConfig: Configuration for inbox behavior
- PDFInboxManager: Main class for managing the PDF inbox
"""

from src.pdf_inbox.manager import PDFInboxManager
from src.pdf_inbox.models import (
    InboxConfig,
    PDFFile,
    PostConversionAction,
    ProcessingStatus,
)

__all__ = [
    "PDFFile",
    "ProcessingStatus",
    "PostConversionAction",
    "InboxConfig",
    "PDFInboxManager",
]
