"""PDF Inbox module for managing PDF files before conversion.

This module provides:
- PDFFile: Data model for PDF file metadata
- ProcessingStatus: Enum for tracking PDF processing state
- PostConversionAction: Enum for post-conversion cleanup actions
- SyncStatus: Enum for cloud sync states
- InboxConfig: Configuration for inbox behavior
- SyncResult: Result of sync operations
- CloudSyncConfig: Configuration for cloud sync
- PDFInboxManager: Main class for managing the PDF inbox
- CloudSync: Cloud sync functionality using rclone
"""

from src.pdf_inbox.cloud_sync import CloudSync
from src.pdf_inbox.manager import PDFInboxManager
from src.pdf_inbox.models import (
    CloudSyncConfig,
    InboxConfig,
    PDFFile,
    PostConversionAction,
    ProcessingStatus,
    SyncResult,
    SyncStatus,
)

__all__ = [
    "PDFFile",
    "ProcessingStatus",
    "PostConversionAction",
    "SyncStatus",
    "InboxConfig",
    "SyncResult",
    "CloudSyncConfig",
    "PDFInboxManager",
    "CloudSync",
]
