# PDF Inbox Feature Design Document

## Overview

This document outlines the design for a PDF inbox management feature that allows users to store PDFs in a designated directory and automatically delete them after successful conversion to markdown.

## Problem Statement

Currently, users manually manage PDF files during the conversion process:
1. Upload PDFs to Colab or specify paths manually
2. Run conversion
3. Manually clean up source PDFs after successful conversion

This creates clutter and requires manual intervention. Users want:
- A dedicated "inbox" directory for PDFs awaiting processing
- Automatic cleanup of PDFs after successful conversion
- Clear visibility into processing status

## Feature Requirements

### Functional Requirements

1. **PDF Inbox Directory**
   - Configurable inbox path via CLI config (`pdf_inbox`)
   - Default location: `~/.kidkazz/inbox/` or project-relative `./inbox/`
   - Auto-create directory if it doesn't exist
   - Support both absolute and relative paths

2. **PDF Discovery**
   - Scan inbox directory for PDF files
   - Support recursive scanning (configurable)
   - Filter by file extension (`.pdf`, case-insensitive)
   - Track processing status per file

3. **Auto-Delete After Conversion**
   - Delete source PDF only after successful markdown generation
   - Verify markdown file exists and is non-empty before deletion
   - Support "move to processed" as alternative to deletion
   - Configurable behavior: `delete`, `move`, or `keep`

4. **Processing Status Tracking**
   - Track: pending, processing, completed, failed
   - Persist status across sessions (optional)
   - Provide CLI command to view inbox status

### Non-Functional Requirements

- Thread-safe file operations
- Graceful handling of file permission errors
- Clear error messages for common issues
- Minimal dependencies (use stdlib where possible)

## Architecture

### New Modules

```
src/
├── pdf_inbox/
│   ├── __init__.py
│   ├── manager.py      # PDFInboxManager class
│   ├── watcher.py      # File system watcher (future)
│   └── models.py       # Data models (PDFFile, ProcessingStatus)
```

### Key Classes

#### `PDFFile` (models.py)
```python
@dataclass
class PDFFile:
    path: Path
    name: str
    size: int
    created_at: datetime
    status: ProcessingStatus
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
```

#### `ProcessingStatus` (models.py)
```python
class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
```

#### `PostConversionAction` (models.py)
```python
class PostConversionAction(Enum):
    DELETE = "delete"      # Delete PDF after successful conversion
    MOVE = "move"          # Move to processed/ subdirectory
    KEEP = "keep"          # Keep original PDF
```

#### `PDFInboxManager` (manager.py)
```python
class PDFInboxManager:
    def __init__(
        self,
        inbox_path: Path,
        output_path: Path,
        post_action: PostConversionAction = PostConversionAction.DELETE,
        recursive: bool = False
    ):
        ...

    def scan(self) -> list[PDFFile]:
        """Scan inbox directory for PDF files."""
        ...

    def get_pending(self) -> list[PDFFile]:
        """Get all pending PDFs ready for processing."""
        ...

    def mark_processing(self, pdf_file: PDFFile) -> None:
        """Mark a PDF as currently being processed."""
        ...

    def mark_completed(
        self,
        pdf_file: PDFFile,
        output_path: Path
    ) -> None:
        """Mark conversion as complete and execute post-action."""
        ...

    def mark_failed(
        self,
        pdf_file: PDFFile,
        error: str
    ) -> None:
        """Mark conversion as failed."""
        ...

    def cleanup(self, pdf_file: PDFFile) -> bool:
        """Execute post-conversion action (delete/move/keep)."""
        ...

    def verify_output(self, output_path: Path) -> bool:
        """Verify markdown output exists and is valid."""
        ...
```

### CLI Integration

#### New Config Options
```toml
# ~/.config/kidkazz/config.toml
[inbox]
path = "~/.kidkazz/inbox"
recursive = false
post_action = "delete"  # delete | move | keep
processed_dir = "~/.kidkazz/processed"  # only used if post_action = "move"
```

#### New CLI Commands
```bash
# View inbox status
kidkazz inbox status

# List pending PDFs
kidkazz inbox list

# Process all pending PDFs (future: with local conversion)
kidkazz inbox process

# Clear processed/failed entries
kidkazz inbox clear [--failed] [--processed]
```

### Integration with Existing Workflow

#### Colab Notebook Updates
The Colab notebook will be updated to:
1. Accept inbox path as configurable location
2. Call `PDFInboxManager.mark_completed()` after successful conversion
3. Trigger auto-cleanup based on configured action

#### Local CLI Updates
The `ingest pdf` command will:
1. Use inbox manager to find PDFs
2. Show pending files before processing
3. Integrate with conversion workflow

## Implementation Plan

### Phase 1: Core Module (TDD)
1. Write tests for `PDFFile` and `ProcessingStatus` models
2. Write tests for `PDFInboxManager.scan()`
3. Write tests for `PDFInboxManager.cleanup()`
4. Implement to pass tests

### Phase 2: CLI Integration
1. Add inbox config options to `CLIConfig`
2. Implement `inbox` command group
3. Write integration tests

### Phase 3: Colab Integration
1. Update notebook to use inbox manager
2. Add auto-cleanup after conversion
3. Update documentation

## Test Strategy

### Unit Tests
- `test_pdf_inbox_models.py` - Data models
- `test_pdf_inbox_manager.py` - Manager class
- All tests written BEFORE implementation (TDD)

### Integration Tests
- `test_inbox_cli.py` - CLI commands
- `test_inbox_workflow.py` - End-to-end workflow

### Test Fixtures
- Temporary directories for inbox/output
- Mock PDF files (empty files with .pdf extension)
- Mock markdown output files

## Security Considerations

1. **Path Traversal**: Validate all paths are within allowed directories
2. **File Permissions**: Handle permission errors gracefully
3. **Symlink Handling**: Decide policy for symbolic links (follow or skip)
4. **Race Conditions**: Use atomic file operations where possible

## Future Enhancements

1. **File Watcher**: Real-time monitoring using `watchdog` library
2. **Batch Processing**: Process multiple PDFs in parallel
3. **Progress Tracking**: Rich progress bars for batch operations
4. **Notifications**: Desktop notifications on completion
5. **Web Dashboard**: Simple web UI for inbox management

## Acceptance Criteria

1. Users can configure an inbox directory via CLI config
2. PDFInboxManager correctly scans and lists PDF files
3. PDFs are automatically deleted after successful conversion
4. PDFs are preserved if conversion fails
5. Clear error messages for common failure cases
6. All tests pass (written before implementation)
7. Documentation updated
