"""Progress indicators for CLI operations.

Provides visual feedback for long-running operations.
"""

from contextlib import contextmanager
from typing import Generator, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()


def create_progress() -> Progress:
    """Create a standard progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def create_detailed_progress() -> Progress:
    """Create a detailed progress bar with time remaining."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


@contextmanager
def ingestion_progress() -> Generator[Progress, None, None]:
    """Context manager for ingestion progress bar."""
    progress = create_progress()
    with progress:
        yield progress


@contextmanager
def spinner(message: str) -> Generator[None, None, None]:
    """Context manager for simple spinner."""
    with console.status(f"[bold blue]{message}..."):
        yield


class IngestionTracker:
    """Track multi-stage ingestion progress."""

    def __init__(self, progress: Progress):
        """Initialize tracker with progress instance."""
        self.progress = progress
        self.tasks: dict[str, int] = {}

    def add_stage(self, name: str, total: int = 100) -> int:
        """Add a new stage to track."""
        task_id = self.progress.add_task(name, total=total)
        self.tasks[name] = task_id
        return task_id

    def update(self, name: str, advance: int = 1) -> None:
        """Update stage progress."""
        if name in self.tasks:
            self.progress.update(self.tasks[name], advance=advance)

    def set_progress(self, name: str, completed: int) -> None:
        """Set absolute progress for a stage."""
        if name in self.tasks:
            self.progress.update(self.tasks[name], completed=completed)

    def complete(self, name: str) -> None:
        """Mark stage as complete."""
        if name in self.tasks:
            task = self.progress.tasks[self.tasks[name]]
            self.progress.update(self.tasks[name], completed=task.total)

    def update_description(self, name: str, description: str) -> None:
        """Update the description of a stage."""
        if name in self.tasks:
            self.progress.update(self.tasks[name], description=description)


class BatchTracker:
    """Track batch processing progress."""

    def __init__(self, total: int, description: str = "Processing"):
        """Initialize batch tracker."""
        self.total = total
        self.description = description
        self.processed = 0
        self.errors: list[str] = []
        self.progress: Optional[Progress] = None
        self.task_id: Optional[int] = None

    def __enter__(self) -> "BatchTracker":
        """Enter context manager."""
        self.progress = create_progress()
        self.progress.start()
        self.task_id = self.progress.add_task(self.description, total=self.total)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        if self.progress:
            self.progress.stop()

    def advance(self, error: Optional[str] = None) -> None:
        """Advance progress by one item."""
        self.processed += 1
        if error:
            self.errors.append(error)
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, advance=1)

    def get_summary(self) -> dict:
        """Get processing summary."""
        return {
            "total": self.total,
            "processed": self.processed,
            "successful": self.processed - len(self.errors),
            "errors": len(self.errors),
            "error_messages": self.errors,
        }
