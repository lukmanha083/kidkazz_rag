"""Tests for CLI progress indicators."""

import pytest

from src.cli.progress import (
    BatchTracker,
    IngestionTracker,
    create_detailed_progress,
    create_progress,
    ingestion_progress,
    spinner,
)


class TestCreateProgress:
    """Tests for progress bar creation."""

    def test_create_progress_returns_progress(self):
        """Test create_progress returns Progress instance."""
        progress = create_progress()
        assert progress is not None
        assert hasattr(progress, "add_task")

    def test_create_detailed_progress_returns_progress(self):
        """Test create_detailed_progress returns Progress instance."""
        progress = create_detailed_progress()
        assert progress is not None
        assert hasattr(progress, "add_task")


class TestIngestionProgress:
    """Tests for ingestion progress context manager."""

    def test_context_manager_yields_progress(self):
        """Test context manager yields Progress instance."""
        with ingestion_progress() as progress:
            assert progress is not None
            assert hasattr(progress, "add_task")

    def test_can_add_and_update_task(self):
        """Test adding and updating tasks."""
        with ingestion_progress() as progress:
            task_id = progress.add_task("Test task", total=100)
            progress.update(task_id, advance=50)
            # Should not raise


class TestSpinner:
    """Tests for spinner context manager."""

    def test_spinner_context_manager(self):
        """Test spinner context manager works."""
        with spinner("Loading"):
            pass  # Just verify no error


class TestIngestionTracker:
    """Tests for IngestionTracker class."""

    def test_add_stage(self):
        """Test adding a stage."""
        with ingestion_progress() as progress:
            tracker = IngestionTracker(progress)
            task_id = tracker.add_stage("Processing", total=100)
            assert task_id is not None
            assert "Processing" in tracker.tasks

    def test_update_stage(self):
        """Test updating a stage."""
        with ingestion_progress() as progress:
            tracker = IngestionTracker(progress)
            tracker.add_stage("Processing", total=100)
            tracker.update("Processing", advance=25)
            # Should not raise

    def test_set_progress(self):
        """Test setting absolute progress."""
        with ingestion_progress() as progress:
            tracker = IngestionTracker(progress)
            tracker.add_stage("Processing", total=100)
            tracker.set_progress("Processing", completed=50)
            # Should not raise

    def test_complete_stage(self):
        """Test completing a stage."""
        with ingestion_progress() as progress:
            tracker = IngestionTracker(progress)
            tracker.add_stage("Processing", total=100)
            tracker.complete("Processing")
            # Should not raise

    def test_update_nonexistent_stage(self):
        """Test updating nonexistent stage doesn't raise."""
        with ingestion_progress() as progress:
            tracker = IngestionTracker(progress)
            tracker.update("Nonexistent")  # Should not raise

    def test_update_description(self):
        """Test updating stage description."""
        with ingestion_progress() as progress:
            tracker = IngestionTracker(progress)
            tracker.add_stage("Processing", total=100)
            tracker.update_description("Processing", "Almost done...")
            # Should not raise


class TestBatchTracker:
    """Tests for BatchTracker class."""

    def test_context_manager(self):
        """Test BatchTracker as context manager."""
        with BatchTracker(10, "Testing") as tracker:
            assert tracker.total == 10
            assert tracker.processed == 0

    def test_advance(self):
        """Test advancing progress."""
        with BatchTracker(10) as tracker:
            tracker.advance()
            assert tracker.processed == 1

    def test_advance_with_error(self):
        """Test advancing with error."""
        with BatchTracker(10) as tracker:
            tracker.advance(error="Something failed")
            assert tracker.processed == 1
            assert len(tracker.errors) == 1
            assert "Something failed" in tracker.errors

    def test_get_summary(self):
        """Test getting summary."""
        with BatchTracker(10) as tracker:
            tracker.advance()
            tracker.advance()
            tracker.advance(error="Error 1")

            summary = tracker.get_summary()
            assert summary["total"] == 10
            assert summary["processed"] == 3
            assert summary["successful"] == 2
            assert summary["errors"] == 1
            assert len(summary["error_messages"]) == 1

    def test_multiple_errors(self):
        """Test tracking multiple errors."""
        with BatchTracker(5) as tracker:
            tracker.advance(error="Error 1")
            tracker.advance(error="Error 2")
            tracker.advance()

            summary = tracker.get_summary()
            assert summary["errors"] == 2
            assert len(summary["error_messages"]) == 2
