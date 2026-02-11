"""Tests for CLI output formatting."""

from io import StringIO
from unittest.mock import patch

import pytest

from src.cli.output import (
    _build_content_flags,
    _shorten_doc_id,
    console,
    print_chunks_preview,
    print_db_status,
    print_document_list,
    print_document_stats,
    print_error,
    print_info,
    print_ingestion_summary,
    print_json,
    print_search_results,
    print_success,
    print_warning,
)


class TestPrintFunctions:
    """Tests for basic print functions."""

    def test_print_success_runs(self):
        """Test print_success executes without error."""
        # Just verify it doesn't raise
        print_success("Test message")

    def test_print_error_runs(self):
        """Test print_error executes without error."""
        print_error("Test error")

    def test_print_warning_runs(self):
        """Test print_warning executes without error."""
        print_warning("Test warning")

    def test_print_info_runs(self):
        """Test print_info executes without error."""
        print_info("Test info")


class TestHelperFunctions:
    """Tests for _shorten_doc_id and _build_content_flags helpers."""

    def test_shorten_doc_id_strips_z_library(self):
        """Test _shorten_doc_id strips _Z-Library suffix."""
        result = _shorten_doc_id("Inventory_Accounting_A_Comprehensive_Guide_Steven_M_Bragg_Z-Library")
        assert "Z-Library" not in result
        assert result.startswith("Inventory Accounting")

    def test_shorten_doc_id_replaces_underscores(self):
        """Test _shorten_doc_id replaces underscores with spaces."""
        result = _shorten_doc_id("My_Great_Book")
        assert result == "My Great Book"

    def test_shorten_doc_id_truncates_long(self):
        """Test _shorten_doc_id truncates long names."""
        result = _shorten_doc_id("A" * 100, max_len=20)
        assert len(result) == 20
        assert result.endswith("\u2026")

    def test_shorten_doc_id_empty(self):
        """Test _shorten_doc_id handles empty string."""
        assert _shorten_doc_id("") == ""

    def test_shorten_doc_id_none(self):
        """Test _shorten_doc_id handles None."""
        assert _shorten_doc_id(None) == ""

    def test_build_content_flags_all(self):
        """Test _build_content_flags with all flags set."""
        result = _build_content_flags({"has_table": True, "has_code": True, "has_math": True})
        assert "[TABLE]" in result
        assert "[CODE]" in result
        assert "[MATH]" in result

    def test_build_content_flags_none(self):
        """Test _build_content_flags with no flags."""
        result = _build_content_flags({"has_table": False, "has_code": False, "has_math": False})
        assert result == ""

    def test_build_content_flags_partial(self):
        """Test _build_content_flags with some flags."""
        result = _build_content_flags({"has_table": True})
        assert result == "[TABLE]"

    def test_build_content_flags_missing_keys(self):
        """Test _build_content_flags with missing keys."""
        result = _build_content_flags({})
        assert result == ""


class TestPrintSearchResults:
    """Tests for search results formatting."""

    def test_empty_results(self):
        """Test handling empty results."""
        # Should not raise
        print_search_results([])

    def test_single_result(self):
        """Test formatting single result."""
        results = [
            {
                "chunk_id": "test_chunk_1",
                "content": "Test content",
                "level": 2,
                "semantic_type": "definition",
                "similarity_score": 0.85,
            }
        ]
        print_search_results(results)

    def test_multiple_results(self):
        """Test formatting multiple results."""
        results = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"Content {i}",
                "level": 2,
                "semantic_type": "narrative",
                "similarity_score": 0.9 - i * 0.1,
            }
            for i in range(3)
        ]
        print_search_results(results)

    def test_result_with_context(self):
        """Test formatting result with context."""
        results = [
            {
                "chunk_id": "test_chunk",
                "content": "Main content",
                "level": 2,
                "semantic_type": "example",
                "similarity_score": 0.75,
                "context": ["Before content", "After content"],
            }
        ]
        print_search_results(results, show_context=True)

    def test_long_content_shown_by_default(self):
        """Test long content is shown in full (up to 2000 chars) by default."""
        long_content = "x" * 500
        results = [
            {
                "chunk_id": "long_chunk",
                "content": long_content,
                "level": 1,
                "semantic_type": "narrative",
                "similarity_score": 0.5,
            }
        ]
        buf = StringIO()
        with console.capture() as capture:
            print_search_results(results)
        output = capture.get()
        # 500 chars should NOT be truncated (default limit is 2000)
        assert "..." not in output or output.count("x") >= 490

    def test_compact_mode_truncation(self):
        """Test compact mode truncates to 300 chars."""
        long_content = "x" * 500
        results = [
            {
                "chunk_id": "long_chunk",
                "content": long_content,
                "level": 1,
                "semantic_type": "narrative",
                "similarity_score": 0.5,
            }
        ]
        with console.capture() as capture:
            print_search_results(results, compact=True)
        output = capture.get()
        # Should be truncated — full 500 x's should NOT appear
        assert output.count("x") < 500

    def test_result_with_citation(self):
        """Test citation metadata is displayed."""
        results = [
            {
                "chunk_id": "test_chunk",
                "content": "Test content",
                "level": 2,
                "semantic_type": "definition",
                "similarity_score": 0.9,
                "document_id": "My_Book_Z-Library",
                "section_path": ["Chapter 1", "Section A"],
                "header_text": "Understanding Costs",
                "topic_tags": ["cost", "overhead"],
                "has_table": True,
                "has_code": False,
                "has_math": False,
            }
        ]
        with console.capture() as capture:
            print_search_results(results)
        output = capture.get()
        assert "My Book" in output
        assert "Chapter 1" in output
        assert "Section A" in output
        assert "Understanding Costs" in output
        assert "cost" in output
        assert "[TABLE]" in output

    def test_result_with_content_flags_in_subtitle(self):
        """Test content flags appear in panel subtitle."""
        results = [
            {
                "chunk_id": "test_chunk",
                "content": "Test",
                "level": 2,
                "semantic_type": "narrative",
                "similarity_score": 0.5,
                "has_table": False,
                "has_code": True,
                "has_math": True,
            }
        ]
        with console.capture() as capture:
            print_search_results(results)
        output = capture.get()
        assert "[CODE]" in output
        assert "[MATH]" in output

    def test_rich_markup_in_content_escaped(self):
        """Test that Rich markup in content does not crash."""
        results = [
            {
                "chunk_id": "markup_chunk",
                "content": "[bold]This has [red]markup[/red][/bold]",
                "level": 2,
                "semantic_type": "narrative",
                "similarity_score": 0.5,
            }
        ]
        # Should not raise
        print_search_results(results)


class TestPrintDocumentList:
    """Tests for document list formatting."""

    def test_empty_document_list(self):
        """Test handling empty document list."""
        print_document_list([])

    def test_single_document(self):
        """Test formatting single document."""
        docs = [
            {
                "doc_id": "test_doc",
                "title": "Test Document",
                "chunk_count": 42,
                "created_at": "2024-01-15T10:30:00",
            }
        ]
        print_document_list(docs)

    def test_multiple_documents(self):
        """Test formatting multiple documents."""
        docs = [
            {
                "doc_id": f"doc_{i}",
                "title": f"Document {i}",
                "chunk_count": i * 10,
                "created_at": f"2024-01-{i+1:02d}",
            }
            for i in range(3)
        ]
        print_document_list(docs)


class TestPrintDocumentStats:
    """Tests for document statistics formatting."""

    def test_basic_stats(self):
        """Test basic statistics display."""
        stats = {
            "doc_id": "test_doc",
            "title": "Test Document",
            "total_chunks": 100,
            "total_tokens": 50000,
        }
        print_document_stats(stats)

    def test_stats_with_levels(self):
        """Test stats with level breakdown."""
        stats = {
            "doc_id": "test_doc",
            "title": "Test Document",
            "total_chunks": 50,
            "total_tokens": 25000,
            "chunks_by_level": {"1": 10, "2": 40},
        }
        print_document_stats(stats)

    def test_stats_with_types(self):
        """Test stats with type breakdown."""
        stats = {
            "doc_id": "test_doc",
            "title": "Test Document",
            "total_chunks": 100,
            "total_tokens": 50000,
            "chunks_by_type": {
                "narrative": 50,
                "definition": 30,
                "example": 20,
            },
        }
        print_document_stats(stats)


class TestPrintDbStatus:
    """Tests for database status formatting."""

    def test_connected_status(self):
        """Test connected status display."""
        status = {
            "connected": True,
            "store_type": "mock",
            "documents": 5,
            "chunks": 250,
        }
        print_db_status(status)

    def test_disconnected_status(self):
        """Test disconnected status display."""
        status = {
            "connected": False,
            "store_type": "helix",
            "documents": 0,
            "chunks": 0,
            "port": 6969,
        }
        print_db_status(status)


class TestPrintJson:
    """Tests for JSON output."""

    def test_simple_dict(self):
        """Test printing simple dict as JSON."""
        print_json({"key": "value"})

    def test_nested_dict(self):
        """Test printing nested dict as JSON."""
        print_json({
            "level1": {
                "level2": {
                    "key": "value"
                }
            }
        })

    def test_list_of_dicts(self):
        """Test printing list of dicts as JSON."""
        print_json([
            {"id": 1, "name": "first"},
            {"id": 2, "name": "second"},
        ])


class TestPrintChunksPreview:
    """Tests for chunks preview formatting."""

    def test_empty_chunks(self):
        """Test handling empty chunks list."""
        print_chunks_preview([])

    def test_few_chunks(self):
        """Test preview with few chunks."""

        class MockChunk:
            def __init__(self, id, content, level):
                self.id = id
                self.content = content
                self.level = level

        chunks = [MockChunk(f"chunk_{i}", f"Content {i}", 2) for i in range(3)]
        print_chunks_preview(chunks)

    def test_many_chunks_truncated(self):
        """Test preview truncates many chunks."""

        class MockChunk:
            def __init__(self, id, content, level):
                self.id = id
                self.content = content
                self.level = level

        chunks = [MockChunk(f"chunk_{i}", f"Content {i}", 2) for i in range(10)]
        print_chunks_preview(chunks, max_display=5)


class TestPrintIngestionSummary:
    """Tests for ingestion summary formatting."""

    def test_basic_summary(self):
        """Test basic ingestion summary."""
        print_ingestion_summary(
            doc_id="test_doc",
            title="Test Document",
            chunk_count=42,
            source_file="test.md",
        )
