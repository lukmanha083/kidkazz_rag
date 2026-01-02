"""Tests for MCP response formatters."""

import pytest

from src.chunker import Chunk, EmbeddedChunk
from src.mcp_server.formatters import (
    format_chunk,
    format_chunk_list,
    format_document,
    format_document_stats,
    format_optional_chunk,
    format_search_result,
    format_search_results,
)


@pytest.fixture
def sample_chunk():
    """Create a sample chunk."""
    return Chunk(
        id="test_l2_1",
        content="This is test content",
        level=2,
        token_count=10,
        parent_id="test_l1_1",
        child_ids=[],
        prev_id="test_l2_0",
        next_id="test_l2_2",
        section_path=["Chapter 1", "Section 1"],
        source_section="1.1 Introduction",
    )


@pytest.fixture
def sample_embedded_chunk(sample_chunk):
    """Create a sample embedded chunk."""
    return EmbeddedChunk(
        chunk=sample_chunk,
        embedding=[0.1] * 384,
        model_name="test-model",
    )


class TestFormatChunk:
    """Tests for format_chunk function."""

    def test_basic_fields(self, sample_embedded_chunk):
        """Test basic chunk fields are included."""
        result = format_chunk(sample_embedded_chunk)
        assert result["chunk_id"] == "test_l2_1"
        assert result["content"] == "This is test content"
        assert result["level"] == 2
        assert result["token_count"] == 10

    def test_word_count(self, sample_embedded_chunk):
        """Test word_count is included (computed from content)."""
        result = format_chunk(sample_embedded_chunk)
        # "This is test content" = 4 words
        assert result["word_count"] == 4

    def test_relationship_fields(self, sample_embedded_chunk):
        """Test relationship fields are included."""
        result = format_chunk(sample_embedded_chunk)
        assert result["parent_id"] == "test_l1_1"
        assert result["child_ids"] == []
        assert result["prev_id"] == "test_l2_0"
        assert result["next_id"] == "test_l2_2"

    def test_section_fields(self, sample_embedded_chunk):
        """Test section fields are included."""
        result = format_chunk(sample_embedded_chunk)
        assert result["section_path"] == ["Chapter 1", "Section 1"]
        assert result["source_section"] == "1.1 Introduction"

    def test_embedding_fields(self, sample_embedded_chunk):
        """Test embedding metadata fields."""
        result = format_chunk(sample_embedded_chunk)
        assert result["model_name"] == "test-model"
        assert result["embedding_dim"] == 384

    def test_embedding_not_included(self, sample_embedded_chunk):
        """Test actual embedding vector is not included."""
        result = format_chunk(sample_embedded_chunk)
        assert "embedding" not in result


class TestFormatSearchResult:
    """Tests for format_search_result function."""

    def test_includes_chunk_fields(self, sample_embedded_chunk):
        """Test search result includes all chunk fields."""
        result = format_search_result(sample_embedded_chunk, 0.85)
        assert result["chunk_id"] == "test_l2_1"
        assert result["content"] == "This is test content"

    def test_includes_similarity_score(self, sample_embedded_chunk):
        """Test similarity score is included."""
        result = format_search_result(sample_embedded_chunk, 0.85)
        assert result["similarity_score"] == 0.85

    def test_score_rounding(self, sample_embedded_chunk):
        """Test similarity score is rounded to 4 decimals."""
        result = format_search_result(sample_embedded_chunk, 0.123456789)
        assert result["similarity_score"] == 0.1235

    def test_zero_score(self, sample_embedded_chunk):
        """Test zero similarity score."""
        result = format_search_result(sample_embedded_chunk, 0.0)
        assert result["similarity_score"] == 0.0

    def test_perfect_score(self, sample_embedded_chunk):
        """Test perfect similarity score."""
        result = format_search_result(sample_embedded_chunk, 1.0)
        assert result["similarity_score"] == 1.0


class TestFormatDocument:
    """Tests for format_document function."""

    def test_all_fields(self):
        """Test all document fields are formatted."""
        doc = {
            "doc_id": "textbook",
            "title": "Introduction to ML",
            "chunk_count": 42,
            "created_at": "2024-01-01T00:00:00",
        }
        result = format_document(doc)
        assert result["doc_id"] == "textbook"
        assert result["title"] == "Introduction to ML"
        assert result["chunk_count"] == 42
        assert result["created_at"] == "2024-01-01T00:00:00"

    def test_missing_fields(self):
        """Test missing fields use defaults."""
        doc = {}
        result = format_document(doc)
        assert result["doc_id"] == ""
        assert result["title"] == ""
        assert result["chunk_count"] == 0
        assert result["created_at"] is None


class TestFormatDocumentStats:
    """Tests for format_document_stats function."""

    def test_all_fields(self):
        """Test all stats fields are formatted."""
        stats = {
            "doc_id": "textbook",
            "title": "ML Book",
            "total_chunks": 100,
            "total_tokens": 5000,
            "level_counts": {"1": 10, "2": 90},
            "type_counts": {"definition": 20, "example": 30},
        }
        result = format_document_stats(stats)
        assert result["doc_id"] == "textbook"
        assert result["title"] == "ML Book"
        assert result["total_chunks"] == 100
        assert result["total_tokens"] == 5000
        assert result["chunks_by_level"] == {"1": 10, "2": 90}
        assert result["chunks_by_type"] == {"definition": 20, "example": 30}

    def test_missing_fields(self):
        """Test missing fields use defaults."""
        stats = {}
        result = format_document_stats(stats)
        assert result["doc_id"] == ""
        assert result["title"] == ""
        assert result["total_chunks"] == 0
        assert result["total_tokens"] == 0
        assert result["chunks_by_level"] == {}
        assert result["chunks_by_type"] == {}


class TestFormatChunkList:
    """Tests for format_chunk_list function."""

    def test_empty_list(self):
        """Test formatting empty list."""
        result = format_chunk_list([])
        assert result == []

    def test_multiple_chunks(self, sample_embedded_chunk):
        """Test formatting multiple chunks."""
        result = format_chunk_list([sample_embedded_chunk, sample_embedded_chunk])
        assert len(result) == 2
        assert all("chunk_id" in r for r in result)


class TestFormatSearchResults:
    """Tests for format_search_results function."""

    def test_empty_results(self):
        """Test formatting empty results."""
        result = format_search_results([])
        assert result == []

    def test_multiple_results(self, sample_embedded_chunk):
        """Test formatting multiple search results."""
        results = [
            (sample_embedded_chunk, 0.9),
            (sample_embedded_chunk, 0.8),
        ]
        formatted = format_search_results(results)
        assert len(formatted) == 2
        assert formatted[0]["similarity_score"] == 0.9
        assert formatted[1]["similarity_score"] == 0.8


class TestFormatOptionalChunk:
    """Tests for format_optional_chunk function."""

    def test_none_input(self):
        """Test None input returns None."""
        result = format_optional_chunk(None)
        assert result is None

    def test_chunk_input(self, sample_embedded_chunk):
        """Test valid chunk input is formatted."""
        result = format_optional_chunk(sample_embedded_chunk)
        assert result is not None
        assert result["chunk_id"] == "test_l2_1"
