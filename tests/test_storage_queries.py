"""Tests for HelixQL query classes."""

import pytest

from src.storage.queries import (
    AddChunkRelationship,
    AddChunkWithEmbedding,
    AddDocument,
    DeleteDocument,
    GetChunk,
    GetChunkWithContext,
    GetDocumentChunks,
    ListDocuments,
    QueryResult,
    SearchKeyword,
    SearchSimilarChunks,
)


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_success_result(self):
        """Should create success result."""
        result = QueryResult(success=True, data={"key": "value"})

        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_error_result(self):
        """Should create error result."""
        result = QueryResult(success=False, error="Something went wrong")

        assert result.success is False
        assert result.error == "Something went wrong"


class TestAddDocument:
    """Tests for AddDocument query."""

    def test_creates_query_payload(self):
        """Should create valid query payload."""
        query = AddDocument("doc_1", "Test Title", 10)
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["operation"] == "add_node"
        assert payload[0]["node_type"] == "Document"
        assert payload[0]["properties"]["doc_id"] == "doc_1"
        assert payload[0]["properties"]["title"] == "Test Title"
        assert payload[0]["properties"]["chunk_count"] == 10

    def test_auto_generates_timestamp(self):
        """Should auto-generate created_at timestamp."""
        query = AddDocument("doc_1", "Title", 5)
        payload = query.query()

        assert "created_at" in payload[0]["properties"]
        assert payload[0]["properties"]["created_at"] > 0

    def test_response_processing(self):
        """Should process response correctly."""
        query = AddDocument("doc_1", "Title", 5)
        result = query.response({"status": "ok"})

        assert result.success is True
        assert result.data["doc_id"] == "doc_1"


class TestDeleteDocument:
    """Tests for DeleteDocument query."""

    def test_creates_delete_payload(self):
        """Should create delete cascade payload."""
        query = DeleteDocument("doc_1")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["operation"] == "delete_cascade"
        assert payload[0]["start_node"] == "Document"
        assert payload[0]["filter"]["doc_id"] == "doc_1"


class TestAddChunkWithEmbedding:
    """Tests for AddChunkWithEmbedding query."""

    def test_creates_multi_operation_payload(self):
        """Should create payload with multiple operations."""
        chunk_props = {
            "chunk_id": "chunk_1",
            "content": "Test content",
            "level": 2,
        }
        embedding = [0.1, 0.2, 0.3]

        query = AddChunkWithEmbedding(chunk_props, embedding, "doc_1")
        payload = query.query()

        # Should have 4 operations: add chunk, add vector, add edges
        assert len(payload) == 4

        # Check chunk node addition
        assert payload[0]["operation"] == "add_node"
        assert payload[0]["node_type"] == "Chunk"

        # Check vector addition
        assert payload[1]["operation"] == "add_vector"
        assert payload[1]["properties"]["embedding"] == [0.1, 0.2, 0.3]

    def test_response_returns_chunk_id(self):
        """Should return chunk_id in response."""
        query = AddChunkWithEmbedding({"chunk_id": "test_1"}, [], "doc_1")
        result = query.response({})

        assert result.data["chunk_id"] == "test_1"


class TestAddChunkRelationship:
    """Tests for AddChunkRelationship query."""

    def test_creates_edge_payload(self):
        """Should create add_edge payload."""
        query = AddChunkRelationship("ParentOf", "parent_1", "child_1")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["operation"] == "add_edge"
        assert payload[0]["edge_type"] == "ParentOf"
        assert payload[0]["from"]["filter"]["chunk_id"] == "parent_1"
        assert payload[0]["to"]["filter"]["chunk_id"] == "child_1"


class TestGetChunk:
    """Tests for GetChunk query."""

    def test_creates_get_payload(self):
        """Should create get_node payload."""
        query = GetChunk("chunk_1")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["operation"] == "get_node"
        assert payload[0]["filter"]["chunk_id"] == "chunk_1"
        assert payload[0]["include_vector"] is True

    def test_response_found(self):
        """Should return success when chunk found."""
        query = GetChunk("chunk_1")
        result = query.response({"node": {"chunk_id": "chunk_1"}})

        assert result.success is True
        assert result.data["node"]["chunk_id"] == "chunk_1"

    def test_response_not_found(self):
        """Should return failure when chunk not found."""
        query = GetChunk("chunk_1")
        result = query.response(None)

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()


class TestGetChunkWithContext:
    """Tests for GetChunkWithContext query."""

    def test_creates_traverse_payload(self):
        """Should create traverse payload with multiple paths."""
        query = GetChunkWithContext("chunk_1")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["operation"] == "traverse"
        assert payload[0]["start"]["filter"]["chunk_id"] == "chunk_1"
        assert len(payload[0]["paths"]) == 4  # parent, siblings, next, prev


class TestSearchSimilarChunks:
    """Tests for SearchSimilarChunks query."""

    def test_creates_vector_search_payload(self):
        """Should create vector_search payload."""
        embedding = [0.1] * 384
        query = SearchSimilarChunks(embedding, top_k=5)
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["operation"] == "vector_search"
        assert payload[0]["vector"] == embedding
        assert payload[0]["top_k"] == 5

    def test_includes_filters(self):
        """Should include optional filters."""
        query = SearchSimilarChunks(
            [0.1] * 384,
            top_k=10,
            doc_id="doc_1",
            level=2,
            semantic_type="definition",
            threshold=0.5,
        )
        payload = query.query()

        assert payload[0]["filters"]["doc_id"] == "doc_1"
        assert payload[0]["filters"]["level"] == 2
        assert payload[0]["filters"]["semantic_type"] == "definition"
        assert payload[0]["threshold"] == 0.5

    def test_omits_empty_filters(self):
        """Should omit filters that are None."""
        query = SearchSimilarChunks([0.1] * 384, top_k=5)
        payload = query.query()

        # Filters should be empty dict
        assert payload[0]["filters"] == {}


class TestSearchKeyword:
    """Tests for SearchKeyword query."""

    def test_creates_keyword_search_payload(self):
        """Should create keyword_search payload."""
        query = SearchKeyword("test term")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["operation"] == "keyword_search"
        assert payload[0]["keyword"] == "test term"
        assert payload[0]["field"] == "content"

    def test_includes_doc_filter(self):
        """Should include document filter."""
        query = SearchKeyword("test", doc_id="doc_1")
        payload = query.query()

        assert payload[0]["filters"]["doc_id"] == "doc_1"


class TestGetDocumentChunks:
    """Tests for GetDocumentChunks query."""

    def test_creates_traverse_payload(self):
        """Should create traverse payload for document chunks."""
        query = GetDocumentChunks("doc_1")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["operation"] == "traverse"
        assert payload[0]["start"]["filter"]["doc_id"] == "doc_1"

    def test_includes_level_filter(self):
        """Should include level filter."""
        query = GetDocumentChunks("doc_1", level=2)
        payload = query.query()

        assert payload[0]["filters"]["level"] == 2


class TestListDocuments:
    """Tests for ListDocuments query."""

    def test_creates_get_all_payload(self):
        """Should create get_all payload."""
        query = ListDocuments()
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["operation"] == "get_all"
        assert payload[0]["node_type"] == "Document"
