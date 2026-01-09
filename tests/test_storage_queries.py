"""Tests for HelixQL query classes."""

import pytest

from src.storage.queries import (
    AddChunk,
    AddChunkRelationship,
    AddChunkVector,
    AddChunkWithEmbedding,
    AddDocument,
    AddNextSibling,
    AddParentChild,
    DeleteDocument,
    DropChunk,
    DropDocument,
    GetChunk,
    GetChunksByParentId,
    GetChunkWithContext,
    GetDocumentByDocId,
    GetDocumentChunks,
    LinkChunkVector,
    LinkDocumentChunk,
    ListDocuments,
    QueryResult,
    SearchKeyword,
    SearchSimilarChunks,
    UpdateChunkContent,
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
        """Should create valid query payload matching HelixQL signature."""
        query = AddDocument("doc_1", "Test Title", 10)
        payload = query.query()

        # helix-py expects list of parameter dicts
        assert len(payload) == 1
        assert payload[0]["doc_id"] == "doc_1"
        assert payload[0]["title"] == "Test Title"
        assert payload[0]["chunk_count"] == 10

    def test_auto_generates_timestamp(self):
        """Should auto-generate created_at timestamp."""
        query = AddDocument("doc_1", "Title", 5)
        payload = query.query()

        assert "created_at" in payload[0]
        assert payload[0]["created_at"] > 0

    def test_tags_normalized_and_serialized(self):
        """Should normalize tags and serialize as JSON."""
        query = AddDocument("doc_1", "Title", 5, tags=["INVENTORY", "  accounting  "])
        payload = query.query()

        # Tags should be lowercase, trimmed, and JSON-serialized
        import json
        tags = json.loads(payload[0]["tags"])
        assert tags == ["inventory", "accounting"]

    def test_response_processing(self):
        """Should process response correctly."""
        query = AddDocument("doc_1", "Title", 5)
        result = query.response({"status": "ok"})

        assert result.success is True
        assert result.data["doc_id"] == "doc_1"


class TestDeleteDocument:
    """Tests for DeleteDocument query."""

    def test_creates_delete_payload(self):
        """Should create HelixQL delete parameters."""
        query = DeleteDocument("doc_1")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["doc_id"] == "doc_1"
        assert query.endpoint == "DeleteDocumentByDocId"

    def test_response(self):
        """Should return doc_id in response."""
        query = DeleteDocument("doc_1")
        result = query.response({})

        assert result.success is True
        assert result.data["doc_id"] == "doc_1"


class TestAddChunk:
    """Tests for AddChunk query (maps to HelixQL AddChunk)."""

    def test_creates_query_payload(self):
        """Should create payload matching HelixQL AddChunk signature."""
        chunk_props = {
            "chunk_id": "chunk_1",
            "content": "Test content",
            "level": 2,
            "document_id": "doc_1",
        }

        query = AddChunk(chunk_props)
        payload = query.query()

        # helix-py expects list of parameter dicts
        assert len(payload) == 1
        assert payload[0]["chunk_id"] == "chunk_1"
        assert payload[0]["content"] == "Test content"
        assert payload[0]["level"] == 2
        assert payload[0]["document_id"] == "doc_1"

    def test_default_values(self):
        """Should provide default values for missing properties."""
        query = AddChunk({"chunk_id": "test"})
        payload = query.query()

        assert payload[0]["content"] == ""
        assert payload[0]["level"] == 0
        assert payload[0]["token_count"] == 0
        assert payload[0]["topic_tags"] == "[]"

    def test_response_returns_chunk_id(self):
        """Should return chunk_id in response."""
        query = AddChunk({"chunk_id": "test_1"})
        result = query.response({})

        assert result.data["chunk_id"] == "test_1"


class TestAddChunkWithEmbedding:
    """Tests for AddChunkWithEmbedding (legacy alias for AddChunk)."""

    def test_creates_chunk_payload(self):
        """Should create payload for chunk (embedding stored separately)."""
        chunk_props = {
            "chunk_id": "chunk_1",
            "content": "Test content",
            "level": 2,
        }
        embedding = [0.1, 0.2, 0.3]

        query = AddChunkWithEmbedding(chunk_props, embedding, "doc_1")
        payload = query.query()

        # Returns chunk parameters only (embedding handled separately)
        assert len(payload) == 1
        assert payload[0]["chunk_id"] == "chunk_1"
        assert payload[0]["content"] == "Test content"
        # Embedding is stored but not in query payload
        assert query.embedding == [0.1, 0.2, 0.3]
        assert query.doc_id == "doc_1"

    def test_response_returns_chunk_id(self):
        """Should return chunk_id in response."""
        query = AddChunkWithEmbedding({"chunk_id": "test_1"}, [], "doc_1")
        result = query.response({})

        assert result.data["chunk_id"] == "test_1"


class TestLinkDocumentChunk:
    """Tests for LinkDocumentChunk query."""

    def test_creates_link_payload(self):
        """Should create payload for linking document to chunk."""
        query = LinkDocumentChunk("doc_internal_123", "chunk_internal_456")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["doc_id"] == "doc_internal_123"
        assert payload[0]["chunk_id"] == "chunk_internal_456"


class TestAddParentChild:
    """Tests for AddParentChild query."""

    def test_creates_parent_child_payload(self):
        """Should create payload for parent-child relationship."""
        query = AddParentChild("parent_id_123", "child_id_456")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["parent_id"] == "parent_id_123"
        assert payload[0]["child_id"] == "child_id_456"


class TestAddNextSibling:
    """Tests for AddNextSibling query."""

    def test_creates_next_sibling_payload(self):
        """Should create payload for next sibling relationship."""
        query = AddNextSibling("chunk_id_123", "next_id_456")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["chunk_id"] == "chunk_id_123"
        assert payload[0]["next_id"] == "next_id_456"


class TestAddChunkRelationship:
    """Tests for AddChunkRelationship query (legacy wrapper)."""

    def test_parent_of_creates_correct_payload(self):
        """Should route ParentOf to AddParentChild parameters."""
        query = AddChunkRelationship("ParentOf", "parent_1", "child_1")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["parent_id"] == "parent_1"
        assert payload[0]["child_id"] == "child_1"
        assert query.endpoint == "AddParentChild"

    def test_next_sibling_creates_correct_payload(self):
        """Should route NextSibling to AddNextSibling parameters."""
        query = AddChunkRelationship("NextSibling", "chunk_a", "chunk_b")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["chunk_id"] == "chunk_a"
        assert payload[0]["next_id"] == "chunk_b"
        assert query.endpoint == "AddNextSibling"

    def test_unknown_edge_type_fallback(self):
        """Should use generic format for unknown edge types."""
        query = AddChunkRelationship("SiblingOf", "id_a", "id_b")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["from_id"] == "id_a"
        assert payload[0]["to_id"] == "id_b"
        assert query.endpoint == "AddSiblingOf"


class TestGetChunk:
    """Tests for GetChunk query."""

    def test_creates_get_payload(self):
        """Should create HelixQL parameters."""
        query = GetChunk("chunk_1")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["chunk_id"] == "chunk_1"
        assert query.endpoint == "GetChunkByChunkId"

    def test_response_found(self):
        """Should return success when chunk found."""
        query = GetChunk("chunk_1")
        # New format: HelixQL returns list of nodes
        result = query.response([{"chunk_id": "chunk_1", "content": "test"}])

        assert result.success is True
        assert result.data["node"]["chunk_id"] == "chunk_1"

    def test_response_not_found(self):
        """Should return failure when no results."""
        query = GetChunk("chunk_1")
        result = query.response([])

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_response_none(self):
        """Should return failure when response is None."""
        query = GetChunk("chunk_1")
        result = query.response(None)

        assert result.success is False


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
        """Should create SearchSimilar parameters."""
        embedding = [0.1] * 384
        query = SearchSimilarChunks(embedding, top_k=5)
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["query_vec"] == embedding
        assert payload[0]["top_k"] == 5
        assert query.endpoint == "SearchSimilar"

    def test_stores_filters_for_post_processing(self):
        """Should store filters on query object for client-side filtering."""
        query = SearchSimilarChunks(
            [0.1] * 384,
            top_k=10,
            doc_id="doc_1",
            level=2,
            semantic_type="definition",
            threshold=0.5,
        )

        # Filters stored on query object for client.py post-processing
        assert query.doc_id == "doc_1"
        assert query.level == 2
        assert query.semantic_type == "definition"
        assert query.threshold == 0.5

    def test_requests_extra_results_when_filters(self):
        """Should request extra results when filters present."""
        query = SearchSimilarChunks(
            [0.1] * 384,
            top_k=10,
            doc_id="doc_1",
        )
        payload = query.query()

        # Should request more than top_k for filtering
        assert payload[0]["top_k"] > 10

    def test_no_extra_results_without_filters(self):
        """Should not request extra results without filters."""
        query = SearchSimilarChunks([0.1] * 384, top_k=5)
        payload = query.query()

        assert payload[0]["top_k"] == 5

    def test_response_with_list(self):
        """Should return results from list response."""
        query = SearchSimilarChunks([0.1] * 384, top_k=5)
        result = query.response([{"chunk_id": "c1"}, {"chunk_id": "c2"}])

        assert result.success is True
        assert len(result.data) == 2

    def test_response_with_dict_format(self):
        """Should extract results from dict with 'results' key (helix-py format)."""
        query = SearchSimilarChunks([0.1] * 384, top_k=5)
        result = query.response({"results": [{"chunk_id": "c1", "score": 0.9}]})

        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["chunk_id"] == "c1"


class TestSearchKeyword:
    """Tests for SearchKeyword query."""

    def test_creates_keyword_search_payload(self):
        """Should create BM25 search parameters."""
        query = SearchKeyword("test term")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["keyword"] == "test term"
        assert payload[0]["limit"] == 100
        assert query.endpoint == "SearchKeywordBM25"

    def test_stores_filters_for_post_processing(self):
        """Should store filters on query object for client-side filtering."""
        query = SearchKeyword("test", doc_id="doc_1", case_sensitive=True)

        assert query.doc_id == "doc_1"
        assert query.case_sensitive is True

    def test_response(self):
        """Should return results in QueryResult."""
        query = SearchKeyword("test")
        result = query.response([{"chunk_id": "c1"}, {"chunk_id": "c2"}])

        assert result.success is True
        assert len(result.data) == 2

    def test_response_with_dict_format(self):
        """Should extract results from dict with 'results' key (helix-py format)."""
        query = SearchKeyword("test")
        result = query.response({"results": [{"chunk_id": "c1", "score": 0.8}]})

        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["chunk_id"] == "c1"


class TestGetDocumentChunks:
    """Tests for GetDocumentChunks query."""

    def test_creates_payload_without_level(self):
        """Should create parameters for GetChunksByDocumentId."""
        query = GetDocumentChunks("doc_1")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["document_id"] == "doc_1"
        assert "level" not in payload[0]
        assert query.endpoint == "GetChunksByDocumentId"

    def test_creates_payload_with_level(self):
        """Should create parameters for GetChunksByDocAndLevel."""
        query = GetDocumentChunks("doc_1", level=2)
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["document_id"] == "doc_1"
        assert payload[0]["level"] == 2
        assert query.endpoint == "GetChunksByDocAndLevel"

    def test_response_with_chunks(self):
        """Should return chunks list from response."""
        query = GetDocumentChunks("doc_1")
        result = query.response([{"chunk_id": "c1"}, {"chunk_id": "c2"}])

        assert result.success is True
        assert len(result.data) == 2

    def test_response_with_dict_format(self):
        """Should extract chunks from dict with 'chunks' key (helix-py format)."""
        query = GetDocumentChunks("doc_1")
        result = query.response({"chunks": [{"chunk_id": "c1"}, {"chunk_id": "c2"}]})

        assert result.success is True
        assert len(result.data) == 2
        assert result.data[0]["chunk_id"] == "c1"

    def test_response_empty(self):
        """Should handle empty response."""
        query = GetDocumentChunks("doc_1")
        result = query.response([])

        assert result.success is True
        assert result.data == []


class TestGetChunksByParentId:
    """Tests for GetChunksByParentId query."""

    def test_creates_payload(self):
        """Should create parameters for HelixQL GetChunksByParentId."""
        query = GetChunksByParentId("parent_chunk_1")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["parent_id"] == "parent_chunk_1"
        assert query.endpoint == "GetChunksByParentId"

    def test_response_with_children(self):
        """Should return children list from response."""
        query = GetChunksByParentId("parent_1")
        result = query.response([{"chunk_id": "child1"}, {"chunk_id": "child2"}])

        assert result.success is True
        assert len(result.data) == 2

    def test_response_empty(self):
        """Should handle empty response (no children)."""
        query = GetChunksByParentId("parent_1")
        result = query.response([])

        assert result.success is True
        assert result.data == []


class TestListDocuments:
    """Tests for ListDocuments query."""

    def test_creates_empty_params(self):
        """Should create empty parameters for HelixQL ListDocuments."""
        query = ListDocuments()
        payload = query.query()

        assert len(payload) == 1
        assert payload[0] == {}
        assert query.endpoint == "ListDocuments"

    def test_response_with_documents(self):
        """Should return documents list from response."""
        query = ListDocuments()
        result = query.response([{"doc_id": "doc1"}, {"doc_id": "doc2"}])

        assert result.success is True
        assert len(result.data) == 2

    def test_response_with_dict_format(self):
        """Should extract documents from dict with 'docs' key (helix-py format)."""
        query = ListDocuments()
        # This is the format helix-py passes from raw HTTP response
        result = query.response({"docs": [{"doc_id": "doc1"}, {"doc_id": "doc2"}]})

        assert result.success is True
        assert len(result.data) == 2
        assert result.data[0]["doc_id"] == "doc1"

    def test_response_empty(self):
        """Should handle empty response."""
        query = ListDocuments()
        result = query.response([])

        assert result.success is True
        assert result.data == []


class TestAddChunkVector:
    """Tests for AddChunkVector query."""

    def test_creates_vector_payload(self):
        """Should create payload for HelixQL AddChunkVector."""
        embedding = [0.1, 0.2, 0.3] * 128  # 384-dim vector
        query = AddChunkVector(
            embedding=embedding,
            model_name="all-MiniLM-L6-v2",
            embedding_dim=384,
        )
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["embedding"] == embedding
        assert payload[0]["model_name"] == "all-MiniLM-L6-v2"
        assert payload[0]["embedding_dim"] == 384
        assert query.endpoint == "AddChunkVector"

    def test_stores_attributes(self):
        """Should store embedding attributes on query object."""
        embedding = [0.5] * 768
        query = AddChunkVector(
            embedding=embedding,
            model_name="bge-base-en-v1.5",
            embedding_dim=768,
        )

        assert query.embedding == embedding
        assert query.model_name == "bge-base-en-v1.5"
        assert query.embedding_dim == 768

    def test_response_success(self):
        """Should return success with vector data."""
        query = AddChunkVector([0.1] * 384, "test-model", 384)
        result = query.response({"id": "vec_123"})

        assert result.success is True
        assert result.data == {"id": "vec_123"}


class TestLinkChunkVector:
    """Tests for LinkChunkVector query."""

    def test_creates_link_payload(self):
        """Should create payload for HelixQL LinkChunkVector."""
        query = LinkChunkVector("chunk_internal_123", "vector_internal_456")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["chunk_id"] == "chunk_internal_123"
        assert payload[0]["vector_id"] == "vector_internal_456"
        assert query.endpoint == "LinkChunkVector"

    def test_response_success(self):
        """Should return success for link operation."""
        query = LinkChunkVector("chunk_1", "vec_1")
        result = query.response({})

        assert result.success is True


class TestDropChunk:
    """Tests for DropChunk query."""

    def test_creates_drop_payload(self):
        """Should create payload for HelixQL DropChunk."""
        query = DropChunk("chunk_internal_123")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["chunk_id"] == "chunk_internal_123"
        assert query.endpoint == "DropChunk"

    def test_response_success(self):
        """Should return success for drop operation."""
        query = DropChunk("chunk_1")
        result = query.response("Removed chunk")

        assert result.success is True


class TestDropDocument:
    """Tests for DropDocument query."""

    def test_creates_drop_payload(self):
        """Should create payload for HelixQL DropDocument."""
        query = DropDocument("doc_internal_456")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["doc_id"] == "doc_internal_456"
        assert query.endpoint == "DropDocument"

    def test_response_success(self):
        """Should return success for drop operation."""
        query = DropDocument("doc_1")
        result = query.response("Removed document")

        assert result.success is True


class TestGetDocumentByDocId:
    """Tests for GetDocumentByDocId query."""

    def test_creates_get_payload(self):
        """Should create payload for HelixQL GetDocumentByDocId."""
        query = GetDocumentByDocId("my-doc-id")
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["doc_id"] == "my-doc-id"
        assert query.endpoint == "GetDocumentByDocId"

    def test_response_found(self):
        """Should return success when document found."""
        query = GetDocumentByDocId("doc_1")
        result = query.response([{"doc_id": "doc_1", "title": "Test"}])

        assert result.success is True
        assert result.data["node"]["doc_id"] == "doc_1"

    def test_response_not_found(self):
        """Should return failure when document not found."""
        query = GetDocumentByDocId("doc_1")
        result = query.response([])

        assert result.success is False
        assert result.error is not None


class TestUpdateChunkContent:
    """Tests for UpdateChunkContent query."""

    def test_creates_update_payload(self):
        """Should create payload for HelixQL UpdateChunkContent."""
        query = UpdateChunkContent(
            chunk_internal_id="chunk_internal_123",
            content="Updated content text",
            word_count=3,
        )
        payload = query.query()

        assert len(payload) == 1
        assert payload[0]["chunk_id"] == "chunk_internal_123"
        assert payload[0]["content"] == "Updated content text"
        assert payload[0]["word_count"] == 3
        assert query.endpoint == "UpdateChunkContent"

    def test_stores_attributes(self):
        """Should store attributes on query object."""
        query = UpdateChunkContent("id_1", "content", 5)

        assert query.chunk_internal_id == "id_1"
        assert query.content == "content"
        assert query.word_count == 5

    def test_response_success(self):
        """Should return success with updated node."""
        query = UpdateChunkContent("id_1", "new content", 2)
        result = query.response({"chunk_id": "id_1", "content": "new content"})

        assert result.success is True
        assert result.data["content"] == "new content"
