"""Tests for MockChunkStore in-memory implementation."""

import pytest

from src.chunker import Chunk, ChunkMetadata, EmbeddedChunk
from src.chunker.embedder import MockEmbedder
from src.storage.mock_store import MockChunkStore


@pytest.fixture
def mock_store():
    """Create fresh mock store."""
    return MockChunkStore()


@pytest.fixture
def mock_embedder():
    """Create mock embedder."""
    return MockEmbedder(embedding_dim=384)


@pytest.fixture
def sample_chunks():
    """Create sample chunks with relationships."""
    # Level 1 parent chunk
    parent = Chunk(
        id="doc_l1_1",
        content="Parent section content",
        level=1,
        token_count=20,
        child_ids=["doc_l2_1", "doc_l2_2"],
    )

    # Level 2 child chunks
    child1 = Chunk(
        id="doc_l2_1",
        content="First child chunk content",
        level=2,
        token_count=10,
        parent_id="doc_l1_1",
        next_id="doc_l2_2",
        section_path=["Section 1"],
    )
    child2 = Chunk(
        id="doc_l2_2",
        content="Second child chunk content",
        level=2,
        token_count=12,
        parent_id="doc_l1_1",
        prev_id="doc_l2_1",
        section_path=["Section 1"],
    )

    return [parent, child1, child2]


@pytest.fixture
def sample_embedded_chunks(sample_chunks, mock_embedder):
    """Create sample embedded chunks."""
    return mock_embedder.embed_chunks(sample_chunks)


@pytest.fixture
def sample_metadata(sample_chunks):
    """Create sample metadata for chunks."""
    return [
        ChunkMetadata(
            chunk_id=sample_chunks[0].id,
            document_id="test_doc",
            semantic_type="narrative",
            sequence_position=0,
            sibling_ids=[],
        ),
        ChunkMetadata(
            chunk_id=sample_chunks[1].id,
            document_id="test_doc",
            semantic_type="definition",
            sequence_position=1,
            sibling_ids=["doc_l2_2"],
            has_code=True,
        ),
        ChunkMetadata(
            chunk_id=sample_chunks[2].id,
            document_id="test_doc",
            semantic_type="example",
            sequence_position=2,
            sibling_ids=["doc_l2_1"],
            has_table=True,
        ),
    ]


class TestMockChunkStoreBasic:
    """Basic MockChunkStore tests."""

    def test_store_creates_instance(self, mock_store):
        """Should create empty store."""
        assert mock_store._documents == {}
        assert mock_store._chunks == {}
        assert mock_store._embeddings == {}

    def test_clear_removes_all_data(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should clear all stored data."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)
        mock_store.clear()

        assert mock_store._documents == {}
        assert mock_store._chunks == {}
        assert mock_store._embeddings == {}


class TestStoreDocument:
    """Tests for store_document method."""

    def test_stores_document_metadata(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should store document metadata."""
        mock_store.store_document("doc1", "Test Title", sample_embedded_chunks, sample_metadata)

        assert "doc1" in mock_store._documents
        assert mock_store._documents["doc1"]["title"] == "Test Title"
        assert mock_store._documents["doc1"]["chunk_count"] == 3

    def test_stores_all_chunks(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should store all chunks."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        assert len(mock_store._chunks) == 3

    def test_stores_embeddings(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should store embeddings."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        for ec in sample_embedded_chunks:
            assert ec.chunk.id in mock_store._embeddings
            assert len(mock_store._embeddings[ec.chunk.id]) == 384


class TestGetChunk:
    """Tests for get_chunk method."""

    def test_retrieves_existing_chunk(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should retrieve existing chunk."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        result = mock_store.get_chunk("doc_l2_1")

        assert result is not None
        assert result.chunk.id == "doc_l2_1"
        assert result.chunk.content == "First child chunk content"

    def test_returns_none_for_missing_chunk(self, mock_store):
        """Should return None for non-existent chunk."""
        result = mock_store.get_chunk("nonexistent")

        assert result is None

    def test_includes_embedding(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should include embedding in result."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        result = mock_store.get_chunk("doc_l2_1")

        assert result is not None
        assert len(result.embedding) == 384


class TestDeleteChunk:
    """Tests for delete_chunk method."""

    def test_deletes_existing_chunk(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should delete existing chunk."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        result = mock_store.delete_chunk("doc_l2_1")

        assert result is True
        assert "doc_l2_1" not in mock_store._chunks

    def test_returns_false_for_missing_chunk(self, mock_store):
        """Should return False for non-existent chunk."""
        result = mock_store.delete_chunk("nonexistent")

        assert result is False

    def test_updates_parent_child_ids(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should update parent's child_ids."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        mock_store.delete_chunk("doc_l2_1")

        parent_data = mock_store._chunks["doc_l1_1"]
        assert "doc_l2_1" not in parent_data["child_ids"]

    def test_updates_sequential_links(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should update prev/next links."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        mock_store.delete_chunk("doc_l2_1")

        # doc_l2_2 should have no prev_id now
        chunk_data = mock_store._chunks["doc_l2_2"]
        assert chunk_data["prev_id"] is None


class TestUpdateChunk:
    """Tests for update_chunk method."""

    def test_updates_content(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should update chunk content."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        result = mock_store.update_chunk("doc_l2_1", content="Updated content")

        assert result is True
        assert mock_store._chunks["doc_l2_1"]["content"] == "Updated content"

    def test_updates_embedding(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should update embedding."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)
        new_embedding = [1.0] * 384

        result = mock_store.update_chunk("doc_l2_1", embedding=new_embedding)

        assert result is True
        assert mock_store._embeddings["doc_l2_1"] == new_embedding

    def test_returns_false_for_missing_chunk(self, mock_store):
        """Should return False for non-existent chunk."""
        result = mock_store.update_chunk("nonexistent", content="New")

        assert result is False


class TestDeleteDocument:
    """Tests for delete_document method."""

    def test_deletes_document_and_chunks(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should delete document and all its chunks."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        result = mock_store.delete_document("doc1")

        assert result is True
        assert "doc1" not in mock_store._documents
        assert len(mock_store._chunks) == 0

    def test_returns_false_for_missing_document(self, mock_store):
        """Should return False for non-existent document."""
        result = mock_store.delete_document("nonexistent")

        assert result is False


class TestSearchSimilar:
    """Tests for search_similar method."""

    def test_finds_similar_chunks(self, mock_store, sample_embedded_chunks, sample_metadata, mock_embedder):
        """Should find similar chunks."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)
        query = mock_embedder.embed_text("First child chunk content")

        results = mock_store.search_similar(query, top_k=3)

        assert len(results) > 0
        # First result should be the most similar
        assert results[0][1] >= results[-1][1]

    def test_returns_top_k_results(self, mock_store, sample_embedded_chunks, sample_metadata, mock_embedder):
        """Should return at most top_k results."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)
        query = mock_embedder.embed_text("test query")

        results = mock_store.search_similar(query, top_k=2)

        assert len(results) <= 2

    def test_filters_by_document(self, mock_store, sample_embedded_chunks, sample_metadata, mock_embedder):
        """Should filter by document ID."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)
        query = mock_embedder.embed_text("test")

        results = mock_store.search_similar(query, top_k=5, doc_id="doc1")

        for ec, _ in results:
            chunk_data = mock_store._chunks[ec.chunk.id]
            assert chunk_data["document_id"] == "doc1"

    def test_filters_by_level(self, mock_store, sample_embedded_chunks, sample_metadata, mock_embedder):
        """Should filter by hierarchy level."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)
        query = mock_embedder.embed_text("test")

        results = mock_store.search_similar(query, top_k=5, level=2)

        for ec, _ in results:
            assert ec.chunk.level == 2

    def test_filters_by_semantic_type(self, mock_store, sample_embedded_chunks, sample_metadata, mock_embedder):
        """Should filter by semantic type."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)
        query = mock_embedder.embed_text("test")

        results = mock_store.search_similar(query, top_k=5, semantic_type="definition")

        for ec, _ in results:
            chunk_data = mock_store._chunks[ec.chunk.id]
            assert chunk_data["semantic_type"] == "definition"

    def test_respects_threshold(self, mock_store, sample_embedded_chunks, sample_metadata, mock_embedder):
        """Should filter by similarity threshold."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)
        query = mock_embedder.embed_text("test")

        results = mock_store.search_similar(query, top_k=5, threshold=0.9)

        for _, score in results:
            assert score >= 0.9


class TestSearchKeyword:
    """Tests for search_keyword method."""

    def test_finds_matching_chunks(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should find chunks containing keyword."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        results = mock_store.search_keyword("First")

        assert len(results) >= 1
        assert any("First" in ec.chunk.content for ec in results)

    def test_case_insensitive_by_default(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should be case insensitive by default."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        results = mock_store.search_keyword("first")

        assert len(results) >= 1

    def test_case_sensitive_option(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should support case-sensitive search."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        # Sample data has "First" (capitalized), searching "first" case-sensitively
        # should return no results
        results = mock_store.search_keyword("first", case_sensitive=True)
        assert len(results) == 0

        # Searching with correct case should find results
        results_correct_case = mock_store.search_keyword("First", case_sensitive=True)
        assert len(results_correct_case) >= 1


class TestGraphTraversal:
    """Tests for graph traversal methods."""

    def test_get_parent(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should get parent chunk."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        result = mock_store.get_parent("doc_l2_1")

        assert result is not None
        assert result.chunk.id == "doc_l1_1"

    def test_get_parent_returns_none_for_orphan(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should return None for chunk without parent."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        result = mock_store.get_parent("doc_l1_1")

        assert result is None

    def test_get_children(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should get child chunks."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        results = mock_store.get_children("doc_l1_1")

        assert len(results) == 2
        child_ids = {ec.chunk.id for ec in results}
        assert child_ids == {"doc_l2_1", "doc_l2_2"}

    def test_get_children_returns_empty_for_leaf(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should return empty list for leaf chunk."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        results = mock_store.get_children("doc_l2_1")

        assert results == []

    def test_get_siblings(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should get sibling chunks."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        results = mock_store.get_siblings("doc_l2_1")

        assert len(results) == 1
        assert results[0].chunk.id == "doc_l2_2"

    def test_get_context_window(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should get context window around chunk."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        results = mock_store.get_context_window("doc_l2_2", window_size=1)

        chunk_ids = [ec.chunk.id for ec in results]
        assert "doc_l2_1" in chunk_ids
        assert "doc_l2_2" in chunk_ids

    def test_get_context_window_handles_boundaries(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should handle document boundaries gracefully."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        results = mock_store.get_context_window("doc_l2_1", window_size=5)

        # Should not crash, should return available chunks
        assert len(results) >= 1
        assert any(ec.chunk.id == "doc_l2_1" for ec in results)


class TestDocumentOperations:
    """Tests for document-level operations."""

    def test_get_document_chunks(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should get all chunks for a document."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        results = mock_store.get_document_chunks("doc1")

        assert len(results) == 3

    def test_get_document_chunks_filters_by_level(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should filter by level."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        results = mock_store.get_document_chunks("doc1", level=2)

        assert len(results) == 2
        assert all(ec.chunk.level == 2 for ec in results)

    def test_list_documents(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should list all documents."""
        mock_store.store_document("doc1", "Title 1", sample_embedded_chunks, sample_metadata)
        mock_store.store_document("doc2", "Title 2", sample_embedded_chunks, sample_metadata)

        results = mock_store.list_documents()

        assert len(results) == 2
        doc_ids = {d["doc_id"] for d in results}
        assert doc_ids == {"doc1", "doc2"}

    def test_get_document_stats(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should get document statistics."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        stats = mock_store.get_document_stats("doc1")

        assert stats is not None
        assert stats["doc_id"] == "doc1"
        assert stats["total_chunks"] == 3
        assert 1 in stats["level_counts"]
        assert 2 in stats["level_counts"]

    def test_get_document_stats_returns_none_for_missing(self, mock_store):
        """Should return None for non-existent document."""
        stats = mock_store.get_document_stats("nonexistent")

        assert stats is None
