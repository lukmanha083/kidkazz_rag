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


class TestDocumentTags:
    """Tests for document tagging functionality."""

    def test_stores_document_with_tags(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should store tags with document."""
        mock_store.store_document(
            "doc1", "Title", sample_embedded_chunks, sample_metadata,
            tags=["inventory", "accounting"]
        )

        assert mock_store._documents["doc1"]["tags"] == ["inventory", "accounting"]

    def test_normalizes_tags_to_lowercase(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should normalize tags to lowercase."""
        mock_store.store_document(
            "doc1", "Title", sample_embedded_chunks, sample_metadata,
            tags=["INVENTORY", "Accounting", "  Supply-Chain  "]
        )

        assert mock_store._documents["doc1"]["tags"] == ["inventory", "accounting", "supply-chain"]

    def test_stores_empty_tags_by_default(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should store empty tags list when not provided."""
        mock_store.store_document("doc1", "Title", sample_embedded_chunks, sample_metadata)

        assert mock_store._documents["doc1"]["tags"] == []

    def test_list_documents_returns_tags(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should include tags in document list."""
        mock_store.store_document(
            "doc1", "Title", sample_embedded_chunks, sample_metadata,
            tags=["inventory"]
        )

        docs = mock_store.list_documents()

        assert len(docs) == 1
        assert docs[0]["tags"] == ["inventory"]

    def test_list_documents_filters_by_single_tag(self, mock_store, sample_embedded_chunks, sample_metadata):
        """Should filter documents by single tag."""
        mock_store.store_document(
            "doc1", "Title 1", sample_embedded_chunks, sample_metadata,
            tags=["inventory", "accounting"]
        )
        mock_store.store_document(
            "doc2", "Title 2", sample_embedded_chunks, sample_metadata,
            tags=["supply-chain"]
        )

        results = mock_store.list_documents(tags=["inventory"])

        assert len(results) == 1
        assert results[0]["doc_id"] == "doc1"

    def test_list_documents_filters_by_multiple_tags_and_logic(
        self, mock_store, sample_embedded_chunks, sample_metadata
    ):
        """Should require ALL tags when filtering (AND logic)."""
        mock_store.store_document(
            "doc1", "Title 1", sample_embedded_chunks, sample_metadata,
            tags=["inventory", "accounting"]
        )
        mock_store.store_document(
            "doc2", "Title 2", sample_embedded_chunks, sample_metadata,
            tags=["inventory"]  # Only has inventory, not accounting
        )

        # Filtering by both tags should only return doc1
        results = mock_store.list_documents(tags=["inventory", "accounting"])

        assert len(results) == 1
        assert results[0]["doc_id"] == "doc1"

    def test_list_documents_tag_filter_case_insensitive(
        self, mock_store, sample_embedded_chunks, sample_metadata
    ):
        """Should match tags case-insensitively."""
        mock_store.store_document(
            "doc1", "Title", sample_embedded_chunks, sample_metadata,
            tags=["inventory"]
        )

        results = mock_store.list_documents(tags=["INVENTORY"])

        assert len(results) == 1
        assert results[0]["doc_id"] == "doc1"

    def test_search_similar_filters_by_tags(
        self, mock_store, sample_embedded_chunks, sample_metadata, mock_embedder
    ):
        """Should filter search results by document tags."""
        mock_store.store_document(
            "doc1", "Title 1", sample_embedded_chunks, sample_metadata,
            tags=["inventory"]
        )

        # Create another set of chunks for doc2
        doc2_chunks = [
            Chunk(
                id="doc2_l1_1",
                content="Different document content",
                level=1,
                token_count=15,
            )
        ]
        doc2_embedded = mock_embedder.embed_chunks(doc2_chunks)
        doc2_metadata = [
            ChunkMetadata(
                chunk_id="doc2_l1_1",
                document_id="doc2",
                semantic_type="narrative",
                sequence_position=0,
                sibling_ids=[],
            )
        ]
        mock_store.store_document(
            "doc2", "Title 2", doc2_embedded, doc2_metadata,
            tags=["supply-chain"]
        )

        # Search with inventory tag - should only find doc1 chunks
        query = mock_embedder.embed_text("test query")
        results = mock_store.search_similar(query, top_k=10, tags=["inventory"])

        # All results should be from doc1
        for ec, _ in results:
            chunk_data = mock_store._chunks[ec.chunk.id]
            assert chunk_data["document_id"] == "doc1"

    def test_filters_empty_strings_from_tags(
        self, mock_store, sample_embedded_chunks, sample_metadata
    ):
        """Should filter out empty strings and whitespace-only tags."""
        mock_store.store_document(
            "doc1", "Title", sample_embedded_chunks, sample_metadata,
            tags=["inventory", "", "  ", "accounting"]
        )

        # Empty strings and whitespace should be filtered out
        assert mock_store._documents["doc1"]["tags"] == ["inventory", "accounting"]

    def test_search_similar_filters_by_has_table(
        self, mock_store, mock_embedder
    ):
        """Should filter search results by has_table flag."""
        # Create chunk with table
        chunk_with_table = Chunk(
            id="chunk_table",
            content="Table content here",
            level=2,
            token_count=10,
        )
        meta_with_table = ChunkMetadata(
            chunk_id="chunk_table",
            document_id="doc1",
            semantic_type="narrative",
            sequence_position=0,
            sibling_ids=[],
            has_table=True,
        )

        # Create chunk without table
        chunk_no_table = Chunk(
            id="chunk_no_table",
            content="Text content here",
            level=2,
            token_count=10,
        )
        meta_no_table = ChunkMetadata(
            chunk_id="chunk_no_table",
            document_id="doc1",
            semantic_type="narrative",
            sequence_position=1,
            sibling_ids=[],
            has_table=False,
        )

        embedded = mock_embedder.embed_chunks([chunk_with_table, chunk_no_table])
        mock_store.store_document(
            "doc1", "Title", embedded, [meta_with_table, meta_no_table]
        )

        query = mock_embedder.embed_text("test query")

        # Filter for chunks with tables (use threshold=-1 to accept negative similarity)
        results_table = mock_store.search_similar(
            query, top_k=10, has_table=True, threshold=-1.0
        )
        assert len(results_table) == 1
        assert results_table[0][0].chunk.id == "chunk_table"

        # Filter for chunks without tables
        results_no_table = mock_store.search_similar(
            query, top_k=10, has_table=False, threshold=-1.0
        )
        assert len(results_no_table) == 1
        assert results_no_table[0][0].chunk.id == "chunk_no_table"

    def test_search_similar_filters_by_has_code(
        self, mock_store, mock_embedder
    ):
        """Should filter search results by has_code flag."""
        chunk_with_code = Chunk(
            id="chunk_code",
            content="Code block here",
            level=2,
            token_count=10,
        )
        meta_with_code = ChunkMetadata(
            chunk_id="chunk_code",
            document_id="doc1",
            semantic_type="narrative",
            sequence_position=0,
            sibling_ids=[],
            has_code=True,
        )

        chunk_no_code = Chunk(
            id="chunk_no_code",
            content="Text content here",
            level=2,
            token_count=10,
        )
        meta_no_code = ChunkMetadata(
            chunk_id="chunk_no_code",
            document_id="doc1",
            semantic_type="narrative",
            sequence_position=1,
            sibling_ids=[],
            has_code=False,
        )

        embedded = mock_embedder.embed_chunks([chunk_with_code, chunk_no_code])
        mock_store.store_document(
            "doc1", "Title", embedded, [meta_with_code, meta_no_code]
        )

        query = mock_embedder.embed_text("test query")

        # Filter for chunks with code (use threshold=-1 to accept negative similarity)
        results_code = mock_store.search_similar(
            query, top_k=10, has_code=True, threshold=-1.0
        )
        assert len(results_code) == 1
        assert results_code[0][0].chunk.id == "chunk_code"

    def test_search_similar_filters_by_has_math(
        self, mock_store, mock_embedder
    ):
        """Should filter search results by has_math flag."""
        chunk_with_math = Chunk(
            id="chunk_math",
            content="Math equations here",
            level=2,
            token_count=10,
        )
        meta_with_math = ChunkMetadata(
            chunk_id="chunk_math",
            document_id="doc1",
            semantic_type="narrative",
            sequence_position=0,
            sibling_ids=[],
            has_math=True,
        )

        chunk_no_math = Chunk(
            id="chunk_no_math",
            content="Text content here",
            level=2,
            token_count=10,
        )
        meta_no_math = ChunkMetadata(
            chunk_id="chunk_no_math",
            document_id="doc1",
            semantic_type="narrative",
            sequence_position=1,
            sibling_ids=[],
            has_math=False,
        )

        embedded = mock_embedder.embed_chunks([chunk_with_math, chunk_no_math])
        mock_store.store_document(
            "doc1", "Title", embedded, [meta_with_math, meta_no_math]
        )

        query = mock_embedder.embed_text("test query")

        # Filter for chunks with math (use threshold=-1 to accept negative similarity)
        results_math = mock_store.search_similar(
            query, top_k=10, has_math=True, threshold=-1.0
        )
        assert len(results_math) == 1
        assert results_math[0][0].chunk.id == "chunk_math"

    def test_search_similar_filters_by_header_level(
        self, mock_store, mock_embedder
    ):
        """Should filter search results by header_level."""
        chunk_h1 = Chunk(
            id="chunk_h1",
            content="H1 header content",
            level=2,
            token_count=10,
        )
        meta_h1 = ChunkMetadata(
            chunk_id="chunk_h1",
            document_id="doc1",
            semantic_type="narrative",
            sequence_position=0,
            sibling_ids=[],
            header_level=1,
            header_text="Main Title",
        )

        chunk_h2 = Chunk(
            id="chunk_h2",
            content="H2 header content",
            level=2,
            token_count=10,
        )
        meta_h2 = ChunkMetadata(
            chunk_id="chunk_h2",
            document_id="doc1",
            semantic_type="narrative",
            sequence_position=1,
            sibling_ids=[],
            header_level=2,
            header_text="Subtitle",
        )

        embedded = mock_embedder.embed_chunks([chunk_h1, chunk_h2])
        mock_store.store_document(
            "doc1", "Title", embedded, [meta_h1, meta_h2]
        )

        query = mock_embedder.embed_text("test query")

        # Filter for header level 1 (use threshold=-1 to accept negative similarity)
        results_h1 = mock_store.search_similar(
            query, top_k=10, header_level=1, threshold=-1.0
        )
        assert len(results_h1) == 1
        assert results_h1[0][0].chunk.id == "chunk_h1"

        # Filter for header level 2
        results_h2 = mock_store.search_similar(
            query, top_k=10, header_level=2, threshold=-1.0
        )
        assert len(results_h2) == 1
        assert results_h2[0][0].chunk.id == "chunk_h2"

    def test_search_similar_combines_multiple_filters(
        self, mock_store, mock_embedder
    ):
        """Should combine multiple content filters correctly."""
        # Chunk with table AND code
        chunk_table_code = Chunk(
            id="chunk_table_code",
            content="Table and code content",
            level=2,
            token_count=10,
        )
        meta_table_code = ChunkMetadata(
            chunk_id="chunk_table_code",
            document_id="doc1",
            semantic_type="narrative",
            sequence_position=0,
            sibling_ids=[],
            has_table=True,
            has_code=True,
        )

        # Chunk with table only
        chunk_table_only = Chunk(
            id="chunk_table_only",
            content="Table only content",
            level=2,
            token_count=10,
        )
        meta_table_only = ChunkMetadata(
            chunk_id="chunk_table_only",
            document_id="doc1",
            semantic_type="narrative",
            sequence_position=1,
            sibling_ids=[],
            has_table=True,
            has_code=False,
        )

        embedded = mock_embedder.embed_chunks([chunk_table_code, chunk_table_only])
        mock_store.store_document(
            "doc1", "Title", embedded, [meta_table_code, meta_table_only]
        )

        query = mock_embedder.embed_text("test query")

        # Filter for chunks with both table AND code (use threshold=-1)
        results = mock_store.search_similar(
            query, top_k=10, has_table=True, has_code=True, threshold=-1.0
        )
        assert len(results) == 1
        assert results[0][0].chunk.id == "chunk_table_code"
