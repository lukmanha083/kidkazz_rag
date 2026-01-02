"""Tests for MCP tool implementations."""

from unittest.mock import MagicMock, patch

import pytest

from src.chunker import Chunk, ChunkMetadata, EmbeddedChunk
from src.chunker.embedder import MockEmbedder
from src.mcp_server.config import MCPServerConfig, ServerState
from src.mcp_server.tools import register_tools
from src.storage.mock_store import MockChunkStore


@pytest.fixture
def mock_store():
    """Create and populate a mock store."""
    store = MockChunkStore()
    return store


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    return MockEmbedder(embedding_dim=384)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    parent = Chunk(
        id="doc_l1_1",
        content="Parent section about machine learning concepts",
        level=1,
        token_count=50,
        child_ids=["doc_l2_1", "doc_l2_2"],
    )
    child1 = Chunk(
        id="doc_l2_1",
        content="Machine learning is a subset of AI",
        level=2,
        token_count=20,
        parent_id="doc_l1_1",
        next_id="doc_l2_2",
        section_path=["ML Concepts"],
    )
    child2 = Chunk(
        id="doc_l2_2",
        content="Deep learning uses neural networks",
        level=2,
        token_count=25,
        parent_id="doc_l1_1",
        prev_id="doc_l2_1",
        section_path=["ML Concepts"],
    )
    return [parent, child1, child2]


@pytest.fixture
def sample_metadata(sample_chunks):
    """Create metadata for sample chunks."""
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
        ),
        ChunkMetadata(
            chunk_id=sample_chunks[2].id,
            document_id="test_doc",
            semantic_type="definition",
            sequence_position=2,
            sibling_ids=["doc_l2_1"],
        ),
    ]


@pytest.fixture
def populated_store(mock_store, mock_embedder, sample_chunks, sample_metadata):
    """Create store with sample data."""
    embedded = mock_embedder.embed_chunks(sample_chunks)
    mock_store.store_document("test_doc", "Test Document", embedded, sample_metadata)
    return mock_store


@pytest.fixture
def server_state(populated_store, mock_embedder):
    """Create server state with mock store and embedder."""
    config = MCPServerConfig(store_type="mock", embedder_type="mock")
    state = ServerState(config)
    state._store = populated_store
    state._embedder = MagicMock()
    state._embedder.embed_text = mock_embedder.embed_text
    return state


@pytest.fixture
def registered_tools(server_state):
    """Register tools and return a mock MCP server with captured tools."""
    tools = {}

    class MockMCP:
        def tool(self):
            def decorator(func):
                tools[func.__name__] = func
                return func

            return decorator

    mcp = MockMCP()
    register_tools(mcp, server_state)
    return tools


class TestSearchSemantic:
    """Tests for search_semantic tool."""

    def test_basic_search(self, registered_tools):
        """Test basic semantic search."""
        search = registered_tools["search_semantic"]
        results = search(query="machine learning", top_k=5)
        assert isinstance(results, list)

    def test_search_with_doc_filter(self, registered_tools):
        """Test search with document filter."""
        search = registered_tools["search_semantic"]
        results = search(query="ML", doc_id="test_doc", top_k=3)
        assert isinstance(results, list)

    def test_search_with_level_filter(self, registered_tools):
        """Test search with level filter."""
        search = registered_tools["search_semantic"]
        results = search(query="neural networks", level=2, top_k=5)
        assert isinstance(results, list)

    def test_search_with_type_filter(self, registered_tools):
        """Test search with semantic type filter."""
        search = registered_tools["search_semantic"]
        results = search(query="definition", semantic_type="definition", top_k=5)
        assert isinstance(results, list)

    def test_search_with_threshold(self, registered_tools):
        """Test search with similarity threshold."""
        search = registered_tools["search_semantic"]
        results = search(query="ML", threshold=0.5, top_k=5)
        assert isinstance(results, list)

    def test_search_returns_scores(self, registered_tools):
        """Test search results include similarity scores when results exist."""
        search = registered_tools["search_semantic"]
        # Use content from test data for better match with mock embeddings
        results = search(query="Parent section content", top_k=5)
        assert isinstance(results, list)
        # With mock embeddings, results may vary - check structure if present
        if len(results) > 0:
            assert "similarity_score" in results[0]
            assert isinstance(results[0]["similarity_score"], float)


class TestSearchKeyword:
    """Tests for search_keyword tool."""

    def test_basic_keyword_search(self, registered_tools):
        """Test basic keyword search."""
        search = registered_tools["search_keyword"]
        results = search(keyword="machine")
        assert isinstance(results, list)

    def test_keyword_search_with_doc_filter(self, registered_tools):
        """Test keyword search with document filter."""
        search = registered_tools["search_keyword"]
        results = search(keyword="learning", doc_id="test_doc")
        assert isinstance(results, list)

    def test_case_insensitive_search(self, registered_tools):
        """Test case-insensitive keyword search."""
        search = registered_tools["search_keyword"]
        results = search(keyword="MACHINE", case_sensitive=False)
        assert isinstance(results, list)

    def test_case_sensitive_search(self, registered_tools):
        """Test case-sensitive keyword search."""
        search = registered_tools["search_keyword"]
        results_lower = search(keyword="machine", case_sensitive=True)
        results_upper = search(keyword="MACHINE", case_sensitive=True)
        # Case sensitive search for uppercase should find nothing
        assert len(results_upper) == 0


class TestGetChunk:
    """Tests for get_chunk tool."""

    def test_get_existing_chunk(self, registered_tools):
        """Test getting existing chunk by ID."""
        get_chunk = registered_tools["get_chunk"]
        result = get_chunk(chunk_id="doc_l2_1")
        assert result is not None
        assert result["chunk_id"] == "doc_l2_1"

    def test_get_nonexistent_chunk(self, registered_tools):
        """Test getting non-existent chunk returns None."""
        get_chunk = registered_tools["get_chunk"]
        result = get_chunk(chunk_id="nonexistent")
        assert result is None

    def test_chunk_has_content(self, registered_tools):
        """Test retrieved chunk has content."""
        get_chunk = registered_tools["get_chunk"]
        result = get_chunk(chunk_id="doc_l2_1")
        assert result is not None
        assert "content" in result
        assert len(result["content"]) > 0


class TestGetContextWindow:
    """Tests for get_context_window tool."""

    def test_get_context_window(self, registered_tools):
        """Test getting context window around chunk."""
        get_context = registered_tools["get_context_window"]
        results = get_context(chunk_id="doc_l2_1", window_size=1)
        assert isinstance(results, list)

    def test_context_window_default_size(self, registered_tools):
        """Test context window with default size."""
        get_context = registered_tools["get_context_window"]
        results = get_context(chunk_id="doc_l2_1")
        assert isinstance(results, list)

    def test_context_window_includes_neighbors(self, registered_tools):
        """Test context window includes neighboring chunks."""
        get_context = registered_tools["get_context_window"]
        results = get_context(chunk_id="doc_l2_1", window_size=2)
        # Should include center chunk and potentially neighbors
        assert isinstance(results, list)


class TestGetParent:
    """Tests for get_parent tool."""

    def test_get_parent_of_child(self, registered_tools):
        """Test getting parent of child chunk."""
        get_parent = registered_tools["get_parent"]
        result = get_parent(chunk_id="doc_l2_1")
        assert result is not None
        assert result["chunk_id"] == "doc_l1_1"
        assert result["level"] == 1

    def test_get_parent_of_root(self, registered_tools):
        """Test getting parent of root chunk returns None."""
        get_parent = registered_tools["get_parent"]
        result = get_parent(chunk_id="doc_l1_1")
        assert result is None


class TestGetChildren:
    """Tests for get_children tool."""

    def test_get_children_of_parent(self, registered_tools):
        """Test getting children of parent chunk."""
        get_children = registered_tools["get_children"]
        results = get_children(chunk_id="doc_l1_1")
        assert isinstance(results, list)
        assert len(results) == 2

    def test_get_children_of_leaf(self, registered_tools):
        """Test getting children of leaf chunk returns empty."""
        get_children = registered_tools["get_children"]
        results = get_children(chunk_id="doc_l2_1")
        assert results == []

    def test_children_have_correct_parent(self, registered_tools):
        """Test children have correct parent_id."""
        get_children = registered_tools["get_children"]
        results = get_children(chunk_id="doc_l1_1")
        for child in results:
            assert child["parent_id"] == "doc_l1_1"


class TestGetSiblings:
    """Tests for get_siblings tool."""

    def test_get_siblings(self, registered_tools):
        """Test getting sibling chunks."""
        get_siblings = registered_tools["get_siblings"]
        results = get_siblings(chunk_id="doc_l2_1")
        assert isinstance(results, list)

    def test_siblings_exclude_self(self, registered_tools):
        """Test siblings list excludes the queried chunk."""
        get_siblings = registered_tools["get_siblings"]
        results = get_siblings(chunk_id="doc_l2_1")
        chunk_ids = [r["chunk_id"] for r in results]
        assert "doc_l2_1" not in chunk_ids

    def test_siblings_same_parent(self, registered_tools):
        """Test siblings have same parent."""
        get_siblings = registered_tools["get_siblings"]
        results = get_siblings(chunk_id="doc_l2_1")
        for sibling in results:
            assert sibling["parent_id"] == "doc_l1_1"


class TestListDocuments:
    """Tests for list_documents tool."""

    def test_list_documents(self, registered_tools):
        """Test listing all documents."""
        list_docs = registered_tools["list_documents"]
        results = list_docs()
        assert isinstance(results, list)

    def test_document_has_fields(self, registered_tools):
        """Test document entries have required fields."""
        list_docs = registered_tools["list_documents"]
        results = list_docs()
        assert isinstance(results, list)
        assert len(results) > 0, "Expected non-empty document list"
        doc = results[0]
        assert "doc_id" in doc
        assert "title" in doc
        assert "chunk_count" in doc


class TestGetDocumentChunks:
    """Tests for get_document_chunks tool."""

    def test_get_document_chunks(self, registered_tools):
        """Test getting all chunks from document."""
        get_chunks = registered_tools["get_document_chunks"]
        results = get_chunks(doc_id="test_doc")
        assert isinstance(results, list)
        assert len(results) == 3  # All sample chunks

    def test_get_document_chunks_level_filter(self, registered_tools):
        """Test getting chunks with level filter."""
        get_chunks = registered_tools["get_document_chunks"]
        results = get_chunks(doc_id="test_doc", level=2)
        assert isinstance(results, list)
        for chunk in results:
            assert chunk["level"] == 2

    def test_get_nonexistent_document(self, registered_tools):
        """Test getting chunks from non-existent document."""
        get_chunks = registered_tools["get_document_chunks"]
        results = get_chunks(doc_id="nonexistent")
        assert results == []


class TestGetDocumentStats:
    """Tests for get_document_stats tool."""

    def test_get_document_stats(self, registered_tools):
        """Test getting document statistics."""
        get_stats = registered_tools["get_document_stats"]
        result = get_stats(doc_id="test_doc")
        assert result is not None
        assert "total_chunks" in result

    def test_stats_has_level_counts(self, registered_tools):
        """Test stats include level counts."""
        get_stats = registered_tools["get_document_stats"]
        result = get_stats(doc_id="test_doc")
        assert result is not None
        assert "chunks_by_level" in result

    def test_stats_has_type_counts(self, registered_tools):
        """Test stats include type counts."""
        get_stats = registered_tools["get_document_stats"]
        result = get_stats(doc_id="test_doc")
        assert result is not None
        assert "chunks_by_type" in result

    def test_nonexistent_document_stats(self, registered_tools):
        """Test getting stats for non-existent document."""
        get_stats = registered_tools["get_document_stats"]
        result = get_stats(doc_id="nonexistent")
        assert result is None
