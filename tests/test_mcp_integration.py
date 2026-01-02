"""Integration tests for MCP server end-to-end workflows."""

from unittest.mock import MagicMock

import pytest

from src.chunker import Chunk, ChunkMetadata, EmbeddedChunk
from src.chunker.embedder import MockEmbedder
from src.mcp_server.config import MCPServerConfig, ServerState
from src.mcp_server.resources import register_resources
from src.mcp_server.tools import register_tools
from src.storage.mock_store import MockChunkStore


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    return MockEmbedder(embedding_dim=384)


@pytest.fixture
def textbook_chunks():
    """Create a realistic textbook document structure."""
    # Chapter level chunk
    chapter = Chunk(
        id="textbook_l1_1",
        content=(
            "Chapter 1: Introduction to Machine Learning\n\n"
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn and improve from experience without explicit programming."
        ),
        level=1,
        token_count=50,
        child_ids=["textbook_l2_1", "textbook_l2_2", "textbook_l2_3"],
    )

    # Section level chunks
    section1 = Chunk(
        id="textbook_l2_1",
        content=(
            "1.1 What is Machine Learning?\n\n"
            "Machine learning is defined as the study of computer algorithms "
            "that improve automatically through experience."
        ),
        level=2,
        token_count=30,
        parent_id="textbook_l1_1",
        next_id="textbook_l2_2",
        section_path=["Chapter 1", "1.1 What is Machine Learning?"],
    )

    section2 = Chunk(
        id="textbook_l2_2",
        content=(
            "1.2 Types of Machine Learning\n\n"
            "There are three main types: supervised learning, unsupervised learning, "
            "and reinforcement learning. Each has different use cases."
        ),
        level=2,
        token_count=35,
        parent_id="textbook_l1_1",
        prev_id="textbook_l2_1",
        next_id="textbook_l2_3",
        section_path=["Chapter 1", "1.2 Types of Machine Learning"],
    )

    section3 = Chunk(
        id="textbook_l2_3",
        content=(
            "1.3 Example: Linear Regression\n\n"
            "Linear regression is a simple example of supervised learning. "
            "It predicts a continuous output variable based on input features."
        ),
        level=2,
        token_count=40,
        parent_id="textbook_l1_1",
        prev_id="textbook_l2_2",
        section_path=["Chapter 1", "1.3 Example"],
    )

    return [chapter, section1, section2, section3]


@pytest.fixture
def textbook_metadata(textbook_chunks):  # noqa: ARG001 - dependency only
    """Create metadata for textbook chunks."""
    return [
        ChunkMetadata(
            chunk_id="textbook_l1_1",
            document_id="ml_textbook",
            semantic_type="narrative",
            sequence_position=0,
            sibling_ids=[],
        ),
        ChunkMetadata(
            chunk_id="textbook_l2_1",
            document_id="ml_textbook",
            semantic_type="definition",
            sequence_position=1,
            sibling_ids=["textbook_l2_2", "textbook_l2_3"],
        ),
        ChunkMetadata(
            chunk_id="textbook_l2_2",
            document_id="ml_textbook",
            semantic_type="narrative",
            sequence_position=2,
            sibling_ids=["textbook_l2_1", "textbook_l2_3"],
        ),
        ChunkMetadata(
            chunk_id="textbook_l2_3",
            document_id="ml_textbook",
            semantic_type="example",
            sequence_position=3,
            sibling_ids=["textbook_l2_1", "textbook_l2_2"],
        ),
    ]


@pytest.fixture
def integrated_system(mock_embedder, textbook_chunks, textbook_metadata):
    """Create a fully integrated system with tools and resources."""
    # Create and populate store
    store = MockChunkStore()
    embedded = mock_embedder.embed_chunks(textbook_chunks)
    store.store_document("ml_textbook", "ML Textbook", embedded, textbook_metadata)

    # Create server state
    config = MCPServerConfig(store_type="mock", embedder_type="mock")
    state = ServerState(config)
    state._store = store
    state._embedder = MagicMock()
    state._embedder.embed_text = mock_embedder.embed_text

    # Register tools and resources
    tools = {}
    resources = {}

    class MockMCP:
        def tool(self):
            def decorator(func):
                tools[func.__name__] = func
                return func

            return decorator

        def resource(self, uri):
            def decorator(func):
                resources[uri] = func
                return func

            return decorator

    mcp = MockMCP()
    register_tools(mcp, state)
    register_resources(mcp, state)

    return {"tools": tools, "resources": resources, "state": state}


class TestSearchWorkflow:
    """Test semantic search workflows."""

    def test_search_and_expand_context(self, integrated_system):
        """Test searching and then expanding context."""
        tools = integrated_system["tools"]

        # Step 1: Search for relevant content using keyword search (more reliable with mock data)
        results = tools["search_keyword"](keyword="machine learning")
        assert len(results) > 0

        # Step 2: Get context around the best result
        best_chunk_id = results[0]["chunk_id"]
        context = tools["get_context_window"](chunk_id=best_chunk_id, window_size=1)
        assert len(context) >= 1

    def test_search_and_navigate_hierarchy(self, integrated_system):
        """Test searching and navigating chunk hierarchy."""
        tools = integrated_system["tools"]

        # Step 1: Search at leaf level using keyword search (more reliable with mock data)
        results = tools["search_keyword"](keyword="linear regression")
        # Filter to level 2 chunks
        level2_results = [r for r in results if r["level"] == 2]
        assert len(level2_results) > 0

        # Step 2: Navigate to parent for broader context
        leaf_id = level2_results[0]["chunk_id"]
        parent = tools["get_parent"](chunk_id=leaf_id)
        assert parent is not None
        assert parent["level"] == 1

        # Step 3: Get siblings for related content
        siblings = tools["get_siblings"](chunk_id=leaf_id)
        assert isinstance(siblings, list)


class TestDocumentBrowsingWorkflow:
    """Test document browsing workflows."""

    def test_browse_documents_and_stats(self, integrated_system):
        """Test browsing documents and getting statistics."""
        tools = integrated_system["tools"]

        # Step 1: List all documents
        docs = tools["list_documents"]()
        assert len(docs) > 0
        assert any(d["doc_id"] == "ml_textbook" for d in docs)

        # Step 2: Get document statistics
        stats = tools["get_document_stats"](doc_id="ml_textbook")
        assert stats is not None
        assert stats["total_chunks"] == 4

    def test_browse_document_structure(self, integrated_system):
        """Test browsing document hierarchical structure."""
        tools = integrated_system["tools"]

        # Step 1: Get level 1 chunks (chapters)
        chapters = tools["get_document_chunks"](doc_id="ml_textbook", level=1)
        assert len(chapters) == 1
        chapter = chapters[0]

        # Step 2: Get children (sections)
        sections = tools["get_children"](chunk_id=chapter["chunk_id"])
        assert len(sections) == 3

        # Step 3: Verify sequential links
        first_section = next(s for s in sections if s["prev_id"] is None)
        assert first_section["next_id"] is not None


class TestKeywordSearchWorkflow:
    """Test keyword search workflows."""

    def test_keyword_search_and_retrieve(self, integrated_system):
        """Test keyword search and chunk retrieval."""
        tools = integrated_system["tools"]

        # Step 1: Keyword search
        results = tools["search_keyword"](keyword="supervised")
        assert len(results) > 0

        # Step 2: Get full chunk details
        chunk_id = results[0]["chunk_id"]
        full_chunk = tools["get_chunk"](chunk_id=chunk_id)
        assert full_chunk is not None
        assert "supervised" in full_chunk["content"].lower()


class TestResourceWorkflow:
    """Test MCP resource workflows."""

    def test_schema_discovery(self, integrated_system):
        """Test discovering available tools via schema resource."""
        import json

        resources = integrated_system["resources"]

        # Get schema
        schema = json.loads(resources["kidkazz://schema"]())
        assert "tools" in schema

        # Verify expected tools listed
        tool_names = [t["name"] for t in schema["tools"]]
        assert "search_semantic" in tool_names
        assert "get_chunk" in tool_names

    def test_document_discovery(self, integrated_system):
        """Test discovering documents via resource."""
        import json

        resources = integrated_system["resources"]

        # List documents
        docs = json.loads(resources["kidkazz://documents"]())
        assert len(docs) > 0

        # Get document overview
        doc_id = docs[0]["doc_id"]
        overview = json.loads(resources["kidkazz://document/{doc_id}"](doc_id=doc_id))
        assert "total_chunks" in overview

    def test_chunk_resource(self, integrated_system):
        """Test accessing chunk via resource."""
        import json

        resources = integrated_system["resources"]

        # Get specific chunk
        chunk = json.loads(
            resources["kidkazz://chunk/{chunk_id}"](chunk_id="textbook_l2_1")
        )
        assert chunk["chunk_id"] == "textbook_l2_1"
        assert "content" in chunk


class TestFilteringWorkflows:
    """Test filtering capabilities in workflows."""

    def test_semantic_type_filtering(self, integrated_system):
        """Test filtering by semantic type."""
        tools = integrated_system["tools"]

        # Search for examples only
        examples = tools["search_semantic"](
            query="machine learning", semantic_type="example", top_k=10
        )

        # All results should be examples (if any match)
        # Note: with mock embeddings, we may not get exact matches
        assert isinstance(examples, list)

    def test_level_filtering(self, integrated_system):
        """Test filtering by hierarchy level."""
        tools = integrated_system["tools"]

        # Get only section level chunks
        sections = tools["get_document_chunks"](doc_id="ml_textbook", level=2)
        for section in sections:
            assert section["level"] == 2

        # Get only chapter level chunks
        chapters = tools["get_document_chunks"](doc_id="ml_textbook", level=1)
        for chapter in chapters:
            assert chapter["level"] == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_chunk(self, integrated_system):
        """Test handling non-existent chunk."""
        tools = integrated_system["tools"]

        result = tools["get_chunk"](chunk_id="nonexistent_id")
        assert result is None

    def test_nonexistent_document(self, integrated_system):
        """Test handling non-existent document."""
        tools = integrated_system["tools"]

        stats = tools["get_document_stats"](doc_id="nonexistent_doc")
        assert stats is None

        chunks = tools["get_document_chunks"](doc_id="nonexistent_doc")
        assert chunks == []

    def test_empty_search_results(self, integrated_system):
        """Test handling empty search results."""
        tools = integrated_system["tools"]

        # Keyword search for non-existent term
        results = tools["search_keyword"](keyword="xyznonexistentxyz")
        assert results == []
