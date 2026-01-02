"""Tests for MCP resource implementations."""

import json
from unittest.mock import MagicMock

import pytest

from src.chunker import Chunk, ChunkMetadata, EmbeddedChunk
from src.chunker.embedder import MockEmbedder
from src.mcp_server.config import MCPServerConfig, ServerState
from src.mcp_server.resources import SCHEMA_INFO, register_resources
from src.storage.mock_store import MockChunkStore


@pytest.fixture
def mock_store():
    """Create a mock store."""
    return MockChunkStore()


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    return MockEmbedder(embedding_dim=384)


@pytest.fixture
def sample_chunks():
    """Create sample chunks."""
    return [
        Chunk(
            id="doc_l1_1",
            content="Parent content",
            level=1,
            token_count=20,
            child_ids=["doc_l2_1"],
        ),
        Chunk(
            id="doc_l2_1",
            content="Child content",
            level=2,
            token_count=10,
            parent_id="doc_l1_1",
        ),
    ]


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
            sibling_ids=[],
        ),
    ]


@pytest.fixture
def populated_store(mock_store, mock_embedder, sample_chunks, sample_metadata):
    """Create store with sample data."""
    embedded = mock_embedder.embed_chunks(sample_chunks)
    mock_store.store_document("test_doc", "Test Document", embedded, sample_metadata)
    return mock_store


@pytest.fixture
def server_state(populated_store):
    """Create server state with populated store."""
    config = MCPServerConfig(store_type="mock", embedder_type="mock")
    state = ServerState(config)
    state._store = populated_store
    return state


@pytest.fixture
def registered_resources(server_state):
    """Register resources and return mock MCP with captured resources."""
    resources = {}

    class MockMCP:
        def resource(self, uri):
            def decorator(func):
                resources[uri] = func
                return func

            return decorator

    mcp = MockMCP()
    register_resources(mcp, server_state)
    return resources


class TestSchemaInfo:
    """Tests for SCHEMA_INFO constant."""

    def test_schema_has_name(self):
        """Test schema has name field."""
        assert "name" in SCHEMA_INFO
        assert SCHEMA_INFO["name"] == "KidKazz RAG Knowledge Base"

    def test_schema_has_version(self):
        """Test schema has version field."""
        assert "version" in SCHEMA_INFO
        assert SCHEMA_INFO["version"] == "1.0.0"

    def test_schema_has_chunk_levels(self):
        """Test schema describes chunk levels."""
        assert "chunk_levels" in SCHEMA_INFO
        assert "1" in SCHEMA_INFO["chunk_levels"]
        assert "2" in SCHEMA_INFO["chunk_levels"]

    def test_schema_has_semantic_types(self):
        """Test schema lists semantic types."""
        assert "semantic_types" in SCHEMA_INFO
        expected_types = ["definition", "example", "procedure", "theorem", "narrative"]
        assert SCHEMA_INFO["semantic_types"] == expected_types

    def test_schema_has_tools(self):
        """Test schema lists available tools."""
        assert "tools" in SCHEMA_INFO
        tool_names = [t["name"] for t in SCHEMA_INFO["tools"]]
        assert "search_semantic" in tool_names
        assert "get_chunk" in tool_names


class TestGetSchemaResource:
    """Tests for kidkazz://schema resource."""

    def test_returns_json(self, registered_resources):
        """Test schema resource returns valid JSON."""
        get_schema = registered_resources["kidkazz://schema"]
        result = get_schema()
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_returns_schema_info(self, registered_resources):
        """Test schema resource returns schema info."""
        get_schema = registered_resources["kidkazz://schema"]
        result = json.loads(get_schema())
        assert result["name"] == SCHEMA_INFO["name"]
        assert result["version"] == SCHEMA_INFO["version"]


class TestGetDocumentsResource:
    """Tests for kidkazz://documents resource."""

    def test_returns_json(self, registered_resources):
        """Test documents resource returns valid JSON."""
        get_docs = registered_resources["kidkazz://documents"]
        result = get_docs()
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_returns_document_list(self, registered_resources):
        """Test documents resource returns document list."""
        get_docs = registered_resources["kidkazz://documents"]
        result = json.loads(get_docs())
        assert len(result) > 0
        assert "doc_id" in result[0]
        assert "title" in result[0]


class TestGetDocumentOverviewResource:
    """Tests for kidkazz://document/{doc_id} resource."""

    def test_existing_document(self, registered_resources):
        """Test getting existing document overview."""
        get_doc = registered_resources["kidkazz://document/{doc_id}"]
        result = json.loads(get_doc(doc_id="test_doc"))
        assert "total_chunks" in result
        assert "chunks_by_level" in result

    def test_nonexistent_document(self, registered_resources):
        """Test getting non-existent document returns error."""
        get_doc = registered_resources["kidkazz://document/{doc_id}"]
        result = json.loads(get_doc(doc_id="nonexistent"))
        assert "error" in result


class TestGetChunkContentResource:
    """Tests for kidkazz://chunk/{chunk_id} resource."""

    def test_existing_chunk(self, registered_resources):
        """Test getting existing chunk content."""
        get_chunk = registered_resources["kidkazz://chunk/{chunk_id}"]
        result = json.loads(get_chunk(chunk_id="doc_l2_1"))
        assert "chunk_id" in result
        assert "content" in result
        assert result["chunk_id"] == "doc_l2_1"

    def test_nonexistent_chunk(self, registered_resources):
        """Test getting non-existent chunk returns error."""
        get_chunk = registered_resources["kidkazz://chunk/{chunk_id}"]
        result = json.loads(get_chunk(chunk_id="nonexistent"))
        assert "error" in result
