"""Tests for MCP multimodal image response features."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.chunker import Chunk, ChunkMetadata, EmbeddedChunk
from src.chunker.embedder import MockEmbedder
from src.chunker.metadata import detect_content_features
from src.mcp_server.config import MCPServerConfig, ServerState
from src.mcp_server.formatters import IMAGE_MARKER_RE, extract_chunk_images, format_chunk
from src.mcp_server.tools import register_tools
from src.storage.mock_store import MockChunkStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    """Create an empty mock store."""
    return MockChunkStore()


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    return MockEmbedder(embedding_dim=384)


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary PNG file for testing."""
    img = tmp_path / "test_table.png"
    # Minimal 1x1 PNG (valid header)
    img.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
        b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00"
        b"\x00\x00\x00IEND\xaeB`\x82"
    )
    return img


@pytest.fixture
def sample_chunks_with_images(temp_image):
    """Create sample chunks — some with image markers, some without."""
    parent = Chunk(
        id="doc_l1_1",
        content="Parent section about inventory tables",
        level=1,
        token_count=30,
        child_ids=["doc_l2_1", "doc_l2_2", "doc_l2_3"],
    )
    # Chunk WITH an image marker pointing to temp file
    child_with_img = Chunk(
        id="doc_l2_1",
        content=f"Table 1 shows the data.\n\n![Inventory table]({temp_image})\n\nAs shown above.",
        level=2,
        token_count=20,
        parent_id="doc_l1_1",
        next_id="doc_l2_2",
        section_path=["Tables"],
    )
    # Chunk WITHOUT image
    child_no_img = Chunk(
        id="doc_l2_2",
        content="This section has no images, just text about warehouse management.",
        level=2,
        token_count=15,
        parent_id="doc_l1_1",
        prev_id="doc_l2_1",
        next_id="doc_l2_3",
        section_path=["Text"],
    )
    # Chunk with image marker to non-existent file
    child_broken_img = Chunk(
        id="doc_l2_3",
        content="Figure 2 shows something.\n\n![Missing figure](/tmp/nonexistent_12345.png)",
        level=2,
        token_count=15,
        parent_id="doc_l1_1",
        prev_id="doc_l2_2",
        section_path=["Figures"],
    )
    return [parent, child_with_img, child_no_img, child_broken_img]


@pytest.fixture
def sample_metadata_with_images(sample_chunks_with_images):
    """Create metadata for sample chunks (with has_image flags)."""
    return [
        ChunkMetadata(
            chunk_id=sample_chunks_with_images[0].id,
            document_id="test_doc",
            semantic_type="narrative",
            sequence_position=0,
        ),
        ChunkMetadata(
            chunk_id=sample_chunks_with_images[1].id,
            document_id="test_doc",
            semantic_type="narrative",
            sequence_position=1,
            sibling_ids=["doc_l2_2", "doc_l2_3"],
            has_image=True,
        ),
        ChunkMetadata(
            chunk_id=sample_chunks_with_images[2].id,
            document_id="test_doc",
            semantic_type="narrative",
            sequence_position=2,
            sibling_ids=["doc_l2_1", "doc_l2_3"],
            has_image=False,
        ),
        ChunkMetadata(
            chunk_id=sample_chunks_with_images[3].id,
            document_id="test_doc",
            semantic_type="narrative",
            sequence_position=3,
            sibling_ids=["doc_l2_1", "doc_l2_2"],
            has_image=True,  # Metadata says True but file doesn't exist
        ),
    ]


@pytest.fixture
def populated_store(mock_store, mock_embedder, sample_chunks_with_images, sample_metadata_with_images):
    """Store with sample data including image chunks."""
    embedded = mock_embedder.embed_chunks(sample_chunks_with_images)
    mock_store.store_document("test_doc", "Test Document", embedded, sample_metadata_with_images)
    return mock_store


@pytest.fixture
def server_state(populated_store, mock_embedder):
    """Server state wired to the populated store."""
    config = MCPServerConfig(store_type="mock", embedder_type="mock")
    state = ServerState(config)
    state._store = populated_store
    state._embedder = MagicMock()
    state._embedder.embed_text = mock_embedder.embed_text
    state._embedder.embed_query = mock_embedder.embed_query
    return state


@pytest.fixture
def registered_tools(server_state):
    """Register tools and return dict of tool functions."""
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


# ---------------------------------------------------------------------------
# Tests: extract_chunk_images utility
# ---------------------------------------------------------------------------

class TestExtractChunkImages:
    """Tests for extract_chunk_images helper."""

    def test_valid_paths_return_images(self, temp_image):
        """Should return Image objects for existing image files."""
        content = f"See the table:\n\n![Table]({temp_image})\n\nEnd."
        images = extract_chunk_images(content)
        assert len(images) == 1

    def test_missing_paths_return_empty(self):
        """Should return empty list for non-existent image paths."""
        content = "![Missing](/tmp/does_not_exist_9999.png)"
        images = extract_chunk_images(content)
        assert images == []

    def test_no_markers_return_empty(self):
        """Should return empty list for content without image markers."""
        content = "Just plain text with no images at all."
        images = extract_chunk_images(content)
        assert images == []

    def test_multiple_images(self, tmp_path):
        """Should return Image objects for all existing images."""
        img1 = tmp_path / "a.png"
        img2 = tmp_path / "b.jpg"
        img1.write_bytes(b"\x89PNG" + b"\x00" * 20)
        img2.write_bytes(b"\xff\xd8\xff" + b"\x00" * 20)

        content = f"![A]({img1})\n![B]({img2})"
        images = extract_chunk_images(content)
        assert len(images) == 2

    def test_non_image_extensions_skipped(self, tmp_path):
        """Should skip non-image files like .txt."""
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("not an image")
        content = f"![Notes]({txt_file})"
        images = extract_chunk_images(content)
        assert images == []


# ---------------------------------------------------------------------------
# Tests: has_image metadata detection
# ---------------------------------------------------------------------------

class TestHasImageMetadata:
    """Tests for has_image in detect_content_features."""

    def test_detects_image_marker(self):
        """Should detect markdown image syntax."""
        content = "See ![Table 1](/path/to/table.png) for details."
        features = detect_content_features(content)
        assert features["has_image"] is True

    def test_no_image_marker(self):
        """Should return False for content without images."""
        content = "Just regular text."
        features = detect_content_features(content)
        assert features["has_image"] is False

    def test_image_with_alt_text(self):
        """Should detect images with various alt text."""
        content = "![Warehouse layout diagram](images/layout.jpg)"
        features = detect_content_features(content)
        assert features["has_image"] is True

    def test_image_empty_alt(self):
        """Should detect images with empty alt text."""
        content = "![](path/to/image.png)"
        features = detect_content_features(content)
        assert features["has_image"] is True

    def test_chunk_metadata_has_image_field(self):
        """ChunkMetadata should have has_image field."""
        meta = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            semantic_type="narrative",
            has_image=True,
        )
        assert meta.has_image is True

    def test_chunk_metadata_has_image_defaults_false(self):
        """has_image should default to False."""
        meta = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            semantic_type="narrative",
        )
        assert meta.has_image is False

    def test_to_dict_includes_has_image(self):
        """to_dict should include has_image."""
        meta = ChunkMetadata(
            chunk_id="test",
            document_id="doc",
            semantic_type="narrative",
            has_image=True,
        )
        d = meta.to_dict()
        assert d["has_image"] is True


# ---------------------------------------------------------------------------
# Tests: format_chunk includes has_image
# ---------------------------------------------------------------------------

class TestFormatChunkHasImage:
    """Tests for has_image in format_chunk output."""

    def test_format_chunk_includes_has_image_true(self, temp_image):
        """format_chunk should include has_image: True for chunks with images."""
        chunk = Chunk(
            id="c1",
            content=f"![Table]({temp_image})",
            level=2,
            token_count=5,
            metadata={"has_image": True},
        )
        ec = EmbeddedChunk(chunk=chunk, embedding=[0.0] * 3, model_name="test")
        result = format_chunk(ec)
        assert result["has_image"] is True

    def test_format_chunk_includes_has_image_false(self):
        """format_chunk should include has_image: False for text-only chunks."""
        chunk = Chunk(
            id="c1",
            content="Plain text, no images.",
            level=2,
            token_count=5,
            metadata={},
        )
        ec = EmbeddedChunk(chunk=chunk, embedding=[0.0] * 3, model_name="test")
        result = format_chunk(ec)
        assert result["has_image"] is False

    def test_format_chunk_dynamic_detection(self, temp_image):
        """format_chunk should dynamically detect has_image from content if not in metadata."""
        chunk = Chunk(
            id="c1",
            content=f"![Figure]({temp_image})",
            level=2,
            token_count=5,
            metadata={},  # No has_image key
        )
        ec = EmbeddedChunk(chunk=chunk, embedding=[0.0] * 3, model_name="test")
        result = format_chunk(ec)
        assert result["has_image"] is True


# ---------------------------------------------------------------------------
# Tests: search_semantic with images
# ---------------------------------------------------------------------------

class TestSearchSemanticImages:
    """Tests for search_semantic include_images and has_image filter."""

    def test_default_no_images(self, registered_tools):
        """Default search (include_images=False) returns list[dict]."""
        search = registered_tools["search_semantic"]
        results = search(query="inventory")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, dict)

    def test_include_images_returns_mixed_list(self, registered_tools):
        """include_images=True returns dicts interleaved with Image objects."""
        search = registered_tools["search_semantic"]
        results = search(query="table", include_images=True)
        assert isinstance(results, list)
        # At least one dict should be present
        dicts = [r for r in results if isinstance(r, dict)]
        assert len(dicts) > 0

    def test_has_image_filter_true(self, registered_tools):
        """has_image=True should only return chunks with images."""
        search = registered_tools["search_semantic"]
        results = search(query="table", has_image=True)
        for r in results:
            assert r.get("has_image") is True

    def test_has_image_filter_false(self, registered_tools):
        """has_image=False should only return chunks without images."""
        search = registered_tools["search_semantic"]
        results = search(query="warehouse", has_image=False)
        for r in results:
            assert r.get("has_image") is False


# ---------------------------------------------------------------------------
# Tests: get_chunk_images tool
# ---------------------------------------------------------------------------

class TestGetChunkImages:
    """Tests for get_chunk_images tool."""

    def test_chunk_with_images(self, registered_tools, temp_image):
        """Should return metadata + Image objects for chunk with images."""
        get_images = registered_tools["get_chunk_images"]
        result = get_images(chunk_id="doc_l2_1")
        assert isinstance(result, list)
        assert len(result) >= 1
        # First element is metadata
        assert result[0]["chunk_id"] == "doc_l2_1"
        assert result[0]["images_found"] >= 1

    def test_chunk_no_images(self, registered_tools):
        """Should return images_found: 0 for chunk without images."""
        get_images = registered_tools["get_chunk_images"]
        result = get_images(chunk_id="doc_l2_2")
        assert isinstance(result, list)
        assert result[0]["chunk_id"] == "doc_l2_2"
        assert result[0]["images_found"] == 0

    def test_chunk_broken_image_path(self, registered_tools):
        """Should return images_found: 0 when image path doesn't exist."""
        get_images = registered_tools["get_chunk_images"]
        result = get_images(chunk_id="doc_l2_3")
        assert isinstance(result, list)
        assert result[0]["images_found"] == 0

    def test_nonexistent_chunk(self, registered_tools):
        """Should return error for non-existent chunk."""
        get_images = registered_tools["get_chunk_images"]
        result = get_images(chunk_id="nonexistent")
        assert isinstance(result, list)
        assert "error" in result[0]


# ---------------------------------------------------------------------------
# Tests: has_image filter in MockChunkStore
# ---------------------------------------------------------------------------

class TestMockStoreHasImageFilter:
    """Tests for has_image filter in MockChunkStore.search_similar."""

    def test_filter_has_image_true(self, populated_store, mock_embedder):
        """search_similar with has_image=True should only return image chunks."""
        query_emb = mock_embedder.embed_query("test")
        results = populated_store.search_similar(
            query_embedding=query_emb,
            top_k=10,
            has_image=True,
        )
        for ec, score in results:
            meta = ec.chunk.metadata or {}
            assert meta.get("has_image", False) is True

    def test_filter_has_image_false(self, populated_store, mock_embedder):
        """search_similar with has_image=False should exclude image chunks."""
        query_emb = mock_embedder.embed_query("test")
        results = populated_store.search_similar(
            query_embedding=query_emb,
            top_k=10,
            has_image=False,
        )
        for ec, score in results:
            meta = ec.chunk.metadata or {}
            assert meta.get("has_image", False) is False

    def test_filter_has_image_none(self, populated_store, mock_embedder):
        """search_similar with has_image=None should not filter by image status."""
        query_emb = mock_embedder.embed_query("test")
        results_all = populated_store.search_similar(
            query_embedding=query_emb,
            top_k=10,
            has_image=None,
        )
        results_true = populated_store.search_similar(
            query_embedding=query_emb,
            top_k=10,
            has_image=True,
        )
        results_false = populated_store.search_similar(
            query_embedding=query_emb,
            top_k=10,
            has_image=False,
        )
        # Unfiltered should return at least as many as either filtered set
        assert len(results_all) >= len(results_true)
        assert len(results_all) >= len(results_false)
