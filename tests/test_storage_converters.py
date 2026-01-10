"""Tests for storage converters."""

import json

import pytest

from src.chunker import Chunk, ChunkMetadata, EmbeddedChunk
from src.storage.converters import (
    chunk_to_helix_node,
    document_to_helix_node,
    embedded_chunk_to_helix_vector,
    helix_node_to_chunk,
    helix_node_to_document,
    helix_node_to_metadata,
    helix_to_embedded_chunk,
)


class TestChunkToHelixNode:
    """Tests for chunk_to_helix_node function."""

    def test_converts_basic_chunk(self):
        """Should convert basic chunk fields."""
        chunk = Chunk(
            id="test_1",
            content="Test content",
            level=2,
            token_count=10,
        )
        result = chunk_to_helix_node(chunk)

        assert result["chunk_id"] == "test_1"
        assert result["content"] == "Test content"
        assert result["level"] == 2
        assert result["token_count"] == 10

    def test_converts_relationships(self):
        """Should convert relationship fields."""
        chunk = Chunk(
            id="test_1",
            content="Test",
            level=2,
            token_count=5,
            parent_id="parent_1",
            child_ids=["child_1", "child_2"],
            prev_id="prev_1",
            next_id="next_1",
        )
        result = chunk_to_helix_node(chunk)

        assert result["parent_id"] == "parent_1"
        assert json.loads(result["child_ids"]) == ["child_1", "child_2"]
        assert result["prev_id"] == "prev_1"
        assert result["next_id"] == "next_1"

    def test_converts_section_path(self):
        """Should convert section_path as JSON."""
        chunk = Chunk(
            id="test_1",
            content="Test",
            level=2,
            token_count=5,
            section_path=["Chapter 1", "Section 1.1"],
        )
        result = chunk_to_helix_node(chunk)

        assert json.loads(result["section_path"]) == ["Chapter 1", "Section 1.1"]

    def test_includes_metadata_when_provided(self):
        """Should include metadata fields when metadata provided."""
        chunk = Chunk(id="test_1", content="Test", level=2, token_count=5)
        metadata = ChunkMetadata(
            chunk_id="test_1",
            document_id="doc_1",
            semantic_type="definition",
            topic_tags=["topic1", "topic2"],
            sequence_position=3,
            has_table=True,
            has_code=False,
            has_math=True,
            has_list=False,
        )

        result = chunk_to_helix_node(chunk, metadata)

        assert result["document_id"] == "doc_1"
        assert result["semantic_type"] == "definition"
        assert json.loads(result["topic_tags"]) == ["topic1", "topic2"]
        assert result["sequence_position"] == 3
        # Helix-DB uses U32 for boolean fields, so values are 0 or 1
        assert result["has_table"] == 1
        assert result["has_code"] == 0
        assert result["has_math"] == 1
        assert result["has_list"] == 0

    def test_handles_none_optional_fields(self):
        """Should handle None values for optional fields."""
        chunk = Chunk(
            id="test_1",
            content="Test",
            level=2,
            token_count=5,
            parent_id=None,
            source_section=None,
        )
        result = chunk_to_helix_node(chunk)

        assert result["parent_id"] == ""
        assert result["source_section"] == ""


class TestHelixNodeToChunk:
    """Tests for helix_node_to_chunk function."""

    def test_reconstructs_basic_chunk(self):
        """Should reconstruct chunk from node properties."""
        node = {
            "chunk_id": "test_1",
            "content": "Test content",
            "level": 2,
            "token_count": 10,
        }
        result = helix_node_to_chunk(node)

        assert result.id == "test_1"
        assert result.content == "Test content"
        assert result.level == 2
        assert result.token_count == 10

    def test_reconstructs_relationships(self):
        """Should reconstruct relationship fields."""
        node = {
            "chunk_id": "test_1",
            "content": "Test",
            "level": 2,
            "token_count": 5,
            "parent_id": "parent_1",
            "child_ids": '["child_1", "child_2"]',
            "prev_id": "prev_1",
            "next_id": "next_1",
        }
        result = helix_node_to_chunk(node)

        assert result.parent_id == "parent_1"
        assert result.child_ids == ["child_1", "child_2"]
        assert result.prev_id == "prev_1"
        assert result.next_id == "next_1"

    def test_reconstructs_section_path(self):
        """Should reconstruct section_path from JSON."""
        node = {
            "chunk_id": "test_1",
            "content": "Test",
            "level": 2,
            "token_count": 5,
            "section_path": '["Chapter 1", "Section 1.1"]',
        }
        result = helix_node_to_chunk(node)

        assert result.section_path == ["Chapter 1", "Section 1.1"]

    def test_handles_empty_optional_fields(self):
        """Should handle empty strings as None."""
        node = {
            "chunk_id": "test_1",
            "content": "Test",
            "level": 2,
            "token_count": 5,
            "parent_id": "",
            "prev_id": "",
            "next_id": "",
        }
        result = helix_node_to_chunk(node)

        assert result.parent_id is None
        assert result.prev_id is None
        assert result.next_id is None


class TestHelixNodeToMetadata:
    """Tests for helix_node_to_metadata function."""

    def test_reconstructs_metadata(self):
        """Should reconstruct metadata from node properties."""
        node = {
            "chunk_id": "test_1",
            "document_id": "doc_1",
            "semantic_type": "definition",
            "topic_tags": '["topic1", "topic2"]',
            "token_count": 10,
            "word_count": 5,
            "level": 2,
            "sequence_position": 3,
            "has_table": True,
            "has_code": False,
        }
        result = helix_node_to_metadata(node)

        assert result.chunk_id == "test_1"
        assert result.document_id == "doc_1"
        assert result.semantic_type == "definition"
        assert result.topic_tags == ["topic1", "topic2"]
        assert result.token_count == 10
        assert result.word_count == 5
        assert result.hierarchy_level == 2
        assert result.sequence_position == 3
        assert result.has_table is True
        assert result.has_code is False

    def test_uses_defaults_for_missing_fields(self):
        """Should use defaults for missing optional fields."""
        node = {
            "chunk_id": "test_1",
        }
        result = helix_node_to_metadata(node)

        assert result.document_id == ""
        assert result.semantic_type == "narrative"
        assert result.topic_tags == []
        assert result.token_count == 0


class TestEmbeddedChunkToHelixVector:
    """Tests for embedded_chunk_to_helix_vector function."""

    def test_extracts_embedding(self):
        """Should extract embedding vector."""
        chunk = Chunk(id="test_1", content="Test", level=2, token_count=5)
        ec = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model",
        )
        result = embedded_chunk_to_helix_vector(ec)

        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["model_name"] == "test-model"
        assert result["embedding_dim"] == 3


class TestHelixToEmbeddedChunk:
    """Tests for helix_to_embedded_chunk function."""

    def test_reconstructs_embedded_chunk(self):
        """Should reconstruct EmbeddedChunk from Helix data."""
        node = {
            "chunk_id": "test_1",
            "content": "Test content",
            "level": 2,
            "token_count": 10,
        }
        embedding = [0.1, 0.2, 0.3]
        model_name = "test-model"

        result = helix_to_embedded_chunk(node, embedding, model_name)

        assert result.chunk.id == "test_1"
        assert result.chunk.content == "Test content"
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.model_name == "test-model"


class TestDocumentConverters:
    """Tests for document conversion functions."""

    def test_document_to_helix_node(self):
        """Should convert document properties."""
        result = document_to_helix_node(
            doc_id="doc_1",
            title="Test Document",
            chunk_count=10,
            created_at=1234567890,
        )

        assert result["doc_id"] == "doc_1"
        assert result["title"] == "Test Document"
        assert result["chunk_count"] == 10
        assert result["created_at"] == 1234567890

    def test_document_to_helix_node_auto_timestamp(self):
        """Should auto-generate timestamp if not provided."""
        result = document_to_helix_node(
            doc_id="doc_1",
            title="Test",
            chunk_count=5,
        )

        assert result["created_at"] > 0

    def test_helix_node_to_document(self):
        """Should reconstruct document from node."""
        node = {
            "doc_id": "doc_1",
            "title": "Test Document",
            "chunk_count": 10,
            "created_at": 1234567890,
        }
        result = helix_node_to_document(node)

        assert result["doc_id"] == "doc_1"
        assert result["title"] == "Test Document"
        assert result["chunk_count"] == 10
        assert result["created_at"] == 1234567890


class TestRoundTrip:
    """Tests for round-trip conversion."""

    def test_chunk_roundtrip(self):
        """Should preserve chunk data through conversion."""
        original = Chunk(
            id="test_1",
            content="Test content",
            level=2,
            token_count=10,
            parent_id="parent_1",
            child_ids=["child_1"],
            prev_id="prev_1",
            next_id="next_1",
            section_path=["Chapter 1"],
            source_section="Introduction",
        )

        helix_node = chunk_to_helix_node(original)
        reconstructed = helix_node_to_chunk(helix_node)

        assert reconstructed.id == original.id
        assert reconstructed.content == original.content
        assert reconstructed.level == original.level
        assert reconstructed.token_count == original.token_count
        assert reconstructed.parent_id == original.parent_id
        assert reconstructed.child_ids == original.child_ids
        assert reconstructed.prev_id == original.prev_id
        assert reconstructed.next_id == original.next_id
        assert reconstructed.section_path == original.section_path
        assert reconstructed.source_section == original.source_section

    def test_embedded_chunk_roundtrip(self):
        """Should preserve embedded chunk data through conversion."""
        chunk = Chunk(id="test_1", content="Test", level=2, token_count=5)
        original = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model",
        )

        helix_node = chunk_to_helix_node(original.chunk)
        vector_data = embedded_chunk_to_helix_vector(original)
        reconstructed = helix_to_embedded_chunk(
            helix_node,
            vector_data["embedding"],
            vector_data["model_name"],
        )

        assert reconstructed.chunk.id == original.chunk.id
        assert reconstructed.embedding == original.embedding
        assert reconstructed.model_name == original.model_name
