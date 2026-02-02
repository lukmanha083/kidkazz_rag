"""Tests for storage schema definitions."""

import pytest

from src.storage.schema import (
    CHUNK_NODE_SCHEMA,
    DOCUMENT_NODE_SCHEMA,
    EDGE_TYPES,
    VECTOR_NODE_SCHEMA,
    SchemaConfig,
    create_kidkazz_schema,
    get_schema_definition,
    validate_chunk_node,
    validate_document_node,
)


class TestSchemaConfig:
    """Tests for SchemaConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = SchemaConfig()

        assert config.vector_dimensions == 1536
        assert config.vector_type == "Vec32"
        assert config.db_name == "kidkazz_rag"

    def test_custom_values(self):
        """Should accept custom values."""
        config = SchemaConfig(
            vector_dimensions=768,
            vector_type="Vec64",
            db_name="custom_db",
        )

        assert config.vector_dimensions == 768
        assert config.vector_type == "Vec64"
        assert config.db_name == "custom_db"


class TestNodeSchemas:
    """Tests for node schema definitions."""

    def test_document_schema_has_required_fields(self):
        """Document schema should have required fields."""
        assert "doc_id" in DOCUMENT_NODE_SCHEMA
        assert "title" in DOCUMENT_NODE_SCHEMA
        assert "created_at" in DOCUMENT_NODE_SCHEMA
        assert "chunk_count" in DOCUMENT_NODE_SCHEMA

    def test_chunk_schema_has_required_fields(self):
        """Chunk schema should have required fields."""
        required = ["chunk_id", "content", "level", "token_count"]
        for field in required:
            assert field in CHUNK_NODE_SCHEMA

    def test_chunk_schema_has_relationship_fields(self):
        """Chunk schema should have relationship fields."""
        relationship_fields = ["parent_id", "child_ids", "prev_id", "next_id"]
        for field in relationship_fields:
            assert field in CHUNK_NODE_SCHEMA

    def test_chunk_schema_has_metadata_fields(self):
        """Chunk schema should have metadata fields."""
        metadata_fields = [
            "semantic_type",
            "topic_tags",
            "has_table",
            "has_code",
            "has_math",
            "has_list",
        ]
        for field in metadata_fields:
            assert field in CHUNK_NODE_SCHEMA

    def test_vector_schema_has_embedding(self):
        """Vector schema should have embedding field."""
        assert "embedding" in VECTOR_NODE_SCHEMA
        assert VECTOR_NODE_SCHEMA["embedding"] == "Vec32"


class TestEdgeTypes:
    """Tests for edge type definitions."""

    def test_has_document_to_chunk_edge(self):
        """Should have HasChunk edge."""
        assert "HasChunk" in EDGE_TYPES
        assert EDGE_TYPES["HasChunk"] == ("Document", "Chunk")

    def test_has_parent_child_edge(self):
        """Should have ParentOf edge."""
        assert "ParentOf" in EDGE_TYPES
        assert EDGE_TYPES["ParentOf"] == ("Chunk", "Chunk")

    def test_has_sequential_edges(self):
        """Should have NextSibling and PrevSibling edges."""
        assert "NextSibling" in EDGE_TYPES
        assert "PrevSibling" in EDGE_TYPES
        assert EDGE_TYPES["NextSibling"] == ("Chunk", "Chunk")
        assert EDGE_TYPES["PrevSibling"] == ("Chunk", "Chunk")

    def test_has_embedding_edge(self):
        """Should have HasEmbedding edge."""
        assert "HasEmbedding" in EDGE_TYPES
        assert EDGE_TYPES["HasEmbedding"] == ("Chunk", "ChunkVector")


class TestGetSchemaDefinition:
    """Tests for get_schema_definition function."""

    def test_returns_complete_schema(self):
        """Should return complete schema definition."""
        schema = get_schema_definition()

        assert "nodes" in schema
        assert "edges" in schema

    def test_nodes_include_all_types(self):
        """Should include all node types."""
        schema = get_schema_definition()

        assert "Document" in schema["nodes"]
        assert "Chunk" in schema["nodes"]
        assert "ChunkVector" in schema["nodes"]

    def test_edges_match_edge_types(self):
        """Should include all edge types."""
        schema = get_schema_definition()

        for edge_name in EDGE_TYPES:
            assert edge_name in schema["edges"]


class TestCreateKidkazzSchema:
    """Tests for create_kidkazz_schema function."""

    def test_returns_schema_or_dict(self):
        """Should return schema object or dictionary."""
        result = create_kidkazz_schema()

        # Without helix-py, returns dict
        if isinstance(result, dict):
            assert "nodes" in result
            assert "edges" in result

    def test_accepts_custom_config(self):
        """Should accept custom configuration."""
        config = SchemaConfig(vector_dimensions=768)
        result = create_kidkazz_schema(config)

        # Should not raise
        assert result is not None


class TestValidateChunkNode:
    """Tests for validate_chunk_node function."""

    def test_valid_chunk_returns_empty_list(self):
        """Should return empty list for valid chunk."""
        node = {
            "chunk_id": "test_1",
            "content": "Test content",
            "level": 2,
            "token_count": 10,
        }
        errors = validate_chunk_node(node)

        assert errors == []

    def test_missing_required_field(self):
        """Should return error for missing required field."""
        node = {
            "content": "Test",
            "level": 2,
            "token_count": 10,
        }
        errors = validate_chunk_node(node)

        assert any("chunk_id" in e for e in errors)

    def test_invalid_level_type(self):
        """Should return error for invalid level type."""
        node = {
            "chunk_id": "test_1",
            "content": "Test",
            "level": "two",  # Should be int
            "token_count": 10,
        }
        errors = validate_chunk_node(node)

        assert any("level" in e and "integer" in e for e in errors)

    def test_invalid_level_value(self):
        """Should return error for invalid level value."""
        node = {
            "chunk_id": "test_1",
            "content": "Test",
            "level": 5,  # Should be 0, 1, or 2
            "token_count": 10,
        }
        errors = validate_chunk_node(node)

        assert any("level" in e and ("0" in e or "1" in e or "2" in e) for e in errors)

    def test_invalid_semantic_type(self):
        """Should return error for invalid semantic type."""
        node = {
            "chunk_id": "test_1",
            "content": "Test",
            "level": 2,
            "token_count": 10,
            "semantic_type": "invalid_type",
        }
        errors = validate_chunk_node(node)

        assert any("semantic_type" in e for e in errors)

    def test_valid_semantic_types(self):
        """Should accept all valid semantic types."""
        valid_types = ["definition", "example", "procedure", "theorem", "narrative"]

        for sem_type in valid_types:
            node = {
                "chunk_id": "test_1",
                "content": "Test",
                "level": 2,
                "token_count": 10,
                "semantic_type": sem_type,
            }
            errors = validate_chunk_node(node)
            assert errors == [], f"Failed for semantic_type: {sem_type}"


class TestValidateDocumentNode:
    """Tests for validate_document_node function."""

    def test_valid_document_returns_empty_list(self):
        """Should return empty list for valid document."""
        node = {
            "doc_id": "doc_1",
            "title": "Test Document",
            "chunk_count": 10,
        }
        errors = validate_document_node(node)

        assert errors == []

    def test_missing_doc_id(self):
        """Should return error for missing doc_id."""
        node = {
            "title": "Test Document",
        }
        errors = validate_document_node(node)

        assert any("doc_id" in e for e in errors)

    def test_missing_title(self):
        """Should return error for missing title."""
        node = {
            "doc_id": "doc_1",
        }
        errors = validate_document_node(node)

        assert any("title" in e for e in errors)

    def test_invalid_chunk_count_type(self):
        """Should return error for invalid chunk_count type."""
        node = {
            "doc_id": "doc_1",
            "title": "Test",
            "chunk_count": "ten",  # Should be int
        }
        errors = validate_document_node(node)

        assert any("chunk_count" in e for e in errors)


class TestHeaderSchemaFields:
    """Tests for header-related schema fields (TDD)."""

    def test_chunk_schema_has_header_text_field(self):
        """Chunk schema should have header_text field."""
        assert "header_text" in CHUNK_NODE_SCHEMA
        assert CHUNK_NODE_SCHEMA["header_text"] == "String"

    def test_chunk_schema_has_header_level_field(self):
        """Chunk schema should have header_level field."""
        assert "header_level" in CHUNK_NODE_SCHEMA
        assert CHUNK_NODE_SCHEMA["header_level"] == "U32"

    def test_chunk_schema_has_block_type_field(self):
        """Chunk schema should have block_type field."""
        assert "block_type" in CHUNK_NODE_SCHEMA
        assert CHUNK_NODE_SCHEMA["block_type"] == "String"

    def test_validate_chunk_with_header_fields(self):
        """Should validate chunk with header fields."""
        node = {
            "chunk_id": "test_1",
            "content": "Test content",
            "level": 2,
            "token_count": 10,
            "header_text": "Introduction",
            "header_level": 1,
            "block_type": "Header",
        }
        errors = validate_chunk_node(node)
        assert errors == []

    def test_validate_chunk_with_invalid_header_level(self):
        """Should return error for invalid header_level value."""
        node = {
            "chunk_id": "test_1",
            "content": "Test content",
            "level": 2,
            "token_count": 10,
            "header_level": 7,  # Should be 0-6
        }
        errors = validate_chunk_node(node)
        assert any("header_level" in e for e in errors)

    def test_validate_chunk_with_valid_header_levels(self):
        """Should accept valid header levels 0-6 (0 means no header)."""
        for level in range(0, 7):  # 0-6 are valid
            node = {
                "chunk_id": "test_1",
                "content": "Test content",
                "level": 2,
                "token_count": 10,
                "header_level": level,
            }
            errors = validate_chunk_node(node)
            assert errors == [], f"Failed for header_level: {level}"
