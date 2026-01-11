"""Helix-DB schema definitions for KidKazz RAG.

This module defines the schema for storing hierarchical chunks with vector
embeddings in Helix-DB.

Schema Overview:
- Document node: Top-level container for chunks
- Chunk node: Text content with metadata
- ChunkVector: 384-dimensional embedding vector

Relationships:
- Document --HasChunk--> Chunk
- Chunk --ParentOf--> Chunk (L1 -> L2)
- Chunk --NextSibling--> Chunk (sequential)
- Chunk --HasEmbedding--> ChunkVector
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SchemaConfig:
    """Configuration for Helix-DB schema.

    Attributes:
        vector_dimensions: Embedding vector size (default: 384 for bge-small)
        vector_type: Helix-DB vector type (Vec32 for float32)
        db_name: Database name for initialization
    """

    vector_dimensions: int = 384
    vector_type: str = "Vec32"
    db_name: str = "kidkazz_rag"


# Node type definitions (for documentation and validation)
DOCUMENT_NODE_SCHEMA: dict[str, str] = {
    "doc_id": "String",
    "title": "String",
    "tags": "String",  # JSON array of document tags
    "created_at": "I64",
    "chunk_count": "U32",
}

CHUNK_NODE_SCHEMA: dict[str, str] = {
    "chunk_id": "String",
    "content": "String",
    "level": "U32",
    "token_count": "U32",
    "word_count": "U32",
    "document_id": "String",
    "semantic_type": "String",
    "topic_tags": "String",  # JSON array
    "section_path": "String",  # JSON array
    "source_section": "String",
    "sequence_position": "U32",
    "parent_id": "String",
    "child_ids": "String",  # JSON array
    "sibling_ids": "String",  # JSON array
    "prev_id": "String",
    "next_id": "String",
    "has_table": "Bool",
    "has_code": "Bool",
    "has_math": "Bool",
    "has_list": "Bool",
    # Header/block metadata (from Reducto block mode)
    "header_text": "String",  # Actual header text if this is a header block
    "header_level": "U32",  # 0 if not a header, 1-6 for h1-h6
    "block_type": "String",  # Block type from Reducto: "Header", "Text", "Table", etc.
}

VECTOR_NODE_SCHEMA: dict[str, str] = {
    "embedding": "Vec32",  # 384-dimensional float32 vector
    "model_name": "String",
    "embedding_dim": "U32",
}

# Table node schema (Phase 2: Table Processing)
TABLE_NODE_SCHEMA: dict[str, str] = {
    "table_id": "String",
    "raw_markdown": "String",
    "summary_text": "String",
    "column_names": "String",  # JSON array
    "column_types": "String",  # JSON array
    "row_count": "U32",
    "column_count": "U32",
    "source_chunk_id": "String",
    "source_document_id": "String",
    "surrounding_context": "String",
    "key_columns": "String",  # JSON array
    "key_values": "String",  # JSON array
}

# Table vector schema for summary embeddings
TABLE_VECTOR_SCHEMA: dict[str, str] = {
    "embedding": "Vec32",  # Summary embedding for retrieval
    "model_name": "String",
    "embedding_dim": "U32",
}

# Edge type definitions
EDGE_TYPES: dict[str, tuple[str, str]] = {
    "HasChunk": ("Document", "Chunk"),
    "ParentOf": ("Chunk", "Chunk"),
    "NextSibling": ("Chunk", "Chunk"),
    "PrevSibling": ("Chunk", "Chunk"),
    "SiblingOf": ("Chunk", "Chunk"),
    "HasEmbedding": ("Chunk", "ChunkVector"),
    # Table edges (Phase 2)
    "HasTable": ("Document", "Table"),
    "ChunkContainsTable": ("Chunk", "Table"),
    "TableRelatedToConcept": ("Table", "Concept"),
    "HasTableEmbedding": ("Table", "TableVector"),
}


def get_schema_definition() -> dict[str, Any]:
    """
    Get complete schema definition as a dictionary.

    Returns:
        Dictionary with nodes, edges, and vectors definitions
    """
    return {
        "nodes": {
            "Document": DOCUMENT_NODE_SCHEMA,
            "Chunk": CHUNK_NODE_SCHEMA,
            "ChunkVector": VECTOR_NODE_SCHEMA,
            "Table": TABLE_NODE_SCHEMA,
            "TableVector": TABLE_VECTOR_SCHEMA,
        },
        "edges": EDGE_TYPES,
    }


def create_kidkazz_schema(config: Optional[SchemaConfig] = None) -> Any:
    """
    Create Helix-DB schema for KidKazz RAG.

    Args:
        config: Optional schema configuration

    Returns:
        Helix Schema object (if helix-py installed) or schema dict

    Note:
        Requires helix-py to be installed for actual Schema creation.
        Returns a dictionary definition if helix-py is not available.
    """
    config = config or SchemaConfig()

    try:
        from helix import Schema

        schema = Schema()

        # Create Document node
        schema.create_node("Document", DOCUMENT_NODE_SCHEMA)

        # Create Chunk node
        schema.create_node("Chunk", CHUNK_NODE_SCHEMA)

        # Create Vector node with configured dimensions
        vector_schema = {
            "embedding": config.vector_type,
            "model_name": "String",
            "embedding_dim": "U32",
        }
        schema.create_vector("ChunkVector", vector_schema)

        # Create edges
        for edge_name, (from_node, to_node) in EDGE_TYPES.items():
            schema.create_edge(edge_name, from_node, to_node)

        return schema

    except ImportError:
        # Return dict definition if helix-py not installed
        return get_schema_definition()


def initialize_schema(
    config: Optional[SchemaConfig] = None,
) -> bool:
    """
    Initialize schema in a Helix-DB instance.

    Args:
        config: Optional schema configuration

    Returns:
        True if schema was initialized successfully

    Raises:
        ImportError: If helix-py is not installed
        RuntimeError: If schema initialization fails
    """
    try:
        from helix import Schema

        schema = create_kidkazz_schema(config)

        if isinstance(schema, Schema):
            schema.save()
            return True
        return False

    except ImportError as e:
        raise ImportError(
            "helix-py is not installed. Run: pip install helix-py"
        ) from e


def validate_chunk_node(node: dict[str, Any]) -> list[str]:
    """
    Validate a chunk node against the schema.

    Args:
        node: Dictionary of chunk node properties

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    # Required fields
    required_fields = ["chunk_id", "content", "level", "token_count"]
    for field_name in required_fields:
        if field_name not in node:
            errors.append(f"Missing required field: {field_name}")

    # Type checks
    if "level" in node and not isinstance(node["level"], int):
        errors.append("Field 'level' must be an integer")

    if "level" in node and node["level"] not in (0, 1, 2):
        errors.append("Field 'level' must be 0, 1, or 2")

    if "semantic_type" in node:
        valid_types = {"definition", "example", "procedure", "theorem", "narrative"}
        if node["semantic_type"] not in valid_types:
            errors.append(f"Invalid semantic_type: {node['semantic_type']}")

    # Validate header_level (must be 0 or 1-6; 0 means no header)
    if "header_level" in node and node["header_level"] is not None:
        if not isinstance(node["header_level"], int) or node["header_level"] < 0 or node["header_level"] > 6:
            errors.append("Field 'header_level' must be an integer between 0 and 6 (0 means no header)")

    return errors


def validate_document_node(node: dict[str, Any]) -> list[str]:
    """
    Validate a document node against the schema.

    Args:
        node: Dictionary of document node properties

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    required_fields = ["doc_id", "title"]
    for field_name in required_fields:
        if field_name not in node:
            errors.append(f"Missing required field: {field_name}")

    if "chunk_count" in node and not isinstance(node["chunk_count"], int):
        errors.append("Field 'chunk_count' must be an integer")

    return errors
