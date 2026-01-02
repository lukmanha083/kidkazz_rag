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
}

VECTOR_NODE_SCHEMA: dict[str, str] = {
    "embedding": "Vec32",  # 384-dimensional float32 vector
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
}


def get_schema_definition() -> dict[str, Any]:
    """
    Return the complete KidKazz schema definition.
    
    Returns:
        dict: Mapping with keys "nodes" (node name -> node schema) and "edges" (edge name -> edge definition).
    """
    return {
        "nodes": {
            "Document": DOCUMENT_NODE_SCHEMA,
            "Chunk": CHUNK_NODE_SCHEMA,
            "ChunkVector": VECTOR_NODE_SCHEMA,
        },
        "edges": EDGE_TYPES,
    }


def create_kidkazz_schema(config: Optional[SchemaConfig] = None) -> Any:
    """
    Constructs the KidKazz Helix-DB schema for documents, hierarchical chunks, and chunk embeddings.
    
    Parameters:
        config (SchemaConfig, optional): Overrides defaults for vector type, dimensions, and database name.
    
    Returns:
        Schema object if helix-py is available; otherwise a dict representing the schema definition.
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
    Initialize the KidKazz Helix-DB schema in a Helix-DB instance.
    
    Parameters:
        config (Optional[SchemaConfig]): Optional configuration for schema creation; uses defaults when omitted.
    
    Returns:
        True if a Helix Schema object was created and persisted, False if a dictionary schema definition was returned instead.
    
    Raises:
        ImportError: If helix-py is not installed.
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
    Validate a chunk node dictionary against the KidKazz Chunk schema.
    
    Performs presence checks for required fields and basic value/type validations:
    - Required: "chunk_id", "content", "level", "token_count".
    - "level" must be an integer equal to 0, 1, or 2.
    - If present, "semantic_type" must be one of: "definition", "example", "procedure", "theorem", "narrative".
    
    Parameters:
        node (dict[str, Any]): Chunk node properties to validate.
    
    Returns:
        list[str]: Validation error messages; empty list if the node is valid.
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

    return errors


def validate_document_node(node: dict[str, Any]) -> list[str]:
    """
    Validate a Document node dictionary against the module's Document schema.
    
    Checks that required fields 'doc_id' and 'title' are present, and that 'chunk_count' is an int when provided.
    
    Parameters:
        node (dict[str, Any]): Document node properties to validate.
    
    Returns:
        list[str]: Validation error messages; empty list if the node is valid.
    """
    errors: list[str] = []

    required_fields = ["doc_id", "title"]
    for field_name in required_fields:
        if field_name not in node:
            errors.append(f"Missing required field: {field_name}")

    if "chunk_count" in node and not isinstance(node["chunk_count"], int):
        errors.append("Field 'chunk_count' must be an integer")

    return errors