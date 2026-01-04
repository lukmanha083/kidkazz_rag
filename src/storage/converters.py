"""Converters between dataclasses and Helix-DB format."""

import json
from typing import Any, Optional

from src.chunker import Chunk, ChunkMetadata, EmbeddedChunk


def chunk_to_helix_node(
    chunk: Chunk,
    metadata: Optional[ChunkMetadata] = None,
) -> dict[str, Any]:
    """
    Convert Chunk and optional ChunkMetadata to Helix-DB node properties.

    Args:
        chunk: The chunk dataclass
        metadata: Optional metadata for additional fields

    Returns:
        Dictionary of node properties for Helix-DB
    """
    node: dict[str, Any] = {
        "chunk_id": chunk.id,
        "content": chunk.content,
        "level": chunk.level,
        "token_count": chunk.token_count,
        "word_count": chunk.word_count,
        "section_path": json.dumps(chunk.section_path),
        "source_section": chunk.source_section or "",
        "parent_id": chunk.parent_id or "",
        "child_ids": json.dumps(chunk.child_ids),
        "prev_id": chunk.prev_id or "",
        "next_id": chunk.next_id or "",
    }

    # Add metadata fields if provided
    if metadata:
        node.update({
            "document_id": metadata.document_id,
            "semantic_type": metadata.semantic_type,
            "topic_tags": json.dumps(metadata.topic_tags),
            "sequence_position": metadata.sequence_position,
            "sibling_ids": json.dumps(metadata.sibling_ids),
            "has_table": metadata.has_table,
            "has_code": metadata.has_code,
            "has_math": metadata.has_math,
            "has_list": metadata.has_list,
        })

    return node


def helix_node_to_chunk(node: dict[str, Any]) -> Chunk:
    """
    Convert Helix-DB node properties to Chunk dataclass.

    Args:
        node: Dictionary of node properties from Helix-DB

    Returns:
        Reconstructed Chunk dataclass
    """
    # Parse JSON fields
    section_path = json.loads(node.get("section_path", "[]"))
    child_ids = json.loads(node.get("child_ids", "[]"))

    # Preserve semantic_type and other metadata in chunk.metadata
    metadata = {}
    if "semantic_type" in node:
        metadata["semantic_type"] = node["semantic_type"]
    if "has_table" in node:
        metadata["has_table"] = node["has_table"]
    if "has_code" in node:
        metadata["has_code"] = node["has_code"]
    if "has_math" in node:
        metadata["has_math"] = node["has_math"]
    if "has_list" in node:
        metadata["has_list"] = node["has_list"]

    return Chunk(
        id=node["chunk_id"],
        content=node["content"],
        level=node["level"],
        token_count=node["token_count"],
        parent_id=node.get("parent_id") or None,
        child_ids=child_ids,
        prev_id=node.get("prev_id") or None,
        next_id=node.get("next_id") or None,
        source_section=node.get("source_section") or None,
        section_path=section_path,
        metadata=metadata,
    )


def helix_node_to_metadata(node: dict[str, Any]) -> ChunkMetadata:
    """
    Convert Helix-DB node properties to ChunkMetadata dataclass.

    Args:
        node: Dictionary of node properties from Helix-DB

    Returns:
        Reconstructed ChunkMetadata dataclass
    """
    # Parse JSON fields
    topic_tags = json.loads(node.get("topic_tags", "[]"))
    sibling_ids = json.loads(node.get("sibling_ids", "[]"))
    child_ids = json.loads(node.get("child_ids", "[]"))
    section_path = json.loads(node.get("section_path", "[]"))

    return ChunkMetadata(
        chunk_id=node["chunk_id"],
        document_id=node.get("document_id", ""),
        semantic_type=node.get("semantic_type", "narrative"),
        topic_tags=topic_tags,
        token_count=node.get("token_count", 0),
        word_count=node.get("word_count", 0),
        hierarchy_level=node.get("level", 2),
        section_path=section_path,
        sequence_position=node.get("sequence_position", 0),
        parent_id=node.get("parent_id") or None,
        child_ids=child_ids,
        sibling_ids=sibling_ids,
        prev_id=node.get("prev_id") or None,
        next_id=node.get("next_id") or None,
        has_table=node.get("has_table", False),
        has_code=node.get("has_code", False),
        has_math=node.get("has_math", False),
        has_list=node.get("has_list", False),
    )


def embedded_chunk_to_helix_vector(
    embedded_chunk: EmbeddedChunk,
) -> dict[str, Any]:
    """
    Convert EmbeddedChunk embedding to Helix-DB vector format.

    Args:
        embedded_chunk: Chunk with embedding attached

    Returns:
        Dictionary with embedding vector for Helix-DB
    """
    return {
        "embedding": embedded_chunk.embedding,
        "model_name": embedded_chunk.model_name,
        "embedding_dim": embedded_chunk.embedding_dim,
    }


def helix_to_embedded_chunk(
    node: dict[str, Any],
    embedding: list[float],
    model_name: str,
) -> EmbeddedChunk:
    """
    Reconstruct EmbeddedChunk from Helix-DB data.

    Args:
        node: Node properties from Helix-DB
        embedding: Vector embedding
        model_name: Name of embedding model

    Returns:
        Reconstructed EmbeddedChunk
    """
    chunk = helix_node_to_chunk(node)
    return EmbeddedChunk(
        chunk=chunk,
        embedding=embedding,
        model_name=model_name,
    )


def document_to_helix_node(
    doc_id: str,
    title: str,
    chunk_count: int,
    created_at: Optional[int] = None,
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Create Helix-DB document node properties.

    Args:
        doc_id: Unique document identifier
        title: Document title
        chunk_count: Number of chunks in document
        created_at: Unix timestamp (auto-generated if not provided)
        tags: Optional list of document tags

    Returns:
        Dictionary of document node properties
    """
    import time

    # Normalize tags to lowercase and filter empty strings
    normalized_tags = [t.lower().strip() for t in (tags or []) if t and t.strip()]

    return {
        "doc_id": doc_id,
        "title": title,
        "tags": json.dumps(normalized_tags),
        "chunk_count": chunk_count,
        "created_at": created_at or int(time.time()),
    }


def helix_node_to_document(node: dict[str, Any]) -> dict[str, Any]:
    """
    Convert Helix-DB document node to dictionary.

    Args:
        node: Document node properties from Helix-DB

    Returns:
        Dictionary with document metadata
    """
    # Parse tags from JSON string with error handling
    tags_raw = node.get("tags", "[]")
    tags: list[str] = []

    if isinstance(tags_raw, str):
        try:
            parsed = json.loads(tags_raw)
            if isinstance(parsed, list):
                tags = parsed
        except json.JSONDecodeError:
            # Invalid JSON, use empty list
            tags = []
    elif isinstance(tags_raw, list):
        tags = tags_raw
    # else: tags remains empty list for other types

    return {
        "doc_id": node["doc_id"],
        "title": node.get("title", ""),
        "tags": tags,
        "chunk_count": node.get("chunk_count", 0),
        "created_at": node.get("created_at", 0),
    }
