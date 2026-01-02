"""Converters between dataclasses and Helix-DB format."""

import json
from typing import Any, Optional

from src.chunker import Chunk, ChunkMetadata, EmbeddedChunk


def chunk_to_helix_node(
    chunk: Chunk,
    metadata: Optional[ChunkMetadata] = None,
) -> dict[str, Any]:
    """
    Convert a Chunk (and optional ChunkMetadata) into a Helix-DB node property dictionary.
    
    Parameters:
        chunk (Chunk): Chunk dataclass to convert.
        metadata (Optional[ChunkMetadata]): Optional metadata to include additional document-level fields.
    
    Returns:
        dict[str, Any]: Helix-DB node properties. Always includes keys like `chunk_id`, `content`, `level`, `token_count`, `word_count`, `section_path` (JSON string), `source_section`, `parent_id`, `child_ids` (JSON string), `prev_id`, and `next_id`. If `metadata` is provided, also includes `document_id`, `semantic_type`, `topic_tags` (JSON string), `sequence_position`, `sibling_ids` (JSON string), and boolean flags `has_table`, `has_code`, `has_math`, `has_list`.
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
    Reconstruct a Chunk dataclass from a Helix-DB node property dictionary.
    
    Parses JSON-encoded 'section_path' and 'child_ids' fields when present and maps node properties to the corresponding Chunk fields; optional relational IDs default to None if missing.
    
    Parameters:
        node (dict[str, Any]): Helix-DB node properties.
    
    Returns:
        Chunk: The reconstructed Chunk with fields populated from the node.
    """
    # Parse JSON fields
    section_path = json.loads(node.get("section_path", "[]"))
    child_ids = json.loads(node.get("child_ids", "[]"))

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
    )


def helix_node_to_metadata(node: dict[str, Any]) -> ChunkMetadata:
    """
    Convert a Helix-DB node dictionary into a ChunkMetadata instance.
    
    Parses JSON-encoded list fields (topic_tags, sibling_ids, child_ids, section_path)
    and applies sensible defaults for optional or missing properties.
    
    Parameters:
        node (dict[str, Any]): Helix-DB node properties for a chunk.
    
    Returns:
        ChunkMetadata: Reconstructed ChunkMetadata with parsed list fields and defaults for missing optional values.
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
    Create an EmbeddedChunk from a Helix-DB node and the provided embedding information.
    
    Parameters:
        node: Helix-DB node properties used to reconstruct the underlying Chunk.
        embedding: Embedding vector associated with the chunk.
        model_name: Name of the embedding model that produced the embedding.
    
    Returns:
        An EmbeddedChunk composed of the reconstructed Chunk, the provided embedding, and the model name.
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
) -> dict[str, Any]:
    """
    Create a Helix-DB document node property dictionary for the given document.
    
    Parameters:
        created_at (Optional[int]): Unix timestamp to set as the document creation time; if `None`, the current Unix time is used.
    
    Returns:
        dict[str, Any]: Document node properties with keys `doc_id`, `title`, `chunk_count`, and `created_at`.
    """
    import time

    return {
        "doc_id": doc_id,
        "title": title,
        "chunk_count": chunk_count,
        "created_at": created_at or int(time.time()),
    }


def helix_node_to_document(node: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a Helix-DB document node dictionary into a plain document metadata dictionary.
    
    Parameters:
        node (dict): Helix-DB document node properties; must contain the key `doc_id`.
    
    Returns:
        dict: A dictionary with keys:
            - `doc_id` (str): Document identifier from `node["doc_id"]`.
            - `title` (str): Document title, empty string if missing.
            - `chunk_count` (int): Number of chunks, `0` if missing.
            - `created_at` (int): Unix timestamp of creation, `0` if missing.
    """
    return {
        "doc_id": node["doc_id"],
        "title": node.get("title", ""),
        "chunk_count": node.get("chunk_count", 0),
        "created_at": node.get("created_at", 0),
    }