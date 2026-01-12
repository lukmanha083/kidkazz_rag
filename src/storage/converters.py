"""Converters between dataclasses and Helix-DB format."""

import json
from typing import Any, Optional

from src.chunker import Chunk, ChunkMetadata, EmbeddedChunk
from src.chunker.table_parser import ParsedTable
from src.chunker.table_summarizer import TableSummary


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
            # Convert booleans to integers (Helix-DB schema uses U32)
            "has_table": int(metadata.has_table),
            "has_code": int(metadata.has_code),
            "has_math": int(metadata.has_math),
            "has_list": int(metadata.has_list),
            # Header/block metadata (convert None to empty/0 for Helix-DB)
            "header_text": metadata.header_text or "",
            "header_level": metadata.header_level or 0,
            "block_type": metadata.block_type or "",
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
    sibling_ids = json.loads(node.get("sibling_ids", "[]"))
    topic_tags = json.loads(node.get("topic_tags", "[]"))

    # Handle header fields (convert empty/0 back to None)
    header_text = node.get("header_text")
    header_text = header_text if header_text else None
    header_level = node.get("header_level")
    header_level = header_level if header_level else None
    block_type = node.get("block_type")
    block_type = block_type if block_type else None

    # Include all metadata for MCP tools
    metadata = {
        "document_id": node.get("document_id"),
        "semantic_type": node.get("semantic_type"),
        "topic_tags": topic_tags,
        "has_table": bool(node.get("has_table", 0)),
        "has_code": bool(node.get("has_code", 0)),
        "has_math": bool(node.get("has_math", 0)),
        "has_list": bool(node.get("has_list", 0)),
        "header_text": header_text,
        "header_level": header_level,
        "block_type": block_type,
        "sibling_ids": sibling_ids,
        "sequence_position": node.get("sequence_position", 0),
    }

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

    # Handle header fields (convert empty/0 back to None)
    header_text = node.get("header_text")
    header_text = header_text if header_text else None

    header_level = node.get("header_level")
    header_level = header_level if header_level else None

    block_type = node.get("block_type")
    block_type = block_type if block_type else None

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
        header_text=header_text,
        header_level=header_level,
        block_type=block_type,
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


# ========== TABLE CONVERTERS ==========


def parsed_table_to_helix_node(
    table: ParsedTable,
    summary: TableSummary,
    doc_id: str,
) -> dict[str, Any]:
    """
    Convert ParsedTable and TableSummary to Helix-DB node properties.

    Args:
        table: The parsed table dataclass
        summary: The table summary with text description
        doc_id: Document ID for document filtering

    Returns:
        Dictionary of node properties for Helix-DB Table node
    """
    return {
        "table_id": summary.table_id,
        "raw_markdown": table.raw_markdown,
        "summary_text": summary.summary_text,
        "column_names": json.dumps(table.column_names),
        "column_types": json.dumps(table.column_types),
        "row_count": table.row_count,
        "column_count": table.column_count,
        "rows": json.dumps(table.rows),
        "has_header_row": int(table.has_header_row),
        "surrounding_context": table.surrounding_context or "",
        "source_chunk_id": table.source_chunk_id or "",
        "document_id": doc_id,
        "key_columns": json.dumps(summary.key_columns),
        "key_values": json.dumps(summary.key_values),
    }


def helix_node_to_parsed_table(node: dict[str, Any]) -> ParsedTable:
    """
    Convert Helix-DB node properties to ParsedTable dataclass.

    Args:
        node: Dictionary of node properties from Helix-DB

    Returns:
        Reconstructed ParsedTable dataclass
    """
    # Parse JSON fields
    column_names = json.loads(node.get("column_names", "[]"))
    column_types = json.loads(node.get("column_types", "[]"))
    rows = json.loads(node.get("rows", "[]"))

    return ParsedTable(
        raw_markdown=node.get("raw_markdown", ""),
        column_names=column_names,
        rows=rows,
        row_count=node.get("row_count", 0),
        column_count=node.get("column_count", 0),
        column_types=column_types,
        has_header_row=bool(node.get("has_header_row", 0)),
        surrounding_context=node.get("surrounding_context", ""),
        source_chunk_id=node.get("source_chunk_id", ""),
    )


def helix_node_to_table_summary(node: dict[str, Any]) -> TableSummary:
    """
    Convert Helix-DB node properties to TableSummary dataclass.

    Args:
        node: Dictionary of node properties from Helix-DB

    Returns:
        Reconstructed TableSummary dataclass (without embedding)
    """
    # Parse JSON fields
    key_columns = json.loads(node.get("key_columns", "[]"))
    key_values = json.loads(node.get("key_values", "[]"))

    return TableSummary(
        table_id=node.get("table_id", ""),
        summary_text=node.get("summary_text", ""),
        key_columns=key_columns,
        key_values=key_values,
        embedding=None,  # Embedding stored separately in TableVector
    )


def table_summary_to_helix_vector(
    summary: TableSummary,
    model_name: str,
) -> dict[str, Any]:
    """
    Convert TableSummary embedding to Helix-DB vector format.

    Args:
        summary: TableSummary with embedding attached
        model_name: Name of embedding model used

    Returns:
        Dictionary with embedding vector for Helix-DB TableVector node
    """
    embedding = summary.embedding or []
    return {
        "embedding": embedding,
        "model_name": model_name,
        "embedding_dim": len(embedding),
    }
