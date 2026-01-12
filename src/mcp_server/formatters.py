"""Response formatters for MCP outputs."""

from typing import Any, Optional

from src.chunker import EmbeddedChunk


def format_chunk(ec: EmbeddedChunk) -> dict[str, Any]:
    """Format an EmbeddedChunk for MCP response.

    Args:
        ec: EmbeddedChunk to format

    Returns:
        Dictionary with chunk data suitable for MCP response
    """
    chunk = ec.chunk
    meta = chunk.metadata or {}

    return {
        # Core chunk fields
        "chunk_id": chunk.id,
        "content": chunk.content,
        "level": chunk.level,
        "token_count": chunk.token_count,
        "word_count": chunk.word_count,
        # Graph relationships
        "parent_id": chunk.parent_id,
        "child_ids": chunk.child_ids,
        "prev_id": chunk.prev_id,
        "next_id": chunk.next_id,
        "sibling_ids": meta.get("sibling_ids", []),
        # Section info
        "section_path": chunk.section_path,
        "source_section": chunk.source_section,
        # Embedding info
        "model_name": ec.model_name,
        "embedding_dim": ec.embedding_dim,
        # Metadata for filtering and context
        "document_id": meta.get("document_id"),
        "semantic_type": meta.get("semantic_type"),
        "topic_tags": meta.get("topic_tags", []),
        # Content flags
        "has_table": meta.get("has_table", False),
        "has_code": meta.get("has_code", False),
        "has_math": meta.get("has_math", False),
        "has_list": meta.get("has_list", False),
        # Header metadata (from Reducto block mode)
        "header_text": meta.get("header_text"),
        "header_level": meta.get("header_level"),
        "block_type": meta.get("block_type"),
    }


def format_search_result(ec: EmbeddedChunk, score: float) -> dict[str, Any]:
    """Format a search result with similarity score.

    Args:
        ec: EmbeddedChunk from search results
        score: Similarity score (0.0 to 1.0)

    Returns:
        Dictionary with chunk data and similarity score
    """
    result = format_chunk(ec)
    result["similarity_score"] = round(score, 4)
    return result


def format_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Format document metadata for MCP response.

    Args:
        doc: Document metadata dictionary

    Returns:
        Formatted document dictionary
    """
    return {
        "doc_id": doc.get("doc_id", ""),
        "title": doc.get("title", ""),
        "tags": doc.get("tags", []),
        "chunk_count": doc.get("chunk_count", 0),
        "created_at": doc.get("created_at"),
    }


def format_document_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Format document statistics for MCP response.

    Args:
        stats: Document statistics dictionary

    Returns:
        Formatted statistics dictionary
    """
    return {
        "doc_id": stats.get("doc_id", ""),
        "title": stats.get("title", ""),
        "total_chunks": stats.get("total_chunks", 0),
        "total_tokens": stats.get("total_tokens", 0),
        "chunks_by_level": stats.get("level_counts", {}),
        "chunks_by_type": stats.get("type_counts", {}),
    }


def format_chunk_list(chunks: list[EmbeddedChunk]) -> list[dict[str, Any]]:
    """Format a list of chunks for MCP response.

    Args:
        chunks: List of EmbeddedChunks to format

    Returns:
        List of formatted chunk dictionaries
    """
    return [format_chunk(ec) for ec in chunks]


def format_search_results(
    results: list[tuple[EmbeddedChunk, float]]
) -> list[dict[str, Any]]:
    """Format search results for MCP response.

    Args:
        results: List of (EmbeddedChunk, score) tuples

    Returns:
        List of formatted search result dictionaries
    """
    return [format_search_result(ec, score) for ec, score in results]


def format_optional_chunk(ec: Optional[EmbeddedChunk]) -> Optional[dict[str, Any]]:
    """Format an optional chunk for MCP response.

    Args:
        ec: Optional EmbeddedChunk to format

    Returns:
        Formatted chunk dictionary or None
    """
    if ec is None:
        return None
    return format_chunk(ec)


# ============================================================================
# Concept Formatters
# ============================================================================


def format_concept(concept: dict[str, Any]) -> dict[str, Any]:
    """Format a single concept for MCP output.

    Args:
        concept: Concept dictionary from storage

    Returns:
        Formatted concept dictionary with parsed JSON fields
    """
    import json

    # Parse JSON string fields
    aliases_raw = concept.get("aliases", "[]")
    try:
        aliases = json.loads(aliases_raw) if isinstance(aliases_raw, str) else aliases_raw
    except json.JSONDecodeError:
        aliases = []

    source_docs_raw = concept.get("source_documents", "[]")
    try:
        source_documents = json.loads(source_docs_raw) if isinstance(source_docs_raw, str) else source_docs_raw
    except json.JSONDecodeError:
        source_documents = []

    return {
        "concept_id": concept.get("concept_id", ""),
        "name": concept.get("name", ""),
        "definition": concept.get("definition", ""),
        "concept_type": concept.get("concept_type", ""),
        "aliases": aliases,
        "source_documents": source_documents,
    }


def format_concept_list(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format a list of concepts.

    Args:
        concepts: List of concept dictionaries

    Returns:
        List of formatted concept dictionaries
    """
    return [format_concept(c) for c in concepts]


def format_concept_with_context(
    concept: dict[str, Any],
    definition_chunks: list[dict[str, Any]],
    related_concepts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Format concept with full context for rich responses.

    Args:
        concept: The concept dictionary
        definition_chunks: Chunks that define this concept
        related_concepts: Concepts related to this one

    Returns:
        Dictionary with concept, citations, and related concepts
    """
    import json

    formatted_chunks = []
    for chunk in definition_chunks:
        section_path_raw = chunk.get("section_path", "[]")
        try:
            section_path = json.loads(section_path_raw) if isinstance(section_path_raw, str) else section_path_raw
        except json.JSONDecodeError:
            section_path = []

        content = chunk.get("content", "")
        content_preview = content[:200] + "..." if len(content) > 200 else content

        formatted_chunks.append({
            "document_id": chunk.get("document_id", ""),
            "section_path": section_path,
            "content_preview": content_preview,
            "chunk_id": chunk.get("chunk_id", ""),
        })

    return {
        "concept": format_concept(concept),
        "citations": formatted_chunks,
        "related_concepts": format_concept_list(related_concepts),
    }


# ============================================================================
# Table Formatters
# ============================================================================


def format_table(table) -> dict[str, Any]:
    """Format a ParsedTable for MCP output.

    Args:
        table: ParsedTable instance

    Returns:
        Formatted table dictionary
    """
    return {
        "table_id": f"table_{table.source_chunk_id}",
        "columns": table.column_names,
        "column_types": table.column_types,
        "rows": table.rows,
        "row_count": table.row_count,
        "column_count": table.column_count,
        "raw_markdown": table.raw_markdown,
        "context": table.surrounding_context,
    }


def format_table_search_result(table, score: float) -> dict[str, Any]:
    """Format a table search result with similarity score.

    Args:
        table: ParsedTable from search results
        score: Similarity score (0.0 to 1.0)

    Returns:
        Dictionary with table data and similarity score
    """
    result = format_table(table)
    result["score"] = round(score, 4)
    return result


def format_table_list(tables: list) -> list[dict[str, Any]]:
    """Format a list of tables for MCP response.

    Args:
        tables: List of ParsedTable instances

    Returns:
        List of formatted table dictionaries
    """
    return [format_table(t) for t in tables]


def format_table_search_results(
    results: list[tuple],
) -> list[dict[str, Any]]:
    """Format table search results for MCP response.

    Args:
        results: List of (ParsedTable, score) tuples

    Returns:
        List of formatted search result dictionaries
    """
    return [format_table_search_result(table, score) for table, score in results]


def format_table_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Format table metadata dict for MCP response.

    Used by list_tables which returns lightweight metadata without full table data.
    Provides consistent structure with format_table but marks missing fields as null.

    Args:
        metadata: Table metadata dict from list_tables

    Returns:
        Formatted metadata dictionary with consistent keys
    """
    return {
        "table_id": metadata.get("table_id", ""),
        "document_id": metadata.get("document_id", ""),
        "source_chunk_id": metadata.get("source_chunk_id", ""),
        "summary_text": metadata.get("summary_text", ""),
        "row_count": metadata.get("row_count", 0),
        "column_count": metadata.get("column_count", 0),
        # These fields are not loaded in metadata-only listing
        "columns": None,
        "column_types": None,
        "rows": None,
        "raw_markdown": None,
        "context": None,
    }


def format_table_metadata_list(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format a list of table metadata dicts for MCP response.

    Used by list_tables for lightweight table listing without loading full data.

    Args:
        tables: List of table metadata dicts

    Returns:
        List of formatted metadata dictionaries
    """
    return [format_table_metadata(t) for t in tables]
