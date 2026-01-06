"""MCP tool implementations for KidKazz RAG."""

import logging
from typing import Any, Optional

from .config import ServerState
from .formatters import (
    format_chunk,
    format_chunk_list,
    format_document,
    format_document_stats,
    format_optional_chunk,
    format_search_results,
)

logger = logging.getLogger(__name__)


def register_tools(mcp: Any, state: ServerState) -> None:
    """Register all MCP tools with the server.

    Args:
        mcp: FastMCP server instance
        state: Server state with store and embedder
    """

    @mcp.tool()
    def search_semantic(
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        level: Optional[int] = None,
        semantic_type: Optional[str] = None,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Search for document chunks semantically similar to the query.

        Args:
            query: Natural language search query
            top_k: Number of results to return (default: 5)
            doc_id: Filter to specific document (optional)
            level: Filter by hierarchy level - 1 (section) or 2 (leaf) (optional)
            semantic_type: Filter by type - definition, example, procedure, theorem, narrative (optional)
            threshold: Minimum similarity score 0.0-1.0 (default: 0.0)
            tags: Filter by document tags - documents must have ALL specified tags (optional)

        Returns:
            List of matching chunks with content, metadata, and similarity scores
        """
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"search_semantic: query='{query_preview}', top_k={top_k}, tags={tags}")

        # Generate query embedding server-side
        query_embedding = state.embedder.embed_text(query)

        # Search storage
        results = state.store.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            doc_id=doc_id,
            level=level,
            semantic_type=semantic_type,
            threshold=threshold,
            tags=tags,
        )

        logger.info(f"search_semantic: found {len(results)} results")
        return format_search_results(results)

    @mcp.tool()
    def search_keyword(
        keyword: str,
        doc_id: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        """Search for document chunks containing a specific keyword or phrase.

        Args:
            keyword: Text to search for
            doc_id: Filter to specific document (optional)
            case_sensitive: Whether search is case-sensitive (default: False)

        Returns:
            List of matching chunks with content and metadata
        """
        logger.info(f"search_keyword: keyword='{keyword}', doc_id={doc_id}")

        results = state.store.search_keyword(
            keyword=keyword,
            doc_id=doc_id,
            case_sensitive=case_sensitive,
        )

        logger.info(f"search_keyword: found {len(results)} results")
        return format_chunk_list(results)

    @mcp.tool()
    def get_chunk(chunk_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a specific chunk by its ID.

        Args:
            chunk_id: Unique identifier of the chunk (e.g., "textbook_l2_5")

        Returns:
            Chunk with content, metadata, and embedding info, or null if not found
        """
        logger.info(f"get_chunk: chunk_id={chunk_id}")

        result = state.store.get_chunk(chunk_id)
        return format_optional_chunk(result)

    @mcp.tool()
    def get_context_window(
        chunk_id: str,
        window_size: int = 2,
    ) -> list[dict[str, Any]]:
        """Get a chunk with its surrounding context (previous and next chunks).

        Args:
            chunk_id: Center chunk ID
            window_size: Number of chunks before and after (default: 2)

        Returns:
            List of chunks in document order, centered on the specified chunk
        """
        logger.info(f"get_context_window: chunk_id={chunk_id}, window_size={window_size}")

        results = state.store.get_context_window(chunk_id, window_size=window_size)

        logger.info(f"get_context_window: found {len(results)} chunks")
        return format_chunk_list(results)

    @mcp.tool()
    def get_parent(chunk_id: str) -> Optional[dict[str, Any]]:
        """Get the parent chunk of a given chunk (Level 2 -> Level 1).

        Useful for expanding context to understand the broader section.

        Args:
            chunk_id: Child chunk ID

        Returns:
            Parent chunk with content and metadata, or null if no parent
        """
        logger.info(f"get_parent: chunk_id={chunk_id}")

        result = state.store.get_parent(chunk_id)
        return format_optional_chunk(result)

    @mcp.tool()
    def get_children(chunk_id: str) -> list[dict[str, Any]]:
        """Get all child chunks of a given chunk (Level 1 -> Level 2).

        Useful for drilling down into section details.

        Args:
            chunk_id: Parent chunk ID

        Returns:
            List of child chunks with content and metadata
        """
        logger.info(f"get_children: chunk_id={chunk_id}")

        results = state.store.get_children(chunk_id)

        logger.info(f"get_children: found {len(results)} children")
        return format_chunk_list(results)

    @mcp.tool()
    def get_siblings(chunk_id: str) -> list[dict[str, Any]]:
        """Get sibling chunks (same parent, same level).

        Useful for exploring related content in the same section.

        Args:
            chunk_id: Chunk ID

        Returns:
            List of sibling chunks (excluding the specified chunk)
        """
        logger.info(f"get_siblings: chunk_id={chunk_id}")

        results = state.store.get_siblings(chunk_id)

        logger.info(f"get_siblings: found {len(results)} siblings")
        return format_chunk_list(results)

    @mcp.tool()
    def list_documents(
        tags: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """List all documents in the knowledge base.

        Args:
            tags: Filter by document tags - documents must have ALL specified tags (optional)

        Returns:
            List of documents with doc_id, title, tags, chunk_count, and created_at
        """
        logger.info(f"list_documents: tags={tags}")

        docs = state.store.list_documents(tags=tags)

        logger.info(f"list_documents: found {len(docs)} documents")
        return [format_document(doc) for doc in docs]

    @mcp.tool()
    def get_document_chunks(
        doc_id: str,
        level: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Get all chunks from a specific document.

        Args:
            doc_id: Document identifier
            level: Filter by hierarchy level - 1 or 2 (optional)

        Returns:
            List of chunks sorted by sequence position
        """
        logger.info(f"get_document_chunks: doc_id={doc_id}, level={level}")

        results = state.store.get_document_chunks(doc_id, level=level)

        logger.info(f"get_document_chunks: found {len(results)} chunks")
        return format_chunk_list(results)

    @mcp.tool()
    def get_document_stats(doc_id: str) -> Optional[dict[str, Any]]:
        """Get statistics and summary for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Statistics including total_chunks, level_counts, type_counts, total_tokens
        """
        logger.info(f"get_document_stats: doc_id={doc_id}")

        stats = state.store.get_document_stats(doc_id)

        if stats is None:
            logger.warning(f"get_document_stats: document '{doc_id}' not found")
            return None

        return format_document_stats(stats)
