"""MCP resource implementations for KidKazz RAG."""

import json
import logging
from typing import Any

from .config import ServerState
from .formatters import format_chunk, format_document, format_document_stats

logger = logging.getLogger(__name__)

# Schema information for the knowledge base
SCHEMA_INFO = {
    "name": "KidKazz RAG Knowledge Base",
    "version": "1.0.0",
    "description": "Hierarchical document storage with semantic search capabilities",
    "chunk_levels": {
        "1": "Section-level chunks (broader context)",
        "2": "Leaf-level chunks (detailed content)",
    },
    "semantic_types": [
        "definition",
        "example",
        "procedure",
        "theorem",
        "narrative",
    ],
    "relationships": {
        "parent_child": "Level 1 chunks contain Level 2 children",
        "siblings": "Chunks at same level with same parent",
        "sequential": "prev_id/next_id for document order",
    },
    "tools": [
        {
            "name": "search_semantic",
            "description": "Vector similarity search using natural language queries",
        },
        {
            "name": "search_keyword",
            "description": "Full-text keyword search",
        },
        {
            "name": "get_chunk",
            "description": "Retrieve a specific chunk by ID",
        },
        {
            "name": "get_context_window",
            "description": "Get chunk with surrounding context",
        },
        {
            "name": "get_parent",
            "description": "Navigate to parent chunk",
        },
        {
            "name": "get_children",
            "description": "Get all child chunks",
        },
        {
            "name": "get_siblings",
            "description": "Get sibling chunks",
        },
        {
            "name": "list_documents",
            "description": "List all documents",
        },
        {
            "name": "get_document_chunks",
            "description": "Get all chunks from a document",
        },
        {
            "name": "get_document_stats",
            "description": "Get document statistics",
        },
    ],
}


def register_resources(mcp: Any, state: ServerState) -> None:
    """Register all MCP resources with the server.

    Args:
        mcp: FastMCP server instance
        state: Server state with store and embedder
    """

    @mcp.resource("kidkazz://schema")
    def get_schema() -> str:
        """Get the knowledge base schema information.

        Returns:
            JSON string with schema details including chunk levels,
            semantic types, relationships, and available tools.
        """
        logger.info("resource: kidkazz://schema")
        return json.dumps(SCHEMA_INFO, indent=2)

    @mcp.resource("kidkazz://documents")
    def get_documents() -> str:
        """List all documents in the knowledge base.

        Returns:
            JSON string with list of documents including doc_id,
            title, chunk_count, and created_at.
        """
        logger.info("resource: kidkazz://documents")
        docs = state.store.list_documents()
        formatted = [format_document(doc) for doc in docs]
        return json.dumps(formatted, indent=2)

    @mcp.resource("kidkazz://document/{doc_id}")
    def get_document_overview(doc_id: str) -> str:
        """Get overview of a specific document.

        Args:
            doc_id: Document identifier

        Returns:
            JSON string with document statistics and metadata,
            or error message if document not found.
        """
        logger.info(f"resource: kidkazz://document/{doc_id}")
        stats = state.store.get_document_stats(doc_id)

        if stats is None:
            return json.dumps({"error": f"Document '{doc_id}' not found"})

        return json.dumps(format_document_stats(stats), indent=2)

    @mcp.resource("kidkazz://chunk/{chunk_id}")
    def get_chunk_content(chunk_id: str) -> str:
        """Get content of a specific chunk.

        Args:
            chunk_id: Chunk identifier (e.g., "textbook_l2_5")

        Returns:
            JSON string with chunk content and metadata,
            or error message if chunk not found.
        """
        logger.info(f"resource: kidkazz://chunk/{chunk_id}")
        chunk = state.store.get_chunk(chunk_id)

        if chunk is None:
            return json.dumps({"error": f"Chunk '{chunk_id}' not found"})

        return json.dumps(format_chunk(chunk), indent=2)
