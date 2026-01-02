"""HelixQL query classes for Helix-DB operations.

This module defines query classes following the helix-py pattern.
Each query class encapsulates a specific database operation.

Usage:
    from src.storage.queries import AddChunkWithEmbedding

    query = AddChunkWithEmbedding(chunk_props, embedding, doc_id)
    result = db.query(query)

Note:
    Requires helix-py to be installed. Query classes inherit from
    helix.Query base class.
"""

import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class QueryResult:
    """Result container for query operations."""

    success: bool
    data: Any = None
    error: Optional[str] = None


# Base query class for when helix-py is not installed
class BaseQuery:
    """
    Base class for queries when helix-py is not available.

    This provides a fallback implementation that can be used for
    testing or when building query definitions without executing them.
    """

    def query(self) -> list[dict[str, Any]]:
        """Return query payload."""
        raise NotImplementedError

    def response(self, response: Any) -> Any:
        """Process response from database."""
        return response


def _get_query_base_class() -> type:
    """Get the appropriate Query base class."""
    try:
        from helix import Query
        return Query
    except ImportError:
        return BaseQuery


# Document Queries
class AddDocument(_get_query_base_class()):
    """Add a document node to the database."""

    def __init__(self, doc_id: str, title: str, chunk_count: int) -> None:
        """
        Initialize AddDocument query.

        Args:
            doc_id: Unique document identifier
            title: Document title
            chunk_count: Number of chunks in document
        """
        super().__init__()
        self.doc_id = doc_id
        self.title = title
        self.chunk_count = chunk_count
        self.created_at = int(time.time())

    def query(self) -> list[dict[str, Any]]:
        """Return query payload for adding document."""
        return [{
            "operation": "add_node",
            "node_type": "Document",
            "properties": {
                "doc_id": self.doc_id,
                "title": self.title,
                "chunk_count": self.chunk_count,
                "created_at": self.created_at,
            },
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True, data={"doc_id": self.doc_id})


class DeleteDocument(_get_query_base_class()):
    """Delete a document and all its chunks."""

    def __init__(self, doc_id: str) -> None:
        """
        Initialize DeleteDocument query.

        Args:
            doc_id: Document identifier to delete
        """
        super().__init__()
        self.doc_id = doc_id

    def query(self) -> list[dict[str, Any]]:
        """Return query payload for deleting document."""
        return [{
            "operation": "delete_cascade",
            "start_node": "Document",
            "filter": {"doc_id": self.doc_id},
            "edges": ["HasChunk"],
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True, data={"doc_id": self.doc_id})


# Chunk Queries
class AddChunkWithEmbedding(_get_query_base_class()):
    """Add a chunk node with its embedding vector."""

    def __init__(
        self,
        chunk_props: dict[str, Any],
        embedding: list[float],
        doc_id: str,
    ) -> None:
        """
        Initialize AddChunkWithEmbedding query.

        Args:
            chunk_props: Dictionary of chunk properties
            embedding: Vector embedding (384 dims)
            doc_id: Parent document ID
        """
        super().__init__()
        self.chunk_props = chunk_props
        self.embedding = embedding
        self.doc_id = doc_id

    def query(self) -> list[dict[str, Any]]:
        """Return query payload for adding chunk with embedding."""
        return [
            # Add chunk node
            {
                "operation": "add_node",
                "node_type": "Chunk",
                "properties": self.chunk_props,
            },
            # Add vector node
            {
                "operation": "add_vector",
                "node_type": "ChunkVector",
                "properties": {"embedding": self.embedding},
            },
            # Create HasEmbedding edge
            {
                "operation": "add_edge",
                "edge_type": "HasEmbedding",
                "from": {"node_type": "Chunk", "filter": {"chunk_id": self.chunk_props["chunk_id"]}},
                "to": {"node_type": "ChunkVector"},
            },
            # Create HasChunk edge from document
            {
                "operation": "add_edge",
                "edge_type": "HasChunk",
                "from": {"node_type": "Document", "filter": {"doc_id": self.doc_id}},
                "to": {"node_type": "Chunk", "filter": {"chunk_id": self.chunk_props["chunk_id"]}},
            },
        ]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(
            success=True,
            data={"chunk_id": self.chunk_props["chunk_id"]},
        )


class AddChunkRelationship(_get_query_base_class()):
    """Add a relationship edge between chunks."""

    def __init__(
        self,
        edge_type: str,
        from_chunk_id: str,
        to_chunk_id: str,
    ) -> None:
        """
        Initialize AddChunkRelationship query.

        Args:
            edge_type: Type of edge (ParentOf, NextSibling, etc.)
            from_chunk_id: Source chunk ID
            to_chunk_id: Target chunk ID
        """
        super().__init__()
        self.edge_type = edge_type
        self.from_chunk_id = from_chunk_id
        self.to_chunk_id = to_chunk_id

    def query(self) -> list[dict[str, Any]]:
        """Return query payload for adding chunk relationship."""
        return [{
            "operation": "add_edge",
            "edge_type": self.edge_type,
            "from": {"node_type": "Chunk", "filter": {"chunk_id": self.from_chunk_id}},
            "to": {"node_type": "Chunk", "filter": {"chunk_id": self.to_chunk_id}},
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True)


class GetChunk(_get_query_base_class()):
    """Get a single chunk by ID."""

    def __init__(self, chunk_id: str) -> None:
        """
        Initialize GetChunk query.

        Args:
            chunk_id: Chunk identifier
        """
        super().__init__()
        self.chunk_id = chunk_id

    def query(self) -> list[dict[str, Any]]:
        """Return query payload for getting chunk."""
        return [{
            "operation": "get_node",
            "node_type": "Chunk",
            "filter": {"chunk_id": self.chunk_id},
            "include_vector": True,
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        if response:
            return QueryResult(success=True, data=response)
        return QueryResult(success=False, error="Chunk not found")


class GetChunkWithContext(_get_query_base_class()):
    """Get a chunk with its parent, siblings, and neighbors."""

    def __init__(self, chunk_id: str) -> None:
        """
        Initialize GetChunkWithContext query.

        Args:
            chunk_id: Chunk identifier
        """
        super().__init__()
        self.chunk_id = chunk_id

    def query(self) -> list[dict[str, Any]]:
        """Return query payload for getting chunk with context."""
        return [{
            "operation": "traverse",
            "start": {"node_type": "Chunk", "filter": {"chunk_id": self.chunk_id}},
            "paths": [
                {"edge": "ParentOf", "direction": "in", "as": "parent"},
                {"edge": "SiblingOf", "direction": "out", "as": "siblings"},
                {"edge": "NextSibling", "direction": "out", "as": "next"},
                {"edge": "PrevSibling", "direction": "out", "as": "prev"},
            ],
            "return": ["chunk", "parent", "siblings", "next", "prev"],
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True, data=response)


# Search Queries
class SearchSimilarChunks(_get_query_base_class()):
    """Vector similarity search for chunks."""

    def __init__(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_id: Optional[str] = None,
        level: Optional[int] = None,
        semantic_type: Optional[str] = None,
        threshold: float = 0.0,
    ) -> None:
        """
        Initialize SearchSimilarChunks query.

        Args:
            query_embedding: Query vector (384 dims)
            top_k: Number of results
            doc_id: Filter by document (optional)
            level: Filter by hierarchy level (optional)
            semantic_type: Filter by semantic type (optional)
            threshold: Minimum similarity score
        """
        super().__init__()
        self.query_embedding = query_embedding
        self.top_k = top_k
        self.doc_id = doc_id
        self.level = level
        self.semantic_type = semantic_type
        self.threshold = threshold

    def query(self) -> list[dict[str, Any]]:
        """Return query payload for vector search."""
        filters: dict[str, Any] = {}
        if self.doc_id:
            filters["document_id"] = self.doc_id
        if self.level is not None:
            filters["level"] = self.level
        if self.semantic_type:
            filters["semantic_type"] = self.semantic_type

        return [{
            "operation": "vector_search",
            "vector": self.query_embedding,
            "node_type": "ChunkVector",
            "top_k": self.top_k,
            "threshold": self.threshold,
            "filters": filters,
            "return_nodes": True,
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True, data=response)


class SearchKeyword(_get_query_base_class()):
    """Full-text keyword search in chunk content."""

    def __init__(
        self,
        keyword: str,
        doc_id: Optional[str] = None,
    ) -> None:
        """
        Initialize SearchKeyword query.

        Args:
            keyword: Search term
            doc_id: Filter by document (optional)
        """
        super().__init__()
        self.keyword = keyword
        self.doc_id = doc_id

    def query(self) -> list[dict[str, Any]]:
        """Return query payload for keyword search."""
        filters: dict[str, Any] = {}
        if self.doc_id:
            filters["document_id"] = self.doc_id

        return [{
            "operation": "keyword_search",
            "keyword": self.keyword,
            "node_type": "Chunk",
            "field": "content",
            "filters": filters,
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True, data=response)


# Graph Traversal Queries
class GetDocumentChunks(_get_query_base_class()):
    """Get all chunks for a document."""

    def __init__(
        self,
        doc_id: str,
        level: Optional[int] = None,
    ) -> None:
        """
        Initialize GetDocumentChunks query.

        Args:
            doc_id: Document identifier
            level: Filter by hierarchy level (optional)
        """
        super().__init__()
        self.doc_id = doc_id
        self.level = level

    def query(self) -> list[dict[str, Any]]:
        """Return query payload for getting document chunks."""
        filters: dict[str, Any] = {"document_id": self.doc_id}
        if self.level is not None:
            filters["level"] = self.level

        return [{
            "operation": "traverse",
            "start": {"node_type": "Document", "filter": {"doc_id": self.doc_id}},
            "paths": [{"edge": "HasChunk", "direction": "out"}],
            "filters": filters,
            "order_by": "sequence_position",
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True, data=response)


class ListDocuments(_get_query_base_class()):
    """List all documents."""

    def query(self) -> list[dict[str, Any]]:
        """Return query payload for listing documents."""
        return [{
            "operation": "get_all",
            "node_type": "Document",
            "order_by": "created_at",
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True, data=response)
