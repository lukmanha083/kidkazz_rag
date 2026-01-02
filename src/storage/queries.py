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
        """
        Builds the list of operation payloads for this query.
        
        Returns:
            list[dict[str, Any]]: A list of operation payload dictionaries describing the actions to execute.
        
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def response(self, response: Any) -> Any:
        """
        Return the database response unchanged.
        
        Returns:
            The original `response` value unchanged.
        """
        return response


def _get_query_base_class() -> type:
    """
    Selects the Query base class to use.
    
    Attempts to import and return `helix.Query`; if that import fails, returns the module's `BaseQuery` fallback.
    
    Returns:
        The `helix.Query` class when helix-py is available, otherwise `BaseQuery`.
    """
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
        Create an AddDocument query that stores document metadata and a creation timestamp.
        
        Parameters:
            doc_id (str): Unique identifier for the document.
            title (str): Human-readable title of the document.
            chunk_count (int): Number of chunks contained in the document.
        
        Notes:
            Sets the `created_at` attribute to the current time as integer seconds since the epoch.
        """
        super().__init__()
        self.doc_id = doc_id
        self.title = title
        self.chunk_count = chunk_count
        self.created_at = int(time.time())

    def query(self) -> list[dict[str, Any]]:
        """
        Builds the payload to add a Document node to Helix-DB.
        
        Returns:
            list[dict[str, Any]]: A list with a single operation dict for `add_node` targeting `Document`,
            containing `doc_id`, `title`, `chunk_count`, and `created_at` in `properties`.
        """
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
        """
        Return a successful QueryResult containing this query's document id.
        
        Parameters:
            response (Any): Raw response from the database or execution engine (ignored).
        
        Returns:
            QueryResult: success=True and data set to {'doc_id': self.doc_id}.
        """
        return QueryResult(success=True, data={"doc_id": self.doc_id})


class DeleteDocument(_get_query_base_class()):
    """Delete a document and all its chunks."""

    def __init__(self, doc_id: str) -> None:
        """
        Create a DeleteDocument query for the specified document.
        
        Parameters:
            doc_id (str): The document identifier to delete.
        """
        super().__init__()
        self.doc_id = doc_id

    def query(self) -> list[dict[str, Any]]:
        """
        Builds the operation payload to delete a Document and cascade-remove its HasChunk edges.
        
        Returns:
            list[dict[str, Any]]: A list containing a single delete_cascade operation payload filtered by the instance's `doc_id`.
        """
        return [{
            "operation": "delete_cascade",
            "start_node": "Document",
            "filter": {"doc_id": self.doc_id},
            "edges": ["HasChunk"],
        }]

    def response(self, response: Any) -> QueryResult:
        """
        Return a successful QueryResult containing this query's document id.
        
        Parameters:
            response (Any): Raw response from the database or execution engine (ignored).
        
        Returns:
            QueryResult: success=True and data set to {'doc_id': self.doc_id}.
        """
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
        Create a query that adds a Chunk node with its embedding and links it to a parent Document.
        
        Parameters:
            chunk_props (dict[str, Any]): Properties for the Chunk node; must include `chunk_id`.
            embedding (list[float]): Embedding vector for the chunk (expected dimensionality: 384).
            doc_id (str): Identifier of the parent Document to link the chunk to.
        """
        super().__init__()
        self.chunk_props = chunk_props
        self.embedding = embedding
        self.doc_id = doc_id

    def query(self) -> list[dict[str, Any]]:
        """
        Builds the operation payloads to add a Chunk node, its embedding vector, and edges linking the vector to the chunk and the chunk to its parent Document.
        
        Returns:
            list[dict[str, Any]]: A list of operation payload dictionaries: an `add_node` for `Chunk`, an `add_vector` for `ChunkVector`, an `add_edge` (`HasEmbedding`) from the chunk to its vector, and an `add_edge` (`HasChunk`) from the document to the chunk.
        """
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
        """
        Return a QueryResult indicating the chunk was added and include the chunk's id.
        
        Returns:
            QueryResult: success is `True`; `data` contains `{"chunk_id": <chunk_id from self.chunk_props>}`.
        """
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
        Initialize an AddChunkRelationship query that adds an edge between two chunk nodes.
        
        Parameters:
            edge_type (str): Name/type of the relationship edge (e.g., "ParentOf", "NextSibling").
            from_chunk_id (str): ID of the source chunk node.
            to_chunk_id (str): ID of the target chunk node.
        """
        super().__init__()
        self.edge_type = edge_type
        self.from_chunk_id = from_chunk_id
        self.to_chunk_id = to_chunk_id

    def query(self) -> list[dict[str, Any]]:
        """
        Add an edge of the specified type between two Chunk nodes identified by their chunk IDs.
        
        Returns:
            list[dict[str, Any]]: A single-element list containing the operation payload to add the edge between the source and target Chunk nodes.
        """
        return [{
            "operation": "add_edge",
            "edge_type": self.edge_type,
            "from": {"node_type": "Chunk", "filter": {"chunk_id": self.from_chunk_id}},
            "to": {"node_type": "Chunk", "filter": {"chunk_id": self.to_chunk_id}},
        }]

    def response(self, response: Any) -> QueryResult:
        """
        Produce a QueryResult indicating success with no payload.
        
        Parameters:
            response (Any): The raw response from the query (ignored).
        
        Returns:
            QueryResult: `success=True` with `data=None` and `error=None`.
        """
        return QueryResult(success=True)


class GetChunk(_get_query_base_class()):
    """Get a single chunk by ID."""

    def __init__(self, chunk_id: str) -> None:
        """
        Initialize a GetChunk query for retrieving a chunk by its identifier.
        
        Parameters:
            chunk_id (str): Identifier of the chunk to retrieve.
        """
        super().__init__()
        self.chunk_id = chunk_id

    def query(self) -> list[dict[str, Any]]:
        """
        Return the payload for a get_node query that retrieves a Chunk by its chunk_id and includes its vector.
        
        Returns:
            list[dict[str, Any]]: A single-item list containing an operation dictionary with keys "operation", "node_type", "filter", and "include_vector"; the "filter" matches this instance's `chunk_id`.
        """
        return [{
            "operation": "get_node",
            "node_type": "Chunk",
            "filter": {"chunk_id": self.chunk_id},
            "include_vector": True,
        }]

    def response(self, response: Any) -> QueryResult:
        """
        Convert a raw chunk retrieval response into a standardized QueryResult.
        
        Parameters:
            response (Any): Raw payload returned by the chunk query; treated as "found" if truthy.
        
        Returns:
            QueryResult: If `response` is truthy, `success=True` and `data` contains the response.
                         If `response` is falsy, `success=False` and `error` is "Chunk not found".
        """
        if response:
            return QueryResult(success=True, data=response)
        return QueryResult(success=False, error="Chunk not found")


class GetChunkWithContext(_get_query_base_class()):
    """Get a chunk with its parent, siblings, and neighbors."""

    def __init__(self, chunk_id: str) -> None:
        """
        Create a query to retrieve a chunk and its related context (parent, siblings, neighbors).
        
        Parameters:
            chunk_id (str): Identifier of the chunk to retrieve.
        """
        super().__init__()
        self.chunk_id = chunk_id

    def query(self) -> list[dict[str, Any]]:
        """
        Builds a traverse query payload to retrieve a Chunk and its related context (parent, siblings, next, prev).
        
        Returns:
            list[dict[str, Any]]: A single-element list containing a traversal operation payload that starts from the Chunk filtered by self.chunk_id and requests the 'chunk', 'parent', 'siblings', 'next', and 'prev' nodes.
        """
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
        """
        Wraps a raw backend response into a successful QueryResult.
        
        Parameters:
            response (Any): Raw response payload produced by executing the query.
        
        Returns:
            QueryResult: A result object with `success` set to `True` and `data` containing the provided response.
        """
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
        Create a SearchSimilarChunks query configured with a query vector and optional filters.
        
        Parameters:
            query_embedding (list[float]): Query embedding vector used for similarity search.
            top_k (int): Maximum number of results to return.
            doc_id (Optional[str]): If provided, restrict results to this document ID.
            level (Optional[int]): If provided, restrict results to this hierarchy level.
            semantic_type (Optional[str]): If provided, restrict results to this semantic type.
            threshold (float): Minimum similarity score for returned results.
        """
        super().__init__()
        self.query_embedding = query_embedding
        self.top_k = top_k
        self.doc_id = doc_id
        self.level = level
        self.semantic_type = semantic_type
        self.threshold = threshold

    def query(self) -> list[dict[str, Any]]:
        """
        Builds the payload for a vector similarity search over chunk vectors.
        
        Returns:
            A list containing a single operation dictionary for a `vector_search` targeting `ChunkVector` with the configured `query_embedding`, `top_k`, `threshold`, and any provided filters (`doc_id`, `level`, `semantic_type`); `return_nodes` is set to `True`.
        """
        filters: dict[str, Any] = {}
        if self.doc_id:
            filters["doc_id"] = self.doc_id
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
        """
        Wraps a raw backend response into a successful QueryResult.
        
        Parameters:
            response (Any): Raw response payload produced by executing the query.
        
        Returns:
            QueryResult: A result object with `success` set to `True` and `data` containing the provided response.
        """
        return QueryResult(success=True, data=response)


class SearchKeyword(_get_query_base_class()):
    """Full-text keyword search in chunk content."""

    def __init__(
        self,
        keyword: str,
        doc_id: Optional[str] = None,
    ) -> None:
        """
        Create a SearchKeyword query for full-text searching chunk content.
        
        Parameters:
            keyword (str): Term to search for within chunk content.
            doc_id (Optional[str]): Optional document ID to restrict results to a single document.
        """
        super().__init__()
        self.keyword = keyword
        self.doc_id = doc_id

    def query(self) -> list[dict[str, Any]]:
        """
        Builds the Helix-DB payload for a keyword search against chunk content.
        
        Returns:
            payload (list[dict[str, Any]]): A single-operation list containing a `keyword_search` dictionary targeting node_type `Chunk`, field `content`, with `keyword` set from the instance and `filters` including `doc_id` when provided.
        """
        filters: dict[str, Any] = {}
        if self.doc_id:
            filters["doc_id"] = self.doc_id

        return [{
            "operation": "keyword_search",
            "keyword": self.keyword,
            "node_type": "Chunk",
            "field": "content",
            "filters": filters,
        }]

    def response(self, response: Any) -> QueryResult:
        """
        Wraps a raw backend response into a successful QueryResult.
        
        Parameters:
            response (Any): Raw response payload produced by executing the query.
        
        Returns:
            QueryResult: A result object with `success` set to `True` and `data` containing the provided response.
        """
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
        Initialize a GetDocumentChunks query for retrieving chunks of a document.
        
        Parameters:
            doc_id (str): Document identifier to retrieve chunks for.
            level (Optional[int]): Optional hierarchy level to filter returned chunks.
        """
        super().__init__()
        self.doc_id = doc_id
        self.level = level

    def query(self) -> list[dict[str, Any]]:
        """
        Build a traverse payload to retrieve chunks belonging to a document.
        
        Includes an optional `level` filter and orders returned chunks by `sequence_position`.
        
        Returns:
            list[dict[str, Any]]: A single-operation payload for a "traverse" operation that starts from the Document identified by `doc_id`, traverses `HasChunk` edges outward, applies filters (`doc_id` and optional `level`), and requests results ordered by `sequence_position`.
        """
        filters: dict[str, Any] = {"doc_id": self.doc_id}
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
        """
        Wraps a raw backend response into a successful QueryResult.
        
        Parameters:
            response (Any): Raw response payload produced by executing the query.
        
        Returns:
            QueryResult: A result object with `success` set to `True` and `data` containing the provided response.
        """
        return QueryResult(success=True, data=response)


class ListDocuments(_get_query_base_class()):
    """List all documents."""

    def query(self) -> list[dict[str, Any]]:
        """
        Builds the query payload to retrieve all Document nodes ordered by creation time.
        
        Returns:
            list[dict[str, Any]]: A single-item list containing a `get_all` operation payload for node_type `Document` ordered by `created_at`.
        """
        return [{
            "operation": "get_all",
            "node_type": "Document",
            "order_by": "created_at",
        }]

    def response(self, response: Any) -> QueryResult:
        """
        Wraps a raw backend response into a successful QueryResult.
        
        Parameters:
            response (Any): Raw response payload produced by executing the query.
        
        Returns:
            QueryResult: A result object with `success` set to `True` and `data` containing the provided response.
        """
        return QueryResult(success=True, data=response)