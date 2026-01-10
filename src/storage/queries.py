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

import json
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

    def __init__(self, endpoint: Optional[str] = None) -> None:
        """Initialize with optional endpoint name."""
        self.endpoint = endpoint or self.__class__.__name__

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

    def __init__(
        self,
        doc_id: str,
        title: str,
        chunk_count: int,
        tags: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize AddDocument query.

        Args:
            doc_id: Unique document identifier
            title: Document title
            chunk_count: Number of chunks in document
            tags: Optional list of document tags
        """
        super().__init__()
        self.doc_id = doc_id
        self.title = title
        self.chunk_count = chunk_count
        self.created_at = int(time.time())
        # Normalize tags to lowercase and filter empty strings
        self.tags = [t.lower().strip() for t in (tags or []) if t and t.strip()]

    def query(self) -> list[dict[str, Any]]:
        """Return query parameters for HelixQL AddDocument query."""
        return [{
            "doc_id": self.doc_id,
            "title": self.title,
            "tags": json.dumps(self.tags),
            "chunk_count": self.chunk_count,
            "created_at": self.created_at,
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True, data={"doc_id": self.doc_id})


class DeleteDocument(_get_query_base_class()):
    """Delete a document by doc_id string.

    Note: Cascade delete of chunks should be handled in client.py
    by first deleting chunks then the document.
    """

    def __init__(self, doc_id: str) -> None:
        """
        Initialize DeleteDocument query.

        Args:
            doc_id: User-facing document identifier to delete
        """
        super().__init__(endpoint="DeleteDocumentByDocId")
        self.doc_id = doc_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL DeleteDocumentByDocId query."""
        return [{"doc_id": self.doc_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True, data={"doc_id": self.doc_id})


# Chunk Queries - matching HelixQL query signatures in db/queries.hx
class AddChunk(_get_query_base_class()):
    """Add a chunk node to the database.

    Maps to HelixQL AddChunk query with all chunk properties.
    """

    def __init__(self, chunk_props: dict[str, Any]) -> None:
        """
        Initialize AddChunk query.

        Args:
            chunk_props: Dictionary containing all chunk properties
        """
        super().__init__()
        self.chunk_props = chunk_props

    def query(self) -> list[dict[str, Any]]:
        """Return query parameters matching HelixQL AddChunk signature."""
        p = self.chunk_props
        return [{
            "chunk_id": p.get("chunk_id", ""),
            "content": p.get("content", ""),
            "level": p.get("level", 0),
            "token_count": p.get("token_count", 0),
            "word_count": p.get("word_count", 0),
            "document_id": p.get("document_id", ""),
            "semantic_type": p.get("semantic_type", ""),
            "topic_tags": p.get("topic_tags", "[]"),
            "section_path": p.get("section_path", "[]"),
            "source_section": p.get("source_section", ""),
            "sequence_position": p.get("sequence_position", 0),
            "parent_id": p.get("parent_id", ""),
            "child_ids": p.get("child_ids", "[]"),
            "sibling_ids": p.get("sibling_ids", "[]"),
            "prev_id": p.get("prev_id", ""),
            "next_id": p.get("next_id", ""),
            "has_table": p.get("has_table", 0),
            "has_code": p.get("has_code", 0),
            "has_math": p.get("has_math", 0),
            "has_list": p.get("has_list", 0),
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(
            success=True,
            data={"chunk_id": self.chunk_props.get("chunk_id")},
        )


class LinkDocumentChunk(_get_query_base_class()):
    """Link a document to a chunk via HasChunk edge.

    Maps to HelixQL LinkDocumentChunk query.
    Note: doc_id and chunk_id are internal Helix node IDs.
    """

    def __init__(self, doc_id: str, chunk_id: str) -> None:
        """
        Initialize LinkDocumentChunk query.

        Args:
            doc_id: Internal document node ID from AddDocument response
            chunk_id: Internal chunk node ID from AddChunk response
        """
        super().__init__()
        self.doc_id = doc_id
        self.chunk_id = chunk_id

    def query(self) -> list[dict[str, Any]]:
        """Return query parameters."""
        return [{
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True)


class AddParentChild(_get_query_base_class()):
    """Add parent-child relationship between chunks.

    Maps to HelixQL AddParentChild query.
    """

    def __init__(self, parent_id: str, child_id: str) -> None:
        """
        Initialize AddParentChild query.

        Args:
            parent_id: Internal parent chunk node ID
            child_id: Internal child chunk node ID
        """
        super().__init__()
        self.parent_id = parent_id
        self.child_id = child_id

    def query(self) -> list[dict[str, Any]]:
        """Return query parameters."""
        return [{
            "parent_id": self.parent_id,
            "child_id": self.child_id,
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True)


class AddNextSibling(_get_query_base_class()):
    """Add next sibling relationship between chunks.

    Maps to HelixQL AddNextSibling query.
    """

    def __init__(self, chunk_id: str, next_id: str) -> None:
        """
        Initialize AddNextSibling query.

        Args:
            chunk_id: Internal chunk node ID
            next_id: Internal next chunk node ID
        """
        super().__init__()
        self.chunk_id = chunk_id
        self.next_id = next_id

    def query(self) -> list[dict[str, Any]]:
        """Return query parameters."""
        return [{
            "chunk_id": self.chunk_id,
            "next_id": self.next_id,
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True)


# Legacy compatibility aliases
class AddChunkWithEmbedding(AddChunk):
    """Legacy alias for AddChunk (embeddings handled separately)."""

    def __init__(
        self,
        chunk_props: dict[str, Any],
        embedding: list[float],
        doc_id: str,
    ) -> None:
        # Store embedding for potential future use, but AddChunk doesn't use it
        self.embedding = embedding
        self.doc_id = doc_id
        super().__init__(chunk_props)


class AddChunkRelationship(_get_query_base_class()):
    """Legacy wrapper that routes to appropriate relationship query."""

    def __init__(
        self,
        edge_type: str,
        from_chunk_id: str,
        to_chunk_id: str,
    ) -> None:
        self.edge_type = edge_type
        self.from_chunk_id = from_chunk_id
        self.to_chunk_id = to_chunk_id
        # Compute endpoint before calling super()
        endpoint = self._compute_endpoint()
        super().__init__(endpoint=endpoint)

    def _compute_endpoint(self) -> str:
        """Compute endpoint based on edge type."""
        if self.edge_type == "ParentOf":
            return "AddParentChild"
        elif self.edge_type == "NextSibling":
            return "AddNextSibling"
        return f"Add{self.edge_type}"

    def query(self) -> list[dict[str, Any]]:
        """Return query parameters based on edge type."""
        if self.edge_type == "ParentOf":
            return [{
                "parent_id": self.from_chunk_id,
                "child_id": self.to_chunk_id,
            }]
        elif self.edge_type == "NextSibling":
            return [{
                "chunk_id": self.from_chunk_id,
                "next_id": self.to_chunk_id,
            }]
        else:
            # Fallback for other edge types
            return [{
                "from_id": self.from_chunk_id,
                "to_id": self.to_chunk_id,
            }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True)


class GetChunk(_get_query_base_class()):
    """Get a single chunk by user-facing chunk_id."""

    def __init__(self, chunk_id: str) -> None:
        """
        Initialize GetChunk query.

        Args:
            chunk_id: User-facing chunk identifier
        """
        super().__init__(endpoint="GetChunkByChunkId")
        self.chunk_id = chunk_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL GetChunkByChunkId query."""
        return [{"chunk_id": self.chunk_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects list of chunk nodes."""
        # HelixQL returns list of matching nodes
        if isinstance(response, list) and len(response) > 0:
            node = response[0]
            return QueryResult(success=True, data={"node": node})
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
    """Vector similarity search - filtering done in Python post-processing."""

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
            query_embedding: Query vector
            top_k: Number of results desired
            doc_id: Filter by document (post-filtered in client.py)
            level: Filter by hierarchy level (post-filtered in client.py)
            semantic_type: Filter by semantic type (post-filtered in client.py)
            threshold: Minimum similarity score (post-filtered in client.py)
        """
        super().__init__(endpoint="SearchSimilar")
        self.query_embedding = query_embedding
        self.top_k = top_k
        # Store filters for post-processing in client.py
        self.doc_id = doc_id
        self.level = level
        self.semantic_type = semantic_type
        self.threshold = threshold

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL SearchSimilar query."""
        # Request extra results if we have filters to apply
        has_filters = any([self.doc_id, self.level is not None, self.semantic_type, self.threshold > 0])
        request_top_k = self.top_k * 3 if has_filters else self.top_k

        return [{
            "query_vec": self.query_embedding,
            "top_k": request_top_k,
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects dict with 'results' key or list of nodes."""
        if isinstance(response, dict) and 'results' in response:
            results = response['results']
        elif isinstance(response, list):
            results = response
        else:
            results = []
        return QueryResult(success=True, data=results)


class SearchKeyword(_get_query_base_class()):
    """BM25 keyword search - filtering done in Python post-processing."""

    def __init__(
        self,
        keyword: str,
        doc_id: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> None:
        """
        Initialize SearchKeyword query.

        Args:
            keyword: Search term
            doc_id: Filter by document (post-filtered in client.py)
            case_sensitive: Whether to apply case-sensitive filter (post-filtered in client.py)
                           Note: BM25 doesn't support case-sensitive natively
        """
        super().__init__(endpoint="SearchKeywordBM25")
        self.keyword = keyword
        # Store filters for post-processing in client.py
        self.doc_id = doc_id
        self.case_sensitive = case_sensitive

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL SearchKeywordBM25 query."""
        return [{
            "keyword": self.keyword,
            "limit": 100,  # Get more results for filtering
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects dict with 'results' key or list of nodes."""
        if isinstance(response, dict) and 'results' in response:
            results = response['results']
        elif isinstance(response, list):
            results = response
        else:
            results = []
        return QueryResult(success=True, data=results)


# Graph Traversal Queries
class GetDocumentChunks(_get_query_base_class()):
    """Get all chunks for a document by document_id."""

    def __init__(
        self,
        doc_id: str,
        level: Optional[int] = None,
    ) -> None:
        """
        Initialize GetDocumentChunks query.

        Args:
            doc_id: Document identifier (user-facing document_id)
            level: Filter by hierarchy level (optional)
        """
        # Dynamic endpoint based on whether level is specified
        endpoint = "GetChunksByDocAndLevel" if level is not None else "GetChunksByDocumentId"
        super().__init__(endpoint=endpoint)
        self.doc_id = doc_id
        self.level = level

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL query."""
        params: dict[str, Any] = {"document_id": self.doc_id}
        if self.level is not None:
            params["level"] = self.level
        return [params]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects dict with 'chunks' key or list of nodes."""
        if isinstance(response, dict) and 'chunks' in response:
            chunks = response['chunks']
        elif isinstance(response, list):
            chunks = response
        else:
            chunks = []
        return QueryResult(success=True, data=chunks)


class GetChunksByParentId(_get_query_base_class()):
    """Get chunks by parent_id field (for finding children)."""

    def __init__(self, parent_id: str) -> None:
        """
        Initialize GetChunksByParentId query.

        Args:
            parent_id: The chunk_id of the parent to find children for
        """
        super().__init__(endpoint="GetChunksByParentId")
        self.parent_id = parent_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL GetChunksByParentId query."""
        return [{"parent_id": self.parent_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects dict with 'chunks' key or list of nodes."""
        if isinstance(response, dict) and 'chunks' in response:
            chunks = response['chunks']
        elif isinstance(response, list):
            chunks = response
        else:
            chunks = []
        return QueryResult(success=True, data=chunks)


class ListDocuments(_get_query_base_class()):
    """List all documents."""

    def __init__(self) -> None:
        """Initialize ListDocuments query."""
        super().__init__(endpoint="ListDocuments")

    def query(self) -> list[dict[str, Any]]:
        """Return empty parameters for HelixQL ListDocuments query."""
        return [{}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects dict with 'docs' key or list of nodes."""
        # helix-py passes raw HTTP response: {'docs': [...]}
        if isinstance(response, dict) and 'docs' in response:
            docs = response['docs']
        elif isinstance(response, list):
            docs = response
        else:
            docs = []
        return QueryResult(success=True, data=docs)


# Vector Queries
class AddChunkVector(_get_query_base_class()):
    """Add vector embedding to database.

    Maps to HelixQL AddChunkVector query which stores the embedding
    in a ChunkVector node.
    """

    def __init__(
        self,
        embedding: list[float],
        model_name: str,
        embedding_dim: int,
    ) -> None:
        """
        Initialize AddChunkVector query.

        Args:
            embedding: The embedding vector
            model_name: Name of the embedding model used
            embedding_dim: Dimension of the embedding
        """
        super().__init__(endpoint="AddChunkVector")
        self.embedding = embedding
        self.model_name = model_name
        self.embedding_dim = embedding_dim

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL AddChunkVector query."""
        return [{
            "embedding": self.embedding,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response - returns vector node ID."""
        return QueryResult(success=True, data=response)


class LinkChunkVector(_get_query_base_class()):
    """Link a chunk to its vector embedding via HasEmbedding edge.

    Maps to HelixQL LinkChunkVector query.
    Note: Uses internal Helix node IDs, not user-facing chunk_id.
    """

    def __init__(self, chunk_id: str, vector_id: str) -> None:
        """
        Initialize LinkChunkVector query.

        Args:
            chunk_id: Internal chunk node ID from AddChunk response
            vector_id: Internal vector node ID from AddChunkVector response
        """
        super().__init__(endpoint="LinkChunkVector")
        self.chunk_id = chunk_id
        self.vector_id = vector_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL LinkChunkVector query."""
        return [{
            "chunk_id": self.chunk_id,
            "vector_id": self.vector_id,
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True)


# Delete Queries (using internal IDs)
class DropChunk(_get_query_base_class()):
    """Drop a chunk by internal ID.

    Maps to HelixQL DropChunk query. Dropping a node also
    removes all connected edges (HasChunk, ParentOf, NextSibling, HasEmbedding).
    """

    def __init__(self, chunk_internal_id: str) -> None:
        """
        Initialize DropChunk query.

        Args:
            chunk_internal_id: Internal Helix node ID
        """
        super().__init__(endpoint="DropChunk")
        self.chunk_internal_id = chunk_internal_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL DropChunk query."""
        return [{"chunk_id": self.chunk_internal_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True)


class DropDocument(_get_query_base_class()):
    """Drop a document by internal ID.

    Maps to HelixQL DropDocument query.
    Note: Chunks should be dropped first to avoid orphans.
    """

    def __init__(self, doc_internal_id: str) -> None:
        """
        Initialize DropDocument query.

        Args:
            doc_internal_id: Internal Helix node ID
        """
        super().__init__(endpoint="DropDocument")
        self.doc_internal_id = doc_internal_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL DropDocument query."""
        return [{"doc_id": self.doc_internal_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True)


class GetDocumentByDocId(_get_query_base_class()):
    """Get document by user-facing doc_id string.

    Used to retrieve internal ID for delete operations.
    """

    def __init__(self, doc_id: str) -> None:
        """
        Initialize GetDocumentByDocId query.

        Args:
            doc_id: User-facing document identifier
        """
        super().__init__(endpoint="GetDocumentByDocId")
        self.doc_id = doc_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL GetDocumentByDocId query."""
        return [{"doc_id": self.doc_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects list of document nodes."""
        if isinstance(response, list) and len(response) > 0:
            node = response[0]
            return QueryResult(success=True, data={"node": node})
        return QueryResult(success=False, error="Document not found")


# Update Queries
class UpdateChunkContent(_get_query_base_class()):
    """Update chunk content by internal ID.

    Maps to HelixQL UpdateChunkContent query.
    Note: Uses internal Helix node ID, not user-facing chunk_id.
    """

    def __init__(
        self,
        chunk_internal_id: str,
        content: str,
        word_count: int,
    ) -> None:
        """
        Initialize UpdateChunkContent query.

        Args:
            chunk_internal_id: Internal Helix node ID
            content: New content for the chunk
            word_count: New word count
        """
        super().__init__(endpoint="UpdateChunkContent")
        self.chunk_internal_id = chunk_internal_id
        self.content = content
        self.word_count = word_count

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL UpdateChunkContent query."""
        return [{
            "chunk_id": self.chunk_internal_id,
            "content": self.content,
            "word_count": self.word_count,
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response - returns updated node."""
        return QueryResult(success=True, data=response)


# ============================================================================
# Concept Queries
# ============================================================================


class AddConcept(_get_query_base_class()):
    """Add a concept node to the database.

    Maps to HelixQL AddConcept query.
    """

    def __init__(
        self,
        concept_id: str,
        name: str,
        definition: str,
        concept_type: str,
        source_documents: list[str],
        aliases: list[str],
    ) -> None:
        """
        Initialize AddConcept query.

        Args:
            concept_id: Slugified unique ID (e.g., "cost-of-goods-sold")
            name: Display name (e.g., "Cost of Goods Sold")
            definition: 1-2 sentence definition
            concept_type: Type (term, method, principle, formula, account)
            source_documents: List of document IDs where concept is defined
            aliases: Alternative names or abbreviations
        """
        super().__init__()
        self.concept_id = concept_id
        self.name = name
        self.definition = definition
        self.concept_type = concept_type
        self.source_documents = source_documents
        self.aliases = aliases

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL AddConcept query."""
        return [{
            "concept_id": self.concept_id,
            "name": self.name,
            "definition": self.definition,
            "concept_type": self.concept_type,
            "source_documents": json.dumps(self.source_documents),
            "aliases": json.dumps(self.aliases),
        }]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True, data={"concept_id": self.concept_id})


class UpdateConcept(_get_query_base_class()):
    """Update an existing concept's source_documents and aliases.

    Maps to HelixQL UpdateConcept or UpdateConceptWithDefinition query.
    Used for cross-document concept merging.
    """

    def __init__(
        self,
        concept_id: str,
        source_documents: list[str],
        aliases: list[str],
        definition: Optional[str] = None,
    ) -> None:
        """
        Initialize UpdateConcept query.

        Args:
            concept_id: Slugified unique ID of concept to update
            source_documents: Updated list of document IDs
            aliases: Updated list of aliases
            definition: Optional updated definition (None to keep existing)
        """
        # Use different endpoint based on whether definition is provided
        endpoint = "UpdateConceptWithDefinition" if definition else "UpdateConcept"
        super().__init__(endpoint=endpoint)
        self.concept_id = concept_id
        self.source_documents = source_documents
        self.aliases = aliases
        self.definition = definition

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL UpdateConcept query."""
        params = {
            "concept_id": self.concept_id,
            "source_documents": json.dumps(self.source_documents),
            "aliases": json.dumps(self.aliases),
        }
        if self.definition is not None:
            params["definition"] = self.definition
        return [params]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True, data={"concept_id": self.concept_id})


class GetConceptByName(_get_query_base_class()):
    """Get concept by name.

    Maps to HelixQL GetConceptByName query.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize GetConceptByName query.

        Args:
            name: Concept name to search for
        """
        super().__init__(endpoint="GetConceptByName")
        self.name = name

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL GetConceptByName query."""
        return [{"name": self.name}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects list of concept nodes."""
        if isinstance(response, list) and len(response) > 0:
            return QueryResult(success=True, data={"node": response[0]})
        return QueryResult(success=False, error="Concept not found")


class GetConceptById(_get_query_base_class()):
    """Get concept by concept_id.

    Maps to HelixQL GetConceptById query.
    """

    def __init__(self, concept_id: str) -> None:
        """
        Initialize GetConceptById query.

        Args:
            concept_id: Slugified concept identifier
        """
        super().__init__(endpoint="GetConceptById")
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL GetConceptById query."""
        return [{"concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects list of concept nodes."""
        if isinstance(response, list) and len(response) > 0:
            return QueryResult(success=True, data={"node": response[0]})
        return QueryResult(success=False, error="Concept not found")


class ListConcepts(_get_query_base_class()):
    """List all concepts.

    Maps to HelixQL ListConcepts query.
    """

    def __init__(self) -> None:
        """Initialize ListConcepts query."""
        super().__init__(endpoint="ListConcepts")

    def query(self) -> list[dict[str, Any]]:
        """Return empty parameters for HelixQL ListConcepts query."""
        return [{}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects list of concept nodes."""
        if isinstance(response, list):
            return QueryResult(success=True, data=response)
        return QueryResult(success=True, data=[])


class ListDocumentConcepts(_get_query_base_class()):
    """List concepts from a specific document.

    Maps to HelixQL ListDocumentConcepts query.
    """

    def __init__(self, document_id: str) -> None:
        """
        Initialize ListDocumentConcepts query.

        Args:
            document_id: Document identifier to filter by
        """
        super().__init__(endpoint="ListDocumentConcepts")
        self.document_id = document_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL ListDocumentConcepts query."""
        return [{"document_id": self.document_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects list of concept nodes."""
        if isinstance(response, list):
            return QueryResult(success=True, data=response)
        return QueryResult(success=True, data=[])


class LinkChunkDefinesConcept(_get_query_base_class()):
    """Link chunk to concept it defines.

    Maps to HelixQL LinkChunkDefinesConcept query.
    Creates DefinesConcept edge from Chunk to Concept.
    """

    def __init__(self, chunk_id: str, concept_id: str) -> None:
        """
        Initialize LinkChunkDefinesConcept query.

        Args:
            chunk_id: Internal chunk node ID
            concept_id: Internal concept node ID
        """
        super().__init__(endpoint="LinkChunkDefinesConcept")
        self.chunk_id = chunk_id
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL LinkChunkDefinesConcept query."""
        return [{"chunk_id": self.chunk_id, "concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True)


class LinkChunkMentionsConcept(_get_query_base_class()):
    """Link chunk to concept it mentions.

    Maps to HelixQL LinkChunkMentionsConcept query.
    Creates MentionsConcept edge from Chunk to Concept.
    """

    def __init__(self, chunk_id: str, concept_id: str) -> None:
        """
        Initialize LinkChunkMentionsConcept query.

        Args:
            chunk_id: Internal chunk node ID
            concept_id: Internal concept node ID
        """
        super().__init__(endpoint="LinkChunkMentionsConcept")
        self.chunk_id = chunk_id
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL LinkChunkMentionsConcept query."""
        return [{"chunk_id": self.chunk_id, "concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True)


class LinkConceptRelatesTo(_get_query_base_class()):
    """Link two concepts with a relationship.

    Maps to HelixQL LinkConceptRelatesTo query.
    Creates RelatesTo edge from Concept to Concept with relation_type.
    """

    def __init__(self, from_id: str, to_id: str, relation_type: str = "relates_to") -> None:
        """
        Initialize LinkConceptRelatesTo query.

        Args:
            from_id: Internal source concept node ID
            to_id: Internal target concept node ID
            relation_type: Relationship type (uses, requires, calculated_from, etc.)
        """
        super().__init__(endpoint="LinkConceptRelatesTo")
        self.from_id = from_id
        self.to_id = to_id
        self.relation_type = relation_type

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL LinkConceptRelatesTo query."""
        return [{"from_id": self.from_id, "to_id": self.to_id, "relation_type": self.relation_type}]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True)


class GetConceptDefinitionChunks(_get_query_base_class()):
    """Get chunks that define a concept (for citations).

    Maps to HelixQL GetConceptDefinitionChunks query.
    Traverses DefinesConcept edges in reverse.
    """

    def __init__(self, concept_id: str) -> None:
        """
        Initialize GetConceptDefinitionChunks query.

        Args:
            concept_id: Internal concept node ID
        """
        super().__init__(endpoint="GetConceptDefinitionChunks")
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL GetConceptDefinitionChunks query."""
        return [{"concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects list of chunk nodes."""
        if isinstance(response, list):
            return QueryResult(success=True, data=response)
        return QueryResult(success=True, data=[])


class GetConceptMentionChunks(_get_query_base_class()):
    """Get chunks that mention a concept.

    Maps to HelixQL GetConceptMentionChunks query.
    Traverses MentionsConcept edges in reverse.
    """

    def __init__(self, concept_id: str) -> None:
        """
        Initialize GetConceptMentionChunks query.

        Args:
            concept_id: Internal concept node ID
        """
        super().__init__(endpoint="GetConceptMentionChunks")
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL GetConceptMentionChunks query."""
        return [{"concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects list of chunk nodes."""
        if isinstance(response, list):
            return QueryResult(success=True, data=response)
        return QueryResult(success=True, data=[])


class GetRelatedConcepts(_get_query_base_class()):
    """Get concepts related to a given concept.

    Maps to HelixQL GetRelatedConcepts query.
    Traverses RelatesTo edges outward.
    """

    def __init__(self, concept_id: str) -> None:
        """
        Initialize GetRelatedConcepts query.

        Args:
            concept_id: Internal concept node ID
        """
        super().__init__(endpoint="GetRelatedConcepts")
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL GetRelatedConcepts query."""
        return [{"concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects list of concept nodes."""
        if isinstance(response, list):
            return QueryResult(success=True, data=response)
        return QueryResult(success=True, data=[])


class GetConceptDependents(_get_query_base_class()):
    """Get concepts that relate TO this one (reverse traversal).

    Maps to HelixQL GetConceptDependents query.
    Traverses RelatesTo edges inward.
    """

    def __init__(self, concept_id: str) -> None:
        """
        Initialize GetConceptDependents query.

        Args:
            concept_id: Internal concept node ID
        """
        super().__init__(endpoint="GetConceptDependents")
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL GetConceptDependents query."""
        return [{"concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects list of concept nodes."""
        if isinstance(response, list):
            return QueryResult(success=True, data=response)
        return QueryResult(success=True, data=[])


class GetRelatedConceptsWithTypes(_get_query_base_class()):
    """Get related concepts with relationship types.

    Maps to HelixQL GetRelatedConceptsWithTypes query.
    Returns edges with relation_type property and target concept info.
    """

    def __init__(self, concept_id: str) -> None:
        """
        Initialize GetRelatedConceptsWithTypes query.

        Args:
            concept_id: Internal concept node ID
        """
        super().__init__(endpoint="GetRelatedConceptsWithTypes")
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL GetRelatedConceptsWithTypes query."""
        return [{"concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects list of edges with relation_type."""
        if isinstance(response, list):
            return QueryResult(success=True, data=response)
        return QueryResult(success=True, data=[])


class GetConceptDependentsWithTypes(_get_query_base_class()):
    """Get concepts that depend on this one with relationship types.

    Maps to HelixQL GetConceptDependentsWithTypes query.
    Returns edges with relation_type property and source concept info.
    """

    def __init__(self, concept_id: str) -> None:
        """
        Initialize GetConceptDependentsWithTypes query.

        Args:
            concept_id: Internal concept node ID
        """
        super().__init__(endpoint="GetConceptDependentsWithTypes")
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL GetConceptDependentsWithTypes query."""
        return [{"concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response - expects list of edges with relation_type."""
        if isinstance(response, list):
            return QueryResult(success=True, data=response)
        return QueryResult(success=True, data=[])


class DropConcept(_get_query_base_class()):
    """Delete a concept by internal ID.

    Maps to HelixQL DropConcept query.
    Note: Also removes connected edges (DefinesConcept, MentionsConcept, RelatesTo).
    """

    def __init__(self, concept_id: str) -> None:
        """
        Initialize DropConcept query.

        Args:
            concept_id: Internal concept node ID
        """
        super().__init__(endpoint="DropConcept")
        self.concept_id = concept_id

    def query(self) -> list[dict[str, Any]]:
        """Return parameters for HelixQL DropConcept query."""
        return [{"concept_id": self.concept_id}]

    def response(self, response: Any) -> QueryResult:
        """Process response."""
        return QueryResult(success=True)
