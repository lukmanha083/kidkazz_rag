"""Main Helix-DB client for chunk storage.

This module provides HelixChunkStore, the primary interface for storing
and retrieving hierarchical chunks with vector embeddings in Helix-DB.

Usage:
    from src.storage import HelixChunkStore, HelixConfig

    # Connect to local Helix-DB
    config = HelixConfig(local=True, port=6969)
    store = HelixChunkStore(config)

    # Store document
    store.store_document("doc1", "Title", embedded_chunks, metadata_list)

    # Search
    results = store.search_similar(query_embedding, top_k=5)

    # Clean up
    store.close()
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from src.chunker import ChunkMetadata, EmbeddedChunk

from .converters import (
    chunk_to_helix_node,
    document_to_helix_node,
    embedded_chunk_to_helix_vector,
    helix_to_embedded_chunk,
)
from .queries import (
    AddChunk,
    AddChunkVector,
    AddConcept,
    AddDocument,
    AddNextSibling,
    AddParentChild,
    DeleteDocument,
    DropChunk,
    DropDocument,
    GetChunk,
    GetChunkForVector,
    GetChunksByParentId,
    GetChunkWithContext,
    GetConceptById,
    GetConceptByName,
    GetConceptDefinitionChunks,
    GetDocumentByDocId,
    GetDocumentChunks,
    GetRelatedConcepts,
    GetSummaryForVector,
    GetConceptDependents,
    BatchDropChunkConceptEdges,
    BatchLinkChunkDefinesConcept,
    DropConcept,
    LinkChunkDefinesConcept,
    LinkChunkMentionsConcept,
    LinkChunkVector,
    LinkConceptRelatesTo,
    LinkDocumentChunk,
    ListConcepts,
    ListDocumentConcepts,
    ListDocuments,
    QueryResult,
    SearchKeyword,
    SearchSimilarChunks,
    UpdateChunkContent,
    UpdateConcept,
    TYPED_CONCEPT_QUERIES_FORWARD,
    TYPED_CONCEPT_QUERIES_REVERSE,
)

logger = logging.getLogger(__name__)


@dataclass
class HelixConfig:
    """Configuration for Helix-DB connection.

    Attributes:
        local: Whether to connect to local instance
        port: Port number for local instance (default: 6969)
        api_endpoint: API endpoint for cloud instance
        verbose: Enable verbose logging
        db_path: Path to database files (for local instance)
    """

    local: bool = True
    port: int = 6969
    api_endpoint: Optional[str] = None
    verbose: bool = False
    db_path: Optional[str] = None


@runtime_checkable
class ChunkStoreProtocol(Protocol):
    """Protocol defining the chunk storage interface.

    This protocol defines the methods that any chunk store implementation
    must provide. Both HelixChunkStore and MockChunkStore implement this.
    """

    def store_document(
        self,
        doc_id: str,
        title: str,
        embedded_chunks: list[EmbeddedChunk],
        metadata_list: list[ChunkMetadata],
        tags: Optional[list[str]] = None,
        book_type: str = "general",
    ) -> None:
        """Store a document with all chunks."""
        ...

    def get_chunk(self, chunk_id: str) -> Optional[EmbeddedChunk]:
        """Get a chunk by ID."""
        ...

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk."""
        ...

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks."""
        ...

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_id: Optional[str] = None,
        level: Optional[int] = None,
        semantic_type: Optional[str] = None,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
        has_table: Optional[bool] = None,
        has_code: Optional[bool] = None,
        has_math: Optional[bool] = None,
        has_image: Optional[bool] = None,
        header_level: Optional[int] = None,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """Find similar chunks by vector search."""
        ...

    def get_parent(self, chunk_id: str) -> Optional[EmbeddedChunk]:
        """Get parent chunk."""
        ...

    def get_children(self, chunk_id: str) -> list[EmbeddedChunk]:
        """Get child chunks."""
        ...

    def get_siblings(self, chunk_id: str) -> list[EmbeddedChunk]:
        """Get sibling chunks."""
        ...

    def get_context_window(
        self,
        chunk_id: str,
        window_size: int = 1,
    ) -> list[EmbeddedChunk]:
        """Get chunk with surrounding context."""
        ...

    def list_documents(self, tags: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """List all documents, optionally filtered by tags."""
        ...


class HelixChunkStore:
    """
    Main storage interface for chunks in Helix-DB.

    Provides CRUD operations, vector similarity search, and graph traversal
    for hierarchical chunks with embeddings.

    Attributes:
        config: Connection configuration
    """

    def __init__(self, config: Optional[HelixConfig] = None) -> None:
        """
        Initialize Helix-DB connection.

        Args:
            config: Optional connection configuration

        Note:
            Connection is lazy-initialized on first operation.
        """
        self.config = config or HelixConfig()
        self._client: Any = None
        self._initialized = False

    def check_embedding_compatibility(self, new_dim: int, new_model: str = "") -> None:
        """Check if new embeddings are compatible with existing vectors in DB.

        Probes the vector index by sending a dummy search. If Helix returns
        a dimension error, it means stored vectors have a different size.
        Raises RuntimeError so the user can wipe the DB before proceeding.

        Args:
            new_dim: Dimension of the embeddings about to be stored/queried
            new_model: Model name for the new embeddings (for display)
        """
        self._ensure_connected()

        try:
            from src.storage.queries import SearchSimilarChunks

            probe = SearchSimilarChunks(
                query_embedding=[0.0] * new_dim, top_k=1,
            )
            # Call helix directly to catch dimension errors from the server
            self._client.query(probe)
            # If it succeeds or returns empty, dimensions are compatible

        except RuntimeError:
            raise
        except Exception as e:
            err_msg = str(e).lower()
            if "invalid vector dimensions" in err_msg:
                raise RuntimeError(
                    f"Embedding dimension mismatch! "
                    f"Current embedder produces dim={new_dim} "
                    f"(model: {new_model}), but database vectors use a "
                    f"different dimension. You must wipe the database and "
                    f"re-ingest all documents. To wipe: stop helix, run "
                    f"'docker run --rm -v .helix/.volumes/dev:/data "
                    f"alpine rm -rf /data/*', then 'helix push dev'."
                ) from e
            # Other errors (e.g., no vectors yet) are fine
            logger.debug("Embedding compatibility check: %s", e)

    def _ensure_connected(self) -> None:
        """Ensure connection to Helix-DB is established."""
        if self._initialized:
            return

        try:
            import helix

            self._client = helix.Client(
                local=self.config.local,
                port=self.config.port,
                verbose=self.config.verbose,
            )
            self._initialized = True

        except ImportError as e:
            raise ImportError(
                "helix-py is not installed. Run: pip install helix-py"
            ) from e

    def _execute_query(self, query: Any) -> Any:
        """Execute a query and process the response through the query's response method.

        Args:
            query: Query object with query() and response() methods

        Returns:
            QueryResult from the query's response() method
        """
        raw_result = self._client.query(query)

        # If helix-py already returned a QueryResult-like object, use it directly
        if hasattr(raw_result, 'success') and hasattr(raw_result, 'data'):
            # Extract actual data if it's nested in 'docs' key
            if isinstance(raw_result.data, dict) and 'docs' in raw_result.data:
                return QueryResult(
                    success=raw_result.success,
                    data=raw_result.data['docs'],
                    error=getattr(raw_result, 'error', None)
                )
            return raw_result

        # If helix-py returned a list of QueryResult objects, extract the first one
        if isinstance(raw_result, list) and len(raw_result) > 0:
            first = raw_result[0]
            if hasattr(first, 'success') and hasattr(first, 'data'):
                # Extract actual data if nested in 'docs' key
                if isinstance(first.data, dict) and 'docs' in first.data:
                    return QueryResult(
                        success=first.success,
                        data=first.data['docs'],
                        error=getattr(first, 'error', None)
                    )
                return first

        # Call the query's response method to convert raw response to QueryResult
        if hasattr(query, 'response'):
            return query.response(raw_result)

        # Fallback: wrap in QueryResult
        return QueryResult(success=True, data=raw_result)

    def _extract_node_id(self, response: Any) -> Optional[str]:
        """Extract internal node ID from helix-py response.

        Args:
            response: Response from helix-py query

        Returns:
            Internal node ID string, or None if not found
        """
        # Response could be a list of results
        if isinstance(response, list) and len(response) > 0:
            response = response[0]

        # Check for ID in various formats helix-py might return
        if hasattr(response, 'id'):
            return str(response.id)
        if hasattr(response, 'data'):
            data = response.data
            if isinstance(data, dict):
                # Check for nested node data under various keys
                for key in ['chunk', 'vec', 'doc', 'node', 'N', 'summary', 'Summary', 'concept', 'Concept']:
                    if key in data and isinstance(data[key], dict):
                        node_id = data[key].get('id') or data[key].get('Id')
                        if node_id:
                            return str(node_id)
                # Direct ID in data
                result = data.get('id') or data.get('_id') or data.get('node_id')
                if result:
                    return str(result)
            if hasattr(data, 'id'):
                return str(data.id)
        if isinstance(response, dict):
            # Check for nested node data first
            for key in ['chunk', 'vec', 'doc', 'node', 'N', 'summary', 'Summary', 'concept', 'Concept']:
                if key in response and isinstance(response[key], dict):
                    node_id = response[key].get('id') or response[key].get('Id')
                    if node_id:
                        return str(node_id)
            # Check common ID locations
            result = response.get('id') or response.get('_id') or response.get('node_id')
            if result:
                return result

        return None

    def store_document(
        self,
        doc_id: str,
        title: str,
        embedded_chunks: list[EmbeddedChunk],
        metadata_list: list[ChunkMetadata],
        tags: Optional[list[str]] = None,
        book_type: str = "general",
    ) -> dict[str, str]:
        """
        Store a complete document with all chunks, embeddings, and relationships.

        Args:
            doc_id: Unique document identifier
            title: Document title
            embedded_chunks: List of chunks with embeddings
            metadata_list: Corresponding metadata for each chunk
            tags: Optional list of document tags (e.g., ["inventory", "accounting"])
            book_type: Extraction profile name (programming, erp, stem, etc.)

        Returns:
            Dictionary mapping chunk string IDs to internal database IDs.
            Useful for creating concept links after ingestion.

        Raises:
            ImportError: If helix-py is not installed
            RuntimeError: If storage operation fails
        """
        self._ensure_connected()

        # Add document node with tags and book type
        doc_query = AddDocument(doc_id, title, len(embedded_chunks), tags=tags, book_type=book_type)
        doc_response = self._client.query(doc_query)
        doc_internal_id = self._extract_node_id(doc_response)

        # Track chunk string IDs to internal IDs mapping
        chunk_id_map: dict[str, str] = {}

        # Add chunks with their embeddings
        for ec, meta in zip(embedded_chunks, metadata_list, strict=True):
            # Add chunk node
            chunk_props = chunk_to_helix_node(ec.chunk, meta)
            chunk_query = AddChunk(chunk_props)
            chunk_response = self._client.query(chunk_query)
            chunk_internal_id = self._extract_node_id(chunk_response)

            # Map string chunk_id to internal ID
            if chunk_internal_id:
                chunk_id_map[ec.chunk.id] = chunk_internal_id

            # Link document to chunk if we have both IDs
            if doc_internal_id and chunk_internal_id:
                link_query = LinkDocumentChunk(doc_internal_id, chunk_internal_id)
                self._client.query(link_query)

            # Store vector embedding if chunk has one
            if ec.embedding and chunk_internal_id:
                vec_query = AddChunkVector(
                    embedding=ec.embedding,
                    model_name=ec.model_name,
                    embedding_dim=ec.embedding_dim,
                )
                vec_response = self._client.query(vec_query)
                vec_internal_id = self._extract_node_id(vec_response)

                # Link chunk to its embedding vector
                if vec_internal_id:
                    link_vec_query = LinkChunkVector(chunk_internal_id, vec_internal_id)
                    self._client.query(link_vec_query)

        # Add relationship edges using internal IDs
        for ec in embedded_chunks:
            chunk = ec.chunk
            chunk_internal = chunk_id_map.get(chunk.id)

            # Parent-child relationships
            if chunk.parent_id and chunk.parent_id in chunk_id_map:
                parent_internal = chunk_id_map[chunk.parent_id]
                if parent_internal and chunk_internal:
                    edge_query = AddParentChild(parent_internal, chunk_internal)
                    self._client.query(edge_query)

            # Sequential relationships
            if chunk.next_id and chunk.next_id in chunk_id_map:
                next_internal = chunk_id_map[chunk.next_id]
                if chunk_internal and next_internal:
                    edge_query = AddNextSibling(chunk_internal, next_internal)
                    self._client.query(edge_query)

        return chunk_id_map

    def get_chunk(self, chunk_id: str) -> Optional[EmbeddedChunk]:
        """
        Retrieve a single chunk by ID with its embedding.

        Args:
            chunk_id: Unique chunk identifier

        Returns:
            EmbeddedChunk if found, None otherwise
        """
        self._ensure_connected()

        query = GetChunk(chunk_id)
        result = self._execute_query(query)

        if not result.success or not result.data:
            return None

        # Handle response format from HelixQL GetChunkByChunkId
        # Response may be {"node": {...}} or direct node dict
        node = result.data.get("node") if isinstance(result.data, dict) else result.data
        if not node:
            return None

        # Embedding may be in response or empty (HelixQL doesn't return embeddings by default)
        embedding = result.data.get("embedding", []) if isinstance(result.data, dict) else []
        model_name = result.data.get("model_name", "unknown") if isinstance(result.data, dict) else "unknown"

        return helix_to_embedded_chunk(node, embedding, model_name)

    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk and its relationships using HelixQL.

        Dropping a chunk also removes all connected edges (HasChunk,
        ParentOf, NextSibling, HasEmbedding) and cascades to vectors.

        Args:
            chunk_id: Unique chunk identifier (user-facing)

        Returns:
            True if deleted, False if not found
        """
        self._ensure_connected()

        # First, get the chunk to retrieve its internal ID
        get_query = GetChunk(chunk_id)
        get_result = self._execute_query(get_query)

        if not get_result.success or not get_result.data:
            return False

        # Extract internal ID from response
        node = get_result.data.get("node") if isinstance(get_result.data, dict) else get_result.data
        internal_id = self._extract_node_id(get_result.data) or node.get("id") if node else None

        if not internal_id:
            # Fallback: try to get internal ID from the raw response
            # Some queries return ID in the node itself
            return False

        # Drop the chunk using internal ID
        drop_query = DropChunk(internal_id)
        drop_result = self._execute_query(drop_query)

        return drop_result.success

    def update_chunk(
        self,
        chunk_id: str,
        content: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        model_name: Optional[str] = None,
    ) -> bool:
        """
        Update chunk content and/or embedding using HelixQL.

        Args:
            chunk_id: Unique chunk identifier (user-facing)
            content: New content (optional)
            embedding: New embedding (optional)
            model_name: Model name for new embedding (required if embedding provided)

        Returns:
            True if updated, False if not found

        Note:
            Embedding updates require dropping and re-adding the vector
            since HelixDB doesn't support direct vector updates.
        """
        self._ensure_connected()

        # Get chunk to verify existence and get internal ID
        get_query = GetChunk(chunk_id)
        get_result = self._execute_query(get_query)

        if not get_result.success or not get_result.data:
            return False

        # Extract internal ID from response
        node = get_result.data.get("node") if isinstance(get_result.data, dict) else get_result.data
        internal_id = self._extract_node_id(get_result.data) or (node.get("id") if node else None)

        if not internal_id:
            return False

        # Update content if provided
        if content is not None:
            word_count = len(content.split())
            update_query = UpdateChunkContent(internal_id, content, word_count)
            update_result = self._execute_query(update_query)
            if not update_result.success:
                return False

        # Update embedding if provided
        # Note: HelixDB doesn't support direct vector updates
        # We need to delete the old vector and add a new one
        if embedding is not None:
            if not model_name:
                model_name = "unknown"

            # Add new vector embedding
            vec_query = AddChunkVector(
                embedding=embedding,
                model_name=model_name,
                embedding_dim=len(embedding),
            )
            vec_response = self._client.query(vec_query)
            vec_internal_id = self._extract_node_id(vec_response)

            # Link chunk to new embedding vector
            if vec_internal_id:
                link_vec_query = LinkChunkVector(internal_id, vec_internal_id)
                self._client.query(link_vec_query)

        return True

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks using HelixQL CASCADE.

        Steps:
        1. Get the document to verify it exists and get internal ID
        2. Get all chunks for this document
        3. Drop each chunk (cascades to edges and vectors)
        4. Drop the document

        Args:
            doc_id: Unique document identifier (user-facing)

        Returns:
            True if deleted, False if not found
        """
        self._ensure_connected()

        # Step 1: Get document to verify existence and get internal ID
        doc_query = GetDocumentByDocId(doc_id)
        doc_result = self._execute_query(doc_query)

        if not doc_result.success or not doc_result.data:
            return False

        # Extract document internal ID
        doc_node = doc_result.data.get("node") if isinstance(doc_result.data, dict) else doc_result.data
        doc_internal_id = self._extract_node_id(doc_result.data) or (doc_node.get("id") if doc_node else None)

        # Step 2: Get all chunks for this document
        chunks_query = GetDocumentChunks(doc_id)
        chunks_result = self._execute_query(chunks_query)

        # Step 3: Drop each chunk (this cascades to edges and vectors)
        if chunks_result.success and chunks_result.data:
            for chunk_data in chunks_result.data:
                # Extract chunk internal ID
                chunk_node = chunk_data.get("node", chunk_data) if isinstance(chunk_data, dict) else chunk_data
                chunk_internal_id = self._extract_node_id(chunk_data) or (chunk_node.get("id") if chunk_node else None)

                if chunk_internal_id:
                    drop_chunk_query = DropChunk(chunk_internal_id)
                    self._execute_query(drop_chunk_query)

        # Step 4: Drop the document itself
        if doc_internal_id:
            drop_doc_query = DropDocument(doc_internal_id)
            drop_result = self._execute_query(drop_doc_query)
            return drop_result.success

        return False

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_id: Optional[str] = None,
        level: Optional[int] = None,
        semantic_type: Optional[str] = None,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,
        has_table: Optional[bool] = None,
        has_code: Optional[bool] = None,
        has_math: Optional[bool] = None,
        has_image: Optional[bool] = None,
        header_level: Optional[int] = None,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """
        Find chunks similar to query embedding with post-filtering.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            doc_id: Filter by document (post-filtered)
            level: Filter by hierarchy level (post-filtered)
            semantic_type: Filter by semantic type (post-filtered)
            threshold: Minimum similarity score (post-filtered)
            tags: Filter by document tags (post-filtered, AND logic)
            has_table: Filter to chunks containing tables (post-filtered)
            has_code: Filter to chunks containing code (post-filtered)
            has_math: Filter to chunks containing math (post-filtered)
            has_image: Filter to chunks containing images (post-filtered)
            header_level: Filter by header level 1-6 (post-filtered)

        Returns:
            List of (EmbeddedChunk, similarity_score) tuples, sorted by score
        """
        self._ensure_connected()

        # If filtering by tags, get matching doc_ids first
        matching_doc_ids: Optional[set[str]] = None
        if tags:
            normalized_tags = {t.lower().strip() for t in tags}
            all_docs = self.list_documents()
            matching_doc_ids = {
                d["doc_id"] for d in all_docs
                if normalized_tags.issubset(set(d.get("tags", [])))
            }
            # If no documents match tags, return empty
            if not matching_doc_ids:
                return []

        # Query requests extra results for post-filtering
        query = SearchSimilarChunks(
            query_embedding=query_embedding,
            top_k=top_k,
            doc_id=doc_id,
            level=level,
            semantic_type=semantic_type,
            threshold=threshold,
        )
        result = self._execute_query(query)

        if not result.success:
            return []

        # Post-filter results
        results: list[tuple[EmbeddedChunk, float]] = []
        for item in result.data or []:
            # Extract vector info and score from response
            # SearchSimilar returns ChunkVector nodes with scores
            if isinstance(item, dict):
                score = item.get("score", 0.0)
                vector_id = item.get("id")
                embedding = item.get("data", [])  # Vector data is in 'data' field
                model_name = item.get("model_name", "unknown")
            else:
                # Handle QueryResult or other response types
                if hasattr(item, 'data') and isinstance(item.data, dict):
                    score = item.data.get("score", 0.0)
                    vector_id = item.data.get("id")
                    embedding = item.data.get("data", [])
                    model_name = item.data.get("model_name", "unknown")
                else:
                    continue

            # Apply threshold filter early
            if threshold > 0 and score < threshold:
                continue

            # Fetch the chunk linked to this vector
            if not vector_id:
                continue

            chunk_query = GetChunkForVector(vector_id)
            chunk_result = self._execute_query(chunk_query)

            if not chunk_result.success or not chunk_result.data:
                continue

            # Handle nested response structure
            # Response could be: {"chunk": [{...}]} or {"chunk": {...}} or [{...}]
            node = chunk_result.data
            if isinstance(node, dict) and "chunk" in node:
                node = node["chunk"]
            # Now node could be a list of chunks
            if isinstance(node, list) and len(node) > 0:
                node = node[0]
            # Final check - node should be a dict now
            if not isinstance(node, dict):
                continue

            # Apply doc_id filter
            if doc_id and node.get("document_id") != doc_id:
                continue

            # Apply level filter
            if level is not None and node.get("level") != level:
                continue

            # Apply semantic_type filter
            if semantic_type and node.get("semantic_type") != semantic_type:
                continue

            # Apply tag filter
            if matching_doc_ids is not None:
                chunk_doc_id = node.get("document_id", "")
                if chunk_doc_id not in matching_doc_ids:
                    continue

            # Apply content flag filters
            if has_table is not None and bool(node.get("has_table", 0)) != has_table:
                continue
            if has_code is not None and bool(node.get("has_code", 0)) != has_code:
                continue
            if has_math is not None and bool(node.get("has_math", 0)) != has_math:
                continue
            if has_image is not None and bool(node.get("has_image", 0)) != has_image:
                continue

            # Apply header_level filter
            if header_level is not None and node.get("header_level") != header_level:
                continue

            ec = helix_to_embedded_chunk(node, embedding, model_name)
            results.append((ec, score))

        # Sort by similarity score descending and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_text(
        self,
        query_text: str,
        embedder: Any,
        top_k: int = 5,
        **filters: Any,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """
        Search by text query (embeds query first).

        Args:
            query_text: Natural language query
            embedder: Embedder instance to generate query embedding
            top_k: Number of results
            **filters: Additional filters (doc_id, level, semantic_type)

        Returns:
            List of (EmbeddedChunk, similarity_score) tuples
        """
        embedding = embedder.embed_text(query_text)
        return self.search_similar(embedding, top_k, **filters)

    def search_keyword(
        self,
        keyword: str,
        doc_id: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> list[EmbeddedChunk]:
        """
        BM25 keyword search with post-filtering.

        Args:
            keyword: Search term
            doc_id: Filter by document (post-filtered)
            case_sensitive: Whether to apply case-sensitive filter (post-filtered)
                           Note: BM25 doesn't support case-sensitive natively

        Returns:
            List of matching EmbeddedChunks
        """
        self._ensure_connected()

        query = SearchKeyword(keyword, doc_id, case_sensitive)
        result = self._execute_query(query)

        if not result.success:
            return []

        # Post-filter results
        results: list[EmbeddedChunk] = []
        for item in result.data or []:
            # HelixQL may return {"node": {...}} or direct node
            node = item.get("node", item) if isinstance(item, dict) else item

            # Apply doc_id filter
            if doc_id and node.get("document_id") != doc_id:
                continue

            # Apply case-sensitive filter in Python (BM25 doesn't support it)
            if case_sensitive:
                content = node.get("content", "")
                if keyword not in content:
                    continue

            embedding = item.get("embedding", []) if isinstance(item, dict) else []
            model_name = item.get("model_name", "unknown") if isinstance(item, dict) else "unknown"

            ec = helix_to_embedded_chunk(node, embedding, model_name)
            results.append(ec)

        return results

    def get_parent(self, chunk_id: str) -> Optional[EmbeddedChunk]:
        """
        Get parent chunk using parent_id field lookup.

        Args:
            chunk_id: Child chunk ID

        Returns:
            Parent EmbeddedChunk if exists, None otherwise
        """
        self._ensure_connected()

        # First get the chunk to find its parent_id
        chunk = self.get_chunk(chunk_id)
        if not chunk or not chunk.chunk.parent_id:
            return None

        # Then get the parent by its chunk_id
        return self.get_chunk(chunk.chunk.parent_id)

    def get_children(self, chunk_id: str) -> list[EmbeddedChunk]:
        """
        Get child chunks by querying parent_id field.

        Args:
            chunk_id: Parent chunk ID

        Returns:
            List of child EmbeddedChunks
        """
        self._ensure_connected()

        # Query for chunks where parent_id = chunk_id
        query = GetChunksByParentId(chunk_id)
        result = self._execute_query(query)

        if not result.success:
            return []

        results: list[EmbeddedChunk] = []
        for item in result.data or []:
            # HelixQL may return direct nodes or {"node": {...}}
            node = item.get("node", item) if isinstance(item, dict) else item
            embedding = item.get("embedding", []) if isinstance(item, dict) else []
            model_name = item.get("model_name", "unknown") if isinstance(item, dict) else "unknown"

            ec = helix_to_embedded_chunk(node, embedding, model_name)
            results.append(ec)

        return results

    def get_siblings(self, chunk_id: str) -> list[EmbeddedChunk]:
        """
        Get sibling chunks (same parent, excluding self).

        Args:
            chunk_id: Chunk ID

        Returns:
            List of sibling EmbeddedChunks
        """
        self._ensure_connected()

        # First get the chunk to find its parent_id
        chunk = self.get_chunk(chunk_id)
        if not chunk or not chunk.chunk.parent_id:
            return []

        # Get all children of the parent
        siblings = self.get_children(chunk.chunk.parent_id)

        # Filter out self
        return [s for s in siblings if s.chunk.id != chunk_id]

    def get_context_window(
        self,
        chunk_id: str,
        window_size: int = 1,
    ) -> list[EmbeddedChunk]:
        """
        Get chunk with surrounding context (prev/next neighbors).

        Args:
            chunk_id: Center chunk ID
            window_size: Number of chunks before and after

        Returns:
            List of EmbeddedChunks in document order
        """
        self._ensure_connected()

        result_ids: list[str] = []

        # Get the center chunk first
        center = self.get_chunk(chunk_id)
        if not center:
            return []

        # Traverse backwards
        current_id = chunk_id
        prev_ids: list[str] = []
        for _ in range(window_size):
            chunk = self.get_chunk(current_id)
            if not chunk or not chunk.chunk.prev_id:
                break
            prev_ids.insert(0, chunk.chunk.prev_id)
            current_id = chunk.chunk.prev_id

        result_ids.extend(prev_ids)
        result_ids.append(chunk_id)

        # Traverse forwards
        current_id = chunk_id
        for _ in range(window_size):
            chunk = self.get_chunk(current_id)
            if not chunk or not chunk.chunk.next_id:
                break
            result_ids.append(chunk.chunk.next_id)
            current_id = chunk.chunk.next_id

        # Fetch all chunks
        results: list[EmbeddedChunk] = []
        for cid in result_ids:
            ec = self.get_chunk(cid)
            if ec:
                results.append(ec)

        return results

    def get_document_chunks(
        self,
        doc_id: str,
        level: Optional[int] = None,
    ) -> list[EmbeddedChunk]:
        """
        Get all chunks for a document.

        Args:
            doc_id: Document identifier
            level: Filter by hierarchy level (optional)

        Returns:
            List of EmbeddedChunks for the document
        """
        self._ensure_connected()

        query = GetDocumentChunks(doc_id, level)
        result = self._execute_query(query)

        if not result.success:
            return []

        results: list[EmbeddedChunk] = []
        for item in result.data or []:
            # HelixQL may return direct nodes or {"node": {...}}
            node = item.get("node", item) if isinstance(item, dict) else item
            embedding = item.get("embedding", []) if isinstance(item, dict) else []
            model_name = item.get("model_name", "unknown") if isinstance(item, dict) else "unknown"

            ec = helix_to_embedded_chunk(node, embedding, model_name)
            results.append(ec)

        return results

    def list_documents(self, tags: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """
        List all documents with metadata.

        Args:
            tags: Filter by document tags (optional, AND logic)

        Returns:
            List of document metadata dictionaries
        """
        self._ensure_connected()

        query = ListDocuments()
        result = self._execute_query(query)

        if not result.success:
            return []

        # Convert raw nodes to document dicts using converter
        from .converters import helix_node_to_document

        docs: list[dict[str, Any]] = []
        for item in result.data or []:
            # HelixQL returns direct nodes
            node = item if isinstance(item, dict) else {}
            if node:
                docs.append(helix_node_to_document(node))

        # Filter by tags if specified
        if tags:
            normalized_tags = {t.lower().strip() for t in tags}
            docs = [
                d for d in docs
                if normalized_tags.issubset(set(d.get("tags", [])))
            ]

        return docs

    def get_document_stats(self, doc_id: str) -> Optional[dict[str, Any]]:
        """
        Get statistics for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Dictionary with document statistics, or None if not found
        """
        self._ensure_connected()

        chunks = self.get_document_chunks(doc_id)
        if not chunks:
            return None

        # Get document metadata
        docs = self.list_documents()
        doc_meta = next((d for d in docs if d.get("doc_id") == doc_id), {})

        level_counts = {0: 0, 1: 0, 2: 0}
        type_counts: dict[str, int] = {}
        total_tokens = 0

        for ec in chunks:
            chunk = ec.chunk
            level_counts[chunk.level] = level_counts.get(chunk.level, 0) + 1
            total_tokens += chunk.token_count
            # Get semantic_type from chunk metadata if available
            semantic_type = chunk.metadata.get("semantic_type", "unknown")
            type_counts[semantic_type] = type_counts.get(semantic_type, 0) + 1

        return {
            "doc_id": doc_id,
            "title": doc_meta.get("title", ""),
            "total_chunks": len(chunks),
            "level_counts": level_counts,
            "type_counts": type_counts,
            "total_tokens": total_tokens,
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._client:
            # helix-py may have a close method
            if hasattr(self._client, "close"):
                self._client.close()
            self._client = None
            self._initialized = False

    def __enter__(self) -> "HelixChunkStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    # ========================================================================
    # Concept Storage Methods
    # ========================================================================

    def store_concept(
        self,
        concept_id: str,
        name: str,
        definition: str,
        concept_type: str,
        source_documents: list[str],
        aliases: list[str],
    ) -> Optional[str]:
        """
        Store a concept and return its internal ID.

        Args:
            concept_id: Slugified unique ID (e.g., "cost-of-goods-sold")
            name: Display name (e.g., "Cost of Goods Sold")
            definition: 1-2 sentence definition
            concept_type: Type (term, method, principle, formula, account)
            source_documents: List of document IDs where concept is defined
            aliases: Alternative names or abbreviations

        Returns:
            Internal concept node ID, or None if storage fails
        """
        self._ensure_connected()

        try:
            query = AddConcept(
                concept_id=concept_id,
                name=name,
                definition=definition,
                concept_type=concept_type,
                source_documents=source_documents,
                aliases=aliases,
            )
            response = self._client.query(query)
            return self._extract_node_id(response)
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Network error storing concept '{concept_id}': {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error storing concept '{concept_id}': {e}")
            raise

    def update_concept(
        self,
        concept_id: str,
        source_documents: list[str],
        aliases: list[str],
        definition: Optional[str] = None,
    ) -> bool:
        """
        Update an existing concept's source_documents and aliases.

        Args:
            concept_id: Slugified concept identifier
            source_documents: Updated list of document IDs
            aliases: Updated list of aliases
            definition: Optional updated definition (None to keep existing)

        Returns:
            True if update succeeded, False otherwise
        """
        self._ensure_connected()

        try:
            query = UpdateConcept(
                concept_id=concept_id,
                source_documents=source_documents,
                aliases=aliases,
                definition=definition,
            )
            response = self._client.query(query)
            result = query.response(response)
            return result.success
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Network error updating concept '{concept_id}': {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error updating concept '{concept_id}': {e}")
            raise

    def store_or_merge_concept(
        self,
        concept_id: str,
        name: str,
        definition: str,
        concept_type: str,
        source_documents: list[str],
        aliases: list[str],
    ) -> Optional[str]:
        """
        Store a new concept or merge with existing one.

        If concept with same name exists, merges source_documents and aliases.
        Otherwise creates a new concept.

        Args:
            concept_id: Slugified unique ID
            name: Display name
            definition: 1-2 sentence definition
            concept_type: Type (term, method, principle, formula, account)
            source_documents: List of document IDs
            aliases: Alternative names

        Returns:
            Internal concept node ID, or None if operation fails
        """
        # Check if concept already exists
        existing = self.get_concept_by_name(name)

        if existing is None:
            # Create new concept
            return self.store_concept(
                concept_id=concept_id,
                name=name,
                definition=definition,
                concept_type=concept_type,
                source_documents=source_documents,
                aliases=aliases,
            )

        # Merge with existing concept
        # Parse existing source_documents and aliases
        existing_docs_raw = existing.get("source_documents", "[]")
        existing_aliases_raw = existing.get("aliases", "[]")

        try:
            existing_docs = json.loads(existing_docs_raw) if isinstance(existing_docs_raw, str) else existing_docs_raw
        except json.JSONDecodeError:
            existing_docs = []

        try:
            existing_aliases = json.loads(existing_aliases_raw) if isinstance(existing_aliases_raw, str) else existing_aliases_raw
        except json.JSONDecodeError:
            existing_aliases = []

        # Merge and deduplicate source_documents
        merged_docs = list(set(existing_docs + source_documents))

        # Merge and deduplicate aliases (case-insensitive)
        existing_aliases_lower = {a.lower(): a for a in existing_aliases}
        for alias in aliases:
            if alias.lower() not in existing_aliases_lower:
                existing_aliases_lower[alias.lower()] = alias
        merged_aliases = list(existing_aliases_lower.values())

        # Update the concept (fill in definition if existing is empty)
        existing_definition = existing.get("definition", "")
        merge_definition = definition if definition and not existing_definition.strip() else None

        success = self.update_concept(
            concept_id=existing.get("concept_id", concept_id),
            source_documents=merged_docs,
            aliases=merged_aliases,
            definition=merge_definition,
        )

        if success:
            # Return existing internal ID using same pattern as other methods
            internal_id = self._extract_node_id(existing) or existing.get("id")
            if internal_id:
                return internal_id
            # If we still can't get internal ID, log warning and return None
            logger.warning(
                f"Could not extract internal ID for merged concept '{name}'. "
                "This may cause issues with relationship linking."
            )

        return None

    def delete_concept(self, concept_id: str) -> bool:
        """
        Delete a concept by its user-facing concept_id.

        Fetches the concept, extracts internal node ID, and drops it.
        Edge deletion (DefinesConcept, MentionsConcept, RelatesTo) cascades
        via the HelixQL DropConcept query.

        Args:
            concept_id: Slugified concept identifier

        Returns:
            True if deleted, False otherwise
        """
        self._ensure_connected()

        concept = self.get_concept(concept_id)
        if concept is None:
            logger.warning(f"Concept '{concept_id}' not found for deletion")
            return False

        internal_id = self._extract_node_id(concept) or concept.get("id")
        if not internal_id:
            logger.error(f"Could not extract internal ID for concept '{concept_id}'")
            return False

        return self.drop_concept_by_internal_id(internal_id)

    def drop_concept_by_internal_id(self, internal_id: str) -> bool:
        """
        Drop a concept node by its internal Helix node ID (single HTTP call).

        Use this when you already have the internal ID (e.g., from list_concepts)
        to avoid the extra lookup that delete_concept() performs.

        Args:
            internal_id: Internal Helix node ID

        Returns:
            True if dropped, False otherwise
        """
        self._ensure_connected()

        try:
            query = DropConcept(internal_id)
            self._client.query(query)
            return True
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Network error dropping concept '{internal_id}': {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error dropping concept '{internal_id}': {e}")
            raise

    def get_concept(self, concept_id: str) -> Optional[dict]:
        """
        Get concept by concept_id.

        Args:
            concept_id: Slugified concept identifier

        Returns:
            Concept dictionary if found, None otherwise
        """
        self._ensure_connected()

        query = GetConceptById(concept_id)
        result = self._execute_query(query)

        if result.success and result.data:
            return result.data.get("node")
        return None

    def get_concept_by_name(self, name: str) -> Optional[dict]:
        """
        Get concept by name.

        Args:
            name: Concept display name

        Returns:
            Concept dictionary if found, None otherwise
        """
        self._ensure_connected()

        query = GetConceptByName(name)
        result = self._execute_query(query)

        if result.success and result.data:
            return result.data.get("node")
        return None

    def list_concepts(self, doc_id: Optional[str] = None) -> list[dict]:
        """
        List all concepts, optionally filtered by document.

        Args:
            doc_id: Filter to concepts defined in this document (optional)

        Returns:
            List of concept dictionaries
        """
        self._ensure_connected()

        if doc_id:
            # Try edge traversal first (requires DefinesConcept edges)
            query = ListDocumentConcepts(doc_id)
            result = self._execute_query(query)
            if result.success and result.data:
                return result.data

            # Fallback: filter all concepts by source_documents field
            all_query = ListConcepts()
            all_result = self._execute_query(all_query)
            if all_result.success and all_result.data:
                return [
                    c for c in all_result.data
                    if doc_id in (c.get("source_documents") or "")
                ]
            return []
        else:
            query = ListConcepts()

        result = self._execute_query(query)

        if result.success:
            return result.data or []
        return []

    def get_chunk_internal_id(self, chunk_id: str) -> Optional[str]:
        """Get internal Helix node ID for a user-facing chunk_id.

        Args:
            chunk_id: User-facing chunk identifier (e.g., "doc_l1_5")

        Returns:
            Internal node ID string, or None if not found
        """
        self._ensure_connected()

        query = GetChunk(chunk_id)
        result = self._execute_query(query)

        if not result.success or not result.data:
            return None

        node = result.data.get("node") if isinstance(result.data, dict) else result.data
        if isinstance(node, dict):
            node_id = node.get("id")
            if node_id:
                return str(node_id)

        return self._extract_node_id(result.data)

    def build_chunk_id_cache(self, doc_id: str) -> dict[str, str]:
        """Build chunk_id -> internal_id mapping with a single query.

        Instead of calling get_chunk_internal_id() per chunk (1 HTTP call each),
        this fetches all chunks for a document in one query and builds a lookup
        dict. For a 660-chunk document, this replaces ~660 HTTP calls with 1.

        Args:
            doc_id: Document identifier

        Returns:
            Dictionary mapping user-facing chunk_id to internal Helix node ID
        """
        self._ensure_connected()

        query = GetDocumentChunks(doc_id)
        result = self._execute_query(query)

        cache: dict[str, str] = {}
        if not result.success:
            return cache

        for item in result.data or []:
            node = item.get("node", item) if isinstance(item, dict) else item
            if isinstance(node, dict):
                chunk_id = node.get("chunk_id")
                internal_id = node.get("id")
                if chunk_id and internal_id:
                    cache[str(chunk_id)] = str(internal_id)

        return cache

    def link_chunk_defines_concept(
        self,
        chunk_internal_id: str,
        concept_internal_id: str,
    ) -> bool:
        """
        Create DefinesConcept edge from chunk to concept.

        Args:
            chunk_internal_id: Internal chunk node ID
            concept_internal_id: Internal concept node ID

        Returns:
            True if link created successfully
        """
        self._ensure_connected()

        query = LinkChunkDefinesConcept(chunk_internal_id, concept_internal_id)
        result = self._execute_query(query)
        return result.success

    def batch_link_chunk_defines_concept(
        self,
        edges: list[tuple[str, str]],
    ) -> int:
        """Batch create DefinesConcept edges in a single HTTP request.

        Args:
            edges: List of (chunk_internal_id, concept_internal_id) tuples

        Returns:
            Number of edges created (len of input on success, 0 on failure)
        """
        self._ensure_connected()
        edge_dicts = [{"chunk_id": c, "concept_id": p} for c, p in edges]
        query = BatchLinkChunkDefinesConcept(edge_dicts)
        result = self._execute_query(query)
        return len(edges) if result.success else 0

    def batch_drop_chunk_concept_edges(
        self,
        chunk_internal_ids: list[str],
    ) -> bool:
        """Batch drop all DefinesConcept edges from chunks.

        Args:
            chunk_internal_ids: List of internal chunk node IDs

        Returns:
            True if edges dropped successfully
        """
        self._ensure_connected()
        BATCH_SIZE = 50
        for i in range(0, len(chunk_internal_ids), BATCH_SIZE):
            batch = chunk_internal_ids[i:i + BATCH_SIZE]
            query = BatchDropChunkConceptEdges(batch)
            result = self._execute_query(query)
            if not result.success:
                return False
            time.sleep(0.08)
        return True

    def link_chunk_mentions_concept(
        self,
        chunk_internal_id: str,
        concept_internal_id: str,
    ) -> bool:
        """
        Create MentionsConcept edge from chunk to concept.

        Args:
            chunk_internal_id: Internal chunk node ID
            concept_internal_id: Internal concept node ID

        Returns:
            True if link created successfully
        """
        self._ensure_connected()

        query = LinkChunkMentionsConcept(chunk_internal_id, concept_internal_id)
        result = self._execute_query(query)
        return result.success

    def link_concept_relates_to(
        self,
        from_concept_id: str,
        to_concept_id: str,
        relation_type: str = "relates_to",
    ) -> bool:
        """
        Create RelatesTo edge between concepts with a relation type.

        Args:
            from_concept_id: Internal source concept node ID
            to_concept_id: Internal target concept node ID
            relation_type: Type of relationship (uses, requires, calculated_from,
                          component_of, recorded_in, supersedes, relates_to)

        Returns:
            True if link created successfully
        """
        self._ensure_connected()

        query = LinkConceptRelatesTo(from_concept_id, to_concept_id, relation_type)
        result = self._execute_query(query)
        return result.success

    def get_concept_definition_chunks(self, concept_id: str) -> list[dict]:
        """
        Get chunks that define a concept (for citations).

        Args:
            concept_id: Slugified concept identifier

        Returns:
            List of chunk dictionaries that define the concept
        """
        self._ensure_connected()

        # First get the concept to find its internal ID
        concept = self.get_concept(concept_id)
        if not concept:
            return []

        internal_id = self._extract_node_id(concept) or concept.get("id")
        if not internal_id:
            return []

        query = GetConceptDefinitionChunks(internal_id)
        result = self._execute_query(query)

        if result.success:
            return result.data or []
        return []

    def get_related_concepts(self, concept_id: str) -> list[dict]:
        """
        Get concepts related to a given concept.

        Args:
            concept_id: Slugified concept identifier

        Returns:
            List of related concept dictionaries
        """
        self._ensure_connected()

        # First get the concept to find its internal ID
        concept = self.get_concept(concept_id)
        if not concept:
            return []

        internal_id = self._extract_node_id(concept) or concept.get("id")
        if not internal_id:
            return []

        query = GetRelatedConcepts(internal_id)
        result = self._execute_query(query)

        if result.success:
            return result.data or []
        return []

    def get_related_concepts_with_types(self, concept_id: str) -> list[dict]:
        """
        Get concepts related to a given concept with relationship types.

        Queries all typed edge types (Uses, Requires, CalculatedFrom, etc.)
        and combines results with type annotations.

        Args:
            concept_id: Slugified concept identifier

        Returns:
            List of dicts with 'concept' (target concept data) and 'relation_type'
        """
        self._ensure_connected()

        # First get the concept to find its internal ID
        concept = self.get_concept(concept_id)
        if not concept:
            return []

        internal_id = self._extract_node_id(concept) or concept.get("id")
        if not internal_id:
            return []

        # Query all typed edges and combine results
        results = []
        for relation_type, query_class in TYPED_CONCEPT_QUERIES_FORWARD.items():
            query = query_class(internal_id)
            result = self._execute_query(query)
            if result.success and result.data:
                for target_concept in result.data:
                    results.append({
                        "concept": target_concept,
                        "relation_type": relation_type,
                    })

        return results

    def count_concept_edges_direct(self, internal_id: str) -> int:
        """Count outgoing typed edges for a concept using its internal Helix ID.

        Queries all typed edge types (Uses, Requires, etc.) and returns the
        total count. Skips the get_concept() lookup since the caller already
        has the internal node ID (e.g. from list_concepts()).

        Args:
            internal_id: Internal Helix node ID

        Returns:
            Total number of outgoing typed edges
        """
        self._ensure_connected()
        count = 0
        for query_class in TYPED_CONCEPT_QUERIES_FORWARD.values():
            result = self._execute_query(query_class(internal_id))
            if result.success and result.data:
                count += len(result.data)
        return count

    def get_related_concepts_with_types_direct(self, internal_id: str) -> list[dict]:
        """Get related concepts with types using internal Helix ID directly.

        Same as get_related_concepts_with_types() but skips the get_concept()
        lookup. Use when the internal node ID is already known (e.g. from
        list_concepts()).

        Args:
            internal_id: Internal Helix node ID

        Returns:
            List of dicts with 'concept' (target concept data) and 'relation_type'
        """
        self._ensure_connected()
        results = []
        for relation_type, query_class in TYPED_CONCEPT_QUERIES_FORWARD.items():
            result = self._execute_query(query_class(internal_id))
            if result.success and result.data:
                for target_concept in result.data:
                    results.append({
                        "concept": target_concept,
                        "relation_type": relation_type,
                    })
        return results

    def get_concept_dependents(self, concept_id: str) -> list[dict]:
        """
        Get concepts that relate TO this concept (reverse relationships).

        Args:
            concept_id: Slugified concept identifier

        Returns:
            List of dependent concept dictionaries
        """
        self._ensure_connected()

        # First get the concept to find its internal ID
        concept = self.get_concept(concept_id)
        if not concept:
            return []

        internal_id = self._extract_node_id(concept) or concept.get("id")
        if not internal_id:
            return []

        query = GetConceptDependents(internal_id)
        result = self._execute_query(query)

        if result.success:
            return result.data or []
        return []

    # =========================================================================
    # Summary Methods (Document Summarization Feature)
    # =========================================================================

    def store_summary(
        self,
        summary: Any,
        doc_internal_id: Optional[str] = None,
        chunk_internal_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store a summary and return its internal ID.

        Args:
            summary: Summary dataclass with summary_id, content, level, etc.
            doc_internal_id: Pre-cached document internal ID (skips GetDocumentByDocId lookup)
            chunk_internal_id: Pre-cached chunk internal ID (skips GetChunk lookup)

        Returns:
            Internal summary node ID, or None if storage fails
        """
        from src.storage.queries import (
            AddSummary,
            AddSummaryVector,
            LinkSummaryVector,
            LinkDocumentSummary,
            LinkChunkSummary,
            GetDocumentByDocId,
            GetChunk,
        )

        self._ensure_connected()

        try:
            # Prepare summary properties
            summary_props = {
                "summary_id": summary.summary_id,
                "content": summary.content,
                "level": summary.level,
                "source_id": summary.source_id,
                "document_id": summary.document_id,
                "parent_summary_id": summary.parent_summary_id or "",
                "key_points": json.dumps(summary.key_points),
                "word_count": summary.word_count,
                "created_at": summary.created_at,
            }

            # Add summary node
            add_query = AddSummary(summary_props)
            response = self._client.query(add_query)
            summary_internal_id = self._extract_node_id(response)

            if not summary_internal_id:
                logger.warning(f"Failed to get internal ID for summary {summary.summary_id}")
                return None

            # Add embedding if present
            if summary.embedding:
                vector_query = AddSummaryVector(
                    embedding=summary.embedding,
                    model_name="unknown",
                    embedding_dim=len(summary.embedding),
                )
                vector_response = self._client.query(vector_query)
                vector_id = self._extract_node_id(vector_response)

                if vector_id:
                    link_query = LinkSummaryVector(summary_internal_id, vector_id)
                    self._client.query(link_query)

            # Link to document
            if summary.level == "document":
                _doc_id = doc_internal_id
                if not _doc_id:
                    doc_query = GetDocumentByDocId(summary.document_id)
                    doc_result = self._execute_query(doc_query)
                    if doc_result.success and doc_result.data:
                        doc_node = doc_result.data.get("node", doc_result.data)
                        _doc_id = self._extract_node_id(doc_result.data) or (doc_node.get("id") if doc_node else None)
                if _doc_id:
                    link_query = LinkDocumentSummary(_doc_id, summary_internal_id)
                    self._client.query(link_query)

            # Link to chunk for chapter/section summaries
            elif summary.level in ("chapter", "section"):
                _chunk_id = chunk_internal_id
                if not _chunk_id:
                    chunk_query = GetChunk(summary.source_id)
                    chunk_result = self._execute_query(chunk_query)
                    if chunk_result.success and chunk_result.data:
                        chunk_node = chunk_result.data.get("node", chunk_result.data)
                        _chunk_id = self._extract_node_id(chunk_result.data) or (chunk_node.get("id") if chunk_node else None)
                if _chunk_id:
                    link_query = LinkChunkSummary(_chunk_id, summary_internal_id)
                    self._client.query(link_query)

            return summary_internal_id

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("Network error storing summary '%s': %s", summary.summary_id, e)
            raise
        except Exception:
            logger.exception("Unexpected error storing summary '%s'", summary.summary_id)
            raise

    def get_summary(self, summary_id: str) -> Optional[dict[str, Any]]:
        """
        Get a summary by summary_id.

        Args:
            summary_id: Summary identifier

        Returns:
            Summary dictionary if found, None otherwise
        """
        from src.storage.queries import GetSummary

        self._ensure_connected()

        query = GetSummary(summary_id)
        result = self._execute_query(query)

        if result.success and result.data:
            return result.data.get("node")
        return None

    def get_document_summaries(
        self,
        document_id: str,
        level: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Get all summaries for a document.

        Args:
            document_id: Document identifier
            level: Optional filter by level ("document", "chapter", "section")

        Returns:
            List of summary dictionaries
        """
        from src.storage.queries import GetDocumentSummaries

        self._ensure_connected()

        query = GetDocumentSummaries(document_id, level=level)
        result = self._execute_query(query)

        if not result.success:
            return []

        # Filter out empty/phantom results from HelixQL
        return [
            s for s in (result.data or [])
            if isinstance(s, dict) and s.get("summary_id")
        ]

    def search_summaries(
        self,
        query_embedding: list[float],
        limit: int = 10,
        level: Optional[str] = None,
        document_id: Optional[str] = None,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Search summaries using vector similarity.

        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            level: Optional filter by level
            document_id: Optional filter by document
            threshold: Minimum similarity score (0.0 to disable)

        Returns:
            List of summary dictionaries with scores
        """
        from src.storage.queries import SearchSimilarSummaries

        self._ensure_connected()

        # Request more results than limit to allow for post-filtering
        fetch_limit = limit * 3 if (level or document_id) else limit

        query = SearchSimilarSummaries(
            query_embedding=query_embedding,
            limit=fetch_limit,
            level=level,
            document_id=document_id,
        )
        result = self._execute_query(query)

        if not result.success:
            return []

        # Vector search returns SummaryVector nodes with scores.
        # We need to fetch the actual Summary node for each via edge traversal.
        results = []
        for item in result.data or []:
            if isinstance(item, dict):
                score = item.get("score", 0.0)
                vector_id = item.get("id")
            else:
                continue

            # Apply threshold filter early
            if threshold > 0 and score < threshold:
                continue

            if not vector_id:
                continue

            # Fetch the summary linked to this vector
            summary_query = GetSummaryForVector(vector_id)
            summary_result = self._execute_query(summary_query)

            if not summary_result.success or not summary_result.data:
                continue

            node = summary_result.data
            if not isinstance(node, dict):
                continue

            # Post-filter by level and document_id
            if level and node.get("level") != level:
                continue
            if document_id and node.get("document_id") != document_id:
                continue

            # Merge score into summary data
            node["score"] = score
            results.append(node)

            if len(results) >= limit:
                break

        return results

    def delete_document_summaries(self, document_id: str) -> int:
        """
        Delete all summaries for a document, including linked SummaryVector nodes.

        For each summary: drop embedding edge -> drop vector -> drop summary.
        This prevents orphaned SummaryVector nodes that pollute the vector index.

        Args:
            document_id: Document identifier

        Returns:
            Number of summaries deleted
        """
        from src.storage.queries import (
            DeleteDocumentSummaries,
            DeleteSummary,
            DropSummaryEmbeddingEdge,
            GetSummaryVector,
            DropSummaryVector,
        )

        self._ensure_connected()

        query = DeleteDocumentSummaries(document_id)
        result = self._execute_query(query)

        if not result.success:
            return 0

        deleted = 0
        for summary in result.data or []:
            if not isinstance(summary, dict):
                continue
            internal_id = summary.get("id")
            if not internal_id:
                continue

            try:
                # Step 1: Get linked SummaryVector
                vec_query = GetSummaryVector(internal_id)
                vec_result = self._execute_query(vec_query)
                vector_id = None
                if vec_result.success and vec_result.data:
                    vector_id = self._extract_node_id(vec_result.data)

                # Step 2: Drop embedding edge
                try:
                    edge_query = DropSummaryEmbeddingEdge(internal_id)
                    self._client.query(edge_query)
                except Exception as e:
                    logger.debug("Failed to drop embedding edge for summary %s: %s", internal_id, e)

                # Step 3: Drop the SummaryVector node
                if vector_id:
                    try:
                        drop_vec_query = DropSummaryVector(vector_id)
                        self._client.query(drop_vec_query)
                    except Exception as e:
                        logger.debug("Failed to drop vector %s: %s", vector_id, e)

                # Step 4: Drop the summary node
                drop_query = DeleteSummary(internal_id)
                self._client.query(drop_query)
                deleted += 1

                # Throttle to prevent Helix-DB thread explosion
                time.sleep(0.08)

            except Exception as e:
                logger.warning("Failed to delete summary %s: %s", internal_id, e)

        return deleted

    def delete_summary_by_id(self, summary_id: str) -> bool:
        """
        Delete a single summary by summary_id, including linked SummaryVector.

        Steps: GetSummary → GetSummaryVector → DropSummaryEmbeddingEdge
               → DropSummaryVector → DeleteSummary.

        Args:
            summary_id: Summary identifier (e.g., "summary_inv_document")

        Returns:
            True if deleted, False if not found
        """
        from src.storage.queries import (
            GetSummary,
            GetSummaryVector,
            DropSummaryEmbeddingEdge,
            DropSummaryVector,
            DeleteSummary,
        )

        self._ensure_connected()

        # Step 1: Get the summary to find its internal ID
        query = GetSummary(summary_id)
        result = self._execute_query(query)

        if not result.success or not result.data:
            return False

        node = result.data.get("node", result.data) if isinstance(result.data, dict) else result.data
        internal_id = self._extract_node_id(result.data) or (node.get("id") if isinstance(node, dict) else None)
        if not internal_id:
            return False

        try:
            # Step 2: Get linked SummaryVector
            vec_query = GetSummaryVector(internal_id)
            vec_result = self._execute_query(vec_query)
            vector_id = None
            if vec_result.success and vec_result.data:
                vector_id = self._extract_node_id(vec_result.data)

            # Step 3: Drop embedding edge
            try:
                edge_query = DropSummaryEmbeddingEdge(internal_id)
                self._client.query(edge_query)
            except Exception as e:
                logger.debug("Failed to drop embedding edge for summary %s: %s", internal_id, e)

            # Step 4: Drop the SummaryVector node
            if vector_id:
                try:
                    drop_vec_query = DropSummaryVector(vector_id)
                    self._client.query(drop_vec_query)
                except Exception as e:
                    logger.debug("Failed to drop vector %s: %s", vector_id, e)

            # Step 5: Drop the summary node
            drop_query = DeleteSummary(internal_id)
            self._client.query(drop_query)
            return True

        except Exception as e:
            logger.warning("Failed to delete summary %s: %s", summary_id, e)
            return False

    def list_summarized_documents(self) -> list[dict[str, Any]]:
        """
        List all documents that have summaries.

        Returns:
            List of document info dictionaries with doc_id, title, and summary_count
        """
        from src.storage.queries import ListSummarizedDocuments

        self._ensure_connected()

        query = ListSummarizedDocuments()
        result = self._execute_query(query)

        if not result.success:
            return []

        summaries = result.data or []
        if not summaries:
            return []

        # Aggregate summaries by document_id
        doc_counts: dict[str, int] = {}
        for s in summaries:
            doc_id = s.get("document_id", "")
            if doc_id:
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        # Get document titles
        docs = self.list_documents()
        doc_titles = {d.get("doc_id", ""): d.get("title", "") for d in docs}

        # Build result with doc_id, title, summary_count
        return [
            {
                "doc_id": doc_id,
                "title": doc_titles.get(doc_id, doc_id),
                "summary_count": count,
            }
            for doc_id, count in doc_counts.items()
        ]

    # ========================================================================
    # SKILL STORAGE (procedural skill extraction)
    # ========================================================================

    def store_skill(
        self,
        skill_props: dict,
        steps: list[dict],
        embedding: Optional[list[float]] = None,
        embedding_model: str = "unknown",
        doc_internal_id: Optional[str] = None,
    ) -> Optional[str]:
        """Store a skill with its steps and optional embedding.

        Args:
            skill_props: Skill node properties (skill_id, name, goal, ...)
            steps: List of step dicts (step_id, step_number, action, ...)
            embedding: Optional embedding vector for the skill
            embedding_model: Name of the embedding model
            doc_internal_id: Pre-cached document internal ID

        Returns:
            Internal skill node ID, or None on failure.
        """
        from src.storage.queries import (
            AddSkill,
            AddSkillStep,
            AddSkillVector,
            GetDocumentByDocId,
            LinkHasStep,
            LinkSkillDefinedIn,
            LinkSkillVector,
        )

        self._ensure_connected()

        try:
            add_query = AddSkill(skill_props)
            response = self._client.query(add_query)
            skill_internal_id = self._extract_node_id(response)
            if not skill_internal_id:
                logger.warning("Failed to get internal ID for skill %r", skill_props.get("skill_id"))
                return None

            # Store steps, link via HasStep. Any failure aborts — we don't
            # want a skill persisted with missing steps.
            for step in steps:
                step_query = AddSkillStep(step)
                step_response = self._client.query(step_query)
                step_internal_id = self._extract_node_id(step_response)
                if not step_internal_id:
                    raise RuntimeError(
                        f"Failed to persist step {step.get('step_id')!r} "
                        f"for skill {skill_props.get('skill_id')!r}"
                    )
                link_query = LinkHasStep(skill_internal_id, step_internal_id)
                link_result = self._execute_query(link_query)
                if not link_result.success:
                    raise RuntimeError(
                        f"Failed to link step {step.get('step_id')!r} "
                        f"to skill {skill_props.get('skill_id')!r}"
                    )

            # Link to source document
            _doc_id = doc_internal_id
            doc_string_id = skill_props.get("source_document_id")
            if not _doc_id and doc_string_id:
                doc_query = GetDocumentByDocId(doc_string_id)
                doc_result = self._execute_query(doc_query)
                if doc_result.success and doc_result.data:
                    doc_node = doc_result.data.get("node", doc_result.data)
                    _doc_id = (
                        self._extract_node_id(doc_result.data)
                        or (doc_node.get("id") if doc_node else None)
                    )
            if _doc_id:
                link_query = LinkSkillDefinedIn(skill_internal_id, _doc_id)
                self._client.query(link_query)

            # Add embedding if provided
            if embedding:
                vec_query = AddSkillVector(
                    embedding=embedding,
                    model_name=embedding_model,
                    embedding_dim=len(embedding),
                )
                vec_response = self._client.query(vec_query)
                vec_id = self._extract_node_id(vec_response)
                if vec_id:
                    link_query = LinkSkillVector(skill_internal_id, vec_id)
                    self._client.query(link_query)

            return skill_internal_id

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("Network error storing skill %r: %s", skill_props.get("skill_id"), e)
            raise
        except Exception:
            logger.exception("Unexpected error storing skill %r", skill_props.get("skill_id"))
            raise

    def link_skill_requires_concept(self, skill_internal_id: str, concept_internal_id: str) -> bool:
        """Link a skill to a prerequisite concept."""
        from src.storage.queries import LinkRequiresConcept

        self._ensure_connected()
        result = self._execute_query(LinkRequiresConcept(skill_internal_id, concept_internal_id))
        return result.success

    def link_skill_produces_concept(self, skill_internal_id: str, concept_internal_id: str) -> bool:
        """Link a skill to a produced concept."""
        from src.storage.queries import LinkProducesConcept

        self._ensure_connected()
        result = self._execute_query(LinkProducesConcept(skill_internal_id, concept_internal_id))
        return result.success

    def link_skill_requires_skill(self, skill_internal_id: str, prereq_internal_id: str) -> bool:
        """Link a skill to a prerequisite skill."""
        from src.storage.queries import LinkRequiresSkill

        self._ensure_connected()
        result = self._execute_query(LinkRequiresSkill(skill_internal_id, prereq_internal_id))
        return result.success

    def get_skill(self, skill_id: str) -> Optional[dict]:
        """Fetch a skill by skill_id."""
        from src.storage.queries import GetSkillById

        self._ensure_connected()
        result = self._execute_query(GetSkillById(skill_id))
        if result.success and result.data:
            return result.data.get("node")
        return None

    def get_skill_by_name(self, name: str) -> Optional[dict]:
        """Fetch a skill by display name."""
        from src.storage.queries import GetSkillByName

        self._ensure_connected()
        result = self._execute_query(GetSkillByName(name))
        if result.success and result.data:
            return result.data.get("node")
        return None

    def get_skill_steps(self, skill_id: str) -> list[dict]:
        """Get all steps for a skill, sorted by step_number."""
        from src.storage.queries import GetSkillSteps

        self._ensure_connected()
        result = self._execute_query(GetSkillSteps(skill_id))
        if not result.success:
            return []
        steps = [s for s in (result.data or []) if isinstance(s, dict) and s.get("step_id")]
        steps.sort(key=lambda s: s.get("step_number", 0))
        return steps

    def get_skill_prerequisite_concepts(self, skill_internal_id: str) -> list[dict]:
        """Get concepts the skill requires (prerequisites)."""
        from src.storage.queries import GetSkillPrerequisiteConcepts

        self._ensure_connected()
        result = self._execute_query(GetSkillPrerequisiteConcepts(skill_internal_id))
        return result.data or [] if result.success else []

    def get_skill_produced_concepts(self, skill_internal_id: str) -> list[dict]:
        """Get concepts the skill produces when executed."""
        from src.storage.queries import GetSkillProducedConcepts

        self._ensure_connected()
        result = self._execute_query(GetSkillProducedConcepts(skill_internal_id))
        return result.data or [] if result.success else []

    def get_skill_prerequisite_skills(self, skill_internal_id: str) -> list[dict]:
        """Get skills the given skill requires first."""
        from src.storage.queries import GetSkillPrerequisiteSkills

        self._ensure_connected()
        result = self._execute_query(GetSkillPrerequisiteSkills(skill_internal_id))
        return result.data or [] if result.success else []

    def list_skills(self, doc_id: Optional[str] = None) -> list[dict]:
        """List all skills, optionally filtered by source document."""
        from src.storage.queries import ListAllSkills, ListDocumentSkills

        self._ensure_connected()
        if doc_id:
            result = self._execute_query(ListDocumentSkills(doc_id))
        else:
            result = self._execute_query(ListAllSkills())
        if not result.success:
            return []
        return [
            s for s in (result.data or [])
            if isinstance(s, dict) and s.get("skill_id")
        ]

    def search_skills(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        doc_id: Optional[str] = None,
    ) -> list[tuple[dict, float]]:
        """Vector search over skills. Returns list of (skill_node, score).

        Post-filters by doc_id when provided.
        """
        from src.storage.queries import GetSkillForVector, SearchSimilarSkills

        self._ensure_connected()
        vec_result = self._execute_query(
            SearchSimilarSkills(query_embedding=query_embedding, limit=top_k * 3 if doc_id else top_k)
        )
        if not vec_result.success:
            return []

        out: list[tuple[dict, float]] = []
        for item in vec_result.data or []:
            # Each result is a vector node; score may live under different keys
            node = item if isinstance(item, dict) else {}
            score = float(node.get("score", node.get("distance", 0.0)) or 0.0)
            vector_id = node.get("id") or node.get("vector_id") or ""
            if not vector_id:
                continue
            skill_result = self._execute_query(GetSkillForVector(vector_id))
            if not (skill_result.success and skill_result.data):
                continue
            skill_node = skill_result.data
            if isinstance(skill_node, list):
                skill_node = skill_node[0] if skill_node else None
            if not isinstance(skill_node, dict):
                continue
            if doc_id and skill_node.get("source_document_id") != doc_id:
                continue
            out.append((skill_node, score))
            if len(out) >= top_k:
                break
        return out
