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
    GetChunksByParentId,
    GetChunkWithContext,
    GetConceptById,
    GetConceptByName,
    GetConceptDefinitionChunks,
    GetDocumentByDocId,
    GetDocumentChunks,
    GetRelatedConcepts,
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
)


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
                return data.get('id') or data.get('_id') or data.get('node_id')
            if hasattr(data, 'id'):
                return str(data.id)
        if isinstance(response, dict):
            return response.get('id') or response.get('_id') or response.get('node_id')

        return None

    def store_document(
        self,
        doc_id: str,
        title: str,
        embedded_chunks: list[EmbeddedChunk],
        metadata_list: list[ChunkMetadata],
        tags: Optional[list[str]] = None,
    ) -> None:
        """
        Store a complete document with all chunks, embeddings, and relationships.

        Args:
            doc_id: Unique document identifier
            title: Document title
            embedded_chunks: List of chunks with embeddings
            metadata_list: Corresponding metadata for each chunk
            tags: Optional list of document tags (e.g., ["inventory", "accounting"])

        Raises:
            ImportError: If helix-py is not installed
            RuntimeError: If storage operation fails
        """
        self._ensure_connected()

        # Add document node with tags
        doc_query = AddDocument(doc_id, title, len(embedded_chunks), tags=tags)
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
            # Extract node and score from response
            # HelixQL may return {"node": {...}, "score": ...} or direct node
            node = item.get("node", item) if isinstance(item, dict) else item
            score = item.get("score", 0.0) if isinstance(item, dict) else 0.0

            # Apply threshold filter
            if threshold > 0 and score < threshold:
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

            embedding = item.get("embedding", []) if isinstance(item, dict) else []
            model_name = item.get("model_name", "unknown") if isinstance(item, dict) else "unknown"

            ec = helix_to_embedded_chunk(node, embedding, model_name)
            results.append((ec, score))

            # Stop once we have enough results
            if len(results) >= top_k:
                break

        return results

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
        except Exception:
            return None

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
        except Exception:
            return False

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

        # Update the concept
        success = self.update_concept(
            concept_id=existing.get("concept_id", concept_id),
            source_documents=merged_docs,
            aliases=merged_aliases,
        )

        if success:
            # Return existing internal ID
            node_data = existing.get("N", {})
            if isinstance(node_data, dict):
                return node_data.get("Id")
            return existing.get("concept_id")

        return None

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
            query = ListDocumentConcepts(doc_id)
        else:
            query = ListConcepts()

        result = self._execute_query(query)

        if result.success:
            return result.data or []
        return []

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
    ) -> bool:
        """
        Create RelatesTo edge between concepts.

        Args:
            from_concept_id: Internal source concept node ID
            to_concept_id: Internal target concept node ID

        Returns:
            True if link created successfully
        """
        self._ensure_connected()

        query = LinkConceptRelatesTo(from_concept_id, to_concept_id)
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
