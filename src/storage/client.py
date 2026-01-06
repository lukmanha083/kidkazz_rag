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
    AddChunkRelationship,
    AddChunkWithEmbedding,
    AddDocument,
    DeleteDocument,
    GetChunk,
    GetChunkWithContext,
    GetDocumentChunks,
    ListDocuments,
    SearchKeyword,
    SearchSimilarChunks,
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
        # Call the query's response method to convert raw response to QueryResult
        if hasattr(query, 'response'):
            return query.response(raw_result)
        return raw_result

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
        self._client.query(doc_query)

        # Add chunks with embeddings
        for ec, meta in zip(embedded_chunks, metadata_list, strict=True):
            chunk_props = chunk_to_helix_node(ec.chunk, meta)
            chunk_query = AddChunkWithEmbedding(
                chunk_props=chunk_props,
                embedding=ec.embedding,
                doc_id=doc_id,
            )
            self._client.query(chunk_query)

        # Add relationship edges
        for ec in embedded_chunks:
            chunk = ec.chunk

            # Parent-child relationships
            if chunk.parent_id:
                edge_query = AddChunkRelationship(
                    "ParentOf",
                    chunk.parent_id,
                    chunk.id,
                )
                self._client.query(edge_query)

            # Sequential relationships
            if chunk.next_id:
                edge_query = AddChunkRelationship(
                    "NextSibling",
                    chunk.id,
                    chunk.next_id,
                )
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

        node = result.data.get("node", {})
        embedding = result.data.get("embedding", [])
        model_name = result.data.get("model_name", "unknown")

        return helix_to_embedded_chunk(node, embedding, model_name)

    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk and its relationships.

        Args:
            chunk_id: Unique chunk identifier

        Returns:
            True if deleted, False if not found
        """
        self._ensure_connected()

        # Get chunk first to update relationships
        chunk = self.get_chunk(chunk_id)
        if not chunk:
            return False

        # Delete via direct query
        delete_query = {
            "operation": "delete_node",
            "node_type": "Chunk",
            "filter": {"chunk_id": chunk_id},
        }
        self._client.query(delete_query)
        return True

    def update_chunk(
        self,
        chunk_id: str,
        content: Optional[str] = None,
        embedding: Optional[list[float]] = None,
    ) -> bool:
        """
        Update chunk content and/or embedding.

        Args:
            chunk_id: Unique chunk identifier
            content: New content (optional)
            embedding: New embedding (optional)

        Returns:
            True if updated, False if not found
        """
        self._ensure_connected()

        chunk = self.get_chunk(chunk_id)
        if not chunk:
            return False

        updates: dict[str, Any] = {}
        if content is not None:
            updates["content"] = content
            updates["word_count"] = len(content.split())

        if updates:
            update_query = {
                "operation": "update_node",
                "node_type": "Chunk",
                "filter": {"chunk_id": chunk_id},
                "set": updates,
            }
            self._client.query(update_query)

        if embedding is not None:
            vector_update = {
                "operation": "update_vector",
                "filter": {"chunk_id": chunk_id},
                "embedding": embedding,
            }
            self._client.query(vector_update)

        return True

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            doc_id: Unique document identifier

        Returns:
            True if deleted, False if not found
        """
        self._ensure_connected()

        query = DeleteDocument(doc_id)
        result = self._execute_query(query)
        return result.success

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
        Find chunks similar to query embedding.

        Args:
            query_embedding: Query vector (384 dims)
            top_k: Number of results
            doc_id: Filter by document (optional)
            level: Filter by hierarchy level (optional)
            semantic_type: Filter by semantic type (optional)
            threshold: Minimum similarity score (default: 0.0)
            tags: Filter by document tags (optional, AND logic)

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

        # Request more results if filtering by tags (we'll filter in memory)
        request_top_k = top_k * 3 if tags else top_k

        query = SearchSimilarChunks(
            query_embedding=query_embedding,
            top_k=request_top_k,
            doc_id=doc_id,
            level=level,
            semantic_type=semantic_type,
            threshold=threshold,
        )
        result = self._execute_query(query)

        if not result.success:
            return []

        # Convert results to EmbeddedChunks
        results: list[tuple[EmbeddedChunk, float]] = []
        for item in result.data or []:
            node = item.get("node", {})
            embedding = item.get("embedding", [])
            model_name = item.get("model_name", "unknown")
            score = item.get("score", 0.0)

            # Filter by document tags
            if matching_doc_ids is not None:
                chunk_doc_id = node.get("document_id", "")
                if chunk_doc_id not in matching_doc_ids:
                    continue

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
        Full-text keyword search in chunk content.

        Args:
            keyword: Search term
            doc_id: Filter by document (optional)
            case_sensitive: Whether search is case-sensitive (default: False)

        Returns:
            List of matching EmbeddedChunks
        """
        self._ensure_connected()

        query = SearchKeyword(keyword, doc_id, case_sensitive)
        result = self._execute_query(query)

        if not result.success:
            return []

        results: list[EmbeddedChunk] = []
        for item in result.data or []:
            node = item.get("node", {})
            embedding = item.get("embedding", [])
            model_name = item.get("model_name", "unknown")

            ec = helix_to_embedded_chunk(node, embedding, model_name)
            results.append(ec)

        return results

    def get_parent(self, chunk_id: str) -> Optional[EmbeddedChunk]:
        """
        Get parent chunk (Level 2 -> Level 1).

        Args:
            chunk_id: Child chunk ID

        Returns:
            Parent EmbeddedChunk if exists, None otherwise
        """
        self._ensure_connected()

        query = GetChunkWithContext(chunk_id)
        result = self._execute_query(query)

        if not result.success or not result.data:
            return None

        parent_data = result.data.get("parent")
        if not parent_data:
            return None

        return helix_to_embedded_chunk(
            parent_data.get("node", {}),
            parent_data.get("embedding", []),
            parent_data.get("model_name", "unknown"),
        )

    def get_children(self, chunk_id: str) -> list[EmbeddedChunk]:
        """
        Get child chunks (Level 1 -> Level 2).

        Args:
            chunk_id: Parent chunk ID

        Returns:
            List of child EmbeddedChunks
        """
        self._ensure_connected()

        # Query for chunks where parent_id = chunk_id
        query = {
            "operation": "get_nodes",
            "node_type": "Chunk",
            "filter": {"parent_id": chunk_id},
            "include_vector": True,
        }
        result = self._client.query(query)

        results: list[EmbeddedChunk] = []
        for item in result or []:
            node = item.get("node", {})
            embedding = item.get("embedding", [])
            model_name = item.get("model_name", "unknown")

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

        query = GetChunkWithContext(chunk_id)
        result = self._execute_query(query)

        if not result.success or not result.data:
            return []

        siblings_data = result.data.get("siblings", [])
        results: list[EmbeddedChunk] = []

        for sibling in siblings_data:
            ec = helix_to_embedded_chunk(
                sibling.get("node", {}),
                sibling.get("embedding", []),
                sibling.get("model_name", "unknown"),
            )
            results.append(ec)

        return results

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
            node = item.get("node", {})
            embedding = item.get("embedding", [])
            model_name = item.get("model_name", "unknown")

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

        docs = result.data or []

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
