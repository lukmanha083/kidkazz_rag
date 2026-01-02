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
    ) -> None:
        """
        Store a document and persist its chunks, embeddings, and hierarchical relationships.
        
        Parameters:
            doc_id (str): Unique identifier for the document.
            title (str): Document title to store with the document node.
            embedded_chunks (list[EmbeddedChunk]): Ordered list of chunks containing content and embeddings.
            metadata_list (list[ChunkMetadata]): List of metadata objects corresponding by index to `embedded_chunks`; must be the same length.
        
        Notes:
            - Persists a document node, creates chunk nodes with their embeddings, and creates "ParentOf" and "NextSibling" relationships between chunks where applicable.
            - May raise ImportError if the underlying `helix` client library is not installed.
        """
        ...

    def get_chunk(self, chunk_id: str) -> Optional[EmbeddedChunk]:
        """
        Retrieve a chunk by its ID, including its embedding and model metadata.
        
        Returns:
            EmbeddedChunk or None: The chunk with its embedding and model information if found, `None` if no chunk exists with the given `chunk_id`.
        """
        ...

    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk and its relationships from the store.
        
        Parameters:
            chunk_id (str): Identifier of the chunk to delete.
        
        Returns:
            bool: `True` if the chunk existed and was deleted, `False` if the chunk was not found.
        """
        ...

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all associated chunks from the store.
        
        Returns:
            bool: `True` if the document was deleted, `False` otherwise.
        """
        ...

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_id: Optional[str] = None,
        level: Optional[int] = None,
        semantic_type: Optional[str] = None,
        threshold: float = 0.0,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """
        Perform a vector-based similarity search to find chunks nearest to a query embedding.
        
        Parameters:
            query_embedding (list[float]): Query vector to compare against stored chunk embeddings.
            top_k (int): Maximum number of results to return.
            doc_id (Optional[str]): If provided, restrict results to chunks belonging to this document.
            level (Optional[int]): If provided, restrict results to chunks at this hierarchical level.
            semantic_type (Optional[str]): If provided, restrict results to chunks with this semantic type.
            threshold (float): Minimum similarity score required for a result to be included.
        
        Returns:
            list[tuple[EmbeddedChunk, float]]: A list of (chunk, score) pairs ordered by descending similarity score; `score` is the similarity between `query_embedding` and the chunk's embedding.
        """
        ...

    def get_parent(self, chunk_id: str) -> Optional[EmbeddedChunk]:
        """
        Retrieve the immediate parent chunk of the specified chunk.
        
        Returns:
            EmbeddedChunk: The parent chunk if it exists, `None` otherwise.
        """
        ...

    def get_children(self, chunk_id: str) -> list[EmbeddedChunk]:
        """
        Return the immediate child chunks of the specified parent chunk.
        
        Parameters:
            chunk_id (str): ID of the parent chunk whose direct children should be retrieved.
        
        Returns:
            list[EmbeddedChunk]: List of child chunks (including their embeddings and metadata); empty list if no children are found or the operation fails.
        """
        ...

    def get_siblings(self, chunk_id: str) -> list[EmbeddedChunk]:
        """
        Retrieve sibling chunks that share the same parent as the given chunk, excluding the chunk itself.
        
        Parameters:
        	chunk_id (str): Identifier of the center chunk whose siblings are requested.
        
        Returns:
        	list[EmbeddedChunk]: Sibling chunks in document order; an empty list if the chunk has no parent, the chunk is not found, or there are no siblings.
        """
        ...

    def get_context_window(
        self,
        chunk_id: str,
        window_size: int = 1,
    ) -> list[EmbeddedChunk]:
        """
        Retrieve a window of chunks around a center chunk in document order.
        
        Parameters:
            chunk_id (str): ID of the center chunk to build the window around.
            window_size (int): Number of neighboring chunks to include on each side of the center chunk.
        
        Returns:
            list[EmbeddedChunk]: Chunks ordered from earlier to later in the document (previous neighbors, center, next neighbors). Returns an empty list if the center chunk is not found.
        """
        ...

    def list_documents(self) -> list[dict[str, Any]]:
        """
        Retrieve metadata for all stored documents.
        
        Returns:
            list[dict[str, Any]]: A list of document metadata dictionaries (e.g., containing keys like `doc_id` and `title`). Returns an empty list if no documents are found or the query fails.
        """
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
        Create a HelixChunkStore configured for connecting to Helix-DB.
        
        Parameters:
            config (Optional[HelixConfig]): Configuration for the connection; defaults to a local HelixConfig().
        
        Notes:
            The actual connection is created lazily on first use.
        """
        self.config = config or HelixConfig()
        self._client: Any = None
        self._initialized = False

    def _ensure_connected(self) -> None:
        """
        Ensure a Helix-DB client connection is created and cached.
        
        If a connection has not already been initialized, creates a helix.Client using values from self.config, assigns it to self._client, and sets self._initialized to True.
        
        Raises:
            ImportError: If the `helix` package is not installed (suggests installing `helix-py`).
        """
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

    def store_document(
        self,
        doc_id: str,
        title: str,
        embedded_chunks: list[EmbeddedChunk],
        metadata_list: list[ChunkMetadata],
    ) -> None:
        """
        Store a document and its chunks into Helix-DB, including embeddings and parent/next relationships.
        
        Parameters:
            doc_id (str): Unique identifier for the document.
            title (str): Document title.
            embedded_chunks (list[EmbeddedChunk]): Chunks paired with their embeddings to store.
            metadata_list (list[ChunkMetadata]): Metadata entries corresponding to each chunk.
        
        Raises:
            ImportError: If the helix-py client library is not installed.
        """
        self._ensure_connected()

        # Add document node
        doc_query = AddDocument(doc_id, title, len(embedded_chunks))
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
        result = self._client.query(query)

        if not result.success or not result.data:
            return None

        node = result.data.get("node", {})
        embedding = result.data.get("embedding", [])
        model_name = result.data.get("model_name", "unknown")

        return helix_to_embedded_chunk(node, embedding, model_name)

    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk and all its relationships from the store.
        
        Parameters:
            chunk_id (str): Unique identifier of the chunk to delete.
        
        Returns:
            bool: `True` if the chunk existed and was deleted, `False` if the chunk was not found.
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
        Update the content and/or embedding of an existing chunk.
        
        Parameters:
            chunk_id (str): Unique identifier of the chunk to update.
            content (Optional[str]): New textual content for the chunk. If provided, the chunk's word count will be updated.
            embedding (Optional[list[float]]): New vector embedding for the chunk.
        
        Returns:
            bool: True if the chunk was found and at least one update was applied, False if the chunk does not exist.
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
        result = self._client.query(query)
        return result.success

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_id: Optional[str] = None,
        level: Optional[int] = None,
        semantic_type: Optional[str] = None,
        threshold: float = 0.0,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """
        Retrieve chunks most similar to a query embedding.
        
        Parameters:
            query_embedding (list[float]): Embedding vector for the query.
            top_k (int): Maximum number of results to return.
            doc_id (Optional[str]): If provided, restrict results to this document.
            level (Optional[int]): If provided, restrict results to this hierarchy level.
            semantic_type (Optional[str]): If provided, restrict results to this semantic type.
            threshold (float): Minimum similarity score to include a result.
        
        Returns:
            list[tuple[EmbeddedChunk, float]]: Pairs of matching chunks and their similarity scores, sorted by score descending.
        """
        self._ensure_connected()

        query = SearchSimilarChunks(
            query_embedding=query_embedding,
            top_k=top_k,
            doc_id=doc_id,
            level=level,
            semantic_type=semantic_type,
            threshold=threshold,
        )
        result = self._client.query(query)

        if not result.success:
            return []

        # Convert results to EmbeddedChunks
        results: list[tuple[EmbeddedChunk, float]] = []
        for item in result.data or []:
            node = item.get("node", {})
            embedding = item.get("embedding", [])
            model_name = item.get("model_name", "unknown")
            score = item.get("score", 0.0)

            ec = helix_to_embedded_chunk(node, embedding, model_name)
            results.append((ec, score))

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
    ) -> list[EmbeddedChunk]:
        """
        Find chunks whose content matches a keyword and return them as EmbeddedChunk objects.
        
        Parameters:
            keyword (str): Term to search for in chunk content.
            doc_id (Optional[str]): Optional document ID to restrict the search.
        
        Returns:
            list[EmbeddedChunk]: Matching chunks converted to EmbeddedChunk objects.
        """
        self._ensure_connected()

        query = SearchKeyword(keyword, doc_id)
        result = self._client.query(query)

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
        Return the parent chunk one level up for the given chunk ID.
        
        Returns:
            The parent EmbeddedChunk if it exists, `None` otherwise.
        """
        self._ensure_connected()

        query = GetChunkWithContext(chunk_id)
        result = self._client.query(query)

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
        Retrieve direct child chunks whose `parent_id` equals the given chunk ID.
        
        Parameters:
            chunk_id (str): ID of the parent chunk whose immediate children to retrieve.
        
        Returns:
            list[EmbeddedChunk]: List of child EmbeddedChunk objects (may be empty).
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
        Retrieve sibling chunks that share the same parent as the given chunk, excluding the chunk itself.
        
        Parameters:
            chunk_id (str): ID of the chunk whose siblings to retrieve.
        
        Returns:
            list[EmbeddedChunk]: List of sibling EmbeddedChunk objects; returns an empty list if the chunk has no siblings or the query fails.
        """
        self._ensure_connected()

        query = GetChunkWithContext(chunk_id)
        result = self._client.query(query)

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
        Retrieve all embedded chunks for a document, optionally filtering by hierarchy level.
        
        Parameters:
            level (Optional[int]): If provided, only return chunks at this hierarchy level (e.g., 0 for top-level).
        
        Returns:
            list[EmbeddedChunk]: EmbeddedChunk objects for the document; returns an empty list if no chunks are found or the query fails.
        """
        self._ensure_connected()

        query = GetDocumentChunks(doc_id, level)
        result = self._client.query(query)

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

    def list_documents(self) -> list[dict[str, Any]]:
        """
        List all documents with metadata.
        
        Returns:
            A list of document metadata dictionaries; returns an empty list if no documents are found or the query fails.
        """
        self._ensure_connected()

        query = ListDocuments()
        result = self._client.query(query)

        if not result.success:
            return []

        return result.data or []

    def get_document_stats(self, doc_id: str) -> Optional[dict[str, Any]]:
        """
        Compute basic statistics for a document's chunks.
        
        Returns:
            A dict with the document statistics containing:
                - "doc_id": the requested document id.
                - "title": document title (empty string if unavailable).
                - "total_chunks": number of chunks for the document.
                - "level_counts": mapping of level (0, 1, 2) to chunk counts.
                - "type_counts": mapping of semantic type to counts (may be empty).
                - "total_tokens": sum of token_count across all chunks.
            `None` if the document has no chunks or cannot be found.
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
        """
        Enter a context and return this HelixChunkStore instance.
        
        Returns:
            HelixChunkStore: The same store instance to be used within the context manager.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Close the HelixChunkStore and release its connection resources when exiting a context.
        
        This method is invoked by the context-manager protocol to ensure the store's connection is closed and internal state is reset.
        """
        self.close()