"""In-memory mock implementation of chunk storage for testing.

This module provides MockChunkStore, an in-memory implementation of the
chunk storage interface that doesn't require a real Helix-DB server.

Usage:
    from src.storage import MockChunkStore

    store = MockChunkStore()
    store.store_document("doc1", "My Document", embedded_chunks, metadata_list)
    results = store.search_similar(query_embedding, top_k=5)
"""

from typing import Any, Optional

from src.chunker import ChunkMetadata, EmbeddedChunk
from src.chunker.embedder import cosine_similarity


class MockChunkStore:
    """
    In-memory mock implementation for unit testing.

    Provides the same interface as HelixChunkStore but stores all data
    in dictionaries. No Helix-DB server required.

    Attributes:
        _documents: Document metadata storage
        _chunks: Chunk data storage (chunk_id -> data)
        _embeddings: Embedding storage (chunk_id -> embedding)
    """

    def __init__(self) -> None:
        """Initialize empty in-memory storage."""
        self._documents: dict[str, dict[str, Any]] = {}
        self._chunks: dict[str, dict[str, Any]] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._model_names: dict[str, str] = {}

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
        """
        import time

        # Normalize tags to lowercase and filter empty strings
        normalized_tags = [t.lower().strip() for t in (tags or []) if t and t.strip()]

        # Store document
        self._documents[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "tags": normalized_tags,
            "chunk_count": len(embedded_chunks),
            "created_at": int(time.time()),
        }

        # Store chunks with metadata
        for ec, meta in zip(embedded_chunks, metadata_list, strict=True):
            chunk = ec.chunk
            self._chunks[chunk.id] = {
                "chunk_id": chunk.id,
                "content": chunk.content,
                "level": chunk.level,
                "token_count": chunk.token_count,
                "word_count": chunk.word_count,
                "document_id": doc_id,
                "semantic_type": meta.semantic_type,
                "topic_tags": meta.topic_tags,
                "section_path": chunk.section_path,
                "source_section": chunk.source_section,
                "sequence_position": meta.sequence_position,
                "parent_id": chunk.parent_id,
                "child_ids": chunk.child_ids,
                "sibling_ids": meta.sibling_ids,
                "prev_id": chunk.prev_id,
                "next_id": chunk.next_id,
                "has_table": meta.has_table,
                "has_code": meta.has_code,
                "has_math": meta.has_math,
                "has_list": meta.has_list,
            }
            self._embeddings[chunk.id] = ec.embedding
            self._model_names[chunk.id] = ec.model_name

    def get_chunk(self, chunk_id: str) -> Optional[EmbeddedChunk]:
        """
        Retrieve a single chunk by ID with its embedding.

        Args:
            chunk_id: Unique chunk identifier

        Returns:
            EmbeddedChunk if found, None otherwise
        """
        if chunk_id not in self._chunks:
            return None

        return self._reconstruct_embedded_chunk(chunk_id)

    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk and its embedding.

        Args:
            chunk_id: Unique chunk identifier

        Returns:
            True if deleted, False if not found
        """
        if chunk_id not in self._chunks:
            return False

        # Update parent's child_ids
        chunk_data = self._chunks[chunk_id]
        parent_id = chunk_data.get("parent_id")
        if parent_id and parent_id in self._chunks:
            parent_data = self._chunks[parent_id]
            parent_data["child_ids"] = [
                cid for cid in parent_data["child_ids"] if cid != chunk_id
            ]

        # Update prev/next links
        prev_id = chunk_data.get("prev_id")
        next_id = chunk_data.get("next_id")
        if prev_id and prev_id in self._chunks:
            self._chunks[prev_id]["next_id"] = next_id
        if next_id and next_id in self._chunks:
            self._chunks[next_id]["prev_id"] = prev_id

        # Remove chunk and embedding
        del self._chunks[chunk_id]
        del self._embeddings[chunk_id]
        del self._model_names[chunk_id]

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
        if chunk_id not in self._chunks:
            return False

        if content is not None:
            self._chunks[chunk_id]["content"] = content
            self._chunks[chunk_id]["word_count"] = len(content.split())

        if embedding is not None:
            self._embeddings[chunk_id] = embedding

        return True

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            doc_id: Unique document identifier

        Returns:
            True if deleted, False if not found
        """
        if doc_id not in self._documents:
            return False

        # Find and delete all chunks for this document
        chunk_ids_to_delete = [
            cid for cid, data in self._chunks.items()
            if data.get("document_id") == doc_id
        ]

        for chunk_id in chunk_ids_to_delete:
            del self._chunks[chunk_id]
            del self._embeddings[chunk_id]
            del self._model_names[chunk_id]

        del self._documents[doc_id]
        return True

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
        results: list[tuple[str, float]] = []

        # Pre-compute set of doc_ids matching tag filter
        if tags:
            normalized_tags = {t.lower().strip() for t in tags}
            matching_doc_ids = {
                d["doc_id"] for d in self._documents.values()
                if normalized_tags.issubset(set(d.get("tags", [])))
            }
        else:
            matching_doc_ids = None

        for chunk_id, data in self._chunks.items():
            # Apply filters
            if doc_id and data.get("document_id") != doc_id:
                continue
            if level is not None and data.get("level") != level:
                continue
            if semantic_type and data.get("semantic_type") != semantic_type:
                continue
            # Filter by document tags
            if matching_doc_ids is not None and data.get("document_id") not in matching_doc_ids:
                continue

            # Calculate similarity
            embedding = self._embeddings.get(chunk_id)
            if embedding:
                similarity = cosine_similarity(query_embedding, embedding)
                if similarity >= threshold:
                    results.append((chunk_id, similarity))

        # Sort by similarity (descending) and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        # Reconstruct EmbeddedChunks (reconstruct once per chunk)
        output: list[tuple[EmbeddedChunk, float]] = []
        for chunk_id, score in results:
            ec = self._reconstruct_embedded_chunk(chunk_id)
            if ec is not None:
                output.append((ec, score))
        return output

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
            case_sensitive: Whether search is case-sensitive

        Returns:
            List of matching EmbeddedChunks
        """
        results: list[EmbeddedChunk] = []
        search_term = keyword if case_sensitive else keyword.lower()

        for chunk_id, data in self._chunks.items():
            if doc_id and data.get("document_id") != doc_id:
                continue

            content = data.get("content", "")
            compare_content = content if case_sensitive else content.lower()

            if search_term in compare_content:
                ec = self._reconstruct_embedded_chunk(chunk_id)
                if ec:
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
        if chunk_id not in self._chunks:
            return None

        parent_id = self._chunks[chunk_id].get("parent_id")
        if not parent_id:
            return None

        return self._reconstruct_embedded_chunk(parent_id)

    def get_children(self, chunk_id: str) -> list[EmbeddedChunk]:
        """
        Get child chunks (Level 1 -> Level 2).

        Args:
            chunk_id: Parent chunk ID

        Returns:
            List of child EmbeddedChunks
        """
        if chunk_id not in self._chunks:
            return []

        child_ids = self._chunks[chunk_id].get("child_ids", [])
        return [
            ec for ec in (self._reconstruct_embedded_chunk(cid) for cid in child_ids)
            if ec is not None
        ]

    def get_siblings(self, chunk_id: str) -> list[EmbeddedChunk]:
        """
        Get sibling chunks (same parent, excluding self).

        Args:
            chunk_id: Chunk ID

        Returns:
            List of sibling EmbeddedChunks
        """
        if chunk_id not in self._chunks:
            return []

        sibling_ids = self._chunks[chunk_id].get("sibling_ids", [])
        return [
            ec for ec in (self._reconstruct_embedded_chunk(sid) for sid in sibling_ids)
            if ec is not None
        ]

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
        if chunk_id not in self._chunks:
            return []

        result_ids: list[str] = []

        # Collect previous chunks
        current_id = chunk_id
        prev_ids: list[str] = []
        for _ in range(window_size):
            prev_id = self._chunks.get(current_id, {}).get("prev_id")
            if not prev_id or prev_id not in self._chunks:
                break
            prev_ids.insert(0, prev_id)
            current_id = prev_id

        result_ids.extend(prev_ids)
        result_ids.append(chunk_id)

        # Collect next chunks
        current_id = chunk_id
        for _ in range(window_size):
            next_id = self._chunks.get(current_id, {}).get("next_id")
            if not next_id or next_id not in self._chunks:
                break
            result_ids.append(next_id)
            current_id = next_id

        return [
            ec for ec in (self._reconstruct_embedded_chunk(cid) for cid in result_ids)
            if ec is not None
        ]

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
        results: list[EmbeddedChunk] = []

        for chunk_id, data in self._chunks.items():
            if data.get("document_id") != doc_id:
                continue
            if level is not None and data.get("level") != level:
                continue

            ec = self._reconstruct_embedded_chunk(chunk_id)
            if ec:
                results.append(ec)

        # Sort by sequence position
        results.sort(key=lambda x: self._chunks[x.chunk.id].get("sequence_position", 0))
        return results

    def list_documents(self, tags: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """
        List all documents with metadata.

        Args:
            tags: Filter by document tags (optional, AND logic)

        Returns:
            List of document metadata dictionaries
        """
        if not tags:
            return list(self._documents.values())

        # Filter by tags (AND logic - document must have ALL specified tags)
        normalized_tags = {t.lower().strip() for t in tags}
        return [
            doc for doc in self._documents.values()
            if normalized_tags.issubset(set(doc.get("tags", [])))
        ]

    def get_document_stats(self, doc_id: str) -> Optional[dict[str, Any]]:
        """
        Get statistics for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Dictionary with document statistics, or None if not found
        """
        if doc_id not in self._documents:
            return None

        chunks = [
            data for data in self._chunks.values()
            if data.get("document_id") == doc_id
        ]

        level_counts = {0: 0, 1: 0, 2: 0}
        type_counts: dict[str, int] = {}
        total_tokens = 0

        for chunk in chunks:
            level = chunk.get("level", 2)
            level_counts[level] = level_counts.get(level, 0) + 1

            sem_type = chunk.get("semantic_type", "narrative")
            type_counts[sem_type] = type_counts.get(sem_type, 0) + 1

            total_tokens += chunk.get("token_count", 0)

        return {
            "doc_id": doc_id,
            "title": self._documents[doc_id].get("title", ""),
            "total_chunks": len(chunks),
            "level_counts": level_counts,
            "type_counts": type_counts,
            "total_tokens": total_tokens,
        }

    def clear(self) -> None:
        """Clear all stored data."""
        self._documents.clear()
        self._chunks.clear()
        self._embeddings.clear()
        self._model_names.clear()

    def _reconstruct_embedded_chunk(self, chunk_id: str) -> Optional[EmbeddedChunk]:
        """
        Reconstruct an EmbeddedChunk from stored data.

        Args:
            chunk_id: Chunk identifier

        Returns:
            EmbeddedChunk if found, None otherwise
        """
        from src.chunker import Chunk

        if chunk_id not in self._chunks:
            return None

        data = self._chunks[chunk_id]
        embedding = self._embeddings.get(chunk_id, [])
        model_name = self._model_names.get(chunk_id, "unknown")

        chunk = Chunk(
            id=data["chunk_id"],
            content=data["content"],
            level=data["level"],
            token_count=data["token_count"],
            parent_id=data.get("parent_id"),
            child_ids=data.get("child_ids", []),
            prev_id=data.get("prev_id"),
            next_id=data.get("next_id"),
            source_section=data.get("source_section"),
            section_path=data.get("section_path", []),
        )

        return EmbeddedChunk(
            chunk=chunk,
            embedding=embedding,
            model_name=model_name,
        )
