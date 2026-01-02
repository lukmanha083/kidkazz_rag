"""Storage module for Helix-DB integration.

This module provides storage interfaces for hierarchical chunks with vector
embeddings in Helix-DB (vector + graph database).

Main components:
- HelixChunkStore: Main storage interface for Helix-DB operations
- MockChunkStore: In-memory implementation for unit testing
- Converters: Transform between dataclasses and Helix-DB formats
- Schema: Helix-DB schema definitions

Usage:
    from src.storage import HelixChunkStore, MockChunkStore

    # For testing (no server needed)
    store = MockChunkStore()

    # For production (requires Helix-DB server)
    store = HelixChunkStore()

    # Store document
    store.store_document("doc_id", "Title", embedded_chunks, metadata_list)

    # Search
    results = store.search_similar(query_embedding, top_k=5)
"""

from .converters import (
    chunk_to_helix_node,
    embedded_chunk_to_helix_vector,
    helix_node_to_chunk,
    helix_to_embedded_chunk,
)
from .mock_store import MockChunkStore
from .schema import SchemaConfig, create_kidkazz_schema

# Conditionally import HelixChunkStore (requires helix-py)
try:
    from .client import ChunkStoreProtocol, HelixChunkStore, HelixConfig
except ImportError:
    # helix-py not installed, provide stubs
    HelixChunkStore = None  # type: ignore[misc, assignment]
    HelixConfig = None  # type: ignore[misc, assignment]
    ChunkStoreProtocol = None  # type: ignore[misc, assignment]

__all__ = [
    # Main client
    "HelixChunkStore",
    "HelixConfig",
    "ChunkStoreProtocol",
    # Mock for testing
    "MockChunkStore",
    # Schema
    "create_kidkazz_schema",
    "SchemaConfig",
    # Converters
    "chunk_to_helix_node",
    "helix_node_to_chunk",
    "embedded_chunk_to_helix_vector",
    "helix_to_embedded_chunk",
]
