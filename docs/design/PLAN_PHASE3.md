# Phase 3: Helix-DB Integration - Implementation Plan

## Overview

Build a storage layer for hierarchical chunks with vector embeddings using Helix-DB, a native vector + graph database. This enables semantic search combined with graph traversal for context expansion.

---

## Key Decisions

### 1. Database: Helix-DB (Vector + Graph)

**Why Helix-DB instead of alternatives:**
- Native combination of vector similarity search AND graph traversal
- Built in Rust for performance (LMDB backend)
- Python SDK (`helix-py`) with clean API
- Designed for RAG applications
- Open-source (AGPL license)

**Comparison with alternatives:**

| Database | Vector Search | Graph Traversal | Python SDK |
|----------|--------------|-----------------|------------|
| Helix-DB | ✅ Native | ✅ Native | ✅ helix-py |
| ChromaDB | ✅ Native | ❌ None | ✅ chromadb |
| LanceDB | ✅ Native | ❌ None | ✅ lancedb |
| Neo4j | ❌ Plugin | ✅ Native | ✅ neo4j |

### 2. Storage Strategy: MockChunkStore + HelixChunkStore

**Two implementations for flexibility:**
- **MockChunkStore**: In-memory implementation for unit testing (no server needed)
- **HelixChunkStore**: Real Helix-DB client for production

**Why this approach:**
- Tests run fast without database dependency
- Same interface for both (Protocol-based)
- Easy to swap implementations

### 3. Schema Design: Document → Chunk → Vector

**Three-level node structure:**

| Node Type | Purpose | Key Fields |
|-----------|---------|------------|
| Document | Container | doc_id, title, tags, chunk_count |
| Chunk | Content + metadata | chunk_id, content, level, semantic_type |
| ChunkVector | Embedding | embedding (384 dims) |

**Document tags** enable topic-based filtering (e.g., "inventory", "accounting"). Tags are stored as a JSON array and support AND-logic filtering in search and list operations.

**Edge relationships:**

| Edge Type | From → To | Purpose |
|-----------|-----------|---------|
| HasChunk | Document → Chunk | Document contains chunks |
| ParentOf | Chunk → Chunk | L1 → L2 hierarchy |
| NextSibling | Chunk → Chunk | Sequential order |
| HasEmbedding | Chunk → ChunkVector | Chunk has embedding |

---

## Module Structure

```text
src/
├── chunker/           # Phase 2 (existing)
└── storage/           # Phase 3 (new)
    ├── __init__.py    # Public API exports
    ├── schema.py      # Helix-DB schema definitions
    ├── converters.py  # Dataclass <-> Helix format
    ├── queries.py     # HelixQL query classes
    ├── mock_store.py  # In-memory MockChunkStore
    └── client.py      # HelixChunkStore client

tests/
├── test_storage_schema.py      # Schema tests
├── test_storage_converters.py  # Converter tests
├── test_storage_mock.py        # MockChunkStore tests
├── test_storage_queries.py     # Query class tests
└── test_storage_integration.py # End-to-end tests
```

---

## Implementation Details

### 1. `schema.py` - Helix-DB Schema Definitions

```python
@dataclass
class SchemaConfig:
    """Configuration for Helix-DB schema."""
    vector_dimensions: int = 384
    vector_type: str = "Vec32"
    db_name: str = "kidkazz_rag"

# Node schemas
DOCUMENT_NODE_SCHEMA = {
    "doc_id": "String",
    "title": "String",
    "tags": "String",       # JSON array of document tags
    "created_at": "I64",
    "chunk_count": "U32",
}

CHUNK_NODE_SCHEMA = {
    "chunk_id": "String",
    "content": "String",
    "level": "U32",
    "token_count": "U32",
    "semantic_type": "String",
    "section_path": "String",  # JSON array
    "has_table": "Bool",
    "has_code": "Bool",
    # ... more fields
}

# Edge definitions
EDGE_TYPES = {
    "HasChunk": ("Document", "Chunk"),
    "ParentOf": ("Chunk", "Chunk"),
    "NextSibling": ("Chunk", "Chunk"),
    "HasEmbedding": ("Chunk", "ChunkVector"),
}

def create_kidkazz_schema(config: SchemaConfig = None) -> Schema:
    """Create Helix-DB schema for KidKazz RAG."""

def validate_chunk_node(node: dict) -> list[str]:
    """Validate chunk node against schema."""
```

### 2. `converters.py` - Data Format Conversion

```python
def chunk_to_helix_node(chunk: Chunk, metadata: ChunkMetadata = None) -> dict:
    """Convert Chunk + Metadata to Helix-DB node properties."""

def helix_node_to_chunk(node: dict) -> Chunk:
    """Convert Helix-DB node to Chunk dataclass."""

def embedded_chunk_to_helix_vector(ec: EmbeddedChunk) -> dict:
    """Convert embedding to Helix-DB vector format."""

def helix_to_embedded_chunk(node: dict, embedding: list, model_name: str) -> EmbeddedChunk:
    """Reconstruct EmbeddedChunk from Helix-DB data."""

def document_to_helix_node(doc_id: str, title: str, chunk_count: int) -> dict:
    """Create document node properties."""
```

### 3. `queries.py` - HelixQL Query Classes

```python
class AddDocument(Query):
    """Add a document node."""
    def __init__(self, doc_id: str, title: str, chunk_count: int): ...
    def query(self) -> Payload: ...

class AddChunkWithEmbedding(Query):
    """Add chunk node with embedding vector."""
    def __init__(self, chunk_props: dict, embedding: list, doc_id: str): ...

class SearchSimilarChunks(Query):
    """Vector similarity search."""
    def __init__(self, query_embedding: list, top_k: int = 5,
                 doc_id: str = None, level: int = None): ...

class GetChunkWithContext(Query):
    """Get chunk with parent, siblings, neighbors."""
    def __init__(self, chunk_id: str): ...

class GetDocumentChunks(Query):
    """Get all chunks for a document."""
    def __init__(self, doc_id: str, level: int = None): ...
```

### 4. `mock_store.py` - In-Memory Implementation

```python
class MockChunkStore:
    """In-memory mock for unit testing."""

    def __init__(self):
        self._documents: dict[str, dict] = {}
        self._chunks: dict[str, dict] = {}
        self._embeddings: dict[str, list[float]] = {}

    def store_document(self, doc_id: str, title: str,
                       embedded_chunks: list[EmbeddedChunk],
                       metadata_list: list[ChunkMetadata],
                       tags: list[str] = None) -> None:
        """Store document with all chunks and optional tags."""

    def search_similar(self, query_embedding: list[float], top_k: int = 5,
                       doc_id: str = None, level: int = None,
                       semantic_type: str = None,
                       tags: list[str] = None) -> list[tuple[EmbeddedChunk, float]]:
        """Vector similarity search with optional tag filtering."""

    def get_parent(self, chunk_id: str) -> Optional[EmbeddedChunk]:
        """Get parent chunk."""

    def get_children(self, chunk_id: str) -> list[EmbeddedChunk]:
        """Get child chunks."""

    def get_context_window(self, chunk_id: str, window_size: int = 1) -> list[EmbeddedChunk]:
        """Get chunk with surrounding neighbors."""
```

### 5. `client.py` - Helix-DB Client

```python
@dataclass
class HelixConfig:
    """Helix-DB connection configuration."""
    local: bool = True
    port: int = 6969
    verbose: bool = False

class ChunkStoreProtocol(Protocol):
    """Interface for chunk storage implementations."""
    def store_document(...) -> None: ...
    def search_similar(...) -> list[tuple[EmbeddedChunk, float]]: ...
    def get_parent(...) -> Optional[EmbeddedChunk]: ...
    # ... more methods

class HelixChunkStore:
    """Main Helix-DB client for chunk storage."""

    def __init__(self, config: HelixConfig = None):
        """Initialize with lazy connection."""

    def store_document(self, doc_id: str, title: str,
                       embedded_chunks: list[EmbeddedChunk],
                       metadata_list: list[ChunkMetadata],
                       tags: list[str] = None) -> None:
        """Store document with chunks, embeddings, graph relationships, and optional tags."""

    def search_similar(self, query_embedding: list[float], top_k: int = 5,
                       tags: list[str] = None, **filters) -> list[tuple[EmbeddedChunk, float]]:
        """Vector similarity search with optional tag filtering."""

    def search_text(self, query_text: str, embedder, top_k: int = 5) -> list:
        """Search by text (embeds query first)."""

    def get_parent(self, chunk_id: str) -> Optional[EmbeddedChunk]:
        """Graph traversal: get parent chunk."""

    def get_context_window(self, chunk_id: str, window_size: int = 1) -> list:
        """Graph traversal: get surrounding chunks."""
```

---

## Public API (`__init__.py`)

```python
from .client import HelixChunkStore, HelixConfig, ChunkStoreProtocol
from .mock_store import MockChunkStore
from .schema import create_kidkazz_schema, SchemaConfig
from .converters import (
    chunk_to_helix_node,
    helix_node_to_chunk,
    embedded_chunk_to_helix_vector,
    helix_to_embedded_chunk,
)

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
```

---

## Dependencies (pyproject.toml)

```toml
[project.optional-dependencies]
helixdb = [
    "helix-py>=0.1.0",
]
all = [
    "fastembed>=0.2.0",
    "helix-py>=0.1.0",
]

[tool.pytest.ini_options]
markers = [
    "helix: marks tests requiring real Helix-DB server",
]
```

---

## Test Plan

### test_storage_schema.py (~24 tests)
- SchemaConfig defaults and custom values
- Node schemas have required fields
- Edge types defined correctly
- Schema validation functions work
- Create schema returns valid object

### test_storage_converters.py (~18 tests)
- Chunk to Helix node conversion
- Helix node to Chunk reconstruction
- Metadata conversion
- Embedding conversion
- Round-trip conversion preserves data
- Handle None/empty optional fields

### test_storage_mock.py (~38 tests)
- Store and retrieve documents
- Store and retrieve chunks
- Delete chunks and documents
- Update chunk content and embedding
- Vector similarity search
- Filter by document, level, semantic type
- Graph traversal (parent, children, siblings)
- Context window retrieval
- Keyword search
- Document statistics

### test_storage_queries.py (~23 tests)
- Query payload generation
- Response processing
- Filter inclusion/exclusion
- All query types work correctly

### test_storage_integration.py (~11 tests)
- Full pipeline: markdown → chunks → embeddings → storage
- Semantic search finds relevant chunks
- Graph traversal works after storage
- Multi-document scenarios
- Search quality (exact match ranks highest)

**Total: ~114 tests**

---

## Usage Example

```python
from src.chunker import (
    create_hierarchical_chunks,
    enrich_all_chunks,
    MockEmbedder,  # or ChunkEmbedder
)
from src.storage import MockChunkStore  # or HelixChunkStore

# 1. Create chunks from markdown (Phase 2)
with open("output/textbook.md") as f:
    content = f.read()

chunks = create_hierarchical_chunks(content, doc_id="textbook")
metadata = enrich_all_chunks(chunks, document_id="textbook")

# 2. Generate embeddings
embedder = MockEmbedder()  # or ChunkEmbedder for production
embedded_chunks = embedder.embed_chunks(chunks)

# 3. Store in database (Phase 3) with optional tags
store = MockChunkStore()  # or HelixChunkStore()
store.store_document(
    doc_id="textbook",
    title="My Textbook",
    embedded_chunks=embedded_chunks,
    metadata_list=metadata,
    tags=["inventory", "accounting"],  # Optional: topic-based filtering
)

# 4. Search by text
query_embedding = embedder.embed_text("What is machine learning?")
results = store.search_similar(query_embedding, top_k=5)

for chunk, score in results:
    print(f"[{score:.3f}] {chunk.chunk.content[:100]}...")

# 5. Expand context via graph
top_chunk = results[0][0]
parent = store.get_parent(top_chunk.chunk.id)
context = store.get_context_window(top_chunk.chunk.id, window_size=2)

# 6. Filter search by semantic type
definitions = store.search_similar(
    query_embedding,
    top_k=10,
    semantic_type="definition"
)

# 7. Filter search by document tags
inventory_results = store.search_similar(
    query_embedding,
    top_k=10,
    tags=["inventory"]  # Only chunks from inventory-tagged documents
)
```

---

## Performance Expectations

| Operation | MockChunkStore | HelixChunkStore |
|-----------|---------------|-----------------|
| Store 1000 chunks | < 100ms | ~1-2 seconds |
| Vector search (top-5) | < 10ms | ~2ms |
| Graph traversal | < 1ms | < 1ms |
| Full pipeline | < 1 second | ~3-5 seconds |

---

## Success Criteria

1. All ~114 unit tests pass
2. MockChunkStore provides same interface as HelixChunkStore
3. Vector search returns relevant results (exact match ranks highest)
4. Graph traversal correctly follows relationships
5. Multi-document storage works correctly
6. Coverage > 75% (client.py lower due to requiring real DB)

---

## Key Design Patterns

### 1. Protocol-Based Interface
```python
class ChunkStoreProtocol(Protocol):
    """Both MockChunkStore and HelixChunkStore implement this."""
```
Enables easy swapping between mock and real implementations.

### 2. Lazy Initialization
```python
def _ensure_connected(self):
    if not self._initialized:
        import helix
        self._client = helix.Client(...)
```
Avoids import errors when helix-py not installed.

### 3. JSON for Complex Fields
```python
"section_path": json.dumps(chunk.section_path),  # list -> JSON string
"child_ids": json.dumps(chunk.child_ids),
```
Helix-DB stores strings; JSON preserves list/dict structure.

### 4. Cosine Similarity for MockChunkStore
```python
from src.chunker.embedder import cosine_similarity

similarity = cosine_similarity(query_embedding, chunk_embedding)
```
Reuses existing similarity function for consistency.

---

## Next Phase (Phase 4: MCP Server)

After this phase, the storage layer is ready for:
- MCP server exposing search and traversal tools
- Claude Code integration for document queries
- Natural language interface to stored documents
