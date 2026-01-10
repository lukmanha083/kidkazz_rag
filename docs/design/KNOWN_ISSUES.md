# Known Issues and Technical Debt

This document tracks known issues, limitations, and planned migrations for KidKazz RAG.

## Full Helix-DB Integration

**Status:** Completed (2025-01-07)
**Priority:** N/A (resolved)
**Affects:** All storage operations in `src/storage/`

### Summary

Full migration to HelixQL completed, including:
- Read operations (completed earlier)
- Vector storage (NEW)
- Delete operations (NEW)
- Update operations (NEW)
- Cascade delete (NEW)

---

## Phase 1: Query Format Migration (Completed Earlier)

### Problem (Resolved)

The codebase previously used two different query formats for Helix-DB:
- Write operations used HelixQL format (working)
- Read operations used legacy `{"operation": ...}` format (broken)

### Solution Implemented

1. **Schema updated** (`db/schema.hx`):
   - Added `INDEX` keyword to fields used in WHERE clauses
   - Indexed: `Document.doc_id`, `Chunk.chunk_id`, `Chunk.document_id`, `Chunk.level`, `Chunk.semantic_type`, `Chunk.parent_id`

2. **HelixQL read queries added** (`db/queries.hx`):
   - `ListDocuments()` - List all documents
   - `GetChunkByChunkId(chunk_id: String)` - Get chunk by user-facing ID
   - `GetChunksByDocumentId(document_id: String)` - Get chunks for a document
   - `GetChunksByDocAndLevel(document_id: String, level: U32)` - Filter by doc and level
   - `GetChunksByParentId(parent_id: String)` - Get children of a chunk
   - `SearchSimilar(query_vec: [F64], top_k: U32)` - Vector similarity search
   - `SearchKeywordBM25(keyword: String, limit: U32)` - BM25 keyword search
   - `DeleteDocumentByDocId(doc_id: String)` - Delete document by ID

3. **Python query classes updated** (`src/storage/queries.py`):
   - Each class now has `endpoint` attribute matching HelixQL query name
   - Query parameters passed via `super().__init__(endpoint="...")`
   - Filters (threshold, doc_id, level, semantic_type) stored for post-processing

4. **Client updated** (`src/storage/client.py`):
   - Post-filtering implemented for vector search and keyword search
   - Response handling updated for HelixQL format
   - Graph operations use field lookups instead of edge traversals

---

## Phase 2: Vector Storage (Completed 2025-01-07)

### Problem (Resolved)

Embeddings were computed but NEVER stored to Helix-DB:
- `embedded_chunk_to_helix_vector()` existed but was never called
- `store_document()` added chunks but no vectors
- Vector search returned empty results

### Solution Implemented

1. **HelixQL vector queries added** (`db/queries.hx`):
   - `AddChunkVector(embedding: [F64], model_name: String, embedding_dim: U32)` - Store vector
   - `LinkChunkVector(chunk_id: ID, vector_id: ID)` - Link chunk to vector via HasEmbedding edge

2. **Python query classes added** (`src/storage/queries.py`):
   - `AddChunkVector` - Store embedding vector
   - `LinkChunkVector` - Create HasEmbedding edge

3. **`store_document()` updated** (`src/storage/client.py`):
   - Now stores vector embeddings for each chunk
   - Creates HasEmbedding edge linking chunk to vector

---

## Phase 3: Delete Operations (Completed 2025-01-07)

### Problem (Resolved)

`delete_chunk()` and `delete_document()` used legacy format:
```python
delete_query = {"operation": "delete_node", ...}  # Legacy - broken
```

### Solution Implemented

1. **HelixQL delete queries added** (`db/queries.hx`):
   - `DropChunk(chunk_id: ID)` - Drop chunk node (cascades to edges)
   - `DropDocument(doc_id: ID)` - Drop document node
   - `DropDocumentChunkEdges(doc_id: ID)` - Drop HasChunk edges
   - `DropChunkEmbeddingEdge(chunk_id: ID)` - Drop HasEmbedding edge

2. **Python query classes added** (`src/storage/queries.py`):
   - `DropChunk` - Drop chunk by internal ID
   - `DropDocument` - Drop document by internal ID
   - `GetDocumentByDocId` - Get document internal ID for delete

3. **Cascade delete implemented** (`src/storage/client.py`):
   - `delete_document()` now drops all chunks first, then document
   - Dropping a chunk cascades to edges and vectors

---

## Phase 4: Update Operations (Completed 2025-01-07)

### Problem (Resolved)

`update_chunk()` used legacy format:
```python
update_query = {"operation": "update_node", ...}  # Legacy - broken
```

### Solution Implemented

1. **HelixQL update query added** (`db/queries.hx`):
   - `UpdateChunkContent(chunk_id: ID, content: String, word_count: U32)`

2. **Python query class added** (`src/storage/queries.py`):
   - `UpdateChunkContent` - Update chunk content by internal ID

3. **`update_chunk()` updated** (`src/storage/client.py`):
   - Uses HelixQL UpdateChunkContent for content updates
   - Embedding updates create new vector (HelixDB doesn't support vector updates)

---

## Key Design Decisions

- **Post-filtering**: HelixQL SearchV doesn't support pre-filtering or threshold, so filters are applied in Python
- **String IDs vs Internal IDs**: Use `WHERE` clauses with indexed String fields for user-facing queries; use internal `ID` type for DROP/UPDATE operations
- **Cascade delete**: Manually drop chunks before document to avoid orphans
- **Vector updates**: Create new vector and link (HelixDB doesn't support direct vector updates)
- **DROP syntax**: HelixQL uses `DROP N<Type>(id)` not `DELETE`

---

## Deployment

After code changes, run to deploy schema and queries:
```bash
helix push dev
```

---

## Remaining Limitations

1. **BM25 Search**: Unverified with real HelixDB instance
2. **Vector Updates**: Creates new vector instead of updating (limitation of HelixDB)
3. **No Batch Deletes**: Deleting a document with many chunks makes N+1 queries

---

## Reducto Response Format Handling

**Status:** Fixed (2024-01-06)
**Affects:** `src/pdf_converter/reducto_client.py`

### Problem

Reducto API could return responses as either Python objects or dictionaries. The original code only handled object responses, causing JSON output instead of markdown.

### Solution

Updated `_response_to_markdown()` and `_chunks_to_markdown()` to handle both formats using helper functions:

```python
def get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
```

---

## Boolean to Integer Conversion for Helix-DB

**Status:** Fixed (2024-01-06)
**Affects:** `src/storage/converters.py`

### Problem

Helix-DB schema uses U32 for boolean-like fields (`has_table`, `has_code`, `has_math`, `has_list`), but Python was sending boolean values.

### Solution

Wrap boolean values with `int()` in `chunk_to_helix_node()`:

```python
"has_table": int(metadata.has_table),
"has_code": int(metadata.has_code),
"has_math": int(metadata.has_math),
"has_list": int(metadata.has_list),
```
