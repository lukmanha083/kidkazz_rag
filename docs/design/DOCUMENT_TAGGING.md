# Document Tagging Feature - Design Document

## Overview

Document tagging enables organizing and filtering documents by topic/concept (e.g., "inventory", "accounting", "supply-chain"). Tags are applied at the document level and inherited by all chunks within that document for search filtering.

---

## Key Decisions

### 1. Tag Scope: Document Level

Tags are stored on the Document node, not on individual chunks.

**Benefits:**
- Simpler schema (no tag duplication per chunk)
- Consistent tagging across all chunks in a document
- Easy to update tags without touching chunks
- Aligns with how users think about documents

### 2. Tag Behavior

| Feature | Behavior |
|---------|----------|
| **Case-insensitive** | Tags normalized to lowercase on storage |
| **Whitespace trimmed** | Leading/trailing spaces removed |
| **AND logic** | Multiple tags = document must have ALL specified tags |
| **Stored as JSON** | `["inventory", "accounting"]` in database |

### 3. Storage Format

Tags are stored as a JSON-encoded string in the `tags` field of the Document node:

```python
DOCUMENT_NODE_SCHEMA = {
    "doc_id": "String",
    "title": "String",
    "tags": "String",       # JSON array: '["inventory", "accounting"]'
    "created_at": "I64",
    "chunk_count": "U32",
}
```

---

## Implementation

### Schema Changes

**`src/storage/schema.py`:**
```python
DOCUMENT_NODE_SCHEMA: dict[str, str] = {
    "doc_id": "String",
    "title": "String",
    "tags": "String",  # JSON array of document tags
    "created_at": "I64",
    "chunk_count": "U32",
}
```

### Store Protocol

**`src/storage/client.py`:**
```python
class ChunkStoreProtocol(Protocol):
    def store_document(
        self,
        doc_id: str,
        title: str,
        embedded_chunks: list[EmbeddedChunk],
        metadata_list: list[ChunkMetadata],
        tags: Optional[list[str]] = None,  # NEW
    ) -> None: ...

    def list_documents(
        self,
        tags: Optional[list[str]] = None,  # NEW: filter by tags
    ) -> list[dict[str, Any]]: ...

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_id: Optional[str] = None,
        level: Optional[int] = None,
        semantic_type: Optional[str] = None,
        threshold: float = 0.0,
        tags: Optional[list[str]] = None,  # NEW: filter by document tags
    ) -> list[tuple[EmbeddedChunk, float]]: ...
```

### Tag Normalization

Tags are normalized on storage:

```python
def normalize_tags(tags: Optional[list[str]]) -> list[str]:
    """Normalize tags to lowercase with stripped whitespace."""
    if not tags:
        return []
    return [t.lower().strip() for t in tags]
```

### Filtering Logic

**AND logic for multiple tags:**
```python
def matches_tags(doc_tags: list[str], filter_tags: list[str]) -> bool:
    """Check if document has ALL specified tags."""
    normalized_filter = {t.lower().strip() for t in filter_tags}
    return normalized_filter.issubset(set(doc_tags))
```

---

## CLI Usage

### Ingestion

```bash
# Ingest with tags
kidkazz ingest markdown inventory_book.md --doc-id inv-book --tags inventory,accounting

# Batch ingest with shared tags
kidkazz ingest batch ./docs --pattern "*.md" --tags documentation
```

### Search

```bash
# Search within tagged documents
kidkazz search semantic "safety stock calculation" --tags inventory

# Multiple tags (AND logic)
kidkazz search semantic "depreciation" --tags inventory,accounting
```

### Document Management

```bash
# List documents by tag
kidkazz docs list --tags inventory

# Multiple tags filter
kidkazz docs list --tags inventory,best-practices
```

---

## MCP Integration

### search_semantic Tool

```python
@mcp.tool()
def search_semantic(
    query: str,
    top_k: int = 5,
    doc_id: Optional[str] = None,
    level: Optional[int] = None,
    semantic_type: Optional[str] = None,
    threshold: float = 0.0,
    tags: Optional[list[str]] = None,  # NEW
) -> list[dict[str, Any]]:
    """Search with optional tag filtering."""
```

### list_documents Tool

```python
@mcp.tool()
def list_documents(
    tags: Optional[list[str]] = None,  # NEW
) -> list[dict[str, Any]]:
    """List documents with optional tag filtering."""
```

---

## Test Coverage

**`tests/test_storage_mock.py` - TestDocumentTags class:**

| Test | Description |
|------|-------------|
| `test_stores_document_with_tags` | Tags stored correctly |
| `test_normalizes_tags_to_lowercase` | Tags normalized |
| `test_stores_empty_tags_by_default` | Default empty list |
| `test_list_documents_returns_tags` | Tags in list output |
| `test_list_documents_filters_by_single_tag` | Single tag filter |
| `test_list_documents_filters_by_multiple_tags_and_logic` | AND logic |
| `test_list_documents_tag_filter_case_insensitive` | Case-insensitive |
| `test_search_similar_filters_by_tags` | Search tag filter |
| `test_search_similar_multiple_tags_filter` | Search AND logic |

---

## Files Modified

| File | Changes |
|------|---------|
| `src/storage/schema.py` | Added `tags` to DOCUMENT_NODE_SCHEMA |
| `src/storage/mock_store.py` | Added tags storage and filtering |
| `src/storage/converters.py` | Added tags to document conversion |
| `src/storage/queries.py` | Added tags to AddDocument |
| `src/storage/client.py` | Updated protocol and HelixChunkStore |
| `src/cli/commands/ingest.py` | Added `--tags` option |
| `src/cli/commands/search.py` | Added `--tags` filter |
| `src/cli/commands/docs.py` | Added `--tags` filter to list |
| `src/mcp_server/tools.py` | Added tags to MCP tools |
| `src/mcp_server/formatters.py` | Added tags to document output |
| `tests/test_storage_mock.py` | Added TestDocumentTags class |

---

## Usage Examples

### Python API

```python
from src.storage import MockChunkStore

store = MockChunkStore()

# Store with tags
store.store_document(
    doc_id="inv-textbook",
    title="Inventory Management",
    embedded_chunks=chunks,
    metadata_list=metadata,
    tags=["inventory", "accounting"],
)

# List by tag
inventory_docs = store.list_documents(tags=["inventory"])

# Search with tag filter
results = store.search_similar(
    query_embedding=embedding,
    top_k=10,
    tags=["inventory"],
)
```

### MCP (Claude Code)

```
User: Search for safety stock calculation in inventory documents

Claude: [Calls search_semantic with tags=["inventory"]]
```

---

## Future Considerations

1. **OR logic**: Currently only AND logic. Could add `--tags-or` flag for OR logic.
2. **Tag management**: Add `kidkazz docs tag add/remove` commands.
3. **Tag autocomplete**: Shell completion for existing tags.
4. **Tag statistics**: Show tag distribution in `kidkazz db status`.
