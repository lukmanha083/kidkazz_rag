# Plan: Document Summarization Feature

## Overview

Add hierarchical document summarization to KidKazz RAG with LLM-powered summaries at document, chapter (L1), and section (L2) levels. Summaries are persisted in Helix-DB with embeddings for semantic search.

## Requirements

- **Summary Levels**: Document, Chapter (L1 chunks), Section (L2 chunks)
- **Storage**: Persist in Helix-DB with embeddings
- **Interfaces**: CLI (`kidkazz summarize`) + MCP tools

## Data Flow

```
Document Chunks (L0, L1, L2)
         │
         ▼
┌─────────────────────────┐
│  1. Summarize L2 chunks │  (Section summaries)
│     via LLM             │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  2. Aggregate L2 →      │
│     Summarize L1 chunks │  (Chapter summaries)
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  3. Aggregate L1 →      │
│     Summarize Document  │  (Document summary)
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Store in Helix-DB      │
│  with embeddings        │
└─────────────────────────┘
```

## Schema Changes (`db/schema.hx`)

```helix
N::Summary {
    INDEX summary_id: String,      // "summary_{source_id}_{level}"
    content: String,               // Summary text
    INDEX level: String,           // "document", "chapter", "section"
    INDEX source_id: String,       // doc_id or chunk_id
    INDEX document_id: String,     // For document filtering
    parent_summary_id: String,     // Hierarchy navigation
    key_points: String,            // JSON array
    word_count: U32,
    created_at: I64,
}

V::SummaryVector {
    model_name: String,
    embedding_dim: U32,
}

E::DocumentHasSummary { From: Document, To: Summary }
E::ChunkHasSummary { From: Chunk, To: Summary }
E::SummaryHasChild { From: Summary, To: Summary }
E::SummaryHasEmbedding { From: Summary, To: SummaryVector }
```

## Files to Create/Modify

### New Files

| File | Purpose | Lines |
|------|---------|-------|
| `src/chunker/summarizer.py` | DocumentSummarizer class | ~350 |
| `src/cli/commands/summarize.py` | CLI commands | ~280 |
| `tests/test_summarizer.py` | Summarizer tests | ~200 |
| `tests/test_summary_storage.py` | Storage tests | ~150 |

### Files to Extend

| File | Changes | Lines |
|------|---------|-------|
| `db/schema.hx` | Add Summary schema | +35 |
| `db/queries.hx` | Add HelixQL queries | +150 |
| `src/storage/queries.py` | Summary query classes | +300 |
| `src/storage/converters.py` | Summary converters | +100 |
| `src/storage/client.py` | Summary storage methods | +250 |
| `src/storage/mock_store.py` | Mock summary methods | +150 |
| `src/mcp_server/tools.py` | MCP summary tools | +150 |
| `src/mcp_server/formatters.py` | Summary formatters | +80 |
| `src/cli/config.py` | Summarization config | +30 |
| `src/cli/main.py` | Register summarize app | +5 |

**Total: ~2,300 lines**

## Core Classes

### Summary Dataclass (`src/chunker/summarizer.py`)

```python
@dataclass
class Summary:
    summary_id: str
    content: str
    level: str  # "document", "chapter", "section"
    source_id: str
    document_id: str
    parent_summary_id: Optional[str] = None
    key_points: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    word_count: int = 0
    created_at: int = 0
```

### DocumentSummarizer Class

```python
class DocumentSummarizer:
    def __init__(self, provider: str = "anthropic/claude-sonnet-4-20250514"):
        self.client = instructor.from_provider(provider)

    def summarize_section(chunk, metadata, doc_title) -> Summary
    def summarize_chapter(chunk, section_summaries, doc_title) -> Summary
    def summarize_document(doc_id, title, chapter_summaries) -> Summary
    def generate_all_summaries(doc_id, title, chunks, metadata) -> list[Summary]
```

## CLI Commands (`kidkazz summarize`)

| Command | Description |
|---------|-------------|
| `summarize generate <doc_id>` | Generate all summaries for document |
| `summarize show <doc_id> [--level]` | Display summary (document/chapter/section) |
| `summarize list` | List summarized documents |
| `summarize search "query"` | Search summaries semantically |
| `summarize delete <doc_id>` | Delete summaries for document |

## MCP Tools

| Tool | Description |
|------|-------------|
| `generate_summaries(doc_id)` | Generate hierarchical summaries |
| `get_document_summary(doc_id)` | Get document-level summary |
| `get_chapter_summaries(doc_id)` | Get chapter summaries |
| `get_section_summaries(doc_id)` | Get section summaries |
| `search_summaries(query, level)` | Semantic search summaries |
| `get_summary_hierarchy(doc_id)` | Get complete summary tree |

## Configuration (`.kidkazz.toml`)

```toml
[summarization]
enabled = true
provider = "anthropic/claude-sonnet-4-20250514"
max_tokens_per_summary = 500
include_key_points = true
```

## Implementation Steps

### Phase 1: Schema & Data Model
1. Add Summary dataclass to `src/chunker/summarizer.py`
2. Add schema to `db/schema.hx`
3. Add HelixQL queries to `db/queries.hx`
4. Run `helix push dev`

### Phase 2: Storage Layer
5. Add query classes to `src/storage/queries.py`
6. Add converters to `src/storage/converters.py`
7. Add methods to `src/storage/client.py`
8. Add methods to `src/storage/mock_store.py`
9. Write storage tests

### Phase 3: Summarizer Core
10. Implement `DocumentSummarizer` class
11. Add Pydantic models for structured LLM output
12. Implement hierarchical summarization
13. Add embedding integration
14. Write summarizer tests

### Phase 4: CLI Interface
15. Create `src/cli/commands/summarize.py`
16. Register in `src/cli/main.py`
17. Add config to `src/cli/config.py`
18. Write CLI tests

### Phase 5: MCP Integration
19. Add formatters to `src/mcp_server/formatters.py`
20. Add tools to `src/mcp_server/tools.py`
21. Write MCP tests

### Phase 6: Documentation
22. Update CLAUDE.md
23. Run full test suite

## Verification

```bash
# 1. Deploy schema
helix push dev

# 2. Run tests
PYTHONPATH=. pytest tests/test_summarizer.py tests/test_summary_storage.py -v

# 3. CLI test
kidkazz summarize generate <doc_id>
kidkazz summarize show <doc_id>
kidkazz summarize search "inventory methods"

# 4. MCP test (via Claude Code)
# Call generate_summaries, get_document_summary, search_summaries tools
```

## Critical Files

- `src/chunker/summarizer.py` - Core summarizer (follow `concept_extractor.py` pattern)
- `src/storage/client.py` - Storage methods (follow existing query patterns)
- `src/storage/queries.py` - Query classes (follow existing patterns)
- `db/schema.hx` - Helix schema
- `src/cli/commands/summarize.py` - CLI (follow `concepts.py` pattern)
