# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KidKazz RAG is a production-grade Retrieval-Augmented Generation system for converting PDF textbooks into searchable knowledge bases. It provides semantic and keyword search through documents via Claude Code (MCP integration) or CLI.

## Commands

### Testing
```bash
PYTHONPATH=. pytest tests/ -v                              # Run all tests
PYTHONPATH=. pytest tests/test_chunker.py -v              # Single test file
PYTHONPATH=. pytest tests/ -k "test_parse_tags"           # Run specific test by name
PYTHONPATH=. pytest tests/ --cov=src --cov-report=term-missing  # With coverage
pytest tests/ -m "not helix"                               # Skip Helix-DB tests (requires server)
pytest tests/ -m "not mcp"                                 # Skip MCP tests
```

### Installation (optional dependencies)
```bash
pip install -e ".[dev]"      # Testing (pytest, coverage)
pip install -e ".[chunker]"  # Chunking pipeline
pip install -e ".[helixdb]"  # Storage (helix-py)
pip install -e ".[mcp]"      # MCP server
pip install -e ".[cli]"      # CLI (typer, rich)
pip install -e ".[reducto]"  # Reducto.ai parsing
pip install -e ".[openai]"   # OpenAI embeddings
pip install -e ".[concepts]"    # Concept extraction & summarization (instructor)
pip install -e ".[extractive]"  # Extractive summarization (sumy, yake, nltk)
pip install -e ".[all]"        # Everything
```

### Helix-DB Setup (Production Storage)
Requires Rust 1.88.0+ and Docker Desktop.
```bash
curl -sSL https://install.helix-db.com | bash
helix init
helix push dev  # Starts on port 6969
```

### CLI (requires `.[cli]` installed)
```bash
kidkazz ingest markdown file.md --tags "inventory,accounting"
kidkazz ingest batch ./docs/ --pattern "*.md" --recursive
kidkazz search semantic "query text" --limit 10 --tags "inventory"
kidkazz docs list
kidkazz inbox status
kidkazz inbox list                  # List PDFs in inbox
kidkazz inbox list --output         # List parsed markdown (status + quality)
kidkazz inbox sync --dry-run        # Preview cloud sync
kidkazz inbox parse                 # Parse PDFs with Reducto.ai
kidkazz config show
kidkazz summarize generate doc_id   # Generate summaries + extract concepts (includes table parsing)
kidkazz summarize show doc_id       # Show document summary
kidkazz summarize search "query"    # Search summaries semantically
kidkazz summarize list              # List documents with summaries
```

## Architecture

**Pipeline flow:** PDF → Markdown → Hierarchical Chunks → Embeddings → Helix-DB Storage → MCP/CLI Search

### Core Modules

- **`src/chunker/`** - Hierarchical chunking with graph relationships (parent/child/sibling). Chunks maintain metadata: semantic type, section paths, special content flags (has_table, has_code, has_math), and header metadata (header_text, header_level, block_type) for improved semantic search.

- **`src/chunker/concept_extractor.py`** - LLM-powered concept extraction using Instructor. Features:
  - Header-aware prompts (includes header_text, header_level in context)
  - Semantic type filtering via `filter_chunks_by_semantic_type()`
  - Header hierarchy relationship inference via `infer_relationships_from_headers()`

- **`src/chunker/table_parser.py`** - Markdown table parsing into structured form:
  - `ParsedTable` dataclass with columns, rows, types, context
  - `parse_markdown_table()` for extracting tables from chunks
  - `to_markdown_kv()` for LLM-friendly format (60.7% accuracy)
  - `infer_column_types()` for detecting text/numeric/date columns

- **`src/chunker/table_summarizer.py`** - LLM-powered table summarization:
  - `TableSummary` dataclass for storing summaries with embeddings
  - `TableSummarizer` class with Anthropic/OpenAI support
  - Key column/value extraction for retrieval hints

- **`src/chunker/extractive.py`** - Local extractive summarization (no LLM required):
  - `extractive_summarize()`: TextRank via sumy for key sentence extraction
  - `extract_keywords()`: YAKE unsupervised keyword extraction
  - `_fallback_summarize()`: Naive first-N-sentences when libs unavailable
  - Auto-downloads NLTK `punkt_tab` data if missing

- **`src/chunker/summarizer.py`** - Tiered document summarization:
  - `Summary` dataclass with hierarchy (document → chapter → section)
  - `DocumentSummarizer` class using Instructor for structured output
  - Default provider: `openai/gpt-4o-mini` (cost-effective)
  - **Tiered strategy** (82% API call reduction):
    - Section (L2): Local extractive (TextRank + YAKE keywords, instant, free)
    - Chapter (L1): LLM via Instructor (structured output with concepts)
    - Document: LLM via Instructor (structured output with concepts)
  - **Adaptive processing strategy** based on L1 (chapter) count:
    - < 50 chapters: Sequential (simple, no overhead)
    - 50-500 chapters: Async with rate limiting (parallel, respects limits)
    - > 500 chapters: OpenAI Batch API (50% cheaper, higher limits)
  - **Table parsing integration**: Chunks with `has_table=True` get structured table data (columns, rows, types) for better concept extraction
  - Checkpointing for resume-from-failure (chapter/document phases only)
  - `generate_all_summaries()` for full hierarchical summarization + concept extraction

- **`src/storage/table_store.py`** - Multi-vector table storage:
  - Embed summaries for retrieval, store raw markdown for synthesis
  - `search_tables()` with cosine similarity ranking
  - `link_table_to_concept()` for graph-based retrieval

- **`src/chunker/embedder.py`** - Embedder implementations:
  - `OpenAIEmbedder`: API-based, 1536/3072 dims (default)
  - `MockEmbedder`: Testing

- **`src/storage/`** - Protocol-based storage with `ChunkStoreProtocol`:
  - `MockChunkStore`: In-memory for testing
  - `HelixChunkStore`: Production vector + graph DB

- **`src/mcp_server/`** - FastMCP server exposing 25+ search tools and 4 resource endpoints for Claude Code integration:
  - Chunk tools: `search_semantic` (with filters: has_table, has_code, has_math, header_level), `search_keyword`, `get_chunk`, `get_context_window`, `get_parent`, `get_children`, `get_siblings`, `list_documents`, `get_document_chunks`, `get_document_stats`
  - Concept tools: `search_concepts`, `get_concept`, `get_related_concepts`, `get_concept_chunks`, `explain_concept_with_context`
  - Table tools: `search_tables`, `get_table`, `get_tables_for_concept`, `list_tables`
  - Summary tools: `get_document_summary`, `get_chapter_summaries`, `get_section_summaries`, `search_summaries`, `get_summary_hierarchy`, `list_summarized_documents`

- **`src/cli/`** - Typer-based CLI with Rich formatting. Commands: `ingest`, `search`, `docs`, `db`, `config`, `inbox`, `concepts`, `tables`, `summarize`

- **`src/pdf_inbox/`** - PDF lifecycle management with Reducto.ai integration and optional rclone cloud sync

## Critical Design Patterns

### Embedder Dimension Compatibility
Different embedders produce different vector dimensions. **Changing embedders requires re-ingesting all documents.** Always use the same embedder for ingestion and search.

### Document Tagging
- Tags are document-level (inherited by all chunks)
- Case-insensitive, normalized to lowercase
- Multiple tags use AND logic (document must have ALL specified tags)
- Stored as JSON string in database: `["inventory", "accounting"]`

### Configuration Priority
CLI args > `KIDKAZZ_*` env vars > `.kidkazz.toml` (project) > `~/.config/kidkazz/config.toml` (user) > defaults

### Graph-Aware Chunking
Chunks maintain relationships: `parent_id`, `children_ids`, `sibling_ids`, `prev_chunk_id`, `next_chunk_id`. This enables context retrieval across chunk boundaries.

### Header Metadata Extraction
When using Reducto.ai with block/chunk mode, header information is extracted from block metadata and reconstructed as markdown syntax. The ingestion pipeline then extracts header metadata from markdown content:
- `header_text`: The actual header text (without `#` prefix)
- `header_level`: 1-6 for h1-h6 (stored as 0 if not a header)
- `block_type`: Block type from Reducto (e.g., "Header", "Text", "Table")

**Re-parsing required**: To leverage header detection, existing PDFs must be re-parsed (not just re-ingested) since the markdown files need proper header syntax.

### Content Flag Filters
The `search_semantic` MCP tool supports filtering by content flags:
- `has_table: bool` - Only chunks containing tables
- `has_code: bool` - Only chunks containing code blocks
- `has_math: bool` - Only chunks containing math expressions
- `header_level: int` - Only chunks at specific header level (1-6)

### Table Storage
Tables use in-memory multi-vector storage (`TableStore`):
- Summaries are embedded for semantic search
- Raw markdown is stored for LLM synthesis
- Concept-table relationships are tracked for graph traversal
- **Note**: Table storage is currently in-memory only, not persisted to Helix-DB

### Document Summarization
Hierarchical summaries use a tiered approach - local extractive for sections, LLM for chapters/documents:
- **Document level**: 5-7 sentence LLM summary of entire document
- **Chapter level**: 3-5 sentence LLM summary per L1 chunk (with concept extraction)
- **Section level**: Extractive summary via TextRank + YAKE keywords (no API calls)
- Default LLM provider: `openai/gpt-4o-mini` (configurable)
- For a 1302-chunk document: ~197 API calls instead of ~1107 (82% reduction)
- Summaries are embedded for semantic search via `SummaryVector`
- Parent-child relationships enable hierarchy navigation

## Configuration Files

- **`.kidkazz.toml`** - Project configuration (storage, embeddings, chunking, inbox)
- **`.env`** - Secrets only (REDUCTO_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY)
- **`pyproject.toml`** - Dependencies and pytest configuration

## Testing Policy

- **Use real OpenAI API embeddings** in tests — do NOT use mock embedders with fake dimensions. The OPENAI_API_KEY is available in `.env`. Tests that need embeddings should use `OpenAIEmbedder` with `text-embedding-3-small` (1536 dims) to match production.
- **All tests must pass.** Do not leave broken tests behind. Fix tests when interfaces change, or remove tests that test removed/obsolete functionality.
- **No mock-only embedding tests.** MockEmbedder (384 dims) causes dimension mismatches with the real DB. Only use it for unit tests that never touch storage.

## Test Markers

Tests use pytest markers to control execution:
- `@pytest.mark.helix` - Requires running Helix-DB server
- `@pytest.mark.mcp` - MCP-related tests
- `@pytest.mark.openai` - Requires OpenAI API key
