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

### Helix-DB Deployment (Two-Step Process)

**Important:** `helix push dev` regenerates `.helix/dev/docker-compose.yml` on every run, stripping any custom environment variables. We use custom env vars to limit Helix-DB thread count (see "Thread Limiting" below). Therefore deployment is a two-step process:

```bash
# Step 1: Compile queries & build image (required after db/queries.hx or db/schema.hx changes)
helix push dev

# Step 2: Re-add env vars and restart with them
# Edit .helix/dev/docker-compose.yml to add env vars (see below), fix port to 6969:6969, then:
docker stop helix-kidkazz_rag-dev_app && docker rm helix-kidkazz_rag-dev_app
cd .helix/dev && docker compose up -d
```

**When to use each command:**
- **Schema/query changes** (`db/queries.hx`, `db/schema.hx`): Must run `helix push dev` first (compiles HQL → Rust, rebuilds Docker image), then Step 2
- **Just restarting** (no schema changes): `cd .helix/dev && docker compose up -d` (preserves env vars)
- **Never use `helix push dev` alone** — it strips thread-limiting env vars

**Thread-limiting env vars** (add to `.helix/dev/docker-compose.yml` after each `helix push`):
```yaml
environment:
  # ... existing vars ...
  - HELIX_CORES_OVERRIDE=2    # Gateway: 2*8=16 threads (default: nproc*8=128)
  - TOKIO_WORKER_THREADS=2    # Tokio async runtime
  - RAYON_NUM_THREADS=2        # Rayon parallel compute
```

Without these, Helix-DB detects 16 cores via `nproc` and spawns 128 worker threads, causing CPU spikes even with Docker CPU limits (cgroups limit CPU time, not thread count).

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
kidkazz summarize generate doc_id   # Generate summaries + extract concepts
kidkazz summarize show doc_id       # Show document summary
kidkazz summarize search "query"    # Search summaries semantically
kidkazz summarize list              # List documents with summaries
```

## Architecture

**Pipeline flow:** PDF → Markdown → Hierarchical Chunks → Embeddings → Helix-DB Storage → MCP/CLI Search

### Core Modules

- **`src/chunker/`** - Hierarchical chunking with graph relationships (parent/child/sibling). Chunks maintain metadata: semantic type, section paths, special content flags (has_table, has_code, has_math, has_image), and header metadata (header_text, header_level, block_type) for improved semantic search.

- **`src/chunker/concept_extractor.py`** - LLM-powered concept extraction using Instructor. Features:
  - Header-aware prompts (includes header_text, header_level in context)
  - Semantic type filtering via `filter_chunks_by_semantic_type()`
  - Header hierarchy relationship inference via `infer_relationships_from_headers()`

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
  - Checkpointing for resume-from-failure (chapter/document phases only)
  - `generate_all_summaries()` for full hierarchical summarization + concept extraction

- **`src/chunker/embedder.py`** - Embedder implementations:
  - `CohereEmbedder`: Cohere Embed v4 with Matryoshka dims (256/512/1024/1536), multimodal support
  - `OpenAIEmbedder`: API-based, 1536/3072 dims
  - `MockEmbedder`: Testing
  - `embed_multimodal()`: Fused text+image embedding via Cohere (shared vector space)

- **`src/storage/`** - Protocol-based storage with `ChunkStoreProtocol`:
  - `MockChunkStore`: In-memory for testing
  - `HelixChunkStore`: Production vector + graph DB

- **`src/mcp_server/`** - FastMCP server exposing 27+ search tools and 4 resource endpoints for Claude Code integration:
  - Chunk tools: `search_semantic` (with filters: has_table, has_code, has_math, has_image, header_level; `include_images` flag for multimodal response), `search_keyword`, `get_chunk`, `get_chunk_images`, `get_context_window`, `get_parent`, `get_children`, `get_siblings`, `list_documents`, `get_document_chunks`, `get_document_stats`
  - Concept tools: `search_concepts`, `get_concept`, `get_concept_with_citations`, `get_related_concepts`, `explain_concept_cross_document`, `get_concept_graph_dot`, `list_concepts`
  - Summary tools: `get_document_summary`, `get_chapter_summaries`, `get_section_summaries`, `search_summaries`, `get_summary_hierarchy`, `list_summarized_documents`

- **`src/cli/`** - Typer-based CLI with Rich formatting. Commands: `ingest`, `search`, `docs`, `db`, `config`, `inbox`, `concepts`, `summarize`

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
- `has_image: bool` - Only chunks containing images (detected via `![alt](path)` markdown syntax)
- `header_level: int` - Only chunks at specific header level (1-6)

### Multimodal Image Response (MCP)
The MCP server can return actual table/figure images to Claude, not just text:
- `search_semantic(..., include_images=True)` — returns text metadata interleaved with `ImageContent` blocks (FastMCP auto-converts `Image` objects)
- `get_chunk_images(chunk_id)` — dedicated tool to fetch images from a specific chunk
- `extract_chunk_images()` in `formatters.py` scans chunk content for `![alt](path)` markers and returns `Image` objects for existing PNG/JPG files on disk
- Backward compatible: `include_images` defaults to `False`, existing behavior unchanged

### Multimodal Image Embedding
Tables and figures from PDFs are embedded as images using Cohere Embed v4's multimodal capability:
- Reducto's `return_images=["figure", "table"]` renders blocks as PNG images
- During `kidkazz inbox parse`, images are downloaded to `~/.kidkazz/images/<doc_id>/`
- During `kidkazz ingest markdown --images-dir`, chunks with image markers get fused text+image embeddings
- Text queries naturally match image embeddings (shared 1536-dim vector space)
- The `has_table` and `has_image` metadata flags enable filtering in search
- MCP tools can return actual images to Claude via `include_images=True` or `get_chunk_images`

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
