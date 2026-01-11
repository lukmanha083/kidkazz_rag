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
pip install -e ".[chunker]"  # Embeddings (FastEmbed)
pip install -e ".[helixdb]"  # Storage (helix-py)
pip install -e ".[mcp]"      # MCP server
pip install -e ".[cli]"      # CLI (typer, rich)
pip install -e ".[reducto]"  # Reducto.ai parsing
pip install -e ".[openai]"   # OpenAI embeddings
pip install -e ".[all]"      # Everything
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
```

## Architecture

**Pipeline flow:** PDF → Markdown → Hierarchical Chunks → Embeddings → Helix-DB Storage → MCP/CLI Search

### Core Modules

- **`src/chunker/`** - Hierarchical chunking with graph relationships (parent/child/sibling). Chunks maintain metadata: semantic type, section paths, special content flags (has_table, has_code, has_math), and header metadata (header_text, header_level, block_type) for improved semantic search.

- **`src/chunker/embedder.py`** - Three embedder implementations:
  - `ChunkEmbedder` (FastEmbed): Local CPU, ONNX Runtime, 384/768/1024 dims
  - `OpenAIEmbedder`: API-based, 1536/3072 dims
  - `MockEmbedder`: Testing

- **`src/storage/`** - Protocol-based storage with `ChunkStoreProtocol`:
  - `MockChunkStore`: In-memory for testing
  - `HelixChunkStore`: Production vector + graph DB

- **`src/mcp_server/`** - FastMCP server exposing 10 search tools and 4 resource endpoints for Claude Code integration

- **`src/cli/`** - Typer-based CLI with Rich formatting. Commands: `ingest`, `search`, `docs`, `db`, `config`, `inbox`

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

## Configuration Files

- **`.kidkazz.toml`** - Project configuration (storage, embeddings, chunking, inbox)
- **`.env`** - Secrets only (REDUCTO_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY)
- **`pyproject.toml`** - Dependencies and pytest configuration

## Test Markers

Tests use pytest markers to control execution:
- `@pytest.mark.helix` - Requires running Helix-DB server
- `@pytest.mark.mcp` - MCP-related tests
- `@pytest.mark.fastembed` - Requires FastEmbed (slow)
