# Kidkazz RAG

A RAG (Retrieval-Augmented Generation) system for converting PDF textbooks to searchable knowledge bases.

## Overview

This project provides tools to:
1. Convert PDF documents to Markdown (with table and image support)
2. Chunk documents hierarchically for vector + graph storage
3. Generate embeddings using CPU-optimized FastEmbed
4. Store in Helix-DB (vector + graph database)
5. Query documents via MCP integration with Claude Code

## Current Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | PDF to Markdown Converter | ✅ Complete |
| 2 | Hierarchical Chunking Pipeline | ✅ Complete |
| 3 | Helix-DB Integration | ✅ Complete |
| 4 | MCP Server | ✅ Complete |
| 5 | End-to-End CLI Tool | ✅ Complete |

## Architecture

```text
PDF Document
     |
     v
[PDF to Markdown Converter] ──── Google Colab (GPU)
     |                           ├── Marker (fast, clean PDFs)
     |                           ├── Docling (tables, mixed)
     |                           └── Nougat (math/equations)
     v
Markdown Document
     |
     v
[Hierarchical Chunker] ───────── Local Python (CPU)
     |                           ├── Markdown structure parsing
     |                           ├── Multi-level chunks (doc→section→leaf)
     |                           ├── Graph relationships (parent/child/sibling)
     |                           └── Semantic type detection
     v
[FastEmbed] ──────────────────── CPU-optimized embeddings
     |                           └── BAAI/bge-small-en-v1.5 (384 dims)
     v
[Helix-DB Storage] ─────────── Vector + Graph Storage
     |                           ├── MockChunkStore (testing)
     |                           └── HelixChunkStore (production)
     |
     ├─────────────────────────────────────────────────────────┐
     v                                                         v
[MCP Server] ────────────────── Claude Code Integration  [CLI Tool] ── Command Line
     |                           ├── 10 search/retrieval tools  |        ├── kidkazz ingest
     |                           └── 4 resource endpoints       |        ├── kidkazz search
     v                                                         v        ├── kidkazz docs
Chat with your documents                               Terminal access  └── kidkazz config
```

## Prerequisites

- Python 3.10+
- VS Code with [Google Colab Extension](https://marketplace.visualstudio.com/items?itemName=GoogleCloudPlatform.colab-vscode-plugin) (for PDF conversion)
- Google account (for Colab GPU access)

## Installation

```bash
# Clone the repository
git clone https://github.com/lukmanha083/kidkazz_rag.git
cd kidkazz_rag

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"           # Development (pytest, coverage)
pip install -e ".[chunker]"       # Chunking + embeddings (fastembed)
pip install -e ".[helixdb]"       # Helix-DB storage (helix-py)
pip install -e ".[mcp]"           # MCP server for Claude Code
pip install -e ".[cli]"           # CLI tool (typer, rich)
pip install -e ".[all]"           # All dependencies
```

## Quick Start

### Step 1: PDF to Markdown Conversion (Colab)

1. Open `notebooks/pdf_to_markdown_converter.ipynb` in VS Code
2. Connect to Colab runtime with GPU
3. Upload your PDF and run the cells
4. Download the converted Markdown

### Step 2: Chunk and Embed (Local)

```python
from src.chunker import (
    parse_markdown_structure,
    create_hierarchical_chunks,
    enrich_all_chunks,
    ChunkEmbedder,  # or MockEmbedder for testing
)

# Load markdown from Phase 1
with open("output/textbook.md") as f:
    content = f.read()

# Parse document structure
doc_structure = parse_markdown_structure(content)

# Create hierarchical chunks (3 levels)
chunks = create_hierarchical_chunks(
    content,
    level_sizes=(2048, 512, 256),  # Level 1, Level 2, overlap
    doc_id="textbook"
)

# Enrich with metadata
metadata = enrich_all_chunks(chunks, document_id="textbook")

# Generate embeddings (CPU-optimized)
embedder = ChunkEmbedder(model_name="BAAI/bge-small-en-v1.5")
embedded_chunks = embedder.embed_chunks(chunks, batch_size=32)

# Ready for storage
for ec in embedded_chunks:
    print(f"Chunk {ec.chunk.id}: {ec.embedding_dim} dims")
    print(f"  Level: {ec.chunk.level}")
    print(f"  Parent: {ec.chunk.parent_id}")
    print(f"  Children: {len(ec.chunk.child_ids)}")
```

### Step 3: Store and Search (Local)

```python
from src.storage import MockChunkStore  # or HelixChunkStore for production

# Store document with chunks and embeddings
store = MockChunkStore()
store.store_document(
    doc_id="textbook",
    title="My Textbook",
    embedded_chunks=embedded_chunks,
    metadata_list=metadata,
)

# Vector similarity search
query_embedding = embedder.embed_text("What is machine learning?")
results = store.search_similar(query_embedding, top_k=5)

for chunk, score in results:
    print(f"[{score:.3f}] {chunk.chunk.content[:100]}...")

# Graph traversal - expand context
top_chunk = results[0][0]
parent = store.get_parent(top_chunk.chunk.id)
siblings = store.get_siblings(top_chunk.chunk.id)
context = store.get_context_window(top_chunk.chunk.id, window_size=2)
```

### Step 4: Query via MCP Server (Claude Code)

1. Copy the MCP configuration template:
```bash
cp .mcp.json.example .mcp.json
```

2. Start the MCP server (Claude Code will do this automatically):
```bash
python -m src.mcp_server
# Or use the installed command:
kidkazz-mcp
```

3. Use from Claude Code - available tools:
- `search_semantic` - Vector similarity search
- `search_keyword` - Full-text keyword search
- `get_chunk` - Get chunk by ID
- `get_context_window` - Get chunk with neighbors
- `get_parent` / `get_children` / `get_siblings` - Graph traversal
- `list_documents` / `get_document_chunks` / `get_document_stats` - Document management

### Step 5: Use the CLI Tool (Alternative to MCP)

The CLI tool provides command-line access to all RAG functionality:

```bash
# Install CLI dependencies
pip install -e ".[cli]"

# Ingest a markdown document
kidkazz ingest markdown output/textbook.md --doc-id textbook --title "My Textbook"

# Batch ingest a directory
kidkazz ingest batch ./docs --pattern "*.md"

# Semantic search
kidkazz search semantic "What is machine learning?" --top-k 5

# Keyword search
kidkazz search keyword "neural network" --json

# List documents
kidkazz docs list

# Get document statistics
kidkazz docs stats textbook

# Database operations
kidkazz db status
kidkazz db init

# View/modify configuration
kidkazz config show
kidkazz config set store_type mock
```

All commands support `--json` for machine-readable output and `--help` for detailed usage.

## Project Structure

```text
kidkazz_rag/
├── README.md
├── pyproject.toml
├── notebooks/
│   └── pdf_to_markdown_converter.ipynb   # Phase 1: PDF → Markdown
├── src/
│   ├── pdf_converter/                     # Phase 1: PDF conversion logic
│   │   ├── __init__.py
│   │   ├── analyzer.py                    # Markdown quality analysis
│   │   ├── converter.py                   # PDF conversion wrapper
│   │   └── selector.py                    # Tool recommendation
│   ├── chunker/                           # Phase 2: Chunking pipeline
│   │   ├── __init__.py                    # Public API
│   │   ├── parser.py                      # Markdown structure extraction
│   │   ├── chunker.py                     # Hierarchical chunking
│   │   ├── metadata.py                    # Metadata enrichment
│   │   └── embedder.py                    # FastEmbed integration
│   ├── storage/                           # Phase 3: Helix-DB integration
│   │   ├── __init__.py                    # Public API
│   │   ├── schema.py                      # Helix-DB schema definitions
│   │   ├── converters.py                  # Dataclass <-> Helix format
│   │   ├── queries.py                     # HelixQL query classes
│   │   ├── mock_store.py                  # In-memory MockChunkStore
│   │   └── client.py                      # HelixChunkStore client
│   ├── mcp_server/                        # Phase 4: MCP server
│   │   ├── __init__.py                    # Public API
│   │   ├── config.py                      # Configuration management
│   │   ├── formatters.py                  # Response formatting
│   │   ├── tools.py                       # MCP tool implementations
│   │   ├── resources.py                   # MCP resource implementations
│   │   ├── server.py                      # FastMCP server setup
│   │   └── __main__.py                    # Entry point
│   └── cli/                               # Phase 5: CLI tool
│       ├── __init__.py                    # Public API
│       ├── main.py                        # Typer app entry point
│       ├── config.py                      # Configuration management
│       ├── output.py                      # Rich terminal formatting
│       ├── progress.py                    # Progress indicators
│       ├── utils.py                       # Utility functions
│       └── commands/                      # Command implementations
│           ├── __init__.py
│           ├── ingest.py                  # ingest markdown/batch/pdf
│           ├── search.py                  # search semantic/keyword
│           ├── docs.py                    # docs list/stats/export/delete
│           ├── db.py                      # db init/status/clear
│           └── config_cmd.py              # config show/set/reset/init
├── tests/
│   ├── conftest.py                        # Shared fixtures
│   ├── test_analyzer.py                   # Phase 1 tests
│   ├── test_converter.py
│   ├── test_selector.py
│   ├── test_parser.py                     # Phase 2 tests
│   ├── test_chunker.py
│   ├── test_metadata.py
│   ├── test_embedder.py
│   ├── test_storage_schema.py             # Phase 3 tests
│   ├── test_storage_converters.py
│   ├── test_storage_mock.py
│   ├── test_storage_queries.py
│   ├── test_storage_integration.py
│   ├── test_mcp_config.py                 # Phase 4 tests
│   ├── test_mcp_formatters.py
│   ├── test_mcp_tools.py
│   ├── test_mcp_resources.py
│   ├── test_mcp_server.py
│   ├── test_mcp_integration.py
│   ├── test_cli_config.py                 # Phase 5 tests
│   ├── test_cli_utils.py
│   ├── test_cli_output.py
│   ├── test_cli_progress.py
│   ├── test_cli_commands.py
│   └── test_cli_integration.py
└── docs/
    ├── design/
    │   ├── PLAN_PHASE2.md                 # Phase 2 design document
    │   ├── PLAN_PHASE3.md                 # Phase 3 design document
    │   ├── PLAN_PHASE4.md                 # Phase 4 design document
    │   └── PLAN_PHASE5.md                 # Phase 5 design document
    └── testing/
        └── UNIT_TEST.md                   # Testing guide
```

## Chunking Pipeline Details

### Hierarchical Chunk Levels

| Level | Token Size | Purpose | Graph Role |
|-------|------------|---------|------------|
| 0 | Full doc | Reference | Document node |
| 1 | 1024-2048 | Context synthesis | Section nodes |
| 2 | 256-512 | Vector search | Leaf nodes |

### Graph Relationships

Each chunk maintains relationships for Helix-DB graph traversal:

- `parent_id` - Parent chunk (higher level)
- `child_ids` - Child chunks (lower level)
- `prev_id` / `next_id` - Sequential order
- `sibling_ids` - Same parent (computed in metadata)

### Semantic Type Detection

Chunks are classified automatically:

| Type | Detection Pattern |
|------|-------------------|
| `definition` | "is defined as", "refers to", "is a" |
| `example` | "for example", "for instance", "e.g." |
| `procedure` | "step 1", "first", "then", "finally" |
| `theorem` | "Theorem:", "Proof:", "Lemma:" |
| `narrative` | Default for general content |

### Atomic Block Preservation

The chunker never splits:
- Code blocks (``` or ~~~)
- Tables (markdown tables)
- Math blocks ($$ or \[\])

## Storage Layer Details

### Helix-DB Schema

The storage layer uses a three-node graph structure:

| Node Type | Purpose | Key Fields |
|-----------|---------|------------|
| Document | Container | doc_id, title, chunk_count |
| Chunk | Content + metadata | chunk_id, content, level, semantic_type |
| ChunkVector | Embedding | embedding (384 dims) |

### Edge Relationships

| Edge Type | From → To | Purpose |
|-----------|-----------|---------|
| HasChunk | Document → Chunk | Document contains chunks |
| ParentOf | Chunk → Chunk | L1 → L2 hierarchy |
| NextSibling | Chunk → Chunk | Sequential order |
| HasEmbedding | Chunk → ChunkVector | Chunk has embedding |

### Storage Implementations

| Implementation | Use Case | Database Required |
|----------------|----------|-------------------|
| MockChunkStore | Unit testing, development | No |
| HelixChunkStore | Production | Yes (Helix-DB server) |

Both implementations share the same `ChunkStoreProtocol` interface for easy swapping.

## MCP Server Details

### Available Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `search_semantic` | Vector similarity search | query, top_k, doc_id, level, semantic_type, threshold |
| `search_keyword` | Full-text keyword search | keyword, doc_id, case_sensitive |
| `get_chunk` | Get chunk by ID | chunk_id |
| `get_context_window` | Get chunk with neighbors | chunk_id, window_size |
| `get_parent` | Navigate to parent chunk | chunk_id |
| `get_children` | Get child chunks | chunk_id |
| `get_siblings` | Get sibling chunks | chunk_id |
| `list_documents` | List all documents | - |
| `get_document_chunks` | Get all chunks from doc | doc_id, level |
| `get_document_stats` | Get document statistics | doc_id |

### Available Resources

| Resource URI | Description |
|--------------|-------------|
| `kidkazz://schema` | Knowledge base schema and available tools |
| `kidkazz://documents` | List of all documents |
| `kidkazz://document/{doc_id}` | Document overview and statistics |
| `kidkazz://chunk/{chunk_id}` | Chunk content and metadata |

### Configuration

Environment variables for MCP server:

| Variable | Default | Description |
|----------|---------|-------------|
| `KIDKAZZ_STORE_TYPE` | `mock` | Storage backend: `mock` or `helix` |
| `KIDKAZZ_HELIX_PORT` | `6969` | Helix-DB port (if using helix) |
| `KIDKAZZ_HELIX_LOCAL` | `true` | Connect to local Helix-DB |
| `KIDKAZZ_EMBEDDER_TYPE` | `fastembed` | Embedder: `mock` or `fastembed` |
| `KIDKAZZ_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Embedding model name |
| `KIDKAZZ_LOG_LEVEL` | `INFO` | Logging level |

### Claude Code Integration

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "kidkazz-rag": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/kidkazz_rag",
      "env": {
        "KIDKAZZ_STORE_TYPE": "mock",
        "KIDKAZZ_EMBEDDER_TYPE": "fastembed"
      }
    }
  }
}
```

## CLI Tool Details

The CLI tool (`kidkazz`) provides command-line access to all RAG functionality with rich terminal output.

### Command Groups

| Command | Subcommands | Description |
|---------|-------------|-------------|
| `kidkazz ingest` | markdown, batch, pdf | Ingest documents into knowledge base |
| `kidkazz search` | semantic, keyword | Search the knowledge base |
| `kidkazz docs` | list, stats, export, delete | Manage documents |
| `kidkazz db` | init, status, clear | Database operations |
| `kidkazz config` | show, set, reset, init | Configuration management |

### Ingest Commands

```bash
# Ingest a single markdown file
kidkazz ingest markdown document.md --doc-id my_doc --title "My Document"

# Preview without ingesting
kidkazz ingest markdown document.md --dry-run

# Custom chunk sizes (level1,level2,overlap)
kidkazz ingest markdown document.md --chunk-sizes 1024,256,128

# Batch ingest a directory
kidkazz ingest batch ./docs --pattern "*.md" --recursive

# PDF ingestion (redirects to Colab notebook)
kidkazz ingest pdf document.pdf
```

### Search Commands

```bash
# Semantic search (vector similarity)
kidkazz search semantic "What is machine learning?" --top-k 5

# Filter by document, level, or type
kidkazz search semantic "neural networks" --doc-id textbook --level 2

# Keyword search
kidkazz search keyword "supervised learning" --case-sensitive

# JSON output for scripting
kidkazz search semantic "example" --json
```

### Document Management

```bash
# List all documents
kidkazz docs list --json

# Get document statistics
kidkazz docs stats textbook

# Export document chunks
kidkazz docs export textbook --format json --output chunks.json

# Delete a document
kidkazz docs delete textbook --force
```

### Database Operations

```bash
# Initialize database
kidkazz db init

# Check database status
kidkazz db status --json

# Clear all data
kidkazz db clear --force
```

### Configuration

The CLI uses a layered configuration system:

1. **CLI arguments** (highest priority)
2. **Environment variables** (`KIDKAZZ_*`)
3. **Project config** (`.kidkazz.toml` in current directory)
4. **User config** (`~/.config/kidkazz/config.toml`)
5. **Defaults** (lowest priority)

```bash
# Show current configuration
kidkazz config show

# Set a configuration value
kidkazz config set store_type helix
kidkazz config set helix_port 6969

# Reset to defaults
kidkazz config reset

# Create project config file
kidkazz config init
```

### Configuration File Format

`.kidkazz.toml`:

```toml
[storage]
store_type = "mock"       # mock or helix
helix_port = 6969
helix_local = true

[embedder]
embedder_type = "fastembed"  # mock or fastembed
model_name = "BAAI/bge-small-en-v1.5"

[chunking]
level_1_size = 2048
level_2_size = 512
overlap = 256
```

## Tool Selection Guide

| Your PDF Has | Recommended Tool |
|--------------|------------------|
| Heavy math/equations | Nougat |
| Many tables/data | Docling |
| Mostly text | Marker (fastest) |
| Mixed content | Docling |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific phase
python -m pytest tests/test_analyzer.py tests/test_converter.py -v  # Phase 1
python -m pytest tests/test_parser.py tests/test_chunker.py -v      # Phase 2
python -m pytest tests/test_storage_*.py -v                         # Phase 3
python -m pytest tests/test_mcp_*.py -v                             # Phase 4
python -m pytest tests/test_cli_*.py -v                             # Phase 5
```

**Current Coverage:** 557 tests (547 passed, 10 skipped)

## Troubleshooting

### PDF Conversion (Colab)

| Issue | Solution |
|-------|----------|
| No GPU detected | Runtime → Change runtime type → GPU |
| PDF not found | Upload via VS Code → "Upload to Colab server" |
| Out of memory | Use Marker, restart runtime, or split PDF |
| Poor quality | Try different tool (Nougat for math, Docling for tables) |

### Chunking (Local)

| Issue | Solution |
|-------|----------|
| FastEmbed not installed | `pip install fastembed` |
| Slow embedding | Reduce batch_size, or use MockEmbedder for testing |
| Memory issues | Process documents in smaller sections |

### Storage (Local)

| Issue | Solution |
|-------|----------|
| helix-py not installed | `pip install helix-py` |
| No Helix-DB server | Use MockChunkStore for testing |
| Connection refused | Start Helix-DB server: `helix deploy --local` |

### MCP Server

| Issue | Solution |
|-------|----------|
| mcp not installed | `pip install 'kidkazz-rag[mcp]'` or `pip install mcp` |
| Server not starting | Check `KIDKAZZ_LOG_LEVEL=DEBUG` for detailed logs |
| No search results | Verify documents are loaded in store |
| Slow startup | First query loads embedder; subsequent queries are faster |
| Claude Code not connecting | Verify `.mcp.json` path and Python environment |

### CLI Tool

| Issue | Solution |
|-------|----------|
| CLI dependencies not installed | `pip install 'kidkazz-rag[cli]'` or `pip install typer rich` |
| Command not found | Ensure `pip install -e ".[cli]"` was run |
| Config file not found | Run `kidkazz config init` to create `.kidkazz.toml` |
| Invalid chunk sizes | Use format `level1,level2,overlap` (e.g., `2048,512,256`) |
| No JSON output | Add `--json` flag to command |

## Roadmap

- [x] Phase 1: PDF to Markdown converter (Colab notebook)
- [x] Phase 2: Hierarchical chunking pipeline
- [x] Phase 3: Helix-DB integration (vector + graph storage)
- [x] Phase 4: MCP server for Claude Code
- [x] Phase 5: End-to-end CLI tool

## Documentation

- [Testing Guide](docs/testing/UNIT_TEST.md) - How to run and understand tests
- [Phase 2 Design](docs/design/PLAN_PHASE2.md) - Chunking pipeline architecture
- [Phase 3 Design](docs/design/PLAN_PHASE3.md) - Helix-DB storage integration
- [Phase 4 Design](docs/design/PLAN_PHASE4.md) - MCP server implementation
- [Phase 5 Design](docs/design/PLAN_PHASE5.md) - CLI tool implementation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run `python -m pytest tests/` to verify
5. Submit a pull request

## License

MIT License
