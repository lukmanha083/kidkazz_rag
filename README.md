# KidKazz RAG

A RAG (Retrieval-Augmented Generation) system for converting PDF textbooks to searchable knowledge bases.

## Overview

This project provides tools to:
1. Convert PDF documents to Markdown (with table and image support)
2. Chunk documents hierarchically for vector + graph storage
3. Generate embeddings using CPU-optimized FastEmbed
4. Store in Helix-DB (vector + graph database) - *coming soon*
5. Query documents via MCP integration with Claude Code - *coming soon*

## Current Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | PDF to Markdown Converter | ✅ Complete |
| 2 | Hierarchical Chunking Pipeline | ✅ Complete |
| 3 | Helix-DB Integration | 🔲 Planned |
| 4 | MCP Server | 🔲 Planned |

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
[Helix-DB] ──────────────────── Vector + Graph Storage (coming soon)
     |
     v
[MCP Server] ────────────────── Claude Code Integration (coming soon)
     |
     v
Chat with your documents
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
pip install -e ".[dev,chunker]"   # All dependencies
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
│   └── chunker/                           # Phase 2: Chunking pipeline
│       ├── __init__.py                    # Public API
│       ├── parser.py                      # Markdown structure extraction
│       ├── chunker.py                     # Hierarchical chunking
│       ├── metadata.py                    # Metadata enrichment
│       └── embedder.py                    # FastEmbed integration
├── tests/
│   ├── conftest.py                        # Shared fixtures
│   ├── test_analyzer.py                   # Phase 1 tests
│   ├── test_converter.py
│   ├── test_selector.py
│   ├── test_parser.py                     # Phase 2 tests
│   ├── test_chunker.py
│   ├── test_metadata.py
│   └── test_embedder.py
└── docs/
    ├── design/
    │   └── PLAN_PHASE2.md                 # Phase 2 design document
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
python -m pytest tests/test_parser.py tests/test_chunker.py -v  # Phase 2
python -m pytest tests/test_analyzer.py tests/test_converter.py -v  # Phase 1
```

**Current Coverage:** 223 tests, 89% coverage

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

## Roadmap

- [x] Phase 1: PDF to Markdown converter (Colab notebook)
- [x] Phase 2: Hierarchical chunking pipeline
- [ ] Phase 3: Helix-DB integration (vector + graph storage)
- [ ] Phase 4: MCP server for Claude Code
- [ ] Phase 5: End-to-end CLI tool

## Documentation

- [Testing Guide](docs/testing/UNIT_TEST.md) - How to run and understand tests
- [Phase 2 Design](docs/design/PLAN_PHASE2.md) - Chunking pipeline architecture

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run `python -m pytest tests/` to verify
5. Submit a pull request

## License

MIT License
