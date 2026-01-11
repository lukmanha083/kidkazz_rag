# Kidkazz RAG

[![CodeRabbit Review](https://img.shields.io/badge/CodeRabbit-Reviewed-green?logo=github)](https://coderabbit.ai)
[![Tests](https://img.shields.io/badge/tests-788%20passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)

A RAG (Retrieval-Augmented Generation) system for converting PDF textbooks to searchable knowledge bases.

## Overview

This project provides tools to:
1. Convert PDF documents to Markdown via **Reducto.ai API** (recommended) or **Google Colab** (GPU)
2. Validate conversion quality with automatic quality checks (OCR confidence, structure, content)
3. Chunk documents hierarchically for vector + graph storage
4. Generate embeddings using **FastEmbed** (local CPU, free) or **OpenAI API** (cloud, higher quality)
5. Store in Helix-DB (vector + graph database)
6. Query documents via MCP integration with Claude Code or CLI tool
7. Manage PDF inbox with auto-delete after conversion
8. Tag documents by topic for filtered search (e.g., "inventory", "accounting")
9. Extract concepts and build cross-document knowledge graphs (LLM-powered)

## Current Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | PDF to Markdown Converter | ✅ Complete |
| 2 | Hierarchical Chunking Pipeline | ✅ Complete |
| 3 | Helix-DB Integration | ✅ Complete |
| 4 | MCP Server | ✅ Complete |
| 5 | End-to-End CLI Tool | ✅ Complete |
| 6 | PDF Inbox Management | ✅ Complete |
| 7 | Document Tagging | ✅ Complete |
| 8 | Quality Checker | ✅ Complete |
| 9 | Concept Extraction & Knowledge Graph | ✅ Complete |

## Architecture

```text
PDF Document
     |
     v
[PDF Inbox Manager] ─────────── Auto-manage PDF lifecycle
     |                           ├── Scan inbox directory
     |                           ├── Track processing status
     |                           └── Auto-delete after conversion
     v
[PDF to Markdown Converter] ──── Two Options:
     |                           │
     |                           ├── Option A: Reducto.ai API (Recommended)
     |                           │   └── kidkazz inbox parse
     |                           │       ├── No GPU required
     |                           │       ├── 99%+ accuracy
     |                           │       └── Cloud backup via rclone
     |                           │
     |                           └── Option B: Google Colab (GPU)
     |                               ├── Marker (fast, clean PDFs)
     |                               ├── Docling (tables, mixed)
     |                               └── Nougat (math/equations)
     v
Markdown Document
     |
     v
[Hierarchical Chunker] ───────── Local Python (CPU)
     |                           ├── Markdown structure parsing
     |                           ├── Multi-level chunks (doc→section→leaf)
     |                           ├── Graph relationships (parent/child/sibling)
     |                           ├── Semantic type detection
     |                           └── Optional: LLM concept extraction
     v
[Embeddings] ─────────────────── Two Options:
     |                           ├── FastEmbed (local CPU, free)
     |                           │   └── BAAI/bge-small-en-v1.5 (384 dims)
     |                           └── OpenAI API (cloud, pay-per-use)
     |                               └── text-embedding-3-small (1536 dims)
     v
[Helix-DB Storage] ─────────── Vector + Graph Storage
     |                           ├── MockChunkStore (testing)
     |                           ├── HelixChunkStore (production)
     |                           ├── Document tagging for filtered search
     |                           └── Concept graph (cross-document links)
     |
     ├─────────────────────────────────────────────────────────┐
     v                                                         v
[MCP Server] ────────────────── Claude Code Integration  [CLI Tool] ── Command Line
     |                           ├── 10 search/retrieval tools  |        ├── kidkazz ingest
     |                           ├── 5 concept graph tools      |        ├── kidkazz search
     |                           └── 4 resource endpoints       |        ├── kidkazz docs
     v                                                         v        ├── kidkazz concepts
Chat with your documents                               Terminal access  ├── kidkazz inbox
                                                                        └── kidkazz config
```

## Prerequisites

- Python 3.10+
- Git
- **For Helix-DB (Production storage):**
  - [Rust 1.88.0+](https://rustup.rs/) - Required for Helix CLI
  - [Docker Desktop](https://www.docker.com/products/docker-desktop/) - Required for local Helix-DB
- **For PDF Parsing (choose one):**
  - **Reducto.ai API (Recommended):** [Reducto API key](https://reducto.ai) + [rclone](https://rclone.org/install/) (optional, for cloud backup)
  - **Google Colab (Alternative):** VS Code with [Google Colab Extension](https://marketplace.visualstudio.com/items?itemName=GoogleCloudPlatform.colab-vscode-plugin) + Google account
- **For Embeddings (choose one):**
  - **FastEmbed (Default):** No API key required, runs locally on CPU
  - **OpenAI API (Optional):** [OpenAI API key](https://platform.openai.com/api-keys) for higher quality embeddings

## Getting Started

Follow these steps in order to set up KidKazz RAG:

### Step 1: Clone Repository

```bash
git clone https://github.com/lukmanha083/kidkazz_rag.git
cd kidkazz_rag
```

### Step 2: Install Python Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"  # Or specific: .[cli], .[chunker], etc.
```

### Step 3: Set Up Helix-DB (Production Storage)

Skip this step if using MockChunkStore for testing.

```bash
# Install Helix CLI (requires Rust 1.88.0+ and Docker Desktop)
curl -sSL https://install.helix-db.com | bash

# Initialize and deploy local instance
helix init
helix push dev  # Starts on port 6969
```

### Step 4: Configure API Keys

```bash
cp .env.example .env
# Edit .env and add:
# - REDUCTO_API_KEY (for PDF parsing)
# - OPENAI_API_KEY (optional, for OpenAI embeddings)
```

### Step 5: Set Up rclone (Optional - Cloud Backup)

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive remote
rclone config
# Follow prompts to set up 'gdrive' remote
```

### Step 6: Initialize KidKazz

```bash
kidkazz config init
kidkazz db init
```

### Step 7: Start Using KidKazz

```bash
# Drop PDFs into inbox
cp document.pdf ~/.kidkazz/inbox/

# Parse with Reducto.ai
kidkazz inbox parse

# Ingest into knowledge base
kidkazz ingest markdown ~/.kidkazz/output/document.md
```

## Installation Options

Install only what you need:

| Extra | Command | Description |
|-------|---------|-------------|
| dev | `pip install -e ".[dev]"` | Testing (pytest, coverage) |
| cli | `pip install -e ".[cli]"` | CLI tool (typer, rich) |
| chunker | `pip install -e ".[chunker]"` | Embeddings (FastEmbed) |
| helixdb | `pip install -e ".[helixdb]"` | Helix-DB client (helix-py) |
| mcp | `pip install -e ".[mcp]"` | MCP server for Claude Code |
| reducto | `pip install -e ".[reducto]"` | Reducto.ai PDF parsing |
| openai | `pip install -e ".[openai]"` | OpenAI embeddings |
| all | `pip install -e ".[all]"` | Everything |

## Embeddings

Kidkazz RAG supports two embedding backends:

### FastEmbed (Default - Local)

CPU-optimized embeddings using ONNX Runtime. Free, private, no API required.

```bash
# Use FastEmbed (default)
kidkazz ingest markdown doc.md --embedder fastembed

# Or configure in .kidkazz.toml
kidkazz config set embedder_type fastembed
```

| Model | Dimensions | Speed | Use Case |
|-------|------------|-------|----------|
| BAAI/bge-small-en-v1.5 | 384 | Fast | Default, general use |
| BAAI/bge-base-en-v1.5 | 768 | Medium | Higher quality |
| BAAI/bge-large-en-v1.5 | 1024 | Slow | Best quality |

### OpenAI Embeddings (Cloud API)

Higher quality embeddings via OpenAI API. Requires API key, pay-per-use.

```bash
# Set API key in .env
echo 'OPENAI_API_KEY=sk-xxxxx' >> .env

# Use OpenAI embeddings
kidkazz ingest markdown doc.md --embedder openai

# Or configure in .kidkazz.toml
kidkazz config set embedder_type openai
kidkazz config set model_name text-embedding-3-small
```

| Model | Dimensions | Cost | Use Case |
|-------|------------|------|----------|
| text-embedding-3-small | 1536 | $0.02/1M tokens | Cost-effective |
| text-embedding-3-large | 3072 | $0.13/1M tokens | Highest quality |
| text-embedding-ada-002 | 1536 | $0.10/1M tokens | Legacy |

### Embedding Compatibility Warning

Different embedders produce different dimensions. **You must use the same embedder for ingestion AND search queries.**

```bash
# WARNING: Changing embedder after ingestion
kidkazz config set embedder_type openai
# Warning: Changing embedder may make existing embeddings incompatible.
# You may need to re-ingest documents for consistent search results.
```

**Best practice:** Choose your embedder before ingesting documents and stick with it.

## PDF Inbox Workflow

The PDF Inbox feature provides an end-to-end workflow for managing PDFs from drop to knowledge base ingestion, with automatic cleanup after successful conversion.

### Recommended: Reducto.ai API Parsing

Reducto.ai provides cloud-based PDF parsing with 99%+ accuracy, no GPU required. This is the recommended approach for most users.

**Quick Start:**

```bash
# 1. Set your API key
export REDUCTO_API_KEY="your_api_key_here"

# 2. Drop PDFs into inbox
cp document.pdf ~/.kidkazz/inbox/

# 3. Parse with Reducto.ai
kidkazz inbox parse

# 4. Ingest into knowledge base
kidkazz ingest markdown ~/.kidkazz/output/document.md \
    --doc-id document --title "My Document"
```

**Reducto.ai Workflow Diagram:**

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LOCAL MACHINE                                     │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────────────┐   │
│  │  Drop PDF into │────▶│ kidkazz inbox  │────▶│ kidkazz inbox parse    │   │
│  │  ~/.kidkazz/   │     │ sync (backup)  │     │ (calls Reducto.ai API) │   │
│  │  inbox/        │     │                │     │                        │   │
│  └────────────────┘     └───────┬────────┘     └───────────┬────────────┘   │
│                                 │                          │                 │
└─────────────────────────────────┼──────────────────────────┼─────────────────┘
                                  │ rclone                   │
                                  ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GOOGLE DRIVE (Backup)              REDUCTO.AI CLOUD │
│                    kidkazz_inbox/document.pdf              PDF → Markdown   │
└─────────────────────────────────────────────────────────────────────────────┘
                                                             │
                                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LOCAL MACHINE                                     │
│  ┌────────────────────────┐     ┌────────────────────────────────────────┐  │
│  │ ~/.kidkazz/output/     │────▶│ kidkazz ingest markdown                │  │
│  │ document.md            │     │ (chunk, embed, store → knowledge base) │  │
│  └────────────────────────┘     └────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Reducto.ai Parse Commands:**

```bash
# Parse all PDFs (human-readable output)
kidkazz inbox parse

# Enable AI-enhanced accuracy (uses 2x credits)
kidkazz inbox parse --agentic

# Chunking modes for different use cases
kidkazz inbox parse --chunk-mode variable  # RAG-optimized (~1000 char chunks)
kidkazz inbox parse --chunk-mode block     # Citation-level (each element = chunk)
kidkazz inbox parse --chunk-mode page      # One chunk per page
kidkazz inbox parse --chunk-mode section   # Split by headings

# Combine options
kidkazz inbox parse --agentic --chunk-mode variable  # High-accuracy + RAG chunks

# Preview what would be parsed (no API calls)
kidkazz inbox parse --dry-run

# Skip cloud backup after parsing
kidkazz inbox parse --no-sync-backup

# View parse command help
kidkazz inbox parse --help
```

**Chunk Mode Reference:**

| Mode | Description | Best For |
|------|-------------|----------|
| `disabled` | Continuous human-readable markdown (default) | Reading, archival |
| `variable` | Adaptive chunks ~1000 chars | RAG pipelines |
| `block` | Each element (paragraph, table) as chunk | Legal/regulatory citations |
| `page` | One chunk per page | Page-boundary apps |
| `section` | Split by document headings | Structured documents |

**Reducto.ai Configuration:**

```bash
# Required: Add API key to .env file (secrets only)
echo 'REDUCTO_API_KEY="your_api_key"' >> .env

# Or export directly
export REDUCTO_API_KEY="your_api_key"

# Other settings go in .kidkazz.toml
kidkazz config set cloud_remote gdrive
kidkazz config set cloud_path kidkazz_inbox
kidkazz config set post_action delete  # delete, move, or keep
```

**Reducto.ai vs Google Colab:**

| Feature | Reducto.ai | Google Colab |
|---------|------------|--------------|
| GPU Required | No (cloud API) | Yes (T4/A100) |
| Setup | API key only | Notebook + Drive mount |
| Speed | Fast (cloud infra) | Depends on GPU availability |
| Cost | Pay per page | Free (with session limits) |
| Accuracy | 99%+ (agentic mode) | Varies by tool |
| Batch Processing | Built-in | Manual |
| Offline | No | No |
| Best For | Production, automation | Free tier, experimentation |

---

### Alternative: Google Colab Workflow

For users who prefer free GPU-based conversion or need specific tools (Marker, Docling, Nougat).

### Complete Workflow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LOCAL MACHINE                                     │
│  ┌────────────────┐                                                         │
│  │  Drop PDF into │──┐                                                      │
│  │  ~/.kidkazz/   │  │                                                      │
│  │  inbox/        │  │                                                      │
│  └────────────────┘  │                                                      │
│          │           │                                                      │
│          ▼           │                                                      │
│  ┌────────────────┐  │    ┌─────────────────────────────────────────────┐  │
│  │ kidkazz inbox  │  │    │              GOOGLE COLAB (GPU)             │  │
│  │ status/list    │  │    │  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │  │
│  │ (view pending) │  │    │  │ Upload  │─▶│ Convert │─▶│ Download    │  │  │
│  └────────────────┘  │    │  │ PDF     │  │ (GPU)   │  │ Markdown    │  │  │
│                      │    │  └─────────┘  └─────────┘  └─────────────┘  │  │
│                      │    └────────────────────│────────────────────────┘  │
│                      │                         │                            │
│                      │                         ▼                            │
│                      │    ┌─────────────────────────────────────────────┐  │
│                      └───▶│  Save markdown to ~/.kidkazz/output/        │  │
│                           └─────────────────────────────────────────────┘  │
│                                                │                            │
│                                                ▼                            │
│  ┌────────────────┐       ┌─────────────────────────────────────────────┐  │
│  │ AUTO-DELETE    │◀──────│  kidkazz ingest markdown                    │  │
│  │ PDF from inbox │       │  (chunk, embed, store)                      │  │
│  │ after success  │       └─────────────────────────────────────────────┘  │
│  └────────────────┘                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cloud Sync with rclone (Recommended)

Instead of manually uploading PDFs to Colab, you can use `rclone` to automatically sync your inbox to Google Drive, then access files from Colab via Drive mount.

**One-Time rclone Setup:**

```bash
# Install rclone (Linux/macOS)
curl https://rclone.org/install.sh | sudo bash

# Or on macOS with Homebrew
brew install rclone

# Configure Google Drive remote (interactive)
rclone config
# → n (new remote)
# → name: gdrive
# → storage: Google Drive (number varies)
# → [follow prompts, browser opens for OAuth]
# → y (confirm)
```

**Configure kidkazz cloud sync:**

```bash
# Set your rclone remote name
kidkazz config set cloud_remote gdrive

# Set the folder path on Google Drive
kidkazz config set cloud_path kidkazz_inbox
```

**Automated Workflow with Cloud Sync:**

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LOCAL MACHINE                                     │
│  ┌────────────────┐     ┌────────────────┐                                  │
│  │  Drop PDF into │────▶│ kidkazz inbox  │                                  │
│  │  ~/.kidkazz/   │     │ sync           │                                  │
│  │  inbox/        │     │ (auto-upload)  │                                  │
│  └────────────────┘     └───────┬────────┘                                  │
│                                 │                                            │
└─────────────────────────────────┼────────────────────────────────────────────┘
                                  │ rclone copy
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GOOGLE DRIVE                                         │
│                    kidkazz_inbox/document.pdf                                │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ drive.mount()
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COLAB / VS CODE EXTENSION (GPU)                           │
│  from google.colab import drive                                              │
│  drive.mount('/content/drive')                                               │
│  # PDF at: /content/drive/MyDrive/kidkazz_inbox/document.pdf                │
│  # Convert → Download markdown → Save to ~/.kidkazz/output/                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Cloud Sync Commands:**

```bash
# Check rclone is installed
kidkazz inbox sync --check

# List configured remotes
kidkazz inbox sync --remotes

# Upload inbox PDFs to Google Drive
kidkazz inbox sync

# Preview what would sync (dry run)
kidkazz inbox sync --dry-run

# Download PDFs from Google Drive to local inbox
kidkazz inbox sync --download
```

### Step-by-Step Instructions

**Step 1: Configure Your Inbox**

```bash
# Initialize configuration (creates ~/.kidkazz/ directories)
kidkazz config init

# Set inbox path (default: ~/.kidkazz/inbox)
kidkazz config set inbox_path ~/my_pdfs

# Set output path for converted markdown (default: ~/.kidkazz/output)
kidkazz config set output_path ~/my_markdown

# Set post-conversion action: delete, move, or keep
kidkazz config set post_action delete   # Auto-delete after success (recommended)
kidkazz config set post_action move     # Move to processed/ folder
kidkazz config set post_action keep     # Keep original PDF

# View current configuration
kidkazz config show
```

**Step 2: Drop PDFs into Inbox**

```bash
# Copy PDFs to your inbox directory
cp textbook.pdf ~/.kidkazz/inbox/
cp lecture_notes.pdf ~/.kidkazz/inbox/

# Or move them directly
mv ~/Downloads/*.pdf ~/.kidkazz/inbox/
```

**Step 3: Check Inbox Status**

```bash
# View inbox summary
kidkazz inbox status

# List PDF files in inbox
kidkazz inbox list

# List parsed markdown files in output directory (with status & quality metrics)
kidkazz inbox list --output
kidkazz inbox list --output --json
```

The `--output` view shows:
- **Status**: `ingested` (in database) or `pending` (not yet ingested)
- **Quality metrics**: word count, headers, tables
- **Summary**: total/ingested/pending counts

**Step 4: Convert PDFs in Google Colab**

1. Open `notebooks/pdf_to_markdown_converter.ipynb` in VS Code or Colab
2. Enable GPU runtime: `Runtime → Change runtime type → GPU (T4)`
3. Upload PDFs from your inbox to Colab:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Select PDFs from ~/.kidkazz/inbox/
   ```
4. Run conversion cells
5. Download converted markdown:
   ```python
   from google.colab import files
   files.download('output/textbook.md')
   ```
6. Save markdown to output directory:
   ```bash
   mv ~/Downloads/textbook.md ~/.kidkazz/output/
   ```

**Step 5: Ingest Markdown (Auto-Deletes PDF)**

```bash
# Ingest the converted markdown
kidkazz ingest markdown ~/.kidkazz/output/textbook.md \
    --doc-id textbook \
    --title "My Textbook"

# The PDF is automatically deleted from inbox after successful ingestion!
# (if post_action is set to "delete")
```

**Step 6: Verify and Clean Up**

```bash
# Check ingestion succeeded
kidkazz docs stats textbook

# View inbox status (PDF should be gone)
kidkazz inbox status

# Clear any failed conversions
kidkazz inbox clear --failed

# Clear completed entries from tracking
kidkazz inbox clear --completed
```

### Post-Conversion Actions

| Action | Behavior | Use Case |
|--------|----------|----------|
| `delete` | Remove PDF after successful ingestion | Save disk space (default) |
| `move` | Move PDF to `processed/` directory | Keep backup of originals |
| `keep` | Leave PDF in inbox | Manual management |

### Batch Processing Workflow

For processing multiple PDFs:

```bash
# 1. Drop all PDFs into inbox
cp ~/textbooks/*.pdf ~/.kidkazz/inbox/

# 2. Check what's pending
kidkazz inbox list --status pending

# 3. Upload batch to Colab (use Google Drive mount for large batches)
# In Colab:
from google.colab import drive
drive.mount('/content/drive')
# Copy PDFs from local inbox to Drive, then access in Colab

# 4. Convert all PDFs in Colab

# 5. Download all markdown files to output directory

# 6. Batch ingest
kidkazz ingest batch ~/.kidkazz/output/ --pattern "*.md"

# 7. All successfully ingested PDFs are auto-deleted from inbox
kidkazz inbox status  # Should show fewer pending files
```

### Tips

| Scenario | Recommendation |
|----------|----------------|
| Large PDFs (>50 pages) | Mount Google Drive for faster upload |
| Many PDFs | Use batch ingest after converting all |
| Failed conversions | Check `kidkazz inbox list --status failed` |
| Keep originals | Set `post_action` to `move` |
| Disk space concerns | Set `post_action` to `delete` (default) |

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
│   │   ├── selector.py                    # Tool recommendation
│   │   ├── reducto_client.py              # Reducto.ai API client
│   │   └── quality_checker.py             # Quality validation for parsed output
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
│   ├── pdf_inbox/                         # Phase 6: PDF inbox management
│   │   ├── __init__.py                    # Public API
│   │   ├── models.py                      # Data models (PDFFile, ProcessingStatus, SyncStatus)
│   │   ├── manager.py                     # PDFInboxManager class
│   │   └── cloud_sync.py                  # CloudSync class (rclone wrapper)
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
│           ├── inbox.py                   # inbox status/list/clear
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
│   ├── test_cli_integration.py
│   ├── test_pdf_inbox_models.py           # Phase 6 tests
│   ├── test_pdf_inbox_manager.py
│   ├── test_pdf_inbox_cli.py
│   ├── test_cloud_sync.py                 # Cloud sync tests
│   ├── test_cloud_sync_cli.py             # Cloud sync CLI tests
│   ├── test_reducto_client.py             # Reducto.ai client tests
│   ├── test_inbox_parse_cli.py            # Inbox parse CLI tests
│   ├── test_quality_checker.py            # Quality checker tests
│   └── test_quality_cli.py                # Quality CLI tests
└── docs/
    ├── design/
    │   ├── PLAN_PHASE2.md                 # Phase 2 design document
    │   ├── PLAN_PHASE3.md                 # Phase 3 design document
    │   ├── PLAN_PHASE4.md                 # Phase 4 design document
    │   ├── PLAN_PHASE5.md                 # Phase 5 design document
    │   ├── PDF_INBOX_FEATURE.md           # Phase 6 design document
    │   ├── DOCUMENT_TAGGING.md            # Document tagging design
    │   ├── PLAN_REDUCTO_INTEGRATION.md    # Reducto.ai integration plan
    │   └── QUALITY_CHECKER.md             # Quality checker design
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
| Document | Container | doc_id, title, tags, chunk_count |
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

## Document Tagging

Documents can be tagged with topics/concepts for organized filtering. Tags are stored at the document level and apply to all chunks within that document.

### Tag Behavior

| Feature | Behavior |
|---------|----------|
| **Case-insensitive** | Tags are normalized to lowercase |
| **AND logic** | Multiple tags filter to documents having ALL specified tags |
| **Stored as JSON** | `["inventory", "accounting"]` in database |
| **Inherited by chunks** | Search filters include chunks from tagged documents |

### CLI Usage

```bash
# Ingest with tags
kidkazz ingest markdown inventory_book.md --doc-id inv-book --tags inventory,accounting

# Search within tagged documents
kidkazz search semantic "safety stock calculation" --tags inventory

# List documents by tag
kidkazz docs list --tags inventory

# Multiple tags (AND logic - must have all)
kidkazz docs list --tags inventory,best-practices
```

### MCP Usage

```python
# Search within tagged documents
search_semantic(query="safety stock", tags=["inventory"])

# List documents by tag
list_documents(tags=["inventory"])
```

## Quality Checker

The quality checker validates parsed PDF output before ingestion, catching OCR errors, broken tables, and incomplete content.

### Quality Metrics

| Metric | Description | Warning | Error |
|--------|-------------|---------|-------|
| **Words per page** | Content density | <100 | <50 |
| **OCR confidence** | Average confidence score | <0.8 | <0.6 |
| **Special char ratio** | OCR artifacts | >10% | >20% |
| **Empty line ratio** | Content completeness | >30% | >45% |
| **Broken tables** | Table structure | 1+ | 3+ |
| **Empty chunks** | Chunk quality | >5% | >15% |

### Quality Thresholds

Three preset configurations for different use cases:

| Preset | Use Case | Example |
|--------|----------|---------|
| **strict** | High-quality requirements | Legal documents, textbooks |
| **normal** | Standard documents (default) | General PDFs |
| **lenient** | Permissive checks | Scanned documents, poor quality source |

### CLI Commands

```bash
# Check single file
kidkazz inbox quality ~/.kidkazz/output/document.md

# Check all files in output directory
kidkazz inbox quality --all

# Check with verbose metrics
kidkazz inbox quality document.md --verbose

# JSON output for scripting
kidkazz inbox quality document.md --json

# Summary of all files
kidkazz inbox quality --all --summary
```

### Integration with Parse Command

Quality checks run automatically after parsing:

```bash
# Default: quality check enabled
kidkazz inbox parse

# Disable quality check (not recommended)
kidkazz inbox parse --no-quality-check

# Use strict threshold
kidkazz inbox parse --quality-threshold strict

# Use lenient threshold for poor quality PDFs
kidkazz inbox parse --quality-threshold lenient
```

### Quality Report Output

```
Quality Report: textbook.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Content Stats:
  Words: 4,523 | Pages (est): 15 | Headings: 12
  Tables: 3 | Code blocks: 2 | Lists: 8

Quality Metrics:
  ✓ Content density: 301 words/page (good)
  ✓ Structure: All tables intact
  ✓ Special chars: 2.1% (normal)
  ⚠ Heading hierarchy: Minor gaps detected

Overall Score: 87/100 (PASS)
Recommendation: Safe to ingest
```

### Helix-DB Setup Guide

#### Step 1: Install Helix-DB

Helix-DB is a native vector + graph database written in Rust. Install the CLI tool:

```bash
# Install helix-py Python SDK
pip install helix-py

# Install helix CLI (check https://helix-db.com for latest instructions)
# The helix CLI is used to deploy and manage the database server
curl -sSL https://install.helix-db.com | bash

# Verify installation
helix --version
```

#### Step 2: Start Helix-DB Server

```bash
# Start local Helix-DB instance on port 6969 (default)
helix deploy --local

# Or specify a custom port
helix deploy --local --port 6970

# The server runs in the foreground. Use screen/tmux for background:
screen -S helix
helix deploy --local
# Detach with Ctrl+A, D
```

#### Step 3: Configure Connection

**Option A: Environment variables**
```bash
export KIDKAZZ_STORE_TYPE=helix
export KIDKAZZ_HELIX_PORT=6969
export KIDKAZZ_HELIX_LOCAL=true
```

**Option B: TOML configuration** (`.kidkazz.toml`)
```toml
[storage]
store_type = "helix"
helix_port = 6969
helix_local = true
```

#### Step 4: Initialize Database

```bash
# Initialize with helix storage
kidkazz db init --store helix

# Verify connection
kidkazz db status
```

#### Step 5: Test Connection

```python
from src.storage import HelixChunkStore, HelixConfig

# Connect to local Helix-DB
config = HelixConfig(local=True, port=6969)
store = HelixChunkStore(config)

# Verify connection
print(store.list_documents())
```

#### Helix-DB vs MockChunkStore

| Feature | MockChunkStore | HelixChunkStore |
|---------|----------------|-----------------|
| Server required | No | Yes (`helix deploy --local`) |
| Persistence | In-memory only | Disk (LMDB) |
| Vector search | Brute-force | Optimized |
| Graph traversal | Python dict | Native edges |
| Use case | Testing, development | Production |

**Recommendation**: Use `MockChunkStore` for development and testing, switch to `HelixChunkStore` for production with real data.

## MCP Server Details

### What is MCP?

MCP (Model Context Protocol) is a standard protocol that allows AI assistants like Claude to access external tools and data sources. The kidkazz MCP server exposes your RAG knowledge base to Claude Code, enabling you to **chat with your documents**.

### Starting the MCP Server

#### Manual Start (for testing)

```bash
# Start with mock storage (no Helix-DB required)
KIDKAZZ_STORE_TYPE=mock \
KIDKAZZ_EMBEDDER_TYPE=fastembed \
python -m src.mcp_server

# Start with Helix-DB (requires helix deploy --local)
KIDKAZZ_STORE_TYPE=helix \
KIDKAZZ_HELIX_PORT=6969 \
python -m src.mcp_server

# Using the entry point (if installed)
kidkazz-mcp
```

**Note**: The MCP server uses stdio transport, so it reads from stdin and writes to stdout. For testing, you can send JSON-RPC messages manually, but normally Claude Code manages the server lifecycle.

#### How Claude Code Starts the Server

When configured in `.mcp.json`, Claude Code:
1. Spawns the MCP server as a subprocess
2. Communicates via stdin/stdout (stdio transport)
3. Automatically restarts if the server crashes
4. Passes environment variables for configuration

You don't need to manually start the server - Claude Code does it automatically when you open a project with `.mcp.json`.

### Available Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `search_semantic` | Vector similarity search | query, top_k, doc_id, level, semantic_type, threshold, tags |
| `search_keyword` | Full-text keyword search | keyword, doc_id, case_sensitive |
| `get_chunk` | Get chunk by ID | chunk_id |
| `get_context_window` | Get chunk with neighbors | chunk_id, window_size |
| `get_parent` | Navigate to parent chunk | chunk_id |
| `get_children` | Get child chunks | chunk_id |
| `get_siblings` | Get sibling chunks | chunk_id |
| `list_documents` | List all documents | tags |
| `get_document_chunks` | Get all chunks from doc | doc_id, level |
| `get_document_stats` | Get document statistics | doc_id |
| `search_concepts` | Search for concepts by name | query, top_k |
| `get_concept` | Get concept definition and citations | concept_name |
| `get_related_concepts` | Get related concepts | concept_name, include_reverse |
| `get_concept_chunks` | Get chunks defining/mentioning concept | concept_name, include_mentions |
| `explain_concept_with_context` | Get concept with cross-document context | concept_name |

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
| `KIDKAZZ_EMBEDDER_TYPE` | `fastembed` | Embedder: `fastembed`, `openai`, or `mock` |
| `KIDKAZZ_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Embedding model name |
| `KIDKAZZ_LOG_LEVEL` | `INFO` | Logging level |

### Claude Code Integration

#### Step 1: Create MCP Configuration

Copy the example configuration or create `.mcp.json` in your project root:

```bash
# Copy the template
cp .mcp.json.example .mcp.json

# Or create manually
```

#### Step 2: Configure `.mcp.json`

**For Development** (MockChunkStore, no server required):

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
        "KIDKAZZ_EMBEDDER_TYPE": "fastembed",
        "KIDKAZZ_MODEL_NAME": "BAAI/bge-small-en-v1.5"
      }
    }
  }
}
```

**For Production** (Helix-DB, requires `helix deploy --local`):

```json
{
  "mcpServers": {
    "kidkazz-rag": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/kidkazz_rag",
      "env": {
        "KIDKAZZ_STORE_TYPE": "helix",
        "KIDKAZZ_HELIX_PORT": "6969",
        "KIDKAZZ_HELIX_LOCAL": "true",
        "KIDKAZZ_EMBEDDER_TYPE": "fastembed"
      }
    }
  }
}
```

**Important**: Replace `/path/to/kidkazz_rag` with your actual project path.

#### Step 3: Start Claude Code

1. Open VS Code or your terminal in the project directory
2. Start Claude Code (the CLI tool you're using now)
3. Claude Code automatically discovers `.mcp.json` and starts the MCP server
4. The server runs in the background as a subprocess

#### Step 4: Verify Connection

In Claude Code, you can verify the MCP server is running by asking:

- "What documents are available?" → Calls `list_documents()` tool
- "Search for machine learning" → Calls `search_semantic()` tool
- "Get stats for doc_id" → Calls `get_document_stats()` tool

#### Step 5: Chat with Your Documents

Once connected, you can:

```text
You: "Search for information about neural networks"
Claude: [Calls search_semantic tool, returns relevant chunks]

You: "Show me the parent section of that chunk"
Claude: [Calls get_parent tool, navigates to broader context]

You: "What other topics are in this document?"
Claude: [Calls get_document_chunks tool, explores structure]
```

#### Troubleshooting Connection

| Issue | Solution |
|-------|----------|
| Server not starting | Check Python path in `.mcp.json` |
| "Module not found" | Ensure `cwd` points to project root |
| No search results | Ingest documents first with `kidkazz ingest` |
| Slow first query | Normal - embedder loads lazily on first use |
| Connection timeout | Check if Helix-DB is running (if using helix) |

#### Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────┐
│                        Claude Code                           │
│   (You're talking to me here!)                              │
└──────────────────────────┬──────────────────────────────────┘
                           │ stdio (stdin/stdout)
                           │ JSON-RPC messages
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     MCP Server                               │
│   python -m src.mcp_server                                  │
│   ├── 10 Tools (search_semantic, get_chunk, etc.)          │
│   └── 4 Resources (schema, documents, etc.)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                                 ▼
┌──────────────────┐              ┌──────────────────┐
│  MockChunkStore  │      OR      │ HelixChunkStore  │
│  (In-memory)     │              │ (Helix-DB)       │
│  For testing     │              │ Port 6969        │
└──────────────────┘              └──────────────────┘
```

## CLI Tool Details

The CLI tool (`kidkazz`) provides command-line access to all RAG functionality with rich terminal output.

### Command Groups

| Command | Subcommands | Description |
|---------|-------------|-------------|
| `kidkazz ingest` | markdown, batch, pdf | Ingest documents into knowledge base |
| `kidkazz search` | semantic, keyword | Search the knowledge base |
| `kidkazz docs` | list, stats, export, delete | Manage documents |
| `kidkazz concepts` | list, show, search, related, graph, export | Query concept graph |
| `kidkazz db` | init, status, clear | Database operations |
| `kidkazz inbox` | status, list, clear, sync, parse, quality | Manage PDF inbox and parsed output |
| `kidkazz config` | show, set, reset, init | Configuration management |

### Ingest Commands

```bash
# Ingest a single markdown file
kidkazz ingest markdown document.md --doc-id my_doc --title "My Document"

# Ingest with tags for topic-based filtering
kidkazz ingest markdown textbook.md --doc-id inv-book --tags inventory,accounting

# Preview without ingesting
kidkazz ingest markdown document.md --dry-run

# Custom chunk sizes (level1,level2,overlap)
kidkazz ingest markdown document.md --chunk-sizes 1024,256,128

# Use specific embedder (fastembed, openai, mock)
kidkazz ingest markdown document.md --embedder openai

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

# Filter by document tags (AND logic - must have all specified tags)
kidkazz search semantic "safety stock calculation" --tags inventory
kidkazz search semantic "depreciation" --tags inventory,accounting

# Keyword search
kidkazz search keyword "supervised learning" --case-sensitive

# JSON output for scripting
kidkazz search semantic "example" --json
```

### Document Management

```bash
# List all documents
kidkazz docs list --json

# List documents by tag
kidkazz docs list --tags inventory
kidkazz docs list --tags inventory,accounting  # AND logic

# Get document statistics
kidkazz docs stats textbook

# Export document chunks
kidkazz docs export textbook --format json --output chunks.json

# Delete a document
kidkazz docs delete textbook --force
```

### Concept Commands

Query the extracted concept knowledge graph:

```bash
# List all concepts
kidkazz concepts list
kidkazz concepts list --doc-id textbook
kidkazz concepts list --type method  # Filter by type: term, method, principle, formula, account
kidkazz concepts list --json

# Show concept details with citations
kidkazz concepts show "Cost of Goods Sold"
kidkazz concepts show "FIFO" --json

# Search concepts
kidkazz concepts search "inventory valuation"
kidkazz concepts search "cost" --top-k 10

# Get related concepts
kidkazz concepts related "COGS"
kidkazz concepts related "FIFO" --depth 2  # Traverse deeper

# Generate graph visualization
kidkazz concepts graph --format dot --output concepts.dot
kidkazz concepts graph --doc-id textbook --format png --output graph.png
kidkazz concepts graph --open  # Render and open in viewer

# Export concepts
kidkazz concepts export --format json --output concepts.json
kidkazz concepts export --format csv --output concepts.csv
```

**Ingesting with concept extraction:**

```bash
# Extract concepts during ingestion (requires ANTHROPIC_API_KEY)
kidkazz ingest markdown textbook.md --extract-concepts

# Use specific LLM provider
kidkazz ingest markdown textbook.md --extract-concepts --concept-provider anthropic/claude-opus-4-20250514

# Batch ingest with concept extraction
kidkazz ingest batch ./docs --pattern "*.md" --extract-concepts
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

### PDF Inbox Management

The inbox feature helps manage PDF files before conversion:

```bash
# View inbox status
kidkazz inbox status

# List PDF files in inbox
kidkazz inbox list
kidkazz inbox list --json

# List parsed markdown files (shows status & quality metrics)
kidkazz inbox list --output
kidkazz inbox list --output --json

# Clear inbox files
kidkazz inbox clear --force
kidkazz inbox clear --failed      # Only failed conversions
kidkazz inbox clear --completed   # Only completed conversions

# Cloud sync with rclone
kidkazz inbox sync                # Upload to cloud
kidkazz inbox sync --download     # Download from cloud
kidkazz inbox sync --dry-run      # Preview sync
kidkazz inbox sync --check        # Verify rclone installed
kidkazz inbox sync --remotes      # List configured remotes

# Parse PDFs with Reducto.ai
kidkazz inbox parse                        # Parse all (human-readable)
kidkazz inbox parse --agentic              # High-accuracy mode (2x credits)
kidkazz inbox parse --chunk-mode variable  # RAG-optimized chunks
kidkazz inbox parse --chunk-mode block     # Citation-level chunks
kidkazz inbox parse --chunk-mode page      # One chunk per page
kidkazz inbox parse --chunk-mode section   # Split by headings
kidkazz inbox parse -c variable --agentic  # Combine options
kidkazz inbox parse --dry-run              # Preview without API calls
kidkazz inbox parse --no-sync-backup       # Skip cloud backup
kidkazz inbox parse --no-quality-check     # Skip quality validation
kidkazz inbox parse --quality-threshold strict  # Use strict thresholds

# Quality check markdown files
kidkazz inbox quality output/doc.md        # Check single file
kidkazz inbox quality --all                # Check all files in output
kidkazz inbox quality --dir /path/to/md    # Check specific directory
kidkazz inbox quality doc.md --json        # JSON output for scripting
kidkazz inbox quality doc.md --verbose     # Detailed metrics breakdown
kidkazz inbox quality --all --summary      # Summary of all files
```

**Inbox Configuration:**

```bash
# Set inbox path
kidkazz config set inbox_path ~/my_pdfs

# Set post-conversion action (delete, move, keep)
kidkazz config set post_action delete

# Enable recursive scanning
kidkazz config set inbox_recursive true

# Configure cloud sync (requires rclone)
kidkazz config set cloud_remote gdrive
kidkazz config set cloud_path kidkazz_inbox
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

[embeddings]
embedder_type = "fastembed"  # fastembed (local), openai (API), or mock
model_name = "BAAI/bge-small-en-v1.5"  # Or "text-embedding-3-small" for OpenAI

[chunking]
level_1_size = 2048
level_2_size = 512
overlap = 256

[inbox]
path = "~/.kidkazz/inbox"
output_path = "~/.kidkazz/output"
post_action = "delete"   # delete, move, or keep
recursive = false
processed_dir = "~/.kidkazz/processed"

[cloud_sync]
remote = "gdrive"        # rclone remote name
path = "kidkazz_inbox"   # folder on remote
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

**Current Coverage:** 788 tests passing

## Troubleshooting

### PDF Conversion (Colab)

| Issue | Solution |
|-------|----------|
| No GPU detected | Runtime → Change runtime type → Hardware accelerator → GPU (T4) |
| PDF not uploading | Use Google Drive mount for large files (see workflow above) |
| Upload timeout | Mount Google Drive instead of direct upload |
| Out of memory | Use Marker (most efficient), restart runtime, or split PDF |
| Poor quality | Try different tool (Nougat for math, Docling for tables) |
| Session disconnected | Enable "Keep alive" extension or remount Drive |
| Files disappeared | Colab sessions reset after timeout; save to Drive |

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

### Reducto.ai Parsing

| Issue | Solution |
|-------|----------|
| REDUCTO_API_KEY not set | `export REDUCTO_API_KEY="your_key"` or add to `.env` |
| API key invalid | Verify key at [reducto.ai/dashboard](https://reducto.ai/dashboard) |
| Rate limit exceeded | Wait and retry, or upgrade plan |
| API error | Check Reducto status page, verify PDF is valid |
| No PDFs found | Check inbox path: `kidkazz config show` |
| Output not saved | Check output path exists: `~/.kidkazz/output/` |
| Cloud backup failed | Verify rclone config: `kidkazz inbox sync --check` |

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
- [x] Phase 6: PDF inbox management with auto-delete
- [x] Phase 7: Document tagging for topic-based filtering
- [x] Phase 8: Quality checker for parsed output validation

## Documentation

- [Testing Guide](docs/testing/UNIT_TEST.md) - How to run and understand tests
- [Phase 2 Design](docs/design/PLAN_PHASE2.md) - Chunking pipeline architecture
- [Phase 3 Design](docs/design/PLAN_PHASE3.md) - Helix-DB storage integration
- [Phase 4 Design](docs/design/PLAN_PHASE4.md) - MCP server implementation
- [Phase 5 Design](docs/design/PLAN_PHASE5.md) - CLI tool implementation
- [PDF Inbox Design](docs/design/PDF_INBOX_FEATURE.md) - PDF inbox management
- [Reducto.ai Integration](docs/design/PLAN_REDUCTO_INTEGRATION.md) - Reducto API parsing
- [Document Tagging](docs/design/DOCUMENT_TAGGING.md) - Topic-based document filtering
- [Quality Checker](docs/design/QUALITY_CHECKER.md) - Parsed output quality validation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run `python -m pytest tests/` to verify
5. Submit a pull request

## License

MIT License
