# Kidkazz RAG

A RAG (Retrieval-Augmented Generation) system for converting PDF textbooks to searchable knowledge bases with Claude Code integration.

## Quick Start

### 1. Install

```bash
git clone https://github.com/lukmanha083/kidkazz_rag.git
cd kidkazz_rag
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add:
# - REDUCTO_API_KEY (for PDF parsing)
# - CO_API_KEY (for Cohere embeddings + reranking)
# - OPENAI_API_KEY (optional, for summarization LLM)
```

### 3. Initialize

```bash
kidkazz config init
kidkazz db init
```

## Workflow: PDF to Knowledge Base

```text
PDF Document
     ↓
[1. Drop into inbox]     cp document.pdf ~/.kidkazz/inbox/
     ↓
[2. Parse with Reducto]  kidkazz inbox parse
     ↓  (images extracted to ~/.kidkazz/images/<doc_id>/)
[3. Ingest markdown]     kidkazz ingest markdown ~/.kidkazz/output/document.md
     ↓  (multimodal: text + table/figure images → fused vectors)
[4. Generate summaries]  kidkazz summarize generate <doc_id>
     ↓  (hierarchical summaries + concept extraction)
[5. Search/Query]        kidkazz search semantic "your query"
     ↓
[6. Chat via MCP]        Configure .mcp.json for Claude Code
```

### Step 1: Add PDFs to Inbox

```bash
# Copy PDFs to inbox
cp textbook.pdf ~/.kidkazz/inbox/

# Check inbox status
kidkazz inbox status
kidkazz inbox list
```

### Step 2: Parse PDFs with Reducto.ai

```bash
# Parse all PDFs in inbox
kidkazz inbox parse

# Quality presets:
kidkazz inbox parse -p full                # Max accuracy (~2x credits)
kidkazz inbox parse -p standard            # Accurate, economical (~1x)
kidkazz inbox parse -p full -c variable    # Full preset + RAG chunks

# Individual options:
kidkazz inbox parse --agentic              # Higher accuracy (2x credits)
kidkazz inbox parse --chunk-mode variable  # RAG-optimized chunks
kidkazz inbox parse --dry-run              # Preview without API calls
```

### Step 3: Ingest into Database

```bash
# Ingest a markdown file
kidkazz ingest markdown ~/.kidkazz/output/textbook.md \
    --doc-id textbook \
    --title "My Textbook"

# With tags for filtered search
kidkazz ingest markdown textbook.md --tags inventory,accounting

# With multimodal image embeddings (table/figure images)
kidkazz ingest markdown textbook.md --images-dir ~/.kidkazz/images/textbook/

# Batch ingest a directory
kidkazz ingest batch ./docs --pattern "*.md"
```

### Step 4: Generate Summaries & Extract Concepts

```bash
# Generate hierarchical summaries + extract concepts
kidkazz summarize generate textbook

# If some chapters failed, repair only the missing ones
kidkazz summarize generate textbook --repair

# View document summary
kidkazz summarize show textbook

# Search summaries
kidkazz summarize search "inventory valuation"

# Search concepts across documents
kidkazz concepts search "FIFO"
```

### Step 5: Search and Retrieve

```bash
# Semantic search (vector similarity)
kidkazz search semantic "What is machine learning?" --top-k 5

# Filter by tags
kidkazz search semantic "safety stock" --tags inventory

# Keyword search
kidkazz search keyword "neural network"

# List documents
kidkazz docs list

# Get document stats
kidkazz docs stats textbook
```

### Step 6: Set Up MCP for Claude Code

Create `.mcp.json` in your project root:

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
        "KIDKAZZ_EMBEDDER_TYPE": "cohere"
      }
    }
  }
}
```

For production with Helix-DB:

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
        "KIDKAZZ_EMBEDDER_TYPE": "cohere"
      }
    }
  }
}
```

Claude Code will automatically start the MCP server and provide access to search tools.

## CLI Commands Reference

### Ingest

```bash
kidkazz ingest markdown <file>              # Ingest markdown file
kidkazz ingest markdown <file> --tags a,b   # With tags
kidkazz ingest markdown <file> --images-dir <path>  # With multimodal images
kidkazz ingest markdown <file> --dry-run    # Preview
kidkazz ingest batch <dir> --pattern "*.md" # Batch ingest
```

### Search

```bash
kidkazz search semantic <query>             # Vector search
kidkazz search semantic <query> --tags a    # Filter by tags
kidkazz search semantic <query> --top-k 10  # Limit results
kidkazz search keyword <term>               # Keyword search
```

### Documents

```bash
kidkazz docs list                           # List all documents
kidkazz docs list --tags inventory          # Filter by tag
kidkazz docs stats <doc_id>                 # Show statistics
kidkazz docs delete <doc_id> --force        # Delete document
```

### PDF Inbox

```bash
kidkazz inbox status                        # View status
kidkazz inbox list                          # List PDFs
kidkazz inbox list --output                 # List parsed markdown
kidkazz inbox parse                         # Parse all PDFs
kidkazz inbox parse -p full                 # Max accuracy (~2x credits)
kidkazz inbox parse -p standard             # Accurate, economical (~1x)
kidkazz inbox parse --agentic               # High-accuracy mode
kidkazz inbox sync                          # Backup to cloud (rclone)
kidkazz inbox clear --completed             # Clean up
```

### Summarize & Concepts

```bash
kidkazz summarize generate <doc_id>         # Generate summaries + concepts
kidkazz summarize generate <doc_id> --force # Delete all and regenerate from scratch
kidkazz summarize generate <doc_id> --repair # Fix only missing chapter summaries
kidkazz summarize show <doc_id>             # Show document summary
kidkazz summarize search "query"            # Search summaries
kidkazz summarize list                      # List summarized documents
kidkazz concepts search "concept name"      # Search concept graph
kidkazz concepts list --doc-id <doc_id>     # List concepts for document
kidkazz concepts show "concept name"        # Show concept details + sources
kidkazz concepts related "concept name"     # Show related concepts
kidkazz concepts graph --view               # Interactive graph (opens browser)
kidkazz concepts graph -o graph.html --view # Custom output path
kidkazz concepts graph --min-docs 3 --view  # Only concepts in 3+ books
kidkazz concepts graph <doc_id> --view      # Single document view
kidkazz concepts clean --dry-run            # Preview garbage concepts
kidkazz concepts clean --force              # Delete garbage concepts
kidkazz concepts export -o concepts.json    # Export to JSON
kidkazz concepts export -o concepts.csv -f csv  # Export to CSV
```

> **`--repair` vs `--force`:** Use `--repair` when a few chapters failed (e.g., max_tokens error) — it detects the gaps and regenerates only those chapters + refreshes the document summary (typically 1-3 LLM calls). Use `--force` to wipe everything and start over (225+ LLM calls for a typical textbook).

### Database

```bash
kidkazz db init                             # Initialize database
kidkazz db status                           # Check connection
kidkazz db clear --force                    # Clear all data
```

### Configuration

```bash
kidkazz config show                         # Show current config
kidkazz config set store_type helix         # Set value
kidkazz config init                         # Create config file
```

## Helix-DB Setup (Production)

For production, use Helix-DB instead of the in-memory mock store:

```bash
# Install Helix CLI (requires Rust 1.88.0+ and Docker)
curl -sSL https://install.helix-db.com | bash

# Initialize and start
helix init
helix push dev  # Starts on port 6969

# Configure kidkazz to use Helix
kidkazz config set store_type helix
kidkazz config set helix_port 6969
```

## Remote Deployment (Fly.io)

Deploy the MCP server + Helix-DB to Fly.io for remote access from any machine. Uses scale-to-zero to minimize costs (~$2.45/mo for 2hrs/day usage).

### Prerequisites

1. [Fly.io CLI](https://fly.io/docs/flyctl/install/) installed and authenticated
2. Helix-DB image built locally (`kidkazz db deploy` must have run at least once)
3. Data already ingested into local Helix-DB

### Deploy

```bash
# 1. Create Fly app + volume (first time only)
fly apps create kidkazz-rag
fly volumes create kidkazz_data --region sin --size 1

# 2. Set secrets (API keys)
fly secrets set \
  CO_API_KEY=your_cohere_key \
  OPENAI_API_KEY=your_openai_key \
  KIDKAZZ_API_KEY=your_secret_mcp_auth_key

# 3. Build and deploy (uses local Docker)
fly deploy --local-only
```

### Claude Code Config (Remote HTTP)

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "kidkazz-rag": {
      "type": "streamable-http",
      "url": "https://kidkazz-rag.fly.dev/mcp",
      "headers": {
        "Authorization": "Bearer your_secret_mcp_auth_key"
      }
    }
  }
}
```

### Scale-to-Zero Behavior

- **Idle**: Fly stops the machine after 5 minutes of no HTTP traffic
- **Cold start**: ~10-15s (Helix-DB startup + MCP server init)
- **Persistent data**: Fly volume at `/data` preserves Helix-DB across restarts
- **Health checks**: `/health` endpoint (public, no auth required)

### Data Migration

To copy local Helix-DB data to Fly volume:

```bash
# SSH into the Fly machine
fly ssh console

# Or use fly sftp to copy LMDB files
fly ssh sftp shell
> put .helix/.volumes/dev/user /data/user
```

## MCP Tools Available

When connected via MCP, Claude Code can use:

### Chunk Search & Retrieval

| Tool | Description |
|------|-------------|
| `search_semantic` | Vector similarity search with filters (`has_table`, `has_code`, `has_math`, `has_image`, `header_level`, `tags`). Set `include_images=True` to receive actual table/figure images alongside text. |
| `search_keyword` | Full-text keyword search |
| `get_chunk` | Get chunk by ID |
| `get_chunk_images` | Get actual PNG/JPG images embedded in a chunk (use after finding `has_image=True`) |
| `get_context_window` | Get chunk with neighbors |
| `get_parent` / `get_children` / `get_siblings` | Graph traversal |
| `list_documents` | List all documents |
| `get_document_chunks` | Get all chunks for a document |
| `get_document_stats` | Document statistics |

### Concept Graph

| Tool | Description |
|------|-------------|
| `search_concepts` | Search concept graph by name/definition |
| `get_concept` | Get concept by name |
| `get_concept_with_citations` | Get concept with source chunk citations |
| `get_related_concepts` | Get related concepts (cross-document) |
| `explain_concept_cross_document` | Comprehensive cross-document concept explanation |
| `get_concept_graph_dot` | Export concept graph in DOT format |
| `list_concepts` | List all concepts |

### Document Summaries

| Tool | Description |
|------|-------------|
| `search_summaries` | Search summaries semantically |
| `get_document_summary` | Get document-level summary |
| `get_chapter_summaries` | Get chapter-level summaries |
| `get_section_summaries` | Get section-level summaries |
| `get_summary_hierarchy` | Get full summary tree |
| `list_summarized_documents` | List documents with summaries |

### Multimodal Image Support

Chunks containing tables or figures have `has_image: true` in their metadata. To view the actual images:

```text
# 1. Search with image filter
search_semantic("inventory table", has_image=True)
→ returns chunks with has_image flag

# 2. Get images inline with search results
search_semantic("inventory table", include_images=True)
→ returns text metadata interleaved with ImageContent blocks

# 3. Get images for a specific chunk
get_chunk_images("textbook_l2_42")
→ returns ImageContent blocks for that chunk's images
```

## Configuration

### Environment Variables

```bash
KIDKAZZ_STORE_TYPE=mock|helix
KIDKAZZ_HELIX_PORT=6969
KIDKAZZ_EMBEDDER_TYPE=cohere|openai|mock
KIDKAZZ_MODEL_NAME=embed-v4.0
KIDKAZZ_TRANSPORT=stdio|streamable-http
KIDKAZZ_HOST=0.0.0.0
KIDKAZZ_PORT=8080
KIDKAZZ_API_KEY=your_key     # MCP HTTP auth (Bearer token)
REDUCTO_API_KEY=your_key
CO_API_KEY=your_key          # Cohere (embeddings + reranking)
OPENAI_API_KEY=your_key      # OpenAI (optional, for summarization LLM)
```

### Config File (.kidkazz.toml)

```toml
[storage]
store_type = "helix"
helix_port = 6969

[embeddings]
embedder_type = "cohere"
model_name = "embed-v4.0"        # Multimodal: text + image → single vector

[reranker]
enabled = true
model = "rerank-v3.5"

[inbox]
path = "~/.kidkazz/inbox"
output_path = "~/.kidkazz/output"
post_action = "move"
return_images = ["figure", "table"]   # Extract table/figure images from PDFs
images_path = "~/.kidkazz/images"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| REDUCTO_API_KEY not set | Add to `.env` file |
| CO_API_KEY not set | Add to `.env` file (Cohere embeddings) |
| OPENAI_API_KEY not set | Add to `.env` file (optional, for summarization) |
| No search results | Ingest documents first |
| MCP server not starting | Check `.mcp.json` path |
| Helix connection refused | Run `helix push dev` |

## Testing

```bash
PYTHONPATH=. pytest tests/ -v
```

## License

MIT License
