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
# - OPENAI_API_KEY (for embeddings)
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
     ↓
[3. Ingest markdown]     kidkazz ingest markdown ~/.kidkazz/output/document.md
     ↓
[4. Search/Query]        kidkazz search semantic "your query"
     ↓
[5. Chat via MCP]        Configure .mcp.json for Claude Code
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

# Options:
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

# Batch ingest a directory
kidkazz ingest batch ./docs --pattern "*.md"
```

### Step 4: Search and Retrieve

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

### Step 5: Set Up MCP for Claude Code

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
        "KIDKAZZ_EMBEDDER_TYPE": "openai"
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
        "KIDKAZZ_EMBEDDER_TYPE": "openai"
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
kidkazz inbox parse --agentic               # High-accuracy mode
kidkazz inbox sync                          # Backup to cloud (rclone)
kidkazz inbox clear --completed             # Clean up
```

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

## MCP Tools Available

When connected via MCP, Claude Code can use:

| Tool | Description |
|------|-------------|
| `search_semantic` | Vector similarity search with filters |
| `search_keyword` | Full-text keyword search |
| `get_chunk` | Get chunk by ID |
| `get_context_window` | Get chunk with neighbors |
| `get_parent` / `get_children` / `get_siblings` | Graph traversal |
| `list_documents` | List all documents |
| `get_document_stats` | Document statistics |
| `search_concepts` | Search concept graph |
| `search_tables` | Search tables semantically |
| `search_summaries` | Search document summaries |

## Configuration

### Environment Variables

```bash
KIDKAZZ_STORE_TYPE=mock|helix
KIDKAZZ_HELIX_PORT=6969
KIDKAZZ_EMBEDDER_TYPE=openai|mock
KIDKAZZ_MODEL_NAME=text-embedding-3-small
REDUCTO_API_KEY=your_key
OPENAI_API_KEY=your_key
```

### Config File (.kidkazz.toml)

```toml
[storage]
store_type = "mock"
helix_port = 6969

[embeddings]
embedder_type = "openai"
model_name = "text-embedding-3-small"

[inbox]
path = "~/.kidkazz/inbox"
output_path = "~/.kidkazz/output"
post_action = "delete"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| REDUCTO_API_KEY not set | Add to `.env` file |
| OPENAI_API_KEY not set | Add to `.env` file |
| No search results | Ingest documents first |
| MCP server not starting | Check `.mcp.json` path |
| Helix connection refused | Run `helix push dev` |

## Testing

```bash
PYTHONPATH=. pytest tests/ -v
```

## License

MIT License
