# Phase 4: MCP Server Implementation Plan

## Overview

Build an MCP (Model Context Protocol) server to expose the KidKazz RAG knowledge base to Claude Code. This enables natural language document search, graph traversal, and context expansion directly from the Claude Code interface.

---

## Key Decisions

### 1. MCP SDK: FastMCP

Using the FastMCP high-level Python SDK for cleaner implementation:
- Decorator-based tool/resource registration
- Automatic JSON schema generation from type hints
- Built-in stdio transport support

### 2. Embedding Strategy: Server-Side

The MCP server handles embedding internally:
- Clients send text queries, server generates embeddings
- Consistent embedding model across all queries
- Lazy loading to avoid slow startup

### 3. Transport: Stdio

Standard input/output transport for Claude Code integration:
- No network setup required
- Simple configuration in `.mcp.json`
- Works out of the box with Claude Code

---

## Module Structure

```text
src/
├── chunker/                    # Existing (Phase 2)
├── storage/                    # Existing (Phase 3)
└── mcp_server/                 # NEW (Phase 4)
    ├── __init__.py             # Public API exports
    ├── server.py               # FastMCP server definition
    ├── tools.py                # Tool implementations
    ├── resources.py            # Resource implementations
    ├── config.py               # Configuration management
    ├── formatters.py           # Response formatting helpers
    └── __main__.py             # Entry point for `python -m`

tests/
├── test_mcp_config.py          # Configuration tests
├── test_mcp_formatters.py      # Formatter tests
├── test_mcp_tools.py           # Tool implementation tests
├── test_mcp_resources.py       # Resource tests
├── test_mcp_server.py          # Server initialization tests
└── test_mcp_integration.py     # End-to-end tests
```

---

## MCP Tools Design

### Search Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `search_semantic` | Vector similarity search | query, top_k, doc_id, level, semantic_type, threshold |
| `search_keyword` | Full-text keyword search | keyword, doc_id, case_sensitive |

### Retrieval Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `get_chunk` | Get chunk by ID | chunk_id |
| `get_context_window` | Get chunk with neighbors | chunk_id, window_size |

### Graph Traversal Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `get_parent` | Navigate to parent chunk | chunk_id |
| `get_children` | Navigate to child chunks | chunk_id |
| `get_siblings` | Get sibling chunks | chunk_id |

### Document Management Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `list_documents` | List all documents | - |
| `get_document_chunks` | Get all chunks from doc | doc_id, level |
| `get_document_stats` | Get document statistics | doc_id |

---

## Tool Specifications

### search_semantic

```python
def search_semantic(
    query: str,                          # Natural language query
    top_k: int = 5,                      # Number of results
    doc_id: str | None = None,           # Filter to document
    level: int | None = None,            # 1=section, 2=leaf
    semantic_type: str | None = None,    # definition, example, etc.
    threshold: float = 0.0,              # Min similarity score
) -> list[dict]:
    """Search for chunks semantically similar to the query."""
```

### get_context_window

```python
def get_context_window(
    chunk_id: str,                       # Center chunk ID
    window_size: int = 2,                # Chunks before/after
) -> list[dict]:
    """Get a chunk with surrounding context."""
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KIDKAZZ_STORE_TYPE` | `mock` | `mock` or `helix` |
| `KIDKAZZ_HELIX_PORT` | `6969` | Helix-DB port |
| `KIDKAZZ_EMBEDDER_TYPE` | `fastembed` | `mock` or `fastembed` |
| `KIDKAZZ_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Embedding model |

### Claude Code Configuration (`.mcp.json`)

```json
{
  "mcpServers": {
    "kidkazz-rag": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "${workspaceFolder}",
      "env": {
        "KIDKAZZ_STORE_TYPE": "mock",
        "KIDKAZZ_EMBEDDER_TYPE": "fastembed"
      }
    }
  }
}
```

---

## Implementation Steps

### Step 1: Project Setup
- Add `mcp>=1.2.0` to pyproject.toml optional deps
- Create `src/mcp_server/` directory structure
- Add entry point to pyproject.toml

### Step 2: Configuration (`config.py`)
- `MCPServerConfig` dataclass with all settings
- `from_env()` class method for environment loading
- `create_store()` and `create_embedder()` factory methods

### Step 3: Formatters (`formatters.py`)
- `format_chunk()` - EmbeddedChunk -> dict
- `format_search_result()` - Add similarity score
- `format_document_stats()` - Statistics formatting

### Step 4: Tools (`tools.py`)
- Implement all 9 MCP tools
- Server-side embedding in search_semantic
- Proper error handling with logging

### Step 5: Resources (`resources.py`)
- `kidkazz://schema` - Schema information
- `kidkazz://documents` - Document list
- `kidkazz://document/{doc_id}` - Document overview
- `kidkazz://chunk/{chunk_id}` - Chunk content

### Step 6: Server (`server.py`)
- `create_server()` function with FastMCP setup
- Tool and resource registration
- Lazy loading for performance

### Step 7: Entry Point (`__main__.py`)
- Configure logging to stderr (critical for stdio)
- Load config from environment
- Run server with stdio transport

### Step 8: Tests & Documentation
- Write ~95 tests across 6 test files
- Create `.mcp.json` template
- Update README with MCP usage

---

## Dependencies (pyproject.toml)

```toml
[project.optional-dependencies]
mcp = [
    "mcp>=1.2.0",
]
all = [
    "fastembed>=0.2.0",
    "helix-py>=0.1.0",
    "mcp>=1.2.0",
]

[project.scripts]
kidkazz-mcp = "src.mcp_server:main"
```

---

## Test Plan

| Test File | Tests | Focus |
|-----------|-------|-------|
| test_mcp_config.py | ~12 | Configuration, env vars, factories |
| test_mcp_formatters.py | ~15 | Response formatting |
| test_mcp_tools.py | ~35 | All tool implementations |
| test_mcp_resources.py | ~10 | Resource handlers |
| test_mcp_server.py | ~8 | Server creation, registration |
| test_mcp_integration.py | ~15 | End-to-end workflows |

**Total: ~95 new tests**

---

## Critical Implementation Notes

### 1. Stdio Transport Logging

**NEVER write to stdout** - all logs must go to stderr:

```python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
```

### 2. Type Hints Required

FastMCP generates JSON schemas from type hints:

```python
# Good - schema generated
def search(query: str, top_k: int = 5) -> list[dict]: ...

# Bad - no schema
def search(query, top_k=5): ...
```

### 3. Lazy Loading

Avoid slow startup by lazy-loading embedder:

```python
class ServerState:
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = self._config.create_embedder()
        return self._embedder
```

---

## Success Criteria

1. All ~95 unit tests pass
2. Server starts via `python -m src.mcp_server`
3. Tools callable from Claude Code
4. Search returns relevant results
5. Graph traversal works correctly
6. No stdout pollution (stderr only)
7. Coverage > 80% for mcp_server module

---

## Usage Example

```python
# After starting MCP server, Claude Code can use these tools:

# Search for relevant content
search_semantic(query="What is machine learning?", top_k=5)

# Get chunk with context
get_context_window(chunk_id="textbook_l2_5", window_size=2)

# Navigate hierarchy
get_parent(chunk_id="textbook_l2_5")
get_children(chunk_id="textbook_l1_2")
get_siblings(chunk_id="textbook_l2_5")

# Browse documents
list_documents()
get_document_stats(doc_id="textbook")
```

---

## Critical Files

| File | Purpose |
|------|---------|
| `src/storage/client.py` | ChunkStoreProtocol interface |
| `src/storage/mock_store.py` | MockChunkStore for testing |
| `src/chunker/embedder.py` | EmbeddedChunk, embedder interfaces |
| `pyproject.toml` | Add mcp dependency |
| `README.md` | Update with MCP usage |
