# Concept Extraction Feature Design Document

## Implementation Status

| Phase | Component | Status |
|-------|-----------|--------|
| 6A | Core Infrastructure (extractor, schema, queries) | ✅ Complete |
| 6B | CLI and Ingestion Integration | ✅ Complete |
| 6C | MCP Tools | ✅ Complete |
| 6D | Graph Visualization | ✅ Complete |

**All phases complete.** Feature is production-ready.

### Implemented Files

| File | Description |
|------|-------------|
| `src/chunker/concept_extractor.py` | LLM-powered concept extraction with Instructor |
| `src/cli/commands/concepts.py` | CLI command group for concept queries |
| `src/cli/graph_viz.py` | DOT/PNG graph visualization |
| `src/mcp_server/tools.py` | Extended with 5 concept tools |
| `src/storage/client.py` | Extended with concept storage methods |
| `db/schema.hx` | Extended with Concept node and edges |
| `db/queries.hx` | Extended with concept queries |
| `tests/test_concept_extractor.py` | Extractor unit tests |
| `tests/test_ingest_concepts.py` | Ingestion integration tests |
| `tests/test_storage_client_concepts.py` | Storage client tests |
| `tests/test_mcp_concept_tools.py` | MCP tools tests |

## Overview

This document outlines the design for an LLM-powered concept extraction feature that builds a knowledge graph from textbook content. Concepts are extracted during ingestion, stored in Helix-DB as graph nodes with typed relationships, and queryable via CLI and MCP tools.

## Problem Statement

Current limitations of the RAG system:
1. **No cross-document connections**: Chunks from different textbooks are isolated
2. **No structured concept data**: Only raw text chunks, no extracted entities
3. **Limited citation support**: Hard to trace where a concept is defined
4. **No relationship awareness**: Can't query "what concepts depend on X?"

**Example use case:**
- User asks: "What is COGS and how is it calculated?"
- Current: Returns chunks mentioning COGS from one document
- Desired: Returns definition with citations from Inventory textbook, plus related concepts (FIFO, Journal Entries) from Accounting textbook with typed relationships

## Feature Requirements

### Functional Requirements

1. **Concept Extraction**
   - Extract entities from textbook content using LLM (Claude via Instructor)
   - Identify concept types: term, method, principle, formula, account
   - Extract definitions, aliases, and relationships
   - Deduplicate concepts across chunks

2. **Relationship Extraction**
   - Extract typed relationships between concepts
   - Relationship types: uses, requires, calculated_from, component_of, recorded_in, supersedes
   - Cross-document relationship linking

3. **Graph Storage**
   - Store concepts as Helix-DB nodes
   - Store relationships as Helix-DB edges
   - Link chunks to concepts (defines, mentions)
   - Enable graph traversal queries

4. **Citation Support**
   - Track which chunks define each concept
   - Provide source document and section path
   - Enable "get citations for concept X"

5. **Query Interface**
   - CLI commands for concept queries
   - MCP tools for Claude Code integration
   - Graph visualization export

### Non-Functional Requirements

- Opt-in extraction (costs LLM API credits)
- Configurable LLM provider
- Graceful handling of extraction failures
- Batch processing for efficiency
- Incremental extraction (add to existing graph)

## Architecture

### Design Decision: Helix-DB Native vs Microsoft GraphRAG

| Aspect | Microsoft GraphRAG | Helix-DB Native (Chosen) |
|--------|-------------------|--------------------------|
| Database | Separate (Parquet/Neo4j) | Unified with vectors |
| Vector + Graph | Separate queries | Combined in one query |
| Schema | Fixed | Custom for our domain |
| Maintenance | Two systems | One system |
| Query language | Python API | HelixQL (compiled, fast) |

**Decision:** Extend Helix-DB schema with native Concept nodes and relationship edges.

**Why:**
- Already using Helix-DB for chunks and vectors
- Native graph traversal built-in
- No additional infrastructure
- Matches existing codebase patterns

### New Modules

```
src/
├── chunker/
│   ├── concept_extractor.py   # NEW: LLM-powered extraction
│   └── ...
├── storage/
│   └── ...                     # Extended with concept methods
├── cli/
│   ├── commands/
│   │   └── concepts.py         # NEW: Concept CLI commands
│   └── graph_viz.py            # NEW: Graph visualization
└── mcp_server/
    └── ...                     # Extended with concept tools

db/
├── schema.hx                   # Extended with Concept node
└── queries.hx                  # Extended with concept queries
```

### Schema Extension

#### New Node: Concept

```hx
N::Concept {
    INDEX concept_id: String,      // Slugified unique ID: "cost-of-goods-sold"
    INDEX name: String,            // Display name: "Cost of Goods Sold"
    definition: String,            // 1-2 sentence definition
    concept_type: String,          // term, method, principle, formula, account
    source_documents: String,      // JSON array of doc_ids where defined
    aliases: String,               // JSON array: ["COGS", "cost of sales"]
}
```

#### New Edges

| Edge Type | From | To | Purpose |
|-----------|------|-----|---------|
| DefinesConcept | Chunk | Concept | Chunk contains definition |
| MentionsConcept | Chunk | Concept | Chunk references concept |
| RelatesTo | Concept | Concept | Concepts are related |

#### Relationship Types (stored in RelatesTo properties or tracked in Python)

- `uses`: COGS uses FIFO
- `requires`: LIFO requires inventory tracking
- `calculated_from`: COGS calculated_from Beginning Inventory
- `component_of`: COGS component_of Income Statement
- `recorded_in`: Inventory recorded_in Balance Sheet
- `supersedes`: IFRS supersedes local GAAP

### Data Flow

```
Ingestion with --extract-concepts:

1. Markdown → Chunks (existing)
2. Chunks → Embeddings (existing)
3. Chunks → Concept Extraction (NEW)
   └── LLM extracts: concepts, definitions, relationships
4. Store chunks + embeddings (existing)
5. Store concepts + relationships (NEW)
6. Link chunks to concepts (NEW)
```

## Implementation Phases

This feature is implemented in 4 sub-phases:

- **Phase 6A**: Core Infrastructure (extractor, schema, queries)
- **Phase 6B**: CLI and Ingestion Integration
- **Phase 6C**: MCP Tools
- **Phase 6D**: Graph Visualization

See individual phase documents for detailed implementation plans:
- `docs/design/PLAN_PHASE6A.md`
- `docs/design/PLAN_PHASE6B.md`
- `docs/design/PLAN_PHASE6C.md`
- `docs/design/PLAN_PHASE6D.md`

## Usage Examples

### CLI Usage

```bash
# Ingest with concept extraction
kidkazz ingest markdown inventory.md --extract-concepts --tags "inventory"
kidkazz ingest markdown accounting.md --extract-concepts --tags "accounting"

# List all concepts
kidkazz concepts list
kidkazz concepts list --doc-id inventory_textbook

# Search concepts
kidkazz concepts search "cost calculation"

# Show concept with citations
kidkazz concepts show "Cost of Goods Sold"
# Output:
# Concept: Cost of Goods Sold
# Type: formula
# Definition: The direct costs attributable to goods sold during a period
# Aliases: COGS, cost of sales
# Sources:
#   - inventory_textbook > Chapter 5 > Inventory Valuation
# Related:
#   - FIFO (uses)
#   - Journal Entry (recorded_in)

# Show related concepts
kidkazz concepts related "COGS"

# Generate graph visualization
kidkazz concepts graph inventory_textbook --format png --output concepts.png
```

### MCP Usage (Claude Code)

```
User: "What is COGS and how does it relate to accounting?"

Claude: [Calls get_concept("Cost of Goods Sold")]
        [Calls get_related_concepts("COGS")]
        [Calls get_concept_chunks("COGS")]

Response: "Cost of Goods Sold (COGS) is the direct costs attributable to
goods sold during a period, calculated as:

  Beginning Inventory + Purchases - Ending Inventory

**Source:** Inventory Textbook, Chapter 5, p.123

**Related Concepts:**
- Uses **FIFO** or **LIFO** methods for inventory valuation
  [Inventory Textbook, Chapter 5, p.130]
- Recorded as a **Journal Entry**: Debit COGS, Credit Inventory
  [Accounting Textbook, Chapter 3, p.45]
- Component of the **Income Statement**
  [Accounting Textbook, Chapter 2, p.22]"
```

## Configuration

### .kidkazz.toml

```toml
[concepts]
enabled = false                              # Default off (costs API credits)
provider = "anthropic/claude-sonnet-4-20250514"  # LLM provider
extract_on_ingest = true                     # Extract during ingestion

[concepts.extraction]
max_concepts_per_chunk = 10                  # Limit per chunk
include_mentioned = true                     # Also extract mentioned concepts
```

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-...  # Required for concept extraction
```

## Dependencies

```toml
[project.optional-dependencies]
concepts = [
    "instructor>=1.0.0",    # Structured LLM extraction
    "graphviz>=0.20",       # Graph visualization (optional)
]
```

## Test Strategy

### Unit Tests

- `test_concept_extractor.py` - Pydantic models, extraction logic
- `test_concept_queries.py` - Query classes
- Mock LLM responses for deterministic testing

### Integration Tests

- `test_concept_storage.py` - Store/retrieve concepts (requires Helix)
- `test_concept_cli.py` - CLI commands
- `test_concept_mcp.py` - MCP tools

## Security Considerations

1. **API Key Protection**: Only store in .env, never in config files
2. **Cost Control**: Extraction is opt-in, configurable limits
3. **Input Validation**: Sanitize concept names before storage
4. **Rate Limiting**: Respect LLM provider rate limits

## Success Criteria

1. Concepts extracted from textbooks with >80% accuracy
2. Cross-document relationships correctly identified
3. Citations trace back to source chunks
4. Graph visualization renders correctly
5. MCP tools enable Claude Code to query concepts
6. All tests pass
7. Documentation complete

## Future Enhancements

1. **Concept Embeddings**: Vector search on concept definitions
2. **Hierarchical Concepts**: Concept → Sub-concept relationships
3. **Automatic Linking**: Link new documents to existing concepts
4. **Concept Quality Scoring**: Confidence scores for extractions
5. **Interactive Graph Explorer**: Web UI for concept browsing
