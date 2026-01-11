# Plan: Header Metadata & Table Processing Enhancement

## Overview

This document outlines the implementation plan for enhancing KidKazz RAG with:
1. **Header-aware concept extraction** - Leverage header metadata for smarter concept extraction and relationship inference
2. **Advanced table processing** - Extract, summarize, and embed tables for improved semantic search

## Research Summary

### Key Findings from Industry Research

#### Table Serialization Formats for LLMs

| Format | Accuracy | Token Efficiency | Recommendation |
|--------|----------|------------------|----------------|
| Markdown-KV | 60.7% | Best (15% less than JSON) | Best for accuracy |
| HTML | Good (6.76% better than plain text) | Medium | Good for complex tables |
| Markdown Table | Medium | Good | Balance of readability/cost |
| CSV | 44.3% | Good | Avoid - poor accuracy |
| JSONL | Poor | Medium | Avoid - poor accuracy |

**Source**: [Which Table Format Do LLMs Understand Best?](https://www.improvingagents.com/blog/best-input-data-format-for-llms/)

#### Multi-Vector Retriever Strategy

The recommended approach for table-heavy RAG:
1. Generate LLM summaries of tables (contextual descriptions)
2. Embed the summaries for retrieval (semantic search)
3. Store raw tables separately for answer synthesis
4. When summary matches query, pass raw table to LLM

**Source**: [LangChain Multi-Vector Retriever](https://blog.langchain.com/semi-structured-multi-modal-rag/)

#### Table Chunking Best Practices

- **Never split tables** - Treat as atomic blocks
- **Contextual enrichment** - Add surrounding document context
- **Format standardization** - Convert to uniform markdown
- **Header repetition** - For large tables, repeat headers every ~100 rows

**Source**: [Mastering RAG for Table-Heavy Documents](https://kx.com/blog/mastering-rag-precision-techniques-for-table-heavy-documents/)

---

## Current State Analysis

### Header Metadata (Implemented)

```
Chunk
├── header_text: Optional[str]    # "Chapter 5: Inventory"
├── header_level: Optional[int]   # 1-6 (h1-h6)
└── block_type: Optional[str]     # "Header", "Text", "Table"
```

**Current usage**: Stored in database but NOT used by concept extraction.

### Table Handling (Limited)

| Stage | Current | Limitation |
|-------|---------|------------|
| Detection | `has_table: bool` | Only presence, no parsing |
| Parsing | `SpecialBlock(type="table")` | Raw markdown preserved |
| Chunking | Atomic (never split) | No structure extraction |
| Embedding | Raw markdown text | Poor semantic matching |
| Search | Text similarity | Can't query by column/cell |

---

## Proposed Architecture

### Phase 1: Header-Aware Concept Extraction

```
                    ┌─────────────────────────────────────┐
                    │         Chunk with Metadata          │
                    │  ├── content: "FIFO is a method..."  │
                    │  ├── header_text: "Inventory Methods"│
                    │  ├── header_level: 2                 │
                    │  ├── section_path: ["Ch5", "COGS"]   │
                    │  └── semantic_type: "definition"     │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │    Enhanced Concept Extraction       │
                    │  1. Filter by semantic_type          │
                    │  2. Include header context in prompt │
                    │  3. Infer relationships from headers │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌─────────────────┐               ┌─────────────────┐
          │ Concept: "FIFO" │               │ Relationship    │
          │ type: METHOD    │◄──────────────│ parent: "COGS"  │
          │ context: header │               │ from: header h2 │
          └─────────────────┘               └─────────────────┘
```

### Phase 2: Advanced Table Processing

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Table Processing Pipeline                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: Table Extraction                                            │
│  ├── Parse markdown table structure                                  │
│  ├── Extract: column_names, row_count, cell_types                    │
│  └── Preserve surrounding context (headers, paragraphs)              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: Table Summarization (LLM)                                   │
│  ├── Generate natural language description                          │
│  ├── Include: purpose, columns, key data points                     │
│  └── Format: "This table compares X and Y across Z dimensions..."   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: Multi-Vector Storage                                        │
│  ├── Embed summary → for retrieval (semantic search)                 │
│  ├── Store raw table → for synthesis (LLM answer generation)         │
│  └── Link: TableSummary --HasRawTable--> TableContent                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: Table Metadata in Graph                                     │
│  ├── TableNode: columns, row_count, data_types                       │
│  ├── Edges: Document --HasTable--> Table                             │
│  ├── Edges: Table --RelatedTo--> Concept                             │
│  └── Edges: Table --InSection--> Chunk (header context)              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Header-Aware Concept Extraction

#### 1.1 Enhance Concept Extractor Prompts

**File**: `src/chunker/concept_extractor.py`

```python
# Current prompt (simplified)
prompt = f"""
Extract concepts from this text:
Section: {section_path}
Content: {content}
"""

# Enhanced prompt
prompt = f"""
Extract concepts from this text.

Context:
- Document: {document_title}
- Section: {' > '.join(section_path)}
- Current Header: {header_text} (h{header_level})
- Content Type: {semantic_type}

Content:
{content}

Instructions:
1. The header "{header_text}" indicates the topic context
2. This content is classified as "{semantic_type}" - prioritize accordingly
3. Infer parent concepts from the header hierarchy
"""
```

#### 1.2 Filter Chunks by Semantic Type

```python
def extract_concepts_smart(chunks: list[Chunk], metadata: list[ChunkMetadata]):
    """Extract concepts with priority filtering."""

    # Priority 1: Definition chunks (most likely to contain concept definitions)
    definition_chunks = [
        (c, m) for c, m in zip(chunks, metadata)
        if m.semantic_type == "definition"
    ]

    # Priority 2: Theorem/procedure chunks
    structured_chunks = [
        (c, m) for c, m in zip(chunks, metadata)
        if m.semantic_type in ("theorem", "procedure")
    ]

    # Priority 3: Header chunks (section titles often name concepts)
    header_chunks = [
        (c, m) for c, m in zip(chunks, metadata)
        if m.header_level is not None and m.header_level <= 3
    ]

    # Extract with priority (avoid duplicates)
    extracted = set()
    for chunk, meta in definition_chunks + structured_chunks + header_chunks:
        concepts = extract_from_chunk(chunk, meta)
        extracted.update(concepts)

    return extracted
```

#### 1.3 Infer Concept Relationships from Header Hierarchy

```python
def infer_relationships_from_headers(
    chunks: list[Chunk],
    metadata: list[ChunkMetadata],
    concepts: list[Concept]
) -> list[ConceptRelation]:
    """Infer parent-child relationships from header hierarchy."""

    relations = []
    header_stack = []  # [(header_text, header_level, concepts)]

    for chunk, meta in zip(chunks, metadata):
        if meta.header_level:
            # Pop headers at same or higher level
            while header_stack and header_stack[-1][1] >= meta.header_level:
                header_stack.pop()

            # Find concepts in this chunk
            chunk_concepts = [c for c in concepts if c.source_chunk_id == chunk.id]

            # If parent header exists, create relationships
            if header_stack and chunk_concepts:
                parent_header = header_stack[-1]
                for concept in chunk_concepts:
                    for parent_concept in parent_header[2]:
                        relations.append(ConceptRelation(
                            source=parent_concept.name,
                            target=concept.name,
                            relation_type="parent_of",
                            inferred_from="header_hierarchy"
                        ))

            header_stack.append((meta.header_text, meta.header_level, chunk_concepts))

    return relations
```

#### 1.4 New Tests

**File**: `tests/test_concept_header_integration.py`

- `test_concept_extraction_includes_header_context`
- `test_definition_chunks_prioritized`
- `test_header_hierarchy_infers_relationships`
- `test_semantic_type_filtering`

---

### Phase 2: Advanced Table Processing

#### 2.1 Table Parser

**File**: `src/chunker/table_parser.py` (new)

```python
@dataclass
class ParsedTable:
    """Structured representation of a markdown table."""

    raw_markdown: str
    column_names: list[str]
    rows: list[list[str]]
    row_count: int
    column_count: int
    column_types: list[str]  # "text", "numeric", "date", "mixed"
    has_header_row: bool
    surrounding_context: str  # Text before/after table
    source_chunk_id: str

    def to_markdown_kv(self) -> str:
        """Convert to Markdown-KV format (best for LLM accuracy)."""
        lines = []
        for row in self.rows:
            for col_name, value in zip(self.column_names, row):
                lines.append(f"- {col_name}: {value}")
            lines.append("")  # Blank line between rows
        return "\n".join(lines)

    def to_summary_prompt(self) -> str:
        """Generate prompt for LLM summarization."""
        return f"""
Summarize this table in 2-3 sentences.

Table columns: {', '.join(self.column_names)}
Number of rows: {self.row_count}
Context: {self.surrounding_context[:200]}

Table:
{self.raw_markdown}

Provide a natural language summary describing:
1. What this table contains
2. The key relationships or comparisons shown
3. Any notable values or patterns
"""


def parse_markdown_table(content: str, chunk_id: str) -> Optional[ParsedTable]:
    """Parse a markdown table into structured form."""

    # Regex for markdown table
    table_pattern = r'\|(.+)\|\n\|[-:\s|]+\|\n((?:\|.+\|\n?)+)'
    match = re.search(table_pattern, content)

    if not match:
        return None

    header_row = match.group(1)
    body = match.group(2)

    # Parse columns
    columns = [c.strip() for c in header_row.split('|') if c.strip()]

    # Parse rows
    rows = []
    for line in body.strip().split('\n'):
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if cells:
            rows.append(cells)

    # Infer column types
    column_types = infer_column_types(columns, rows)

    # Get surrounding context
    table_start = content.find(match.group(0))
    context_before = content[:table_start].strip()[-200:]

    return ParsedTable(
        raw_markdown=match.group(0),
        column_names=columns,
        rows=rows,
        row_count=len(rows),
        column_count=len(columns),
        column_types=column_types,
        has_header_row=True,
        surrounding_context=context_before,
        source_chunk_id=chunk_id,
    )


def infer_column_types(columns: list[str], rows: list[list[str]]) -> list[str]:
    """Infer column data types from content."""
    types = []
    for col_idx in range(len(columns)):
        values = [row[col_idx] for row in rows if col_idx < len(row)]

        # Check if numeric
        numeric_count = sum(1 for v in values if re.match(r'^[\d,.$%()-]+$', v.strip()))
        if numeric_count > len(values) * 0.8:
            types.append("numeric")
        # Check if date
        elif any(re.match(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', v) for v in values):
            types.append("date")
        else:
            types.append("text")

    return types
```

#### 2.2 Table Summarizer

**File**: `src/chunker/table_summarizer.py` (new)

```python
@dataclass
class TableSummary:
    """LLM-generated summary of a table for embedding."""

    table_id: str
    summary_text: str  # Natural language description
    key_columns: list[str]
    key_values: list[str]  # Notable data points
    embedding: Optional[list[float]] = None


class TableSummarizer:
    """Generate natural language summaries of tables using LLM."""

    def __init__(self, llm_provider: str = "anthropic"):
        self.client = self._init_client(llm_provider)

    def summarize(self, table: ParsedTable) -> TableSummary:
        """Generate a summary for semantic search."""

        prompt = table.to_summary_prompt()

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        summary_text = response.content[0].text

        # Extract key columns and values
        key_columns = self._identify_key_columns(table)
        key_values = self._extract_key_values(table)

        return TableSummary(
            table_id=f"table_{table.source_chunk_id}",
            summary_text=summary_text,
            key_columns=key_columns,
            key_values=key_values,
        )

    def _identify_key_columns(self, table: ParsedTable) -> list[str]:
        """Identify the most important columns."""
        # Heuristic: first column + any numeric columns
        key = [table.column_names[0]]
        for i, col_type in enumerate(table.column_types):
            if col_type == "numeric" and table.column_names[i] not in key:
                key.append(table.column_names[i])
        return key[:3]  # Max 3 key columns

    def _extract_key_values(self, table: ParsedTable) -> list[str]:
        """Extract notable values from the table."""
        values = []
        # First row values (often important)
        if table.rows:
            values.extend(table.rows[0][:3])
        # Last row values (often totals)
        if len(table.rows) > 1:
            values.extend(table.rows[-1][:3])
        return values
```

#### 2.3 Schema Updates

**File**: `src/storage/schema.py`

```python
# New node type for tables
TABLE_NODE_SCHEMA: dict[str, str] = {
    "table_id": "String",
    "raw_markdown": "String",
    "summary_text": "String",
    "column_names": "String",  # JSON array
    "column_types": "String",  # JSON array
    "row_count": "U32",
    "column_count": "U32",
    "source_chunk_id": "String",
    "source_document_id": "String",
    "surrounding_context": "String",
}

# New edge types
EDGE_TYPES.update({
    "HasTable": ("Document", "Table"),
    "ChunkContainsTable": ("Chunk", "Table"),
    "TableRelatedToConcept": ("Table", "Concept"),
    "HasTableEmbedding": ("Table", "TableVector"),
})
```

#### 2.4 Multi-Vector Retrieval

**File**: `src/storage/table_store.py` (new)

```python
class TableStore:
    """Storage and retrieval for tables with multi-vector approach."""

    def store_table(
        self,
        table: ParsedTable,
        summary: TableSummary,
        document_id: str,
    ) -> str:
        """Store table with summary embedding for retrieval."""

        # Embed the summary (for retrieval)
        summary_embedding = self.embedder.embed_text(summary.summary_text)

        # Store table node
        table_id = self._store_table_node(table, summary)

        # Store summary embedding (linked to table)
        self._store_table_embedding(table_id, summary_embedding)

        # Create edges
        self._create_edge("HasTable", document_id, table_id)
        self._create_edge("ChunkContainsTable", table.source_chunk_id, table_id)

        return table_id

    def search_tables(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[ParsedTable, float]]:
        """Search tables by summary similarity."""

        query_embedding = self.embedder.embed_text(query)

        # Search summary embeddings
        results = self._search_table_embeddings(query_embedding, top_k)

        # Return raw tables (for LLM synthesis)
        tables = []
        for table_id, score in results:
            table = self._get_table_by_id(table_id)
            tables.append((table, score))

        return tables

    def get_tables_for_concept(self, concept_name: str) -> list[ParsedTable]:
        """Get tables related to a concept via graph traversal."""

        # Query: Concept <--TableRelatedToConcept-- Table
        return self._traverse_edge_reverse("TableRelatedToConcept", concept_name)
```

#### 2.5 Integration with Ingestion Pipeline

**File**: `src/cli/commands/ingest.py` (modifications)

```python
def ingest_markdown(...):
    # ... existing stages ...

    # Stage 5: Table Processing (NEW)
    if extract_tables:
        console.print("[bold blue]Stage 5:[/] Processing tables...")

        table_parser = TableParser()
        table_summarizer = TableSummarizer()
        table_store = TableStore(store_instance)

        for chunk, metadata in zip(chunks, metadata_list):
            if metadata.has_table:
                # Parse table structure
                parsed_table = table_parser.parse(chunk.content, chunk.id)

                if parsed_table:
                    # Generate summary
                    summary = table_summarizer.summarize(parsed_table)

                    # Store with multi-vector approach
                    table_id = table_store.store_table(
                        parsed_table, summary, doc_id
                    )

                    # Link to related concepts
                    related_concepts = find_related_concepts(
                        parsed_table, extracted_concepts
                    )
                    for concept in related_concepts:
                        table_store.link_table_to_concept(table_id, concept.name)

        console.print(f"  Processed {table_count} tables")
```

#### 2.6 New MCP Tools

**File**: `src/mcp_server/tools.py` (additions)

```python
@mcp.tool()
def search_tables(
    query: str,
    top_k: int = 5,
    doc_id: Optional[str] = None,
) -> list[dict]:
    """Search tables by semantic similarity to query.

    Returns table summaries and raw markdown for LLM synthesis.
    """
    results = table_store.search_tables(query, top_k, doc_id)
    return [
        {
            "table_id": t.table_id,
            "summary": t.summary_text,
            "columns": t.column_names,
            "row_count": t.row_count,
            "raw_markdown": t.raw_markdown,
            "score": score,
        }
        for t, score in results
    ]


@mcp.tool()
def get_table(table_id: str) -> dict:
    """Get a specific table by ID with full metadata."""
    table = table_store.get_table(table_id)
    return {
        "table_id": table.table_id,
        "columns": table.column_names,
        "column_types": table.column_types,
        "rows": table.rows,
        "raw_markdown": table.raw_markdown,
        "summary": table.summary_text,
        "context": table.surrounding_context,
    }


@mcp.tool()
def get_tables_for_concept(concept_name: str) -> list[dict]:
    """Get all tables related to a concept."""
    tables = table_store.get_tables_for_concept(concept_name)
    return [
        {
            "table_id": t.table_id,
            "summary": t.summary_text,
            "columns": t.column_names,
        }
        for t in tables
    ]
```

#### 2.7 CLI Commands

**File**: `src/cli/commands/tables.py` (new)

```python
@app.command()
def list(
    doc_id: Optional[str] = None,
    json_output: bool = Option(False, "--json"),
):
    """List all tables in the knowledge base."""
    tables = table_store.list_tables(doc_id)
    # ... output formatting ...


@app.command()
def show(table_id: str):
    """Show table details including structure and summary."""
    table = table_store.get_table(table_id)
    # ... rich table output ...


@app.command()
def search(query: str, top_k: int = 5):
    """Search tables by semantic similarity."""
    results = table_store.search_tables(query, top_k)
    # ... output formatting ...
```

---

## Testing Strategy

### Phase 1 Tests

| Test File | Coverage |
|-----------|----------|
| `test_concept_header_integration.py` | Header context in extraction |
| `test_concept_filtering.py` | Semantic type filtering |
| `test_concept_hierarchy.py` | Header-based relationship inference |

### Phase 2 Tests

| Test File | Coverage |
|-----------|----------|
| `test_table_parser.py` | Markdown table parsing |
| `test_table_summarizer.py` | LLM summarization |
| `test_table_store.py` | Multi-vector storage |
| `test_table_search.py` | Table semantic search |
| `test_table_concept_linking.py` | Table-concept relationships |
| `test_table_mcp_tools.py` | MCP tool integration |
| `test_table_cli.py` | CLI commands |

---

## Configuration

### New Config Options

```toml
# .kidkazz.toml

[tables]
enabled = true
summarize = true                    # Generate LLM summaries
summarizer_model = "claude-sonnet-4-20250514"  # Model for summarization
format = "markdown-kv"              # Output format for LLM: markdown-kv, html, raw
max_rows_for_summary = 50           # Skip summarization for very large tables
link_to_concepts = true             # Create table-concept relationships

[concept_extraction]
use_header_context = true           # Include header text in prompts
filter_by_semantic_type = true      # Prioritize definition chunks
infer_header_relationships = true   # Create relationships from h1>h2>h3
priority_types = ["definition", "theorem", "procedure"]
```

---

## Migration Path

### For Existing Documents

1. **Re-parse PDFs** - Required if using Reducto block mode (headers need reconstruction)
2. **Re-ingest markdown** - Required to extract table metadata and summaries
3. **Backward compatible** - Old documents continue to work, just without enhanced features

### CLI Command

```bash
# Re-process existing documents with new features
kidkazz docs reprocess --doc-id my_doc --extract-tables --extract-concepts

# Batch reprocess all documents
kidkazz docs reprocess --all --extract-tables --extract-concepts
```

---

## Dependencies

### New Dependencies

```toml
# pyproject.toml

[project.optional-dependencies]
tables = [
    "tabulate>=0.9.0",      # Table formatting
]
```

### Existing Dependencies Used

- `anthropic` - LLM for table summarization
- `instructor` - Structured extraction
- `fastembed` / `openai` - Embeddings for table summaries

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Table retrieval precision | >80% (vs current ~50%) |
| Concept extraction recall | +20% improvement |
| Cross-document concept linking | +30% more relationships |
| Query latency (tables) | <500ms |

---

## Timeline Estimate

| Phase | Tasks | Effort |
|-------|-------|--------|
| 1.1 | Enhanced prompts | Small |
| 1.2 | Semantic type filtering | Small |
| 1.3 | Header hierarchy relationships | Medium |
| 1.4 | Tests for Phase 1 | Small |
| 2.1 | Table parser | Medium |
| 2.2 | Table summarizer | Medium |
| 2.3 | Schema updates | Small |
| 2.4 | Multi-vector storage | Large |
| 2.5 | Ingestion integration | Medium |
| 2.6 | MCP tools | Small |
| 2.7 | CLI commands | Small |
| 2.8 | Tests for Phase 2 | Medium |

---

## References

### Research Sources

- [Which Table Format Do LLMs Understand Best?](https://www.improvingagents.com/blog/best-input-data-format-for-llms/) - Format comparison study
- [Table Meets LLM](https://arxiv.org/html/2305.13062v4) - Benchmark for LLM table understanding
- [HtmlRAG](https://arxiv.org/html/2411.02959v1) - HTML outperforms plain text for RAG
- [LangChain Multi-Vector Retriever](https://blog.langchain.com/semi-structured-multi-modal-rag/) - Semi-structured RAG
- [Mastering RAG for Table-Heavy Documents](https://kx.com/blog/mastering-rag-precision-techniques-for-table-heavy-documents/) - Table extraction best practices
- [Pinecone: Vectorizing Structured Data](https://www.pinecone.io/learn/structured-data/) - Embedding strategies
- [LlamaIndex Recursive Retrieval](https://www.datacamp.com/tutorial/recursive-retrieval-rag-llamaindex) - Multi-vector approaches
- [Enhancing RAG: Best Practices Study](https://arxiv.org/abs/2501.07391) - 2025 RAG research

### Existing Codebase

- `src/chunker/concept_extractor.py` - Current concept extraction
- `src/chunker/metadata.py` - Header metadata extraction
- `src/chunker/parser.py` - SpecialBlock for tables
- `src/storage/schema.py` - Current schema
- `src/cli/commands/ingest.py` - Ingestion pipeline
