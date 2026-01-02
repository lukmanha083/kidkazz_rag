# Phase 2: LlamaIndex Chunking Pipeline - Implementation Plan

## Overview

Build a chunking pipeline optimized for **both vector search AND graph traversal** (for Helix-DB), running locally on CPU (AMD Ryzen Pro 7).

---

## Key Decisions

### 1. Embedding Model: FastEmbed (CPU-optimized)

**Why FastEmbed instead of GPU:**
- Your AMD Ryzen Pro 7 (Renoir) doesn't support ROCm
- Vulkan GPU acceleration for embeddings is impractical (no mature libraries)
- FastEmbed uses ONNX Runtime with INT8 quantization - **3x faster than baseline**
- CPU inference for embeddings is actually faster than GPU for single queries (no data transfer overhead)
- Model: `BAAI/bge-small-en-v1.5` (state-of-the-art, ~33MB quantized)

### 2. Chunking Strategy: Hierarchical + Markdown-Aware

**Three-level hierarchy for vector + graph:**

| Level | Token Size | Purpose | Graph Role |
|-------|------------|---------|------------|
| 0 | Full doc | Root reference | Document node |
| 1 | 1024-2048 | Context synthesis | Section nodes |
| 2 | 256-512 | Vector search | Leaf nodes |

**Why this works for Helix-DB:**
- **Vector search**: Level 2 chunks (~512 tokens) are optimal for semantic similarity
- **Graph traversal**: Parent-child relationships enable context expansion
- **Hybrid queries**: Find via vector, expand via graph

### 3. Environment: Local Python Module

- Runs on CPU (no Colab needed)
- Integrates with Phase 1 markdown output
- Dependencies: `llama-index-core`, `fastembed`

---

## Module Structure

```
src/
├── pdf_converter/     # Phase 1 (existing)
└── chunker/           # Phase 2 (new)
    ├── __init__.py    # Public API exports
    ├── parser.py      # Markdown structure extraction
    ├── chunker.py     # Hierarchical chunking logic
    ├── metadata.py    # Metadata enrichment for vector+graph
    └── embedder.py    # FastEmbed integration

tests/
├── test_parser.py     # Parser tests
├── test_chunker.py    # Chunker tests
├── test_metadata.py   # Metadata tests
└── test_embedder.py   # Embedder tests
```

---

## Implementation Details

### 1. `parser.py` - Markdown Structure Extraction

```python
@dataclass
class MarkdownSection:
    """Represents a section in the markdown document."""
    heading: str
    level: int  # 1-6 for h1-h6
    content: str
    start_line: int
    end_line: int
    children: list["MarkdownSection"]

def parse_markdown_structure(content: str) -> MarkdownSection:
    """Parse markdown into hierarchical structure based on headings."""

def extract_special_blocks(content: str) -> dict:
    """Extract tables, code blocks, math blocks as atomic units."""

def get_heading_hierarchy(sections: list[MarkdownSection]) -> list[str]:
    """Get breadcrumb path like ['Chapter 1', 'Section 1.2', 'Topic']."""
```

### 2. `chunker.py` - Hierarchical Chunking

```python
@dataclass
class Chunk:
    """A chunk with content and relationships."""
    id: str
    content: str
    level: int  # 0=doc, 1=section, 2=leaf
    token_count: int

    # Graph relationships
    parent_id: Optional[str]
    child_ids: list[str]
    prev_id: Optional[str]
    next_id: Optional[str]

    # Metadata
    metadata: dict

def create_hierarchical_chunks(
    content: str,
    level_sizes: tuple[int, int, int] = (2048, 512, 256),
    overlap_ratio: float = 0.15
) -> list[Chunk]:
    """Create multi-level chunks optimized for vector+graph."""

def preserve_atomic_blocks(content: str, max_tokens: int) -> list[str]:
    """Never split tables, code blocks, or math equations."""
```

### 3. `metadata.py` - Metadata Enrichment

```python
@dataclass
class ChunkMetadata:
    """Metadata for vector search + graph traversal."""
    # Identity
    chunk_id: str
    document_id: str

    # Vector search optimization
    semantic_type: str  # definition, example, procedure, narrative
    topic_tags: list[str]
    token_count: int

    # Graph traversal optimization
    hierarchy_level: int
    section_path: list[str]  # Breadcrumb
    sequence_position: int

    # Relationships (for Helix-DB graph)
    parent_id: Optional[str]
    child_ids: list[str]
    sibling_ids: list[str]

def infer_semantic_type(content: str) -> str:
    """Classify chunk as definition/example/procedure/narrative."""

def extract_topic_tags(content: str) -> list[str]:
    """Extract key topics/entities from chunk."""

def enrich_chunk_metadata(chunk: Chunk, doc_structure: MarkdownSection) -> ChunkMetadata:
    """Add all metadata needed for vector+graph storage."""
```

### 4. `embedder.py` - FastEmbed Integration

```python
from fastembed import TextEmbedding

@dataclass
class EmbeddedChunk:
    """Chunk with embedding vector."""
    chunk: Chunk
    embedding: list[float]
    model_name: str

class ChunkEmbedder:
    """CPU-optimized embedding generator using FastEmbed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)
        self.model_name = model_name

    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        """Generate embedding for a single chunk."""

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> list[EmbeddedChunk]:
        """Batch embed multiple chunks (CPU-optimized)."""

    def get_embedding_dim(self) -> int:
        """Return embedding dimension (384 for bge-small)."""
```

---

## Public API (`__init__.py`)

```python
from .parser import parse_markdown_structure, MarkdownSection
from .chunker import create_hierarchical_chunks, Chunk
from .metadata import enrich_chunk_metadata, ChunkMetadata
from .embedder import ChunkEmbedder, EmbeddedChunk

__all__ = [
    # Parser
    "parse_markdown_structure",
    "MarkdownSection",
    # Chunker
    "create_hierarchical_chunks",
    "Chunk",
    # Metadata
    "enrich_chunk_metadata",
    "ChunkMetadata",
    # Embedder
    "ChunkEmbedder",
    "EmbeddedChunk",
]
```

---

## Dependencies (pyproject.toml)

```toml
[project.optional-dependencies]
chunker = [
    "llama-index-core>=0.10.0",
    "fastembed>=0.2.0",
    "tiktoken>=0.5.0",  # Token counting
]
```

---

## Test Plan

### test_parser.py (~15 tests)
- Parse simple markdown with headings
- Extract heading hierarchy
- Handle nested headings (h1 > h2 > h3)
- Extract tables as atomic blocks
- Extract code blocks as atomic blocks
- Handle empty content
- Handle markdown without headings

### test_chunker.py (~20 tests)
- Create chunks at correct sizes
- Respect token limits
- Maintain parent-child relationships
- Maintain prev/next relationships
- Never split tables
- Never split code blocks
- Apply overlap correctly
- Handle short content (< min chunk size)
- Handle very long sections

### test_metadata.py (~15 tests)
- Infer semantic type (definition, example, etc.)
- Extract topic tags
- Build section path correctly
- Set sequence positions
- Link parent/child/sibling IDs

### test_embedder.py (~10 tests)
- Generate embeddings
- Correct embedding dimension
- Batch embedding works
- Model loads on CPU
- Handle empty content

**Total: ~60 tests**

---

## Usage Example

```python
from src.chunker import (
    parse_markdown_structure,
    create_hierarchical_chunks,
    enrich_chunk_metadata,
    ChunkEmbedder
)

# 1. Load markdown from Phase 1
with open("output/textbook.md") as f:
    content = f.read()

# 2. Parse structure
doc_structure = parse_markdown_structure(content)

# 3. Create hierarchical chunks
chunks = create_hierarchical_chunks(
    content,
    level_sizes=(2048, 512, 256),
    overlap_ratio=0.15
)

# 4. Enrich metadata
for chunk in chunks:
    chunk.metadata = enrich_chunk_metadata(chunk, doc_structure)

# 5. Generate embeddings (CPU-optimized)
embedder = ChunkEmbedder(model_name="BAAI/bge-small-en-v1.5")
embedded_chunks = embedder.embed_chunks(chunks, batch_size=32)

# 6. Ready for Helix-DB storage (Phase 4)
for ec in embedded_chunks:
    print(f"Chunk {ec.chunk.id}: {len(ec.embedding)} dims")
    print(f"  Parent: {ec.chunk.parent_id}")
    print(f"  Children: {ec.chunk.child_ids}")
```

---

## Performance Expectations (AMD Ryzen Pro 7)

| Operation | Expected Speed |
|-----------|----------------|
| Parse 500-page textbook markdown | < 1 second |
| Chunk into ~1000 chunks | < 2 seconds |
| Embed 1000 chunks (batch) | ~30-60 seconds |
| Total pipeline | < 2 minutes |

---

## Success Criteria

1. All ~60 unit tests pass
2. Chunks maintain hierarchical relationships
3. Tables and code blocks never split
4. Embeddings generate on CPU without GPU
5. Metadata supports both vector search and graph traversal
6. Coverage > 85%

---

## Next Phase (Phase 3: Helix-DB Integration)

After this phase, chunks with embeddings and metadata are ready for:
- Vector storage in Helix-DB
- Graph relationships for traversal
- MCP server for Claude Code queries
