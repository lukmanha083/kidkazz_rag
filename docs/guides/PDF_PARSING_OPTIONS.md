# PDF Parsing Options Guide

This guide explains the chunk modes and agentic flag options for `kidkazz inbox parse` to help you choose the optimal settings for your use case.

## Quick Reference

```bash
# Default (human-readable output)
kidkazz inbox parse

# RAG-optimized (recommended for most users)
kidkazz inbox parse --chunk-mode variable

# Maximum feature leverage (recommended for complex documents)
kidkazz inbox parse --chunk-mode block

# High-accuracy for problematic PDFs
kidkazz inbox parse --chunk-mode block --agentic
```

## Chunk Modes Explained

The `--chunk-mode` (or `-c`) flag controls how Reducto.ai splits your PDF into chunks.

| Mode | Chunk Size | Best For | Feature Support |
|------|------------|----------|-----------------|
| `disabled` | Full document | Human reading | Basic |
| `variable` | ~1000 chars | RAG semantic search | Good |
| `block` | 1 element | Citations & precision | **Full** |
| `page` | ~1 page | Page-level citations | Moderate |
| `section` | Varies | Section-level retrieval | Good |

### Variable Mode (Default for RAG)

```bash
kidkazz inbox parse --chunk-mode variable
```

**How it works:** Combines multiple elements (paragraphs, headers) into adaptive ~1000 character chunks.

**Pros:**
- Better semantic context per chunk (more words = better embedding quality)
- Fewer total chunks = faster search
- Lower storage requirements

**Cons:**
- Harder to cite exact source (chunk may span multiple sections)
- Less precise retrieval
- Block metadata may be aggregated

**Best for:** General RAG applications, semantic search, Q&A systems

### Block Mode (Recommended for Full Features)

```bash
kidkazz inbox parse --chunk-mode block
```

**How it works:** Each document element (header, paragraph, table, code block) becomes its own chunk.

**Pros:**
- Preserves block metadata: `header_text`, `header_level`, `block_type`
- Precise citation capability
- Tables isolated with `has_table=True` flag
- More accurate parent/child/sibling graph relationships
- Enables header-aware concept extraction

**Cons:**
- More chunks to store and search
- Smaller context per embedding
- Higher storage requirements

**Best for:**
- Documents requiring precise citations
- Table-heavy content (accounting, scientific data)
- Leveraging full KidKazz RAG feature set
- Graph-based retrieval

### Feature Comparison by Mode

| Feature | disabled | variable | block |
|---------|----------|----------|-------|
| Semantic search | Basic | Good | Good |
| Keyword search | Basic | Good | Good |
| Header metadata | No | Partial | **Full** |
| Table extraction | No | Partial | **Full** |
| Graph traversal | No | Good | **Best** |
| Concept extraction | Basic | Good | **Best** |
| Precise citations | No | No | **Yes** |
| `search_tables` MCP tool | No | Limited | **Full** |
| `get_tables_for_concept` | No | Limited | **Full** |

## Agentic Mode Explained

The `--agentic` flag enables AI-enhanced accuracy in Reducto.ai parsing.

```bash
# Without agentic (standard parsing)
kidkazz inbox parse --chunk-mode block

# With agentic (AI-enhanced)
kidkazz inbox parse --chunk-mode block --agentic
```

### Comparison

| Aspect | Without `--agentic` | With `--agentic` |
|--------|---------------------|------------------|
| **Parsing method** | Standard OCR/extraction | AI-enhanced multi-pass |
| **Accuracy** | ~90-95% (varies by PDF) | 99%+ |
| **Credit cost** | 1x | 2x |
| **Speed** | Faster | Slightly slower |

### When to Use Agentic Mode

**Skip agentic (save credits) for:**
- Digital-native PDFs (Word, LaTeX, Google Docs exports)
- Simple layouts (single column, standard fonts)
- Clean, well-formatted documents
- Budget-conscious batch processing

**Use agentic for:**
- Scanned documents and textbooks
- Complex layouts (multi-column, sidebars)
- Documents with handwriting or annotations
- Tables with merged/complex cells
- Math formulas and equations
- Low-quality scans or photos of documents

### Risks Without Agentic Mode

For **high-quality digital PDFs**, risks are low:
- Standard parsing works well
- Block metadata extraction works normally
- Headers, tables, code blocks detected correctly

For **problematic PDFs**, potential issues include:
- OCR errors in text extraction
- Misidentified block types (header detected as paragraph)
- Table structure errors (merged/split cells)
- Math formulas may be garbled
- Missing or incorrect header levels

## Recommended Configurations

### By Document Type

| Document Type | Recommended Command |
|---------------|---------------------|
| Digital textbook (clean) | `--chunk-mode block` |
| Scanned textbook | `--chunk-mode block --agentic` |
| Research papers | `--chunk-mode block` |
| Financial reports (tables) | `--chunk-mode block --agentic` |
| Simple documentation | `--chunk-mode variable` |
| Legal documents | `--chunk-mode block --agentic` |
| Presentation slides | `--chunk-mode block` |

### By Use Case

| Use Case | Recommended Command |
|----------|---------------------|
| General Q&A system | `--chunk-mode variable` |
| Citation-required research | `--chunk-mode block` |
| Table data extraction | `--chunk-mode block --agentic` |
| Concept/glossary building | `--chunk-mode block` |
| Quick document search | `--chunk-mode variable` |
| Maximum accuracy needed | `--chunk-mode block --agentic` |

## Cost-Saving Strategy

Parse without `--agentic` first, then selectively re-parse problematic files:

```bash
# Step 1: Initial parse (1x credits)
kidkazz inbox parse --chunk-mode block

# Step 2: Check quality scores
kidkazz inbox list --output

# Step 3: Re-parse low-quality files only (2x credits for these only)
kidkazz inbox parse problematic_doc.pdf --chunk-mode block --agentic
```

### Quality Check Output

The `kidkazz inbox list --output` command shows quality metrics:

```
Output files in inbox/output/:
  accounting_101.md  [OK]     Quality: 0.95  Words: 12,450  Tables: 8
  scanned_notes.md   [WARN]   Quality: 0.72  Words: 3,200   Tables: 2
  lecture_slides.md  [OK]     Quality: 0.88  Words: 5,100   Tables: 0
```

Files with quality score below 0.80 may benefit from re-parsing with `--agentic`.

## Feature Integration

### Block Mode + Table Tools

With block mode, tables are properly isolated and indexed:

```bash
# Parse with block mode
kidkazz inbox parse --chunk-mode block

# Ingest with table extraction
kidkazz ingest markdown output/document.md --extract-tables

# Use table tools
kidkazz tables list
kidkazz tables search "inventory valuation methods"
```

### Block Mode + Concept Extraction

Block mode preserves header context for better concept extraction:

```bash
# Parse
kidkazz inbox parse --chunk-mode block

# Ingest
kidkazz ingest markdown output/document.md

# Extract concepts (header-aware)
kidkazz concepts extract doc_id --use-headers
```

### MCP Tools Requiring Block Mode

These MCP tools work best with block-mode parsed documents:

| Tool | Block Mode Benefit |
|------|-------------------|
| `search_tables` | Tables properly isolated with metadata |
| `get_table` | Full table structure preserved |
| `get_tables_for_concept` | Accurate table-concept links |
| `get_context_window` | Precise chunk boundaries |
| `get_parent` / `get_children` | Accurate graph relationships |

## Troubleshooting

### Low Quality Scores

If quality scores are consistently low:
1. Check PDF source quality (re-scan if possible)
2. Use `--agentic` flag
3. Consider alternative PDF source

### Missing Tables

If tables aren't detected:
1. Ensure `--chunk-mode block` is used
2. Use `--agentic` for complex table layouts
3. Check if tables use images instead of text

### Incorrect Headers

If header hierarchy is wrong:
1. Use `--agentic` for better block type detection
2. Check if PDF has proper heading styles (not just bold text)

### High Credit Usage

To reduce Reducto.ai credit usage:
1. Use `--chunk-mode variable` for simple documents
2. Skip `--agentic` for digital-native PDFs
3. Use `--dry-run` to preview before parsing
4. Batch similar-quality documents together

## Command Reference

```bash
kidkazz inbox parse [OPTIONS] [FILENAME]

Options:
  -c, --chunk-mode TEXT  Chunking mode: disabled, variable, block, page, section
                         [default: disabled]
  --agentic              Enable AI-enhanced accuracy (2x credits)
  --dry-run              Preview without making API calls
  --no-quality-check     Skip quality validation
  --no-sync-backup       Skip cloud backup after parsing
  --help                 Show help message

Examples:
  kidkazz inbox parse                         # Parse all, human-readable
  kidkazz inbox parse -c variable             # RAG-optimized chunks
  kidkazz inbox parse -c block                # Citation-level chunks
  kidkazz inbox parse -c block --agentic      # Maximum accuracy
  kidkazz inbox parse document.pdf            # Parse single file
  kidkazz inbox parse --dry-run               # Preview only
```
