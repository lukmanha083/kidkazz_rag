# KidKazz RAG

A RAG (Retrieval-Augmented Generation) system for converting PDF textbooks to searchable knowledge bases.

## Overview

This project provides tools to:
1. Convert PDF documents to Markdown (with table and image support)
2. Chunk and embed documents using LlamaIndex
3. Store embeddings in Helix-DB (vector + graph database)
4. Query documents via MCP integration with Claude Code

## Architecture

```
PDF Document
     |
     v
[PDF to Markdown Converter] ──── Google Colab (GPU)
     |
     v
Markdown Document
     |
     v
[LlamaIndex Chunking/Indexing] ── Local Python
     |
     v
[Helix-DB] ──────────────────── Vector + Graph Storage
     |
     v
[MCP Server] ────────────────── Claude Code Integration
     |
     v
Chat with your documents
```

## Prerequisites

- VS Code with [Google Colab Extension](https://marketplace.visualstudio.com/items?itemName=GoogleCloudPlatform.colab-vscode-plugin)
- Google account (for Colab GPU access)
- Python 3.10+

## Quick Start

### Step 1: PDF to Markdown Conversion

1. **Install VS Code Colab Extension**
   ```
   Open VS Code → Extensions → Search "Colab" → Install
   ```

2. **Enable Experimental Features** (for file upload)
   ```
   VS Code Settings → Search "colab" → Enable experimental upload
   ```

3. **Open the Converter Notebook**
   ```
   Open: notebooks/pdf_to_markdown_converter.ipynb
   Connect to Colab runtime (GPU enabled)
   ```

4. **Upload Your PDF**
   ```
   Right-click your PDF in VS Code Explorer
   Select "Upload to Colab server"
   ```

5. **Run the Notebook**
   - Cell 1: Environment setup + PDF detection
   - Cell 2: Select PDF and conversion tool
   - Cell 3: Install tool + convert
   - Cell 4-5: Preview + quality check
   - Cell 6: Download result

### Tool Selection Guide

| Your PDF Has | Recommended Tool |
|--------------|------------------|
| Heavy math/equations | Nougat |
| Many tables/data | Docling |
| Mostly text | Marker (fastest) |
| Mixed content | Docling |

## Project Structure

```
kidkazz_rag/
├── README.md
├── notebooks/
│   └── pdf_to_markdown_converter.ipynb  # PDF → Markdown (Colab)
├── src/                                  # (Coming soon)
│   ├── chunking/                         # LlamaIndex chunking
│   ├── embedding/                        # Embedding generation
│   └── mcp/                              # MCP server
└── data/                                 # (Coming soon)
    ├── raw/                              # Original PDFs
    ├── markdown/                         # Converted markdown
    └── vectors/                          # Helix-DB storage
```

## Notebook Usage

### PDF to Markdown Converter

The notebook auto-detects uploaded PDFs and guides you through tool selection:

```
==================================================
TOOL RECOMMENDATION QUIZ
==================================================

1. Primary content type?
   [1] Heavy math/equations
   [2] Many tables/data
   [3] Mostly text
   [4] Mixed content
   Choice: 3

2. PDF type?
   [1] Scanned/image-based
   [2] Digital (selectable text)
   [3] Mixed/unsure
   Choice: 2

3. Priority?
   [1] Speed
   [2] Quality
   [3] Balance
   Choice: 1

==================================================
RECOMMENDATION
==================================================
  Tool: MARKER
  Why: Fast, clean layouts
  Est. Time: ~1-2 min/100pg
```

### Quality Analysis

After conversion, the notebook analyzes output quality:

```
==================================================
QUALITY ANALYSIS
==================================================

File: textbook_marker.md

Content Statistics:
  Characters      1,234,567
  Words           205,432
  Lines           15,234
  Headings        342
  Tables          56
  Code blocks     23
  Images          89
  Math blocks     12

Quality Indicators:
  No issues detected

Estimated original pages: ~685
```

## Conversion Tools Comparison

| Tool | Speed | Tables | Math | Images | Best For |
|------|-------|--------|------|--------|----------|
| **Marker** | Fast (~1-2 min/100pg) | Excellent | Good | Basic | Clean PDFs, books |
| **Docling** | Medium (~2-3 min/100pg) | Great | Good | Describes | Mixed content |
| **Nougat** | Slow (~5-10 min/100pg) | Basic | Excellent | Skips | Academic papers |

## Troubleshooting

### No GPU Detected
```
Runtime → Change runtime type → Hardware accelerator → GPU
```

### PDF Not Found
```
Right-click PDF in VS Code → "Upload to Colab server"
Then re-run Cell 1 to detect it
```

### Out of Memory
- Use Marker (most memory efficient)
- Runtime → Restart runtime
- Try smaller PDF or split into sections

### Poor Conversion Quality
- Try different tool (Nougat for math, Docling for tables)
- Check if PDF is scanned vs digital
- Consider preprocessing with OCR first

### File Not Visible in VS Code
- Click refresh icon in Colab workspace panel
- Check `/content/output/` directory

## Roadmap

- [x] PDF to Markdown converter (Colab notebook)
- [ ] LlamaIndex chunking pipeline
- [ ] Embedding generation
- [ ] Helix-DB integration
- [ ] MCP server for Claude Code
- [ ] End-to-end CLI tool

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License
