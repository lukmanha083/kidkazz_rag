# Reducto.ai PDF Parsing Integration Plan

## Overview

Reducto.ai is the PDF parsing solution for KidKazz RAG. It provides API-based processing with rclone + Google Drive for storage and backup.

## Workflow

```text
LOCAL                              CLOUD                           LOCAL
─────────────────────────────────────────────────────────────────────────────
~/.kidkazz/inbox/
    │
    ├──── kidkazz inbox sync ────→ gdrive:kidkazz_inbox/ (backup)
    │
    ▼
kidkazz inbox parse --tool reducto
    │ (calls Reducto.ai API)
    │
    ▼
~/.kidkazz/output/*.md
    │
    └──── rclone copy ───────────→ gdrive:kidkazz_output/ (backup)
    │
    ▼
kidkazz ingest markdown
    │
    ▼
Knowledge Base (Helix-DB)
```

## Research Summary

### Reducto.ai Capabilities

| Feature | Details |
|---------|---------|
| API Endpoint | `https://platform.reducto.ai/parse` |
| Authentication | Bearer token (API key) |
| Python SDK | `pip install reductoai` |
| CLI Tool | `pip install reducto-cli` |
| Output | Structured JSON with chunks, or Markdown via CLI |
| Accuracy | 99.24% extraction accuracy (healthcare benchmark) |
| Features | Tables, figures, OCR, multi-language (100+ langs) |

### SDK Usage

```python
from reducto import Reducto

client = Reducto(api_key=os.environ.get("REDUCTO_API_KEY"))
response = client.parse.run(input="path/to/document.pdf")
```

### CLI Usage

```bash
reducto login                    # Authenticate via browser
reducto parse document.pdf       # Creates document.parse.md
reducto parse ./inbox/           # Batch process directory
```

### Key Configuration Options

| Option | Values | Description |
|--------|--------|-------------|
| `chunking.chunk_mode` | variable, section, page, block | How to chunk output |
| `table_output_format` | html, json, md, csv, dynamic | Table formatting |
| `extraction_mode` | ocr, hybrid | Processing mode |
| `agentic` | true/false | AI-enhanced accuracy (2x credits) |

## Implementation Plan

### Phase 1: Dependencies & Configuration

**Files to modify:**
- `pyproject.toml` - Add `reductoai` optional dependency
- `.env.example` - Add `REDUCTO_API_KEY` placeholder
- `src/cli/config.py` - Add reducto configuration fields

**New configuration:**
```toml
[reducto]
api_key_env = "REDUCTO_API_KEY"
agentic = false
chunk_mode = "variable"
table_format = "md"
```

**Environment variable:**
```bash
REDUCTO_API_KEY=your_api_key_here
```

### Phase 2: Reducto Client Module

**New file:** `src/pdf_converter/reducto_client.py`

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

@dataclass
class ReductoConfig:
    api_key: str
    agentic: bool = False
    chunk_mode: str = "variable"
    table_format: str = "md"

    @classmethod
    def from_env(cls) -> "ReductoConfig":
        api_key = os.environ.get("REDUCTO_API_KEY")
        if not api_key:
            raise ValueError("REDUCTO_API_KEY not set")
        return cls(api_key=api_key)


class ReductoClient:
    """Client for Reducto.ai PDF parsing API."""

    def __init__(self, config: ReductoConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from reducto import Reducto
            self._client = Reducto(api_key=self.config.api_key)
        return self._client

    def parse_pdf(self, pdf_path: Path) -> str:
        """Parse PDF and return markdown content."""
        response = self.client.parse.run(
            input=str(pdf_path),
            enhance={"agentic": self.config.agentic},
            retrieval={"chunking": {"chunk_mode": self.config.chunk_mode}},
            formatting={"table_output_format": self.config.table_format},
        )
        return self._response_to_markdown(response)

    def _response_to_markdown(self, response) -> str:
        """Convert Reducto response to markdown."""
        chunks = response.result.chunks
        return "\n\n".join(chunk.content for chunk in chunks)

    def parse_directory(self, inbox_path: Path) -> list[tuple[Path, str]]:
        """Parse all PDFs in directory."""
        results = []
        for pdf_file in inbox_path.glob("*.pdf"):
            markdown = self.parse_pdf(pdf_file)
            results.append((pdf_file, markdown))
        return results
```

### Phase 3: CLI Command

**Modify:** `src/cli/commands/inbox.py`

Add new command: `kidkazz inbox parse`

```python
@app.command()
def parse(
    tool: str = typer.Option(
        "reducto",
        "--tool", "-t",
        help="Parsing tool: reducto"
    ),
    agentic: bool = typer.Option(
        False,
        "--agentic",
        help="Enable AI-enhanced accuracy (uses more credits)"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be parsed without processing"
    ),
    sync_backup: bool = typer.Option(
        True,
        "--sync-backup/--no-sync-backup",
        help="Sync output to Google Drive as backup"
    ),
) -> None:
    """Parse PDFs in inbox using Reducto.ai API.

    Requires REDUCTO_API_KEY environment variable.

    Examples:
        kidkazz inbox parse                    # Parse all PDFs
        kidkazz inbox parse --agentic          # High-accuracy mode
        kidkazz inbox parse --dry-run          # Preview only
        kidkazz inbox parse --no-sync-backup   # Skip backup
    """
```

### Phase 4: Integration with Cloud Sync

**Workflow integration:**

1. **Before parsing:** Optionally sync inbox to Google Drive (backup source PDFs)
2. **Parse:** Use Reducto.ai API on local files
3. **Save:** Write markdown to `~/.kidkazz/output/`
4. **Backup:** Sync output to Google Drive via rclone

```python
def parse_with_backup(config, pdf_files):
    # 1. Sync inbox to Drive (backup PDFs)
    cloud_sync = CloudSync(config.cloud_remote, config.cloud_path)
    cloud_sync.sync_to_remote(config.inbox_path)

    # 2. Parse with Reducto
    reducto = ReductoClient(ReductoConfig.from_env())
    for pdf in pdf_files:
        markdown = reducto.parse_pdf(pdf)
        output_path = config.output_path / f"{pdf.stem}.md"
        output_path.write_text(markdown)

    # 3. Sync output to Drive (backup results)
    output_sync = CloudSync(config.cloud_remote, "kidkazz_output")
    output_sync.sync_to_remote(config.output_path)
```

### Phase 5: Tests

**New file:** `tests/test_reducto_client.py`

```python
class TestReductoConfig:
    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("REDUCTO_API_KEY", "test_key")
        config = ReductoConfig.from_env()
        assert config.api_key == "test_key"

    def test_from_env_missing_key(self, monkeypatch):
        monkeypatch.delenv("REDUCTO_API_KEY", raising=False)
        with pytest.raises(ValueError):
            ReductoConfig.from_env()


class TestReductoClient:
    @patch("reducto.Reducto")
    def test_parse_pdf(self, mock_reducto, tmp_path):
        # Mock response
        mock_response = MagicMock()
        mock_response.result.chunks = [
            MagicMock(content="# Title"),
            MagicMock(content="Some content"),
        ]
        mock_reducto.return_value.parse.run.return_value = mock_response

        config = ReductoConfig(api_key="test")
        client = ReductoClient(config)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf")

        result = client.parse_pdf(pdf)
        assert "# Title" in result
        assert "Some content" in result
```

**New file:** `tests/test_inbox_parse_cli.py`

```python
class TestParseCommand:
    @patch("src.pdf_converter.reducto_client.ReductoClient")
    def test_parse_with_reducto(self, mock_client, temp_inbox_with_pdfs):
        mock_client.return_value.parse_pdf.return_value = "# Parsed content"

        result = runner.invoke(app, ["inbox", "parse"])

        assert result.exit_code == 0
        mock_client.return_value.parse_pdf.assert_called()
```

## File Structure

```
src/
├── pdf_converter/
│   ├── __init__.py          # Add ReductoClient export
│   └── reducto_client.py    # NEW: Reducto API client
├── cli/
│   ├── commands/
│   │   └── inbox.py         # MODIFY: Add parse command
│   └── config.py            # MODIFY: Add reducto config
tests/
├── test_reducto_client.py   # NEW: Client tests
└── test_inbox_parse_cli.py  # NEW: CLI tests
```

## Configuration Summary

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `REDUCTO_API_KEY` | Yes | API key from reducto.ai |
| `KIDKAZZ_CLOUD_REMOTE` | No | rclone remote (default: gdrive) |
| `KIDKAZZ_CLOUD_PATH` | No | Remote inbox path |

### TOML Configuration

```toml
[reducto]
agentic = false
chunk_mode = "variable"
table_format = "md"

[cloud_sync]
remote = "gdrive"
path = "kidkazz_inbox"
output_path = "kidkazz_output"
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `kidkazz inbox parse` | Parse all PDFs using Reducto.ai |
| `kidkazz inbox parse --agentic` | High-accuracy mode (2x credits) |
| `kidkazz inbox parse --dry-run` | Preview without parsing |
| `kidkazz inbox parse --no-sync-backup` | Skip Google Drive backup |
| `kidkazz inbox sync` | Sync inbox to Google Drive |

## Complete Workflow Example

```bash
# 1. Setup (one-time)
export REDUCTO_API_KEY="your_api_key"
kidkazz config set cloud_remote gdrive

# 2. Add PDFs to inbox
cp document.pdf ~/.kidkazz/inbox/

# 3. Sync to Google Drive (backup)
kidkazz inbox sync

# 4. Parse with Reducto.ai
kidkazz inbox parse
# Output: ~/.kidkazz/output/document.md
# Backup: gdrive:kidkazz_output/document.md

# 5. Ingest into knowledge base
kidkazz ingest markdown ~/.kidkazz/output/document.md \
    --doc-id document --title "My Document"

# 6. Query
kidkazz search semantic "your query"
```

## Reducto.ai Features

| Feature | Details |
|---------|---------|
| GPU Required | No (cloud API) |
| Setup | API key only |
| Speed | Fast (cloud infra) |
| Accuracy | 99%+ (agentic mode) |
| Batch Processing | Built-in |
| Cloud Backup | rclone + Google Drive |

## References

- [Reducto.ai Documentation](https://docs.reducto.ai/)
- [Reducto Parse API](https://docs.reducto.ai/api-reference/parse)
- [Reducto Python SDK](https://github.com/reductoai/reducto-python-sdk)
- [Reducto CLI](https://docs.reducto.ai/cli)
