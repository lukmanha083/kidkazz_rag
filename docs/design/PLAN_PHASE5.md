# Phase 5: End-to-End CLI Tool - Implementation Plan

## Overview

Build a unified command-line interface (CLI) that ties together all previous phases, providing a streamlined workflow for converting PDFs to searchable knowledge bases. The CLI enables users to ingest documents, search content, manage their knowledge base, and configure settings—all from the terminal.

---

## Key Decisions

### 1. CLI Framework: Typer

**Why Typer over alternatives:**

| Framework | Type Hints | Auto Help | Shell Completion | Rich Output |
|-----------|-----------|-----------|------------------|-------------|
| Typer | ✅ Native | ✅ Auto | ✅ Built-in | ✅ Rich |
| Click | ❌ Manual | ✅ Auto | ✅ Plugin | ❌ Manual |
| argparse | ❌ Manual | ✅ Basic | ❌ None | ❌ None |
| Fire | ✅ Auto | ✅ Auto | ❌ None | ❌ None |

**Benefits:**
- Type hints for automatic validation and help generation
- Built on Click (mature, well-tested)
- Rich terminal output support (progress bars, tables, colors)
- Shell completion out of the box
- Clean, Pythonic API

### 2. Architecture: Command Groups

**Organized by domain:**

```text
kidkazz
├── ingest          # Document ingestion pipeline
│   ├── pdf         # Convert PDF → Markdown → Store
│   ├── markdown    # Chunk Markdown → Store
│   └── batch       # Process multiple files
├── search          # Query knowledge base
│   ├── semantic    # Vector similarity search
│   └── keyword     # Full-text keyword search
├── docs            # Document management
│   ├── list        # List all documents
│   ├── stats       # Show document statistics
│   ├── export      # Export document data
│   └── delete      # Remove document
├── db              # Database management
│   ├── init        # Initialize database
│   ├── status      # Check connection status
│   └── clear       # Clear all data
└── config          # Configuration management
    ├── show        # Display current config
    ├── set         # Set configuration value
    └── reset       # Reset to defaults
```

### 3. Configuration: Layered Settings

**Priority order (highest to lowest):**
1. Command-line arguments
2. Environment variables (`KIDKAZZ_*`)
3. Project config file (`.kidkazz.toml`)
4. User config file (`~/.config/kidkazz/config.toml`)
5. Default values

### 4. Output Modes: Human & Machine

**Two output formats:**
- **Human**: Rich formatted output with colors, tables, progress bars
- **JSON**: Machine-readable output for scripting and automation

```bash
# Human-readable (default)
kidkazz docs list

# Machine-readable
kidkazz docs list --json
```

---

## Module Structure

```text
src/
├── chunker/                    # Phase 2 (existing)
├── storage/                    # Phase 3 (existing)
├── mcp_server/                 # Phase 4 (existing)
└── cli/                        # Phase 5 (new)
    ├── __init__.py             # Public API exports
    ├── main.py                 # Main CLI entry point
    ├── commands/               # Command implementations
    │   ├── __init__.py
    │   ├── ingest.py           # Ingestion commands
    │   ├── search.py           # Search commands
    │   ├── docs.py             # Document management
    │   ├── db.py               # Database commands
    │   └── config.py           # Configuration commands
    ├── config.py               # Configuration management
    ├── output.py               # Output formatting (Rich)
    ├── progress.py             # Progress indicators
    └── utils.py                # Shared utilities

tests/
├── test_cli_main.py            # Main entry point tests
├── test_cli_ingest.py          # Ingest command tests
├── test_cli_search.py          # Search command tests
├── test_cli_docs.py            # Document management tests
├── test_cli_db.py              # Database command tests
├── test_cli_config.py          # Configuration tests
├── test_cli_output.py          # Output formatting tests
└── test_cli_integration.py     # End-to-end CLI tests
```

---

## Command Specifications

### Ingest Commands

#### `kidkazz ingest pdf`

Convert PDF to Markdown, chunk, embed, and store.

```bash
kidkazz ingest pdf <file.pdf> [OPTIONS]

Arguments:
  FILE                      Path to PDF file

Options:
  --doc-id TEXT            Document identifier (default: filename)
  --title TEXT             Document title (default: filename)
  --converter [marker|docling|nougat]
                           PDF converter to use (default: marker)
  --chunk-sizes TEXT       Chunk sizes as L1,L2,overlap (default: 2048,512,256)
  --model TEXT             Embedding model (default: BAAI/bge-small-en-v1.5)
  --store [mock|helix]     Storage backend (default: from config)
  --output-dir PATH        Save intermediate markdown (optional)
  --dry-run               Show what would be done without storing
  --json                  Output results as JSON

Example:
  kidkazz ingest pdf textbook.pdf --title "Machine Learning 101"
  kidkazz ingest pdf notes.pdf --converter docling --chunk-sizes 1024,256,128
```

#### `kidkazz ingest markdown`

Chunk existing Markdown file, embed, and store.

```bash
kidkazz ingest markdown <file.md> [OPTIONS]

Arguments:
  FILE                      Path to Markdown file

Options:
  --doc-id TEXT            Document identifier (default: filename)
  --title TEXT             Document title (default: from H1 or filename)
  -g, --tags TEXT          Comma-separated tags (e.g., 'inventory,accounting')
  --chunk-sizes TEXT       Chunk sizes as L1,L2,overlap (default: 2048,512,256)
  --model TEXT             Embedding model
  --store [mock|helix]     Storage backend
  --dry-run               Show chunks without storing
  --json                  Output results as JSON

Example:
  kidkazz ingest markdown output/textbook.md --doc-id ml_textbook
  kidkazz ingest markdown inventory.md --tags inventory,accounting
```

#### `kidkazz ingest batch`

Process multiple files in a directory.

```bash
kidkazz ingest batch <directory> [OPTIONS]

Arguments:
  DIRECTORY                 Directory containing files to process

Options:
  --pattern TEXT           Glob pattern for files (default: *.pdf)
  --recursive             Search subdirectories
  -g, --tags TEXT          Comma-separated tags for all files
  --converter TEXT         PDF converter to use
  --parallel INTEGER       Number of parallel workers (default: 1)
  --skip-existing         Skip files already in database
  --json                  Output results as JSON

Example:
  kidkazz ingest batch ./pdfs --pattern "*.pdf" --recursive --parallel 4
  kidkazz ingest batch ./docs --pattern "*.md" --tags documentation
```

### Search Commands

#### `kidkazz search semantic`

Vector similarity search.

```bash
kidkazz search semantic <query> [OPTIONS]

Arguments:
  QUERY                     Natural language search query

Options:
  -k, --top-k INTEGER      Number of results (default: 5)
  -d, --doc-id TEXT        Filter to specific document
  -g, --tags TEXT          Filter by document tags (comma-separated, AND logic)
  -l, --level INTEGER      Filter by hierarchy level (1 or 2)
  -t, --type TEXT          Filter by semantic type
  --threshold FLOAT        Minimum similarity (0.0-1.0)
  --show-context          Include surrounding chunks
  --show-parent           Include parent chunk
  --json                  Output results as JSON

Example:
  kidkazz search semantic "What is machine learning?" --top-k 10
  kidkazz search semantic "neural networks" --doc-id ml_textbook --level 2
  kidkazz search semantic "safety stock" --tags inventory
```

#### `kidkazz search keyword`

Full-text keyword search.

```bash
kidkazz search keyword <keyword> [OPTIONS]

Arguments:
  KEYWORD                   Text to search for

Options:
  -d, --doc-id TEXT        Filter to specific document
  --case-sensitive        Enable case-sensitive matching
  --json                  Output results as JSON

Example:
  kidkazz search keyword "gradient descent" --doc-id ml_textbook
```

### Document Management Commands

#### `kidkazz docs list`

List all documents in the knowledge base.

```bash
kidkazz docs list [OPTIONS]

Options:
  -g, --tags TEXT            Filter by document tags (comma-separated, AND logic)
  --sort [date|name|chunks]  Sort order (default: date)
  --reverse                 Reverse sort order
  --json                   Output as JSON

Example:
  kidkazz docs list --sort chunks --reverse
  kidkazz docs list --tags inventory
  kidkazz docs list --tags inventory,accounting
```

Output:
```
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Document ID    ┃ Title                  ┃ Tags                   ┃ Chunks ┃ Created            ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ inv_textbook   │ Inventory Management   │ inventory, accounting  │    127 │ 2024-01-15 10:30   │
│ stats_guide    │ Statistics Handbook    │ statistics             │     89 │ 2024-01-14 15:45   │
│ python_notes   │ Python Programming     │                        │     56 │ 2024-01-13 09:00   │
└────────────────┴────────────────────────┴────────────────────────┴────────┴────────────────────┘
```

#### `kidkazz docs stats`

Show detailed statistics for a document.

```bash
kidkazz docs stats <doc-id> [OPTIONS]

Arguments:
  DOC_ID                    Document identifier

Options:
  --json                   Output as JSON

Example:
  kidkazz docs stats ml_textbook
```

Output:
```
Document: Machine Learning 101 (ml_textbook)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overview:
  Total Chunks: 127
  Total Tokens: 45,230
  Created: 2024-01-15 10:30:00

Chunks by Level:
  Level 1 (sections):  12
  Level 2 (leaves):   115

Chunks by Type:
  narrative:    67 (53%)
  definition:   28 (22%)
  example:      19 (15%)
  procedure:     8 (6%)
  theorem:       5 (4%)

Top Sections:
  1. Introduction (15 chunks)
  2. Supervised Learning (23 chunks)
  3. Neural Networks (31 chunks)
```

#### `kidkazz docs export`

Export document data to file.

```bash
kidkazz docs export <doc-id> [OPTIONS]

Arguments:
  DOC_ID                    Document identifier

Options:
  -o, --output PATH        Output file path
  --format [json|markdown|csv]
                           Export format (default: json)
  --include-embeddings    Include embedding vectors
  --level INTEGER          Export only specific level

Example:
  kidkazz docs export ml_textbook -o export.json
  kidkazz docs export ml_textbook --format markdown -o textbook_export.md
```

#### `kidkazz docs delete`

Remove document from knowledge base.

```bash
kidkazz docs delete <doc-id> [OPTIONS]

Arguments:
  DOC_ID                    Document identifier

Options:
  --force                  Skip confirmation prompt
  --json                  Output result as JSON

Example:
  kidkazz docs delete old_document --force
```

### Database Commands

#### `kidkazz db init`

Initialize or verify database connection.

```bash
kidkazz db init [OPTIONS]

Options:
  --store [mock|helix]     Storage backend
  --port INTEGER           Helix-DB port (default: 6969)
  --create-schema         Create database schema

Example:
  kidkazz db init --store helix --port 6969 --create-schema
```

#### `kidkazz db status`

Check database connection and statistics.

```bash
kidkazz db status [OPTIONS]

Options:
  --json                  Output as JSON

Example:
  kidkazz db status
```

Output:
```
Database Status
━━━━━━━━━━━━━━━

Connection: ✓ Connected (Helix-DB on localhost:6969)
Documents:  3
Chunks:     272
Embeddings: 272

Storage:
  Type: helix
  Port: 6969
  Status: healthy
```

#### `kidkazz db clear`

Clear all data from database.

```bash
kidkazz db clear [OPTIONS]

Options:
  --force                  Skip confirmation prompt
  --keep-schema           Keep schema, only delete data

Example:
  kidkazz db clear --force
```

### Configuration Commands

#### `kidkazz config show`

Display current configuration.

```bash
kidkazz config show [OPTIONS]

Options:
  --json                  Output as JSON

Example:
  kidkazz config show
```

Output:
```
Configuration
━━━━━━━━━━━━━

Storage:
  store_type: helix
  helix_port: 6969
  helix_local: true

Embeddings:
  embedder_type: fastembed
  model_name: BAAI/bge-small-en-v1.5

Chunking:
  level_1_size: 2048
  level_2_size: 512
  overlap: 256

Sources:
  store_type: environment (KIDKAZZ_STORE_TYPE)
  model_name: default
```

#### `kidkazz config set`

Set a configuration value.

```bash
kidkazz config set <key> <value> [OPTIONS]

Arguments:
  KEY                       Configuration key
  VALUE                     Configuration value

Options:
  --global                 Set in user config (~/.config/kidkazz/)
  --project               Set in project config (.kidkazz.toml)

Example:
  kidkazz config set store_type helix --project
  kidkazz config set model_name BAAI/bge-base-en-v1.5 --global
```

#### `kidkazz config reset`

Reset configuration to defaults.

```bash
kidkazz config reset [OPTIONS]

Options:
  --global                 Reset user config
  --project               Reset project config
  --all                   Reset all configs

Example:
  kidkazz config reset --project
```

---

## Implementation Details

### 1. `main.py` - CLI Entry Point

```python
import typer
from rich.console import Console

from .commands import config, db, docs, ingest, search

app = typer.Typer(
    name="kidkazz",
    help="KidKazz RAG - Convert PDFs to searchable knowledge bases",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register command groups
app.add_typer(ingest.app, name="ingest", help="Ingest documents")
app.add_typer(search.app, name="search", help="Search knowledge base")
app.add_typer(docs.app, name="docs", help="Manage documents")
app.add_typer(db.app, name="db", help="Database operations")
app.add_typer(config.app, name="config", help="Configuration")

console = Console()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-essential output"),
):
    """KidKazz RAG - Convert PDFs to searchable knowledge bases."""
    # Set global state for verbosity
    pass


def cli():
    """Entry point for console script."""
    app()


if __name__ == "__main__":
    cli()
```

### 2. `config.py` - Configuration Management

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import tomli
import tomli_w


@dataclass
class CLIConfig:
    """CLI configuration with layered sources."""

    # Storage settings
    store_type: str = "mock"
    helix_port: int = 6969
    helix_local: bool = True

    # Embedding settings
    embedder_type: str = "fastembed"
    model_name: str = "BAAI/bge-small-en-v1.5"

    # Chunking settings
    level_1_size: int = 2048
    level_2_size: int = 512
    overlap: int = 256

    # Output settings
    output_format: str = "human"  # or "json"
    color: bool = True

    # Source tracking
    _sources: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "CLIConfig":
        """Load configuration from all sources."""
        config = cls()

        # Load from user config
        user_config_path = Path.home() / ".config" / "kidkazz" / "config.toml"
        if user_config_path.exists():
            config._load_from_file(user_config_path, "user")

        # Load from project config
        project_config_path = Path.cwd() / ".kidkazz.toml"
        if project_config_path.exists():
            config._load_from_file(project_config_path, "project")

        # Load from environment
        config._load_from_env()

        return config

    def _load_from_file(self, path: Path, source: str) -> None:
        """Load settings from TOML file."""
        with open(path, "rb") as f:
            data = tomli.load(f)
        self._apply_settings(data, source)

    def _load_from_env(self) -> None:
        """Load settings from environment variables."""
        import os

        env_mappings = {
            "KIDKAZZ_STORE_TYPE": "store_type",
            "KIDKAZZ_HELIX_PORT": ("helix_port", int),
            "KIDKAZZ_EMBEDDER_TYPE": "embedder_type",
            "KIDKAZZ_MODEL_NAME": "model_name",
        }

        for env_var, mapping in env_mappings.items():
            if value := os.environ.get(env_var):
                if isinstance(mapping, tuple):
                    attr, converter = mapping
                    setattr(self, attr, converter(value))
                else:
                    setattr(self, mapping, value)
                self._sources[mapping if isinstance(mapping, str) else mapping[0]] = f"environment ({env_var})"

    def save_project(self, path: Path = None) -> None:
        """Save configuration to project file."""
        path = path or Path.cwd() / ".kidkazz.toml"
        self._save_to_file(path)

    def save_user(self) -> None:
        """Save configuration to user file."""
        path = Path.home() / ".config" / "kidkazz" / "config.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._save_to_file(path)

    def _save_to_file(self, path: Path) -> None:
        """Save settings to TOML file."""
        data = {
            "storage": {
                "store_type": self.store_type,
                "helix_port": self.helix_port,
                "helix_local": self.helix_local,
            },
            "embeddings": {
                "embedder_type": self.embedder_type,
                "model_name": self.model_name,
            },
            "chunking": {
                "level_1_size": self.level_1_size,
                "level_2_size": self.level_2_size,
                "overlap": self.overlap,
            },
        }
        with open(path, "wb") as f:
            tomli_w.dump(data, f)
```

### 3. `output.py` - Rich Output Formatting

```python
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()


def print_search_results(results: list[dict], show_context: bool = False) -> None:
    """Print search results in rich format."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, result in enumerate(results, 1):
        score = result.get("similarity_score", 0)
        content = result.get("content", "")[:200]
        chunk_id = result.get("chunk_id", "unknown")
        level = result.get("level", "?")
        semantic_type = result.get("semantic_type", "unknown")

        # Color based on score
        if score >= 0.8:
            score_color = "green"
        elif score >= 0.5:
            score_color = "yellow"
        else:
            score_color = "red"

        panel = Panel(
            f"{content}...",
            title=f"[bold]#{i}[/bold] {chunk_id} [{score_color}]{score:.3f}[/{score_color}]",
            subtitle=f"Level {level} | {semantic_type}",
            border_style="blue",
        )
        console.print(panel)


def print_document_list(documents: list[dict]) -> None:
    """Print document list as table."""
    table = Table(title="Documents")
    table.add_column("Document ID", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Chunks", justify="right", style="green")
    table.add_column("Created", style="dim")

    for doc in documents:
        table.add_row(
            doc.get("doc_id", ""),
            doc.get("title", ""),
            str(doc.get("chunk_count", 0)),
            doc.get("created_at", ""),
        )

    console.print(table)


def print_document_stats(stats: dict) -> None:
    """Print document statistics."""
    doc_id = stats.get("doc_id", "unknown")
    title = stats.get("title", "Unknown")

    console.print(f"\n[bold]Document:[/bold] {title} ({doc_id})")
    console.print("━" * 40)

    # Overview section
    console.print("\n[bold]Overview:[/bold]")
    console.print(f"  Total Chunks: {stats.get('total_chunks', 0)}")
    console.print(f"  Total Tokens: {stats.get('total_tokens', 0):,}")

    # Chunks by level
    console.print("\n[bold]Chunks by Level:[/bold]")
    for level, count in stats.get("chunks_by_level", {}).items():
        console.print(f"  Level {level}: {count}")

    # Chunks by type
    console.print("\n[bold]Chunks by Type:[/bold]")
    total = stats.get("total_chunks", 1)
    for stype, count in stats.get("chunks_by_type", {}).items():
        pct = (count / total) * 100
        console.print(f"  {stype}: {count} ({pct:.0f}%)")


def print_progress_summary(
    processed: int,
    total: int,
    errors: list[str] = None,
) -> None:
    """Print ingestion progress summary."""
    console.print(f"\n[bold]Ingestion Complete[/bold]")
    console.print(f"  Processed: {processed}/{total}")

    if errors:
        console.print(f"  [red]Errors: {len(errors)}[/red]")
        for error in errors[:5]:
            console.print(f"    - {error}")
        if len(errors) > 5:
            console.print(f"    ... and {len(errors) - 5} more")


def print_json(data: Any) -> None:
    """Print data as formatted JSON."""
    import json

    console.print_json(json.dumps(data, indent=2, default=str))
```

### 4. `progress.py` - Progress Indicators

```python
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()


@contextmanager
def ingestion_progress() -> Generator[Progress, None, None]:
    """Context manager for ingestion progress bar."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        yield progress


@contextmanager
def spinner(message: str) -> Generator[None, None, None]:
    """Context manager for simple spinner."""
    with console.status(f"[bold blue]{message}..."):
        yield


class IngestionTracker:
    """Track multi-stage ingestion progress."""

    def __init__(self, progress: Progress):
        self.progress = progress
        self.tasks = {}

    def add_stage(self, name: str, total: int = 100) -> int:
        """Add a new stage to track."""
        task_id = self.progress.add_task(name, total=total)
        self.tasks[name] = task_id
        return task_id

    def update(self, name: str, advance: int = 1) -> None:
        """Update stage progress."""
        if name in self.tasks:
            self.progress.update(self.tasks[name], advance=advance)

    def complete(self, name: str) -> None:
        """Mark stage as complete."""
        if name in self.tasks:
            task = self.progress.tasks[self.tasks[name]]
            self.progress.update(self.tasks[name], completed=task.total)
```

### 5. `commands/ingest.py` - Ingestion Commands

```python
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ..config import CLIConfig
from ..output import print_json, print_progress_summary
from ..progress import ingestion_progress, spinner

app = typer.Typer(help="Document ingestion commands")
console = Console()


@app.command("pdf")
def ingest_pdf(
    file: Path = typer.Argument(..., help="Path to PDF file", exists=True),
    doc_id: Optional[str] = typer.Option(None, help="Document identifier"),
    title: Optional[str] = typer.Option(None, help="Document title"),
    converter: str = typer.Option("marker", help="PDF converter to use"),
    chunk_sizes: str = typer.Option("2048,512,256", help="Chunk sizes (L1,L2,overlap)"),
    model: Optional[str] = typer.Option(None, help="Embedding model"),
    store: Optional[str] = typer.Option(None, help="Storage backend"),
    output_dir: Optional[Path] = typer.Option(None, help="Save intermediate markdown"),
    dry_run: bool = typer.Option(False, help="Show what would be done"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Convert PDF to Markdown, chunk, embed, and store."""
    config = CLIConfig.load()

    # Use defaults from config
    doc_id = doc_id or file.stem
    title = title or file.stem
    model = model or config.model_name
    store_type = store or config.store_type

    # Parse chunk sizes
    sizes = [int(s) for s in chunk_sizes.split(",")]
    if len(sizes) != 3:
        raise typer.BadParameter("chunk_sizes must have 3 values: L1,L2,overlap")

    if dry_run:
        console.print(f"[bold]Dry run:[/bold] Would process {file}")
        console.print(f"  Document ID: {doc_id}")
        console.print(f"  Converter: {converter}")
        console.print(f"  Chunk sizes: {sizes}")
        console.print(f"  Store: {store_type}")
        return

    result = {
        "doc_id": doc_id,
        "title": title,
        "source": str(file),
        "chunks": 0,
        "status": "success",
    }

    try:
        with ingestion_progress() as progress:
            # Stage 1: Convert PDF
            task1 = progress.add_task("Converting PDF...", total=100)

            # PDF conversion would happen here
            # For now, we assume markdown is provided or conversion is external
            progress.update(task1, completed=100)

            # Stage 2: Chunk document
            task2 = progress.add_task("Chunking document...", total=100)

            from src.chunker import create_hierarchical_chunks, enrich_all_chunks

            # Read markdown (from output_dir or converted)
            markdown_path = output_dir / f"{doc_id}.md" if output_dir else None
            if markdown_path and markdown_path.exists():
                content = markdown_path.read_text()
            else:
                raise typer.BadParameter(
                    "PDF conversion not yet implemented in CLI. "
                    "Please use the Colab notebook first, then run 'kidkazz ingest markdown'."
                )

            chunks = create_hierarchical_chunks(
                content,
                doc_id=doc_id,
                level_sizes=tuple(sizes),
            )
            metadata = enrich_all_chunks(chunks, document_id=doc_id)
            progress.update(task2, completed=100)

            # Stage 3: Generate embeddings
            task3 = progress.add_task("Generating embeddings...", total=len(chunks))

            from src.chunker import ChunkEmbedder, MockEmbedder

            if config.embedder_type == "mock":
                embedder = MockEmbedder()
            else:
                embedder = ChunkEmbedder(model_name=model)

            embedded_chunks = embedder.embed_chunks(chunks, batch_size=32)
            progress.update(task3, completed=len(chunks))

            # Stage 4: Store in database
            task4 = progress.add_task("Storing in database...", total=100)

            from src.storage import HelixChunkStore, MockChunkStore

            if store_type == "mock":
                store_instance = MockChunkStore()
            else:
                store_instance = HelixChunkStore()

            store_instance.store_document(doc_id, title, embedded_chunks, metadata)
            progress.update(task4, completed=100)

            result["chunks"] = len(chunks)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        if not json_output:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    if json_output:
        print_json(result)
    else:
        console.print(f"\n[green]✓[/green] Ingested {result['chunks']} chunks from {file.name}")


@app.command("markdown")
def ingest_markdown(
    file: Path = typer.Argument(..., help="Path to Markdown file", exists=True),
    doc_id: Optional[str] = typer.Option(None, help="Document identifier"),
    title: Optional[str] = typer.Option(None, help="Document title"),
    chunk_sizes: str = typer.Option("2048,512,256", help="Chunk sizes"),
    model: Optional[str] = typer.Option(None, help="Embedding model"),
    store: Optional[str] = typer.Option(None, help="Storage backend"),
    dry_run: bool = typer.Option(False, help="Show chunks without storing"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Chunk existing Markdown file, embed, and store."""
    config = CLIConfig.load()

    doc_id = doc_id or file.stem
    title = title or file.stem
    model = model or config.model_name
    store_type = store or config.store_type

    sizes = [int(s) for s in chunk_sizes.split(",")]

    content = file.read_text()

    # Extract title from H1 if not provided
    if title == file.stem:
        for line in content.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break

    if dry_run:
        from src.chunker import create_hierarchical_chunks

        chunks = create_hierarchical_chunks(content, doc_id=doc_id, level_sizes=tuple(sizes))
        console.print(f"[bold]Dry run:[/bold] Would create {len(chunks)} chunks")
        for chunk in chunks[:5]:
            console.print(f"  - {chunk.id}: {chunk.content[:50]}...")
        if len(chunks) > 5:
            console.print(f"  ... and {len(chunks) - 5} more")
        return

    result = {"doc_id": doc_id, "title": title, "chunks": 0, "status": "success"}

    try:
        with ingestion_progress() as progress:
            task1 = progress.add_task("Chunking...", total=100)
            from src.chunker import create_hierarchical_chunks, enrich_all_chunks

            chunks = create_hierarchical_chunks(content, doc_id=doc_id, level_sizes=tuple(sizes))
            metadata = enrich_all_chunks(chunks, document_id=doc_id)
            progress.update(task1, completed=100)

            task2 = progress.add_task("Embedding...", total=len(chunks))
            from src.chunker import ChunkEmbedder, MockEmbedder

            embedder = MockEmbedder() if config.embedder_type == "mock" else ChunkEmbedder(model_name=model)
            embedded = embedder.embed_chunks(chunks)
            progress.update(task2, completed=len(chunks))

            task3 = progress.add_task("Storing...", total=100)
            from src.storage import HelixChunkStore, MockChunkStore

            store_instance = MockChunkStore() if store_type == "mock" else HelixChunkStore()
            store_instance.store_document(doc_id, title, embedded, metadata)
            progress.update(task3, completed=100)

            result["chunks"] = len(chunks)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        if not json_output:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    if json_output:
        print_json(result)
    else:
        console.print(f"\n[green]✓[/green] Ingested {result['chunks']} chunks")


@app.command("batch")
def ingest_batch(
    directory: Path = typer.Argument(..., help="Directory containing files"),
    pattern: str = typer.Option("*.md", help="Glob pattern"),
    recursive: bool = typer.Option(False, help="Search subdirectories"),
    parallel: int = typer.Option(1, help="Parallel workers"),
    skip_existing: bool = typer.Option(False, help="Skip existing documents"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Process multiple files in a directory."""
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))

    if not files:
        console.print(f"[yellow]No files matching '{pattern}' found in {directory}[/yellow]")
        return

    console.print(f"Found {len(files)} files to process")

    results = []
    errors = []

    with ingestion_progress() as progress:
        task = progress.add_task("Processing files...", total=len(files))

        for file in files:
            try:
                # Process each file (simplified for now)
                results.append({"file": str(file), "status": "success"})
            except Exception as e:
                errors.append(str(e))
                results.append({"file": str(file), "status": "error", "error": str(e)})
            progress.advance(task)

    if json_output:
        print_json({"processed": len(results), "errors": len(errors), "results": results})
    else:
        print_progress_summary(len(results), len(files), errors)
```

### 6. `commands/search.py` - Search Commands

```python
from typing import Optional

import typer
from rich.console import Console

from ..config import CLIConfig
from ..output import print_json, print_search_results

app = typer.Typer(help="Search commands")
console = Console()


@app.command("semantic")
def search_semantic(
    query: str = typer.Argument(..., help="Natural language search query"),
    top_k: int = typer.Option(5, "-k", "--top-k", help="Number of results"),
    doc_id: Optional[str] = typer.Option(None, "-d", "--doc-id", help="Filter to document"),
    level: Optional[int] = typer.Option(None, "-l", "--level", help="Filter by level"),
    semantic_type: Optional[str] = typer.Option(None, "-t", "--type", help="Filter by type"),
    threshold: float = typer.Option(0.0, help="Minimum similarity"),
    show_context: bool = typer.Option(False, help="Include surrounding chunks"),
    show_parent: bool = typer.Option(False, help="Include parent chunk"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Vector similarity search."""
    config = CLIConfig.load()

    from src.chunker import ChunkEmbedder, MockEmbedder
    from src.storage import HelixChunkStore, MockChunkStore

    # Initialize components
    if config.embedder_type == "mock":
        embedder = MockEmbedder()
    else:
        embedder = ChunkEmbedder(model_name=config.model_name)

    if config.store_type == "mock":
        store = MockChunkStore()
    else:
        store = HelixChunkStore()

    # Generate query embedding
    with console.status("Generating query embedding..."):
        query_embedding = embedder.embed_text(query)

    # Search
    results = store.search_similar(
        query_embedding=query_embedding,
        top_k=top_k,
        doc_id=doc_id,
        level=level,
        semantic_type=semantic_type,
        threshold=threshold,
    )

    # Format results
    formatted = []
    for ec, score in results:
        result = {
            "chunk_id": ec.chunk.id,
            "content": ec.chunk.content,
            "level": ec.chunk.level,
            "semantic_type": getattr(ec.chunk, "semantic_type", "unknown"),
            "similarity_score": score,
        }

        if show_context:
            context = store.get_context_window(ec.chunk.id, window_size=1)
            result["context"] = [c.chunk.content for c in context]

        if show_parent and ec.chunk.parent_id:
            parent = store.get_parent(ec.chunk.id)
            if parent:
                result["parent"] = parent.chunk.content

        formatted.append(result)

    if json_output:
        print_json(formatted)
    else:
        print_search_results(formatted, show_context=show_context)


@app.command("keyword")
def search_keyword(
    keyword: str = typer.Argument(..., help="Text to search for"),
    doc_id: Optional[str] = typer.Option(None, "-d", "--doc-id", help="Filter to document"),
    case_sensitive: bool = typer.Option(False, help="Case-sensitive matching"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Full-text keyword search."""
    config = CLIConfig.load()

    from src.storage import HelixChunkStore, MockChunkStore

    store = MockChunkStore() if config.store_type == "mock" else HelixChunkStore()

    results = store.search_keyword(
        keyword=keyword,
        doc_id=doc_id,
        case_sensitive=case_sensitive,
    )

    formatted = [
        {
            "chunk_id": ec.chunk.id,
            "content": ec.chunk.content,
            "level": ec.chunk.level,
        }
        for ec in results
    ]

    if json_output:
        print_json(formatted)
    else:
        if not formatted:
            console.print(f"[yellow]No results for '{keyword}'[/yellow]")
        else:
            console.print(f"Found {len(formatted)} results for '{keyword}':\n")
            for result in formatted:
                console.print(f"[cyan]{result['chunk_id']}[/cyan]")
                console.print(f"  {result['content'][:150]}...\n")
```

---

## Dependencies (pyproject.toml additions)

```toml
[project.optional-dependencies]
cli = [
    "typer>=0.9.0",
    "rich>=13.0.0",
    "tomli>=2.0.0",
    "tomli-w>=1.0.0",
]
all = [
    "fastembed>=0.2.0",
    "helix-py>=0.1.0",
    "mcp>=1.2.0",
    "typer>=0.9.0",
    "rich>=13.0.0",
    "tomli>=2.0.0",
    "tomli-w>=1.0.0",
]

[project.scripts]
kidkazz = "src.cli:cli"
kidkazz-mcp = "src.mcp_server:main"
```

---

## Test Plan

### test_cli_main.py (~12 tests)
- CLI app creation and structure
- Command group registration
- Global options (verbose, quiet)
- Help text generation
- Version display

### test_cli_ingest.py (~25 tests)
- `ingest pdf` command parsing
- `ingest markdown` command execution
- `ingest batch` directory processing
- Dry run mode
- Error handling for missing files
- Chunk size parsing
- JSON output format

### test_cli_search.py (~18 tests)
- `search semantic` query execution
- `search keyword` matching
- Filter options (doc_id, level, type)
- Context and parent inclusion
- Empty results handling
- JSON output format

### test_cli_docs.py (~20 tests)
- `docs list` output formatting
- `docs stats` statistics display
- `docs export` file generation
- `docs delete` confirmation and execution
- JSON output for all commands

### test_cli_db.py (~12 tests)
- `db init` connection setup
- `db status` health checking
- `db clear` data removal
- Error handling for unavailable database

### test_cli_config.py (~15 tests)
- Configuration loading from files
- Environment variable precedence
- `config show` display
- `config set` value persistence
- `config reset` defaults restoration

### test_cli_output.py (~10 tests)
- Table formatting
- Panel creation
- JSON output
- Color handling
- Progress bar rendering

### test_cli_integration.py (~15 tests)
- Full ingestion workflow
- Search after ingestion
- Configuration persistence
- Error recovery
- Multi-document scenarios

### Total: ~127 new tests

---

## Configuration File Format

### `.kidkazz.toml` (Project Config)

```toml
[storage]
store_type = "helix"
helix_port = 6969
helix_local = true

[embeddings]
embedder_type = "fastembed"
model_name = "BAAI/bge-small-en-v1.5"

[chunking]
level_1_size = 2048
level_2_size = 512
overlap = 256

[output]
format = "human"  # or "json"
color = true
```

### `~/.config/kidkazz/config.toml` (User Config)

```toml
[storage]
store_type = "mock"  # default for development

[embeddings]
model_name = "BAAI/bge-base-en-v1.5"  # user preference

[output]
color = true
```

---

## Usage Examples

### Complete Workflow

```bash
# 1. Initialize database
kidkazz db init --store helix --create-schema

# 2. Ingest a document
kidkazz ingest markdown textbook.md --title "ML Textbook" --doc-id ml_book

# 3. Search for content
kidkazz search semantic "What is gradient descent?" --top-k 5

# 4. View document stats
kidkazz docs stats ml_book

# 5. Export for backup
kidkazz docs export ml_book -o backup.json
```

### Batch Processing

```bash
# Process all markdown files in a directory
kidkazz ingest batch ./documents --pattern "*.md" --recursive

# Check database status
kidkazz db status
```

### Scripting with JSON

```bash
# Get search results as JSON for processing
kidkazz search semantic "neural networks" --json | jq '.[] | .chunk_id'

# List documents in JSON format
kidkazz docs list --json > documents.json
```

---

## Success Criteria

1. All ~127 unit tests pass
2. CLI installs via `pip install kidkazz-rag[cli]`
3. `kidkazz --help` shows all command groups
4. Full ingestion workflow works end-to-end
5. Search returns relevant results
6. Configuration persists correctly
7. JSON output parses correctly
8. Progress bars display smoothly
9. Error messages are helpful
10. Coverage > 80% for cli module

---

## Implementation Order

### Step 1: Project Setup
- Add dependencies to pyproject.toml
- Create `src/cli/` directory structure
- Add entry point for `kidkazz` command

### Step 2: Configuration Module
- Implement `CLIConfig` dataclass
- Add TOML file loading/saving
- Environment variable support

### Step 3: Output Formatting
- Rich console setup
- Table and panel formatters
- JSON output helpers

### Step 4: Progress Indicators
- Progress bar context managers
- Spinner for long operations
- Multi-stage tracker

### Step 5: Core Commands
- Implement `ingest markdown` first (simplest)
- Add `search semantic` and `search keyword`
- Implement `docs list`, `stats`, `delete`

### Step 6: Database Commands
- Implement `db init`, `status`, `clear`
- Add schema creation support

### Step 7: Configuration Commands
- Implement `config show`, `set`, `reset`
- Test configuration precedence

### Step 8: Batch Processing
- Implement `ingest batch`
- Add parallel processing support

### Step 9: Documentation & Polish
- Update README with CLI usage
- Add shell completion instructions
- Create example workflows

---

## Critical Files Reference

| File | Purpose |
|------|---------|
| `src/chunker/__init__.py` | Chunking API |
| `src/chunker/embedder.py` | Embedding generation |
| `src/storage/__init__.py` | Storage API |
| `src/storage/mock_store.py` | MockChunkStore |
| `src/storage/client.py` | HelixChunkStore |
| `src/mcp_server/config.py` | MCP config (reference) |

---

## Future Enhancements (Post-Phase 5)

- **PDF Conversion Integration**: Direct PDF → Markdown in CLI (requires GPU)
- **Interactive Mode**: REPL-style interface for exploration
- **Watch Mode**: Auto-ingest new files in directory
- **Plugin System**: Custom converters and embedders
- **Web UI**: Browser-based interface option
