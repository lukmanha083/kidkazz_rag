# Phase 6B: Concept Extraction - CLI and Ingestion Integration

**Status: ✅ Complete**

## Overview

Integrate concept extraction into the ingestion pipeline and create CLI commands for concept management.

## Prerequisites

- Phase 6A complete (core infrastructure)
- Helix-DB schema deployed with Concept node

---

## Implementation Details

### 1. Configuration Extension (`src/cli/config.py`)

Add concept configuration fields to `CLIConfig`:

```python
@dataclass
class CLIConfig:
    # ... existing fields ...

    # Concept extraction settings
    extract_concepts: bool = False
    concept_provider: str = "anthropic/claude-sonnet-4-20250514"
    max_concepts_per_chunk: int = 10

    @classmethod
    def from_dict(cls, data: dict) -> "CLIConfig":
        # ... existing parsing ...

        # Parse concept settings
        concepts = data.get("concepts", {})
        config.extract_concepts = concepts.get("enabled", False)
        config.concept_provider = concepts.get(
            "provider", "anthropic/claude-sonnet-4-20250514"
        )
        config.max_concepts_per_chunk = concepts.get("max_concepts_per_chunk", 10)

        return config
```

**Example .kidkazz.toml:**

```toml
[concepts]
enabled = false
provider = "anthropic/claude-sonnet-4-20250514"
max_concepts_per_chunk = 10
```

### 2. Ingest Command Extension (`src/cli/commands/ingest.py`)

Modify `ingest_markdown()` to support concept extraction:

```python
@app.command("markdown")
def ingest_markdown(
    file: Path = typer.Argument(..., help="Path to markdown file"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", "-d"),
    title: Optional[str] = typer.Option(None, "--title", "-t"),
    tags: Optional[str] = typer.Option(None, "--tags"),
    embedder: Optional[str] = typer.Option(None, "--embedder", "-e"),
    store: Optional[str] = typer.Option(None, "--store", "-s"),
    chunk_sizes: Optional[str] = typer.Option(None, "--chunk-sizes"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    json_output: bool = typer.Option(False, "--json"),
    # NEW: Concept extraction options
    extract_concepts: Optional[bool] = typer.Option(
        None,
        "--extract-concepts/--no-extract-concepts",
        help="Extract concepts using LLM (costs API credits)",
    ),
    concept_provider: Optional[str] = typer.Option(
        None,
        "--concept-provider",
        help="LLM provider for extraction (e.g., anthropic/claude-sonnet-4-20250514)",
    ),
) -> None:
    """Ingest a markdown file into the knowledge base."""

    config = CLIConfig.load()

    # Apply CLI overrides
    if extract_concepts is not None:
        config.extract_concepts = extract_concepts
    if concept_provider:
        config.concept_provider = concept_provider

    # ... existing chunking, embedding code ...

    # Stage 4: Concept Extraction (NEW)
    concepts = []
    relations = []
    if config.extract_concepts and not dry_run:
        try:
            from src.chunker.concept_extractor import ConceptExtractor, slugify

            with console.status("[bold blue]Extracting concepts..."):
                extractor = ConceptExtractor(provider=config.concept_provider)
                chunk_dicts = [
                    {"content": c.content, "section_path": c.section_path}
                    for c in chunks
                ]
                concepts, relations = extractor.extract_from_chunks(
                    chunk_dicts, document_title
                )

            print_success(f"Extracted {len(concepts)} concepts, {len(relations)} relationships")

        except ImportError:
            print_warning("Instructor not installed. Run: pip install 'kidkazz[concepts]'")
        except Exception as e:
            print_error(f"Concept extraction failed: {e}")

    # Stage 5: Storage
    if not dry_run:
        # Store document with chunks (existing)
        store_instance.store_document(
            doc_id=resolved_doc_id,
            title=document_title,
            embedded_chunks=embedded_chunks,
            metadata_list=metadata_list,
            tags=parsed_tags,
        )

        # Store concepts and relationships (NEW)
        if concepts:
            _store_concepts(
                store_instance,
                concepts,
                relations,
                resolved_doc_id,
                embedded_chunks,
                console,
            )

    # ... rest of function ...


def _store_concepts(
    store: "HelixChunkStore",
    concepts: list,
    relations: list,
    doc_id: str,
    embedded_chunks: list,
    console,
) -> None:
    """Store extracted concepts and relationships."""
    from src.chunker.concept_extractor import slugify

    concept_ids: dict[str, str] = {}  # name -> internal_id

    # Store concepts
    with console.status("[bold blue]Storing concepts..."):
        for concept in concepts:
            concept_id = slugify(concept.name)
            internal_id = store.store_concept(
                concept_id=concept_id,
                name=concept.name,
                definition=concept.definition,
                concept_type=concept.concept_type.value,
                source_documents=[doc_id],
                aliases=concept.aliases,
            )
            if internal_id:
                concept_ids[concept.name.lower()] = internal_id

    # Link chunks to concepts they define
    # (Simplified: link based on concept name appearing in chunk)
    with console.status("[bold blue]Linking chunks to concepts..."):
        for ec in embedded_chunks:
            chunk_content_lower = ec.chunk.content.lower()
            for concept in concepts:
                if concept.name.lower() in chunk_content_lower:
                    concept_internal_id = concept_ids.get(concept.name.lower())
                    if concept_internal_id:
                        # Get chunk internal ID
                        chunk_data = store.get_chunk(ec.chunk.id)
                        if chunk_data:
                            chunk_internal_id = store._extract_node_id(chunk_data)
                            if chunk_internal_id:
                                store.link_chunk_defines_concept(
                                    chunk_internal_id, concept_internal_id
                                )

    # Store relationships
    with console.status("[bold blue]Storing relationships..."):
        for rel in relations:
            from_id = concept_ids.get(rel.from_concept.lower())
            to_id = concept_ids.get(rel.to_concept.lower())
            if from_id and to_id:
                store.link_concept_relates_to(from_id, to_id)

    console.print(f"[green]Stored {len(concepts)} concepts with relationships[/green]")
```

### 3. Concepts CLI Command Group (`src/cli/commands/concepts.py`)

Create new command file:

```python
"""Concept management CLI commands."""

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..config import CLIConfig
from ..output import print_error, print_success, print_warning
from ..utils import get_store

app = typer.Typer(help="Concept extraction and management")
console = Console()


@app.command("list")
def list_concepts(
    doc_id: Optional[str] = typer.Option(None, "--doc-id", "-d", help="Filter by document"),
    concept_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by type"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List all extracted concepts."""
    config = CLIConfig.load()
    store = get_store(config)

    try:
        concepts = store.list_concepts(doc_id=doc_id)

        # Filter by type if specified
        if concept_type:
            concepts = [c for c in concepts if c.get("concept_type") == concept_type]

        if json_output:
            console.print_json(json.dumps(concepts, indent=2))
            return

        if not concepts:
            print_warning("No concepts found")
            return

        table = Table(title="Concepts")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Definition", style="white", max_width=50)
        table.add_column("Aliases", style="yellow")

        for c in concepts:
            aliases = json.loads(c.get("aliases", "[]"))
            table.add_row(
                c.get("name", ""),
                c.get("concept_type", ""),
                c.get("definition", "")[:50] + "..." if len(c.get("definition", "")) > 50 else c.get("definition", ""),
                ", ".join(aliases) if aliases else "-",
            )

        console.print(table)
        console.print(f"\nTotal: {len(concepts)} concepts")

    except Exception as e:
        print_error(f"Failed to list concepts: {e}")
        raise typer.Exit(1)


@app.command("show")
def show_concept(
    name: str = typer.Argument(..., help="Concept name to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show detailed information about a concept."""
    config = CLIConfig.load()
    store = get_store(config)

    try:
        concept = store.get_concept_by_name(name)

        if not concept:
            # Try case-insensitive search
            from src.chunker.concept_extractor import slugify
            concept = store.get_concept(slugify(name))

        if not concept:
            print_error(f"Concept '{name}' not found")
            raise typer.Exit(1)

        if json_output:
            console.print_json(json.dumps(concept, indent=2))
            return

        # Get citations (chunks that define this concept)
        concept_id = concept.get("concept_id")
        definition_chunks = store.get_concept_definition_chunks(concept_id)

        # Get related concepts
        related = store.get_related_concepts(concept_id)

        # Display
        console.print(f"\n[bold cyan]Concept:[/bold cyan] {concept.get('name')}")
        console.print(f"[bold]Type:[/bold] {concept.get('concept_type')}")
        console.print(f"[bold]Definition:[/bold] {concept.get('definition')}")

        aliases = json.loads(concept.get("aliases", "[]"))
        if aliases:
            console.print(f"[bold]Aliases:[/bold] {', '.join(aliases)}")

        if definition_chunks:
            console.print("\n[bold]Sources (Citations):[/bold]")
            for chunk in definition_chunks:
                section_path = json.loads(chunk.get("section_path", "[]"))
                doc_id = chunk.get("document_id", "unknown")
                path_str = " > ".join(section_path) if section_path else "root"
                console.print(f"  - {doc_id} > {path_str}")

        if related:
            console.print("\n[bold]Related Concepts:[/bold]")
            for rel in related:
                console.print(f"  - {rel.get('name')} ({rel.get('concept_type')})")

    except Exception as e:
        print_error(f"Failed to show concept: {e}")
        raise typer.Exit(1)


@app.command("search")
def search_concepts(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Search for concepts by name or definition."""
    config = CLIConfig.load()
    store = get_store(config)

    try:
        all_concepts = store.list_concepts()

        # Simple text search in name, definition, aliases
        query_lower = query.lower()
        matches = []
        for c in all_concepts:
            score = 0
            name = c.get("name", "").lower()
            definition = c.get("definition", "").lower()
            aliases = json.loads(c.get("aliases", "[]"))
            aliases_lower = [a.lower() for a in aliases]

            if query_lower in name:
                score += 10
            if query_lower == name:
                score += 20
            if query_lower in definition:
                score += 5
            if any(query_lower in a for a in aliases_lower):
                score += 8
            if any(query_lower == a for a in aliases_lower):
                score += 15

            if score > 0:
                matches.append((c, score))

        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        matches = matches[:top_k]

        if json_output:
            results = [{"concept": m[0], "score": m[1]} for m in matches]
            console.print_json(json.dumps(results, indent=2))
            return

        if not matches:
            print_warning(f"No concepts found matching '{query}'")
            return

        table = Table(title=f"Search Results for '{query}'")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Score", style="yellow")

        for c, score in matches:
            table.add_row(c.get("name", ""), c.get("concept_type", ""), str(score))

        console.print(table)

    except Exception as e:
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)


@app.command("related")
def related_concepts(
    name: str = typer.Argument(..., help="Concept name"),
    depth: int = typer.Option(1, "--depth", "-d", help="Traversal depth"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show concepts related to a given concept."""
    config = CLIConfig.load()
    store = get_store(config)

    try:
        from src.chunker.concept_extractor import slugify

        concept = store.get_concept_by_name(name) or store.get_concept(slugify(name))

        if not concept:
            print_error(f"Concept '{name}' not found")
            raise typer.Exit(1)

        concept_id = concept.get("concept_id")
        related = store.get_related_concepts(concept_id)

        if json_output:
            console.print_json(json.dumps(related, indent=2))
            return

        if not related:
            print_warning(f"No related concepts found for '{name}'")
            return

        console.print(f"\n[bold]Concepts related to[/bold] [cyan]{name}[/cyan]:\n")

        for rel in related:
            aliases = json.loads(rel.get("aliases", "[]"))
            alias_str = f" ({', '.join(aliases)})" if aliases else ""
            console.print(f"  - [cyan]{rel.get('name')}[/cyan]{alias_str}")
            console.print(f"    Type: {rel.get('concept_type')}")
            console.print(f"    {rel.get('definition', '')[:80]}...")
            console.print()

    except Exception as e:
        print_error(f"Failed to get related concepts: {e}")
        raise typer.Exit(1)
```

### 4. Register Command Group

**`src/cli/commands/__init__.py`:**

```python
from . import concepts, config_cmd, db, docs, inbox, ingest, search

__all__ = [
    "concepts",
    "config_cmd",
    "db",
    "docs",
    "inbox",
    "ingest",
    "search",
]
```

**`src/cli/main.py`:**

```python
from .commands import concepts, config_cmd, db, docs, inbox, ingest, search

# ... existing registrations ...

app.add_typer(concepts.app, name="concepts", help="Concept extraction and management")
```

---

## Test Plan

### test_concepts_cli.py (~15 tests)

- `list` command with no concepts
- `list` command with concepts
- `list` with --doc-id filter
- `list` with --type filter
- `list` with --json output
- `show` command for existing concept
- `show` command for non-existent concept
- `show` with --json output
- `search` command with matches
- `search` command with no matches
- `related` command with related concepts
- `related` command with no related concepts

### test_ingest_concepts.py (~10 tests)

- Ingest without --extract-concepts (no extraction)
- Ingest with --extract-concepts
- Ingest with concept storage
- Ingest with relationship storage
- Ingest with extraction failure (graceful degradation)
- Batch ingest with concept extraction

---

## Verification

1. **Test concept extraction during ingest:**
   ```bash
   kidkazz ingest markdown test_doc.md --extract-concepts --store helix
   ```

2. **Verify concepts stored:**
   ```bash
   kidkazz concepts list
   kidkazz concepts show "Test Concept"
   kidkazz concepts related "Test Concept"
   ```

3. **Test search:**
   ```bash
   kidkazz concepts search "inventory"
   ```

4. **Run tests:**
   ```bash
   PYTHONPATH=. pytest tests/test_concepts_cli.py -v
   PYTHONPATH=. pytest tests/test_ingest_concepts.py -v
   ```

---

## Files Changed/Created

### New Files
- `src/cli/commands/concepts.py`
- `tests/test_concepts_cli.py`
- `tests/test_ingest_concepts.py`

### Modified Files
- `src/cli/config.py` (add concept settings)
- `src/cli/commands/ingest.py` (add extraction to pipeline)
- `src/cli/commands/__init__.py` (register concepts)
- `src/cli/main.py` (register concepts app)

---

## Next Phase

After Phase 6B, proceed to Phase 6C: MCP Tools.
