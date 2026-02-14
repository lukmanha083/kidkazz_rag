"""Concept management commands."""

import json
import time
from typing import Optional

import typer

from ..config import CLIConfig
from pathlib import Path

from ..output import (
    console,
    print_error,
    print_json,
    print_success,
    print_warning,
)
from ..utils import get_store


app = typer.Typer(help="Concept extraction and management commands")


@app.command("list")
def list_concepts(
    doc_id: Optional[str] = typer.Option(
        None,
        "--doc-id",
        "-d",
        help="Filter by document ID",
    ),
    concept_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by concept type (term, method, principle, formula, account)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """List all concepts in the knowledge base."""
    config = CLIConfig.load()
    store = get_store(config)

    try:
        concepts = store.list_concepts(doc_id=doc_id)
    except Exception as e:
        print_error(f"Failed to list concepts: {e}")
        raise typer.Exit(1)

    # Filter by type if specified
    if concept_type:
        concepts = [
            c for c in concepts
            if c.get("concept_type", "").lower() == concept_type.lower()
        ]

    if json_output:
        print_json(concepts)
        return

    if not concepts:
        print_warning("No concepts found.")
        return

    # Print as table
    from rich.table import Table

    table = Table(title="Concepts", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="yellow")
    table.add_column("Definition", style="white")
    table.add_column("Aliases", style="dim")

    for concept in concepts:
        # Parse aliases from JSON string
        aliases_str = concept.get("aliases", "[]")
        try:
            aliases = json.loads(aliases_str) if isinstance(aliases_str, str) else aliases_str
            aliases_display = ", ".join(aliases) if aliases else "-"
        except json.JSONDecodeError:
            aliases_display = "-"

        table.add_row(
            concept.get("name", ""),
            concept.get("concept_type", ""),
            concept.get("definition", ""),
            aliases_display,
        )

    console.print(table)


@app.command("show")
def show_concept(
    name: str = typer.Argument(
        ...,
        help="Concept name or ID to show",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """Show detailed information about a concept."""
    config = CLIConfig.load()
    store = get_store(config)

    try:
        # Try by name first, then by ID
        concept = store.get_concept_by_name(name)
        if concept is None:
            concept = store.get_concept(name)
    except Exception as e:
        print_error(f"Failed to get concept: {e}")
        raise typer.Exit(1)

    if concept is None:
        print_error(f"Concept not found: {name}")
        raise typer.Exit(1)

    if json_output:
        print_json(concept)
        return

    # Print detailed view
    from rich.panel import Panel

    console.print()
    console.print(f"[bold cyan]{concept.get('name', '')}[/bold cyan]")
    console.print(f"[dim]Type: {concept.get('concept_type', 'unknown')}[/dim]")
    console.print()

    # Definition
    console.print("[bold]Definition:[/bold]")
    console.print(concept.get("definition", "No definition available."))
    console.print()

    # Aliases
    aliases_str = concept.get("aliases", "[]")
    try:
        aliases = json.loads(aliases_str) if isinstance(aliases_str, str) else aliases_str
        if aliases:
            console.print(f"[bold]Aliases:[/bold] {', '.join(aliases)}")
            console.print()
    except json.JSONDecodeError:
        pass

    # Source citations
    try:
        chunks = store.get_concept_definition_chunks(concept.get("concept_id", ""))
        if chunks:
            console.print("[bold]Sources:[/bold]")
            for chunk in chunks:
                doc_id = chunk.get("document_id", "unknown")
                section = chunk.get("section_path", "")
                try:
                    section_list = json.loads(section) if isinstance(section, str) else section
                    section_display = " > ".join(section_list) if section_list else ""
                except json.JSONDecodeError:
                    section_display = section
                console.print(f"  - {doc_id}: {section_display}" if section_display else f"  - {doc_id}")
            console.print()
    except Exception:
        pass

    # Related concepts
    try:
        related = store.get_related_concepts(concept.get("concept_id", ""))
        if related:
            console.print("[bold]Related Concepts:[/bold]")
            for rel in related:
                console.print(f"  - {rel.get('name', '')} ({rel.get('concept_type', '')})")
    except Exception:
        pass


@app.command("search")
def search_concepts(
    query: str = typer.Argument(
        ...,
        help="Search query",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """Search for concepts by name, definition, or alias."""
    config = CLIConfig.load()
    store = get_store(config)

    try:
        # Get all concepts and filter locally
        all_concepts = store.list_concepts()
    except Exception as e:
        print_error(f"Failed to search concepts: {e}")
        raise typer.Exit(1)

    # Filter by query (case-insensitive)
    query_lower = query.lower()
    matches = []

    for concept in all_concepts:
        name = concept.get("name", "").lower()
        definition = concept.get("definition", "").lower()
        aliases_str = concept.get("aliases", "[]")

        # Check aliases
        try:
            aliases = json.loads(aliases_str) if isinstance(aliases_str, str) else aliases_str
            aliases_lower = [a.lower() for a in aliases] if aliases else []
        except json.JSONDecodeError:
            aliases_lower = []

        # Match against name, definition, or aliases
        if (
            query_lower in name
            or query_lower in definition
            or any(query_lower in alias for alias in aliases_lower)
        ):
            matches.append(concept)

    if json_output:
        print_json(matches)
        return

    if not matches:
        print_warning(f"No concepts found matching '{query}'.")
        return

    # Print as table
    from rich.table import Table

    table = Table(
        title=f"Search Results for '{query}'",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="yellow")
    table.add_column("Definition", style="white")

    for concept in matches:
        table.add_row(
            concept.get("name", ""),
            concept.get("concept_type", ""),
            concept.get("definition", ""),
        )

    console.print(table)


@app.command("related")
def related_concepts(
    name: str = typer.Argument(
        ...,
        help="Concept name or ID",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """Show concepts related to a given concept."""
    config = CLIConfig.load()
    store = get_store(config)

    try:
        # Find the concept first
        concept = store.get_concept_by_name(name)
        if concept is None:
            concept = store.get_concept(name)
    except Exception as e:
        print_error(f"Failed to get concept: {e}")
        raise typer.Exit(1)

    if concept is None:
        print_error(f"Concept not found: {name}")
        raise typer.Exit(1)

    try:
        related = store.get_related_concepts(concept.get("concept_id", ""))
    except Exception as e:
        print_error(f"Failed to get related concepts: {e}")
        raise typer.Exit(1)

    if json_output:
        print_json(related)
        return

    if not related:
        print_warning(f"No related concepts found for '{concept.get('name', name)}'.")
        return

    # Print as table
    from rich.table import Table

    table = Table(
        title=f"Concepts Related to '{concept.get('name', name)}'",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="yellow")
    table.add_column("Definition", style="white", max_width=60)

    for rel in related:
        definition = rel.get("definition", "")
        if len(definition) > 60:
            definition = definition[:57] + "..."

        table.add_row(
            rel.get("name", ""),
            rel.get("concept_type", ""),
            definition,
        )

    console.print(table)


@app.command("graph")
def graph_concepts(
    doc_id: Optional[str] = typer.Argument(
        None,
        help="Document ID (all concepts if not specified)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path",
    ),
    format: str = typer.Option(
        "png",
        "--format",
        "-f",
        help="Output format: dot, png, svg, pdf",
    ),
    title: Optional[str] = typer.Option(
        None,
        "--title",
        "-t",
        help="Graph title",
    ),
    view: bool = typer.Option(
        False,
        "--view",
        "-v",
        help="Open rendered graph after creation",
    ),
    min_docs: int = typer.Option(
        2,
        "--min-docs",
        help="Minimum number of documents a concept must appear in (default: 2)",
    ),
) -> None:
    """
    Generate a cross-document concept graph.

    Shows concepts shared between multiple textbooks as bridge nodes,
    revealing cross-discipline connections.

    Examples:
        kidkazz concepts graph -o graph.png         # Shared concepts across all docs
        kidkazz concepts graph -o graph.png --min-docs 3  # Only in 3+ books
        kidkazz concepts graph inventory            # All concepts from one doc
        kidkazz concepts graph -f dot -o graph.dot  # Export as DOT
        kidkazz concepts graph --view               # Render and open
    """
    config = CLIConfig.load()
    store = get_store(config)

    try:
        from ..graph_viz import generate_concept_graph

        # Generate graph
        dot_content, rendered_path = generate_concept_graph(
            store=store,
            doc_id=doc_id,
            output_path=output,
            output_format=format,
            title=title,
            min_docs=min_docs,
        )

        if output:
            if rendered_path:
                print_success(f"Graph saved to: {rendered_path}")

                # Open if requested
                if view:
                    import platform
                    import subprocess

                    if platform.system() == "Darwin":
                        subprocess.run(["open", str(rendered_path)])
                    elif platform.system() == "Linux":
                        subprocess.run(["xdg-open", str(rendered_path)])
                    elif platform.system() == "Windows":
                        import os
                        os.startfile(str(rendered_path))
            else:
                print_error("Failed to render graph")
                raise typer.Exit(1)
        else:
            # Print DOT to stdout if no output specified
            console.print(dot_content)

    except ImportError as e:
        print_error(str(e))
        print_warning("For PNG/SVG output, install graphviz:")
        print_warning("  pip install 'kidkazz[concepts]'")
        print_warning("  # Also install system graphviz:")
        print_warning("  # macOS: brew install graphviz")
        print_warning("  # Ubuntu: apt install graphviz")
        raise typer.Exit(1)

    except Exception as e:
        print_error(f"Failed to generate graph: {e}")
        raise typer.Exit(1)


@app.command("export")
def export_concepts(
    doc_id: Optional[str] = typer.Option(
        None,
        "--doc-id",
        "-d",
        help="Filter by document",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file path",
    ),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Export format: json, csv",
    ),
) -> None:
    """
    Export concepts to file.

    Examples:
        kidkazz concepts export -o concepts.json
        kidkazz concepts export -o concepts.csv -f csv
        kidkazz concepts export -d inventory -o inv_concepts.json
    """
    config = CLIConfig.load()
    store = get_store(config)

    try:
        concepts = store.list_concepts(doc_id=doc_id)

        if not concepts:
            print_warning("No concepts to export")
            return

        if format == "json":
            output.write_text(json.dumps(concepts, indent=2))

        elif format == "csv":
            import csv

            with output.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["concept_id", "name", "type", "definition", "aliases"])

                for c in concepts:
                    aliases_str = c.get("aliases", "[]")
                    try:
                        aliases = json.loads(aliases_str) if isinstance(aliases_str, str) else aliases_str
                    except json.JSONDecodeError:
                        aliases = []
                    writer.writerow([
                        c.get("concept_id", ""),
                        c.get("name", ""),
                        c.get("concept_type", ""),
                        c.get("definition", ""),
                        "; ".join(aliases) if aliases else "",
                    ])

        else:
            print_error(f"Unknown format: {format}")
            raise typer.Exit(1)

        print_success(f"Exported {len(concepts)} concepts to: {output}")

    except typer.Exit:
        raise  # Re-raise typer.Exit without catching
    except Exception as e:
        print_error(f"Export failed: {e}")
        raise typer.Exit(1)


def _is_garbage_concept(concept: dict, min_words: int) -> bool:
    """Check if a concept is garbage (empty or too-short definition)."""
    definition = concept.get("definition", "")
    if not definition or not definition.strip():
        return True
    if len(definition.strip().split()) < min_words:
        return True
    return False


@app.command("clean")
def clean_concepts(
    doc_id: Optional[str] = typer.Option(
        None,
        "--doc-id",
        "-d",
        help="Only clean concepts from this document",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be deleted without deleting",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Skip confirmation prompt",
    ),
    min_words: int = typer.Option(
        1,
        "--min-words",
        help="Minimum definition word count to keep (default: 1 = any non-empty definition)",
    ),
) -> None:
    """
    Remove garbage concepts (empty or insufficient definitions).

    By default, identifies concepts with empty definitions (e.g., bold/italic
    text extracted during section summarization) and deletes them.

    Examples:
        kidkazz concepts clean --dry-run          # Preview garbage concepts
        kidkazz concepts clean                    # Delete with confirmation
        kidkazz concepts clean --force            # Delete without confirmation
        kidkazz concepts clean --doc-id inventory # Only from one document
        kidkazz concepts clean --min-words 3      # Require 3+ word definitions
    """
    config = CLIConfig.load()
    store = get_store(config)

    try:
        concepts = store.list_concepts(doc_id=doc_id)
    except Exception as e:
        print_error(f"Failed to list concepts: {e}")
        raise typer.Exit(1)

    if not concepts:
        print_warning("No concepts found.")
        return

    # Identify garbage
    garbage = [c for c in concepts if _is_garbage_concept(c, min_words)]
    valid_count = len(concepts) - len(garbage)

    # Summary
    from rich.table import Table

    console.print()
    console.print(f"[bold]Total concepts:[/bold] {len(concepts)}")
    console.print(f"[bold green]Valid (keeping):[/bold green] {valid_count}")
    console.print(f"[bold red]Garbage (removing):[/bold red] {len(garbage)}")
    console.print()

    if not garbage:
        print_success("No garbage concepts found. Nothing to clean.")
        return

    # Show sample of garbage concepts
    sample = garbage[:20]
    table = Table(
        title=f"Garbage Concepts (showing {len(sample)} of {len(garbage)})",
        show_header=True,
        header_style="bold red",
    )
    table.add_column("Name", style="cyan", no_wrap=True, max_width=40)
    table.add_column("Type", style="yellow")
    table.add_column("Definition", style="dim", max_width=30)
    table.add_column("Concept ID", style="dim", max_width=30)

    for c in sample:
        definition = c.get("definition", "")
        if not definition:
            definition = "[empty]"
        elif len(definition) > 30:
            definition = definition[:27] + "..."

        table.add_row(
            c.get("name", ""),
            c.get("concept_type", ""),
            definition,
            c.get("concept_id", ""),
        )

    console.print(table)
    console.print()

    if dry_run:
        print_warning("Dry run — no concepts were deleted.")
        return

    # Confirmation
    if not force:
        confirmed = typer.confirm(
            f"Delete {len(garbage)} garbage concepts? This cannot be undone"
        )
        if not confirmed:
            print_warning("Aborted.")
            raise typer.Exit(0)

    # Extract internal IDs from already-fetched concept data (avoids extra lookups)
    drop_items = []
    for c in garbage:
        internal_id = c.get("id")
        if internal_id:
            drop_items.append(str(internal_id))
        else:
            failed += 1

    # Delete in parallel using thread pool — 1 HTTP call each (DropConcept only)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    start_time = time.time()
    deleted = 0
    failed = 0
    processed = 0

    def _drop_one(internal_id: str) -> bool:
        try:
            return store.drop_concept_by_internal_id(internal_id)
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_drop_one, iid): iid for iid in drop_items}
        for future in as_completed(futures):
            if future.result():
                deleted += 1
            else:
                failed += 1
            processed += 1
            if processed % 500 == 0:
                console.print(f"[dim]Progress: {processed}/{len(drop_items)} processed...[/dim]")

    elapsed = time.time() - start_time

    console.print()
    print_success(
        f"Cleaned {deleted} garbage concepts in {elapsed:.1f}s "
        f"({failed} failed)"
    )
