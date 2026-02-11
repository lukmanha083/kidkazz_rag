"""Output formatting utilities using Rich.

Provides consistent, beautiful terminal output for all CLI commands.
"""

import json
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Global console instance
console = Console()


def _shorten_doc_id(doc_id: str, max_len: int = 50) -> str:
    """Shorten a document ID for display.

    Strips ``_Z-Library`` suffix, replaces underscores with spaces,
    and truncates to *max_len* characters.
    """
    if not doc_id:
        return ""
    name = doc_id.removesuffix("_Z-Library")
    name = name.replace("_", " ")
    if len(name) > max_len:
        name = name[: max_len - 1] + "\u2026"
    return name


def _build_content_flags(result: dict[str, Any]) -> str:
    """Return content-flag indicators like ``[TABLE] [CODE]``."""
    flags: list[str] = []
    if result.get("has_table"):
        flags.append("[TABLE]")
    if result.get("has_code"):
        flags.append("[CODE]")
    if result.get("has_math"):
        flags.append("[MATH]")
    return " ".join(flags)


def print_search_results(
    results: list[dict[str, Any]],
    show_context: bool = False,
    compact: bool = False,
) -> None:
    """Print search results in rich format."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    content_limit = 300 if compact else 2000
    context_limit = 200 if compact else 500
    parent_limit = 100 if compact else 300

    for i, result in enumerate(results, 1):
        score = result.get("similarity_score", 0)
        content = result.get("content", "")
        chunk_id = result.get("chunk_id", "unknown")
        level = result.get("level", "?")
        semantic_type = result.get("semantic_type", "unknown")

        # Truncate content for display
        if len(content) > content_limit:
            display_content = content[:content_limit] + "..."
        else:
            display_content = content

        # Color based on score
        if score >= 0.8:
            score_color = "green"
        elif score >= 0.5:
            score_color = "yellow"
        else:
            score_color = "red"

        # Build title
        title = f"[bold]#{i}[/bold] {chunk_id}"
        if score > 0:
            title += f" [{score_color}]{score:.3f}[/{score_color}]"

        # Build subtitle with content flags
        subtitle = f"Level {level} | {semantic_type}"
        flags = _build_content_flags(result)
        if flags:
            subtitle += f" {flags}"

        panel = Panel(
            Text(display_content),
            title=title,
            subtitle=subtitle,
            border_style="blue",
        )
        console.print(panel)

        # Citation block
        citation_lines: list[str] = []
        doc_id = result.get("document_id")
        if doc_id:
            citation_lines.append(f"Source: {_shorten_doc_id(doc_id)}")
        section_path = result.get("section_path")
        if section_path:
            if isinstance(section_path, list):
                citation_lines.append(f"Section: {' > '.join(section_path)}")
            else:
                citation_lines.append(f"Section: {section_path}")
        header_text = result.get("header_text")
        if header_text:
            citation_lines.append(f'Header: "{header_text}"')
        topic_tags = result.get("topic_tags")
        if topic_tags:
            if isinstance(topic_tags, list):
                citation_lines.append(f"Tags: {', '.join(topic_tags)}")
            elif isinstance(topic_tags, str) and topic_tags:
                citation_lines.append(f"Tags: {topic_tags}")
        if citation_lines:
            for line in citation_lines:
                console.print(f"  [dim]{line}[/dim]")

        # Show context if requested
        if show_context and "context" in result:
            ctx_list = result["context"][:2]
            labels = ["before", "after"]
            for idx, ctx in enumerate(ctx_list):
                label = labels[idx] if idx < len(labels) else "context"
                ctx_preview = ctx[:context_limit] + "..." if len(ctx) > context_limit else ctx
                console.print(f"  [dim]{label}: {ctx_preview}[/dim]")

        # Show parent if available
        if "parent" in result:
            parent_preview = (
                result["parent"][:parent_limit] + "..."
                if len(result["parent"]) > parent_limit
                else result["parent"]
            )
            console.print(f"  [dim]Parent: {parent_preview}[/dim]")


def print_document_list(documents: list[dict[str, Any]]) -> None:
    """Print document list as table."""
    if not documents:
        console.print("[yellow]No documents found.[/yellow]")
        return

    table = Table(title="Documents", show_header=True, header_style="bold cyan")
    table.add_column("Document ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Chunks", justify="right", style="green")
    table.add_column("Created", style="dim")

    for doc in documents:
        table.add_row(
            doc.get("doc_id", ""),
            doc.get("title", ""),
            str(doc.get("chunk_count", 0)),
            str(doc.get("created_at", ""))[:19],  # Truncate timestamp
        )

    console.print(table)


def print_document_stats(stats: dict[str, Any]) -> None:
    """Print document statistics."""
    doc_id = stats.get("doc_id", "unknown")
    title = stats.get("title", "Unknown")

    console.print(f"\n[bold]Document:[/bold] {title} ({doc_id})")
    console.print("=" * 50)

    # Overview section
    console.print("\n[bold cyan]Overview[/bold cyan]")
    console.print(f"  Total Chunks: {stats.get('total_chunks', 0)}")
    console.print(f"  Total Tokens: {stats.get('total_tokens', 0):,}")

    # Chunks by level
    chunks_by_level = stats.get("chunks_by_level", {})
    if chunks_by_level:
        console.print("\n[bold cyan]Chunks by Level[/bold cyan]")
        for level, count in sorted(chunks_by_level.items()):
            label = "sections" if str(level) == "1" else "leaves"
            console.print(f"  Level {level} ({label}): {count}")

    # Chunks by type
    chunks_by_type = stats.get("chunks_by_type", {})
    if chunks_by_type:
        total = stats.get("total_chunks", 1) or 1
        console.print("\n[bold cyan]Chunks by Type[/bold cyan]")
        for stype, count in sorted(chunks_by_type.items(), key=lambda x: -x[1]):
            pct = (count / total) * 100
            console.print(f"  {stype}: {count} ({pct:.0f}%)")


def print_config(config: dict[str, Any], sources: Optional[dict[str, str]] = None) -> None:
    """Print configuration settings."""
    console.print("\n[bold]Configuration[/bold]")
    console.print("=" * 40)

    for section, settings in config.items():
        console.print(f"\n[bold cyan]{section.title()}[/bold cyan]")
        if isinstance(settings, dict):
            for key, value in settings.items():
                source = ""
                if sources and key in sources:
                    source = f" [dim]({sources[key]})[/dim]"
                console.print(f"  {key}: {value}{source}")
        else:
            console.print(f"  {settings}")


def print_db_status(status: dict[str, Any]) -> None:
    """Print database status."""
    console.print("\n[bold]Database Status[/bold]")
    console.print("=" * 40)

    connected = status.get("connected", False)
    if connected:
        console.print(f"\nConnection: [green]Connected[/green]")
    else:
        console.print(f"\nConnection: [red]Disconnected[/red]")

    console.print(f"Documents:  {status.get('documents', 0)}")
    console.print(f"Chunks:     {status.get('chunks', 0)}")

    console.print("\n[bold cyan]Storage[/bold cyan]")
    console.print(f"  Type: {status.get('store_type', 'unknown')}")
    if status.get("store_type") == "helix":
        console.print(f"  Port: {status.get('port', 6969)}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]{message}[/green]")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red]Error:[/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]Warning:[/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]Info:[/blue] {message}")


def print_json(data: Any) -> None:
    """Print data as formatted JSON."""
    console.print_json(json.dumps(data, indent=2, default=str))


def confirm(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    suffix = "[Y/n]" if default else "[y/N]"
    response = console.input(f"{message} {suffix} ")

    if not response:
        return default

    return response.lower() in ("y", "yes")


def print_chunks_preview(chunks: list[Any], max_display: int = 5) -> None:
    """Print a preview of chunks."""
    table = Table(title=f"Chunks Preview (showing {min(len(chunks), max_display)} of {len(chunks)})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Level", justify="center")
    table.add_column("Content Preview", style="white")

    for chunk in chunks[:max_display]:
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        preview = content[:80] + "..." if len(content) > 80 else content
        chunk_id = chunk.id if hasattr(chunk, "id") else "?"
        level = str(chunk.level) if hasattr(chunk, "level") else "?"
        table.add_row(chunk_id, level, preview)

    console.print(table)

    if len(chunks) > max_display:
        console.print(f"[dim]... and {len(chunks) - max_display} more chunks[/dim]")


def print_ingestion_summary(
    doc_id: str,
    title: str,
    chunk_count: int,
    source_file: str,
) -> None:
    """Print ingestion summary."""
    console.print("\n[bold green]Ingestion Complete[/bold green]")
    console.print(f"  Document ID: [cyan]{doc_id}[/cyan]")
    console.print(f"  Title: {title}")
    console.print(f"  Chunks: {chunk_count}")
    console.print(f"  Source: {source_file}")
