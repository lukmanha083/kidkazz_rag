"""Table management CLI commands."""

from typing import Optional

import typer

from ..output import console, print_error, print_info, print_json, print_warning

app = typer.Typer(help="Table management commands")


# Lazy imports to avoid circular dependencies
def _get_table_store():
    """Get table store instance."""
    try:
        from src.storage.table_store import TableStore
        from src.cli.utils import get_embedder
        from src.cli.config import CLIConfig

        config = CLIConfig.load()
        embedder = get_embedder(config)
        return TableStore(embedder=embedder)
    except ImportError as e:
        print_error(f"Table store not available: {e}")
        raise typer.Exit(1)


@app.command("list")
def list_tables(
    doc_id: Optional[str] = typer.Option(
        None,
        "--doc-id",
        "-d",
        help="Filter by document ID",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """List all tables in the knowledge base."""
    store = _get_table_store()

    tables = store.list_tables(doc_id=doc_id)

    if not tables:
        if json_output:
            print_json({"tables": [], "count": 0})
        else:
            print_info("No tables found.")
        return

    if json_output:
        print_json({"tables": tables, "count": len(tables)})
    else:
        console.print(f"\n[bold]Tables ({len(tables)}):[/bold]\n")
        for table in tables:
            console.print(f"  [cyan]{table['table_id']}[/cyan]")
            console.print(f"    Columns: {', '.join(table['column_names'])}")
            console.print(f"    Rows: {table['row_count']}")
            if table.get("summary"):
                summary = table["summary"][:80] + "..." if len(table.get("summary", "")) > 80 else table.get("summary", "")
                console.print(f"    Summary: {summary}")
            console.print()


@app.command("show")
def show_table(
    table_id: str = typer.Argument(..., help="Table ID to display"),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
    format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format: markdown, kv, raw",
    ),
) -> None:
    """Show table details including structure and content."""
    store = _get_table_store()

    table = store.get_table(table_id)

    if not table:
        print_error(f"Table not found: {table_id}")
        raise typer.Exit(1)

    if json_output:
        print_json({
            "table_id": table_id,
            "columns": table.column_names,
            "column_types": table.column_types,
            "rows": table.rows,
            "row_count": table.row_count,
            "column_count": table.column_count,
            "raw_markdown": table.raw_markdown,
            "context": table.surrounding_context,
        })
    else:
        console.print(f"\n[bold]Table: {table_id}[/bold]\n")
        console.print(f"Columns ({table.column_count}): {', '.join(table.column_names)}")
        console.print(f"Rows: {table.row_count}")
        console.print(f"Types: {', '.join(table.column_types)}")
        console.print()

        if format == "kv":
            console.print("[bold]Content (Key-Value format):[/bold]")
            console.print(table.to_markdown_kv())
        elif format == "raw":
            console.print("[bold]Raw Markdown:[/bold]")
            console.print(table.raw_markdown)
        else:
            console.print("[bold]Markdown:[/bold]")
            console.print(table.raw_markdown)

        if table.surrounding_context:
            console.print(f"\n[dim]Context: {table.surrounding_context[:200]}[/dim]")


@app.command("search")
def search_tables(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(
        5,
        "--limit",
        "-k",
        help="Maximum number of results",
    ),
    doc_id: Optional[str] = typer.Option(
        None,
        "--doc-id",
        "-d",
        help="Filter by document ID",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """Search tables by semantic similarity."""
    store = _get_table_store()

    results = store.search_tables(query, top_k=top_k, doc_id=doc_id)

    if not results:
        if json_output:
            print_json({"results": [], "query": query})
        else:
            print_info(f"No tables found for query: {query}")
        return

    if json_output:
        json_results = [
            {
                "table_id": f"table_{table.source_chunk_id}",
                "columns": table.column_names,
                "row_count": table.row_count,
                "score": round(score, 4),
            }
            for table, score in results
        ]
        print_json({"results": json_results, "query": query})
    else:
        console.print(f"\n[bold]Search results for:[/bold] {query}\n")
        for i, (table, score) in enumerate(results, 1):
            console.print(f"  {i}. [cyan]table_{table.source_chunk_id}[/cyan] (score: {score:.4f})")
            console.print(f"     Columns: {', '.join(table.column_names)}")
            console.print(f"     Rows: {table.row_count}")
            console.print()


@app.command("for-concept")
def tables_for_concept(
    concept_name: str = typer.Argument(..., help="Concept name"),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """Get all tables related to a concept."""
    store = _get_table_store()

    tables = store.get_tables_for_concept(concept_name)

    if not tables:
        if json_output:
            print_json({"tables": [], "concept": concept_name})
        else:
            print_info(f"No tables found for concept: {concept_name}")
        return

    if json_output:
        json_tables = [
            {
                "table_id": f"table_{table.source_chunk_id}",
                "columns": table.column_names,
                "row_count": table.row_count,
            }
            for table in tables
        ]
        print_json({"tables": json_tables, "concept": concept_name, "count": len(tables)})
    else:
        console.print(f"\n[bold]Tables for concept:[/bold] {concept_name}\n")
        for table in tables:
            console.print(f"  [cyan]table_{table.source_chunk_id}[/cyan]")
            console.print(f"    Columns: {', '.join(table.column_names)}")
            console.print(f"    Rows: {table.row_count}")
            console.print()
