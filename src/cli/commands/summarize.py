"""CLI commands for document summarization."""

import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from src.cli.config import CLIConfig
from src.cli.output import print_error, print_json, print_success, print_warning

# Optional import for summarizer
try:
    from src.chunker.summarizer import DocumentSummarizer, Summary, INSTRUCTOR_AVAILABLE

    SUMMARIZER_AVAILABLE = INSTRUCTOR_AVAILABLE
except ImportError:
    DocumentSummarizer = None  # type: ignore
    Summary = None  # type: ignore
    SUMMARIZER_AVAILABLE = False

app = typer.Typer(help="Document summarization commands")
console = Console()


def _get_store(config: CLIConfig):
    """Get storage instance."""
    from src.cli.commands.ingest import get_store
    return get_store(config)


def _get_embedder(config: CLIConfig):
    """Get embedder instance."""
    from src.cli.commands.ingest import get_embedder
    return get_embedder(config)


@app.command("generate")
def generate_summaries(
    doc_id: str = typer.Argument(..., help="Document ID to summarize"),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider for summarization",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Regenerate even if summaries exist",
    ),
) -> None:
    """Generate hierarchical summaries for a document.

    Creates summaries at document, chapter (L1), and section (L2) levels.

    Examples:
        kidkazz summarize generate inventory_accounting
        kidkazz summarize generate inventory_accounting --provider anthropic/claude-opus-4-20250514
    """
    if not SUMMARIZER_AVAILABLE:
        print_error(
            "Summarization requires 'instructor' package. "
            "Install with: pip install instructor"
        )
        raise typer.Exit(1)

    config = CLIConfig.load()
    store = _get_store(config)

    # Check if document exists
    doc = store.get_document(doc_id)
    if not doc:
        print_error(f"Document not found: {doc_id}")
        raise typer.Exit(1)

    doc_title = doc.get("title", doc_id)

    # Check for existing summaries
    if not force:
        existing = store.get_document_summaries(doc_id)
        if existing:
            if json_output:
                print_json({"status": "exists", "count": len(existing)})
            else:
                print_warning(
                    f"Document already has {len(existing)} summaries. "
                    "Use --force to regenerate."
                )
            return

    # Get chunks for document
    chunks_result = store.get_document_chunks(doc_id)
    if not chunks_result:
        print_error(f"No chunks found for document: {doc_id}")
        raise typer.Exit(1)

    # Convert to dict format expected by summarizer
    chunks = []
    for ec in chunks_result:
        chunk = ec.chunk if hasattr(ec, 'chunk') else ec
        chunk_dict = {
            "id": chunk.id if hasattr(chunk, 'id') else chunk.get("chunk_id"),
            "content": chunk.content if hasattr(chunk, 'content') else chunk.get("content"),
            "level": chunk.level if hasattr(chunk, 'level') else chunk.get("level"),
            "section_path": chunk.section_path if hasattr(chunk, 'section_path') else chunk.get("section_path", []),
            "source_section": chunk.source_section if hasattr(chunk, 'source_section') else chunk.get("source_section"),
            "child_ids": chunk.child_ids if hasattr(chunk, 'child_ids') else chunk.get("child_ids", []),
        }
        chunks.append(chunk_dict)

    if not json_output:
        console.print(f"\nGenerating summaries for: [bold]{doc_title}[/bold]")
        console.print(f"Found {len(chunks)} chunks")

    # Initialize summarizer
    final_provider = provider or config.summarization_provider or "anthropic/claude-sonnet-4-20250514"
    summarizer = DocumentSummarizer(provider=final_provider)

    # Generate summaries
    try:
        with console.status("[bold green]Generating summaries...") as status:
            summaries = summarizer.generate_all_summaries(
                document_id=doc_id,
                document_title=doc_title,
                chunks=chunks,
            )
    except Exception as e:
        print_error(f"Summarization failed: {e}")
        raise typer.Exit(1)

    # Store summaries (with embeddings)
    embedder = _get_embedder(config)
    stored_count = 0

    for summary in summaries:
        try:
            # Generate embedding for summary
            embedding = embedder.embed_text(summary.content)
            summary.embedding = embedding

            # Store summary
            store.store_summary(summary)
            stored_count += 1
        except Exception as e:
            if not json_output:
                print_warning(f"Failed to store summary {summary.summary_id}: {e}")

    # Output results
    if json_output:
        print_json({
            "doc_id": doc_id,
            "title": doc_title,
            "summaries_generated": len(summaries),
            "summaries_stored": stored_count,
            "levels": {
                "document": len([s for s in summaries if s.level == "document"]),
                "chapter": len([s for s in summaries if s.level == "chapter"]),
                "section": len([s for s in summaries if s.level == "section"]),
            },
        })
    else:
        console.print()
        print_success(
            f"Generated {len(summaries)} summaries "
            f"({stored_count} stored)"
        )

        # Show breakdown
        doc_count = len([s for s in summaries if s.level == "document"])
        chap_count = len([s for s in summaries if s.level == "chapter"])
        sec_count = len([s for s in summaries if s.level == "section"])
        console.print(f"  Document: {doc_count}")
        console.print(f"  Chapters: {chap_count}")
        console.print(f"  Sections: {sec_count}")


@app.command("show")
def show_summary(
    doc_id: str = typer.Argument(..., help="Document ID"),
    level: str = typer.Option(
        "document",
        "--level",
        "-l",
        help="Summary level: document, chapter, section, all",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """Display summaries for a document.

    Examples:
        kidkazz summarize show inventory_accounting
        kidkazz summarize show inventory_accounting --level chapter
        kidkazz summarize show inventory_accounting --level all
    """
    config = CLIConfig.load()
    store = _get_store(config)

    # Get summaries
    level_filter = None if level == "all" else level
    summaries = store.get_document_summaries(doc_id, level=level_filter)

    if not summaries:
        if json_output:
            print_json({"summaries": []})
        else:
            print_warning(f"No summaries found for document: {doc_id}")
        return

    if json_output:
        print_json({
            "doc_id": doc_id,
            "summaries": [
                {
                    "summary_id": s.get("summary_id"),
                    "level": s.get("level"),
                    "content": s.get("content"),
                    "key_points": json.loads(s.get("key_points", "[]")),
                    "word_count": s.get("word_count"),
                }
                for s in summaries
            ],
        })
    else:
        for summary in summaries:
            level_name = summary.get("level", "unknown").title()
            content = summary.get("content", "")
            key_points = json.loads(summary.get("key_points", "[]"))

            panel = Panel(
                content,
                title=f"[bold]{level_name} Summary[/bold]",
                border_style="blue" if summary.get("level") == "document" else "green",
            )
            console.print(panel)

            if key_points:
                console.print("\n[bold]Key Points:[/bold]")
                for point in key_points:
                    console.print(f"  • {point}")
            console.print()


@app.command("list")
def list_summarized(
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """List all documents that have summaries.

    Examples:
        kidkazz summarize list
        kidkazz summarize list --json
    """
    config = CLIConfig.load()
    store = _get_store(config)

    # Get all documents with summaries
    docs = store.list_summarized_documents()

    if not docs:
        if json_output:
            print_json({"documents": []})
        else:
            console.print("[yellow]No summarized documents found[/yellow]")
        return

    if json_output:
        print_json({"documents": docs})
    else:
        table = Table(title="Summarized Documents")
        table.add_column("Document ID", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Summaries", justify="right")

        for doc in docs:
            doc_id = doc.get("doc_id", "")
            title = doc.get("title", doc_id)
            summary_count = doc.get("summary_count", 0)
            table.add_row(doc_id, title, str(summary_count))

        console.print(table)


@app.command("search")
def search_summaries(
    query: str = typer.Argument(..., help="Search query"),
    level: Optional[str] = typer.Option(
        None,
        "--level",
        "-l",
        help="Filter by level: document, chapter, section",
    ),
    doc_id: Optional[str] = typer.Option(
        None,
        "--doc",
        "-d",
        help="Filter by document ID",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        help="Maximum results",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """Search summaries semantically.

    Examples:
        kidkazz summarize search "inventory valuation methods"
        kidkazz summarize search "FIFO" --level chapter
        kidkazz summarize search "cost accounting" --doc inventory_accounting
    """
    config = CLIConfig.load()
    store = _get_store(config)
    embedder = _get_embedder(config)

    # Generate query embedding
    query_embedding = embedder.embed_text(query)

    # Search summaries
    results = store.search_summaries(
        query_embedding=query_embedding,
        limit=limit,
        level=level,
        document_id=doc_id,
    )

    if not results:
        if json_output:
            print_json({"results": []})
        else:
            console.print("[yellow]No matching summaries found[/yellow]")
        return

    if json_output:
        print_json({
            "query": query,
            "results": [
                {
                    "summary_id": r.get("summary_id"),
                    "document_id": r.get("document_id"),
                    "level": r.get("level"),
                    "content": r.get("content"),
                    "score": r.get("score", 0),
                }
                for r in results
            ],
        })
    else:
        console.print(f"\n[bold]Search results for:[/bold] {query}\n")

        for i, result in enumerate(results, 1):
            level_name = result.get("level", "unknown").title()
            doc = result.get("document_id", "")
            content = result.get("content", "")
            score = result.get("score", 0)

            # Truncate long content
            if len(content) > 200:
                content = content[:200] + "..."

            console.print(f"[bold cyan]{i}. {level_name} Summary[/bold cyan] ({doc})")
            console.print(f"   Score: {score:.3f}")
            console.print(f"   {content}")
            console.print()


@app.command("delete")
def delete_summaries(
    doc_id: str = typer.Argument(..., help="Document ID"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """Delete all summaries for a document.

    Examples:
        kidkazz summarize delete inventory_accounting
        kidkazz summarize delete inventory_accounting --force
    """
    config = CLIConfig.load()
    store = _get_store(config)

    # Check if summaries exist
    existing = store.get_document_summaries(doc_id)
    if not existing:
        if json_output:
            print_json({"status": "not_found", "deleted": 0})
        else:
            print_warning(f"No summaries found for document: {doc_id}")
        return

    # Confirm deletion
    if not force and not json_output:
        confirm = typer.confirm(
            f"Delete {len(existing)} summaries for '{doc_id}'?"
        )
        if not confirm:
            console.print("Cancelled")
            return

    # Delete summaries
    try:
        store.delete_document_summaries(doc_id)
        if json_output:
            print_json({"status": "deleted", "deleted": len(existing)})
        else:
            print_success(f"Deleted {len(existing)} summaries for '{doc_id}'")
    except Exception as e:
        print_error(f"Delete failed: {e}")
        raise typer.Exit(1)


@app.command("hierarchy")
def show_hierarchy(
    doc_id: str = typer.Argument(..., help="Document ID"),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """Show summary hierarchy for a document as a tree.

    Examples:
        kidkazz summarize hierarchy inventory_accounting
    """
    config = CLIConfig.load()
    store = _get_store(config)

    # Get all summaries for document
    summaries = store.get_document_summaries(doc_id)
    if not summaries:
        if json_output:
            print_json({"hierarchy": None})
        else:
            print_warning(f"No summaries found for document: {doc_id}")
        return

    # Build hierarchy
    by_id = {s.get("summary_id"): s for s in summaries}
    doc_summary = next((s for s in summaries if s.get("level") == "document"), None)

    if json_output:
        def build_tree(summary):
            children = [
                s for s in summaries
                if s.get("parent_summary_id") == summary.get("summary_id")
            ]
            return {
                "summary_id": summary.get("summary_id"),
                "level": summary.get("level"),
                "content": summary.get("content", "")[:100] + "...",
                "children": [build_tree(c) for c in children],
            }

        if doc_summary:
            print_json({"hierarchy": build_tree(doc_summary)})
        else:
            print_json({"hierarchy": None})
    else:
        if not doc_summary:
            console.print("[yellow]No document-level summary found[/yellow]")
            return

        tree = Tree(f"[bold blue]{doc_id}[/bold blue]")
        doc_node = tree.add(
            f"[bold]Document Summary[/bold]\n{doc_summary.get('content', '')[:100]}..."
        )

        # Add chapters
        chapters = [s for s in summaries if s.get("level") == "chapter"]
        for chapter in chapters:
            chapter_node = doc_node.add(
                f"[green]Chapter[/green]\n{chapter.get('content', '')[:80]}..."
            )

            # Add sections under this chapter
            sections = [
                s for s in summaries
                if s.get("level") == "section"
                and s.get("parent_summary_id") == chapter.get("summary_id")
            ]
            for section in sections:
                chapter_node.add(
                    f"[dim]Section[/dim]\n{section.get('content', '')[:60]}..."
                )

        console.print(tree)
