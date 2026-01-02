"""Document management commands."""

import json
from pathlib import Path
from typing import Optional

import typer

from ..config import CLIConfig
from ..output import (
    confirm,
    console,
    print_document_list,
    print_document_stats,
    print_error,
    print_json,
    print_success,
    print_warning,
)
from ..utils import get_store

app = typer.Typer(help="Document management commands")


@app.command("list")
def list_documents(
    sort: str = typer.Option(
        "date",
        "--sort",
        "-s",
        help="Sort by: date, name, chunks",
    ),
    reverse: bool = typer.Option(
        False,
        "--reverse",
        "-r",
        help="Reverse sort order",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """List all documents in the knowledge base."""
    config = CLIConfig.load()
    store = get_store(config)

    try:
        docs = store.list_documents()
    except Exception as e:
        print_error(f"Failed to list documents: {e}")
        raise typer.Exit(1)

    # Convert to list of dicts if needed
    doc_list = []
    for doc in docs:
        if hasattr(doc, "doc_id"):
            doc_list.append({
                "doc_id": doc.doc_id,
                "title": getattr(doc, "title", doc.doc_id),
                "chunk_count": getattr(doc, "chunk_count", 0),
                "created_at": str(getattr(doc, "created_at", "")),
            })
        else:
            doc_list.append(doc)

    # Sort documents
    if sort == "name":
        doc_list.sort(key=lambda x: x.get("title", ""), reverse=reverse)
    elif sort == "chunks":
        doc_list.sort(key=lambda x: x.get("chunk_count", 0), reverse=reverse)
    else:  # date
        doc_list.sort(key=lambda x: x.get("created_at", ""), reverse=reverse)

    if json_output:
        print_json(doc_list)
    else:
        print_document_list(doc_list)


@app.command("stats")
def document_stats(
    doc_id: str = typer.Argument(
        ...,
        help="Document identifier",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """Show detailed statistics for a document."""
    config = CLIConfig.load()
    store = get_store(config)

    try:
        stats = store.get_document_stats(doc_id)
    except Exception as e:
        print_error(f"Failed to get stats: {e}")
        raise typer.Exit(1)

    if stats is None:
        print_error(f"Document not found: {doc_id}")
        raise typer.Exit(1)

    # Ensure stats is a dict
    if hasattr(stats, "__dict__"):
        stats = vars(stats)

    # Add doc_id if not present
    if "doc_id" not in stats:
        stats["doc_id"] = doc_id

    if json_output:
        print_json(stats)
    else:
        print_document_stats(stats)


@app.command("export")
def export_document(
    doc_id: str = typer.Argument(
        ...,
        help="Document identifier",
    ),
    output: Path = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file path",
    ),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Export format: json, markdown, csv",
    ),
    include_embeddings: bool = typer.Option(
        False,
        "--include-embeddings",
        help="Include embedding vectors",
    ),
    level: Optional[int] = typer.Option(
        None,
        "--level",
        "-l",
        help="Export only specific level",
    ),
) -> None:
    """Export document data to file."""
    config = CLIConfig.load()
    store = get_store(config)

    try:
        chunks = store.get_document_chunks(doc_id, level=level)
    except Exception as e:
        print_error(f"Failed to get chunks: {e}")
        raise typer.Exit(1)

    if not chunks:
        print_error(f"Document not found or empty: {doc_id}")
        raise typer.Exit(1)

    # Build export data
    export_data = {
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "chunks": [],
    }

    for ec in chunks:
        chunk_data = {
            "chunk_id": ec.chunk.id,
            "content": ec.chunk.content,
            "level": ec.chunk.level,
            "token_count": ec.chunk.token_count,
            "parent_id": ec.chunk.parent_id,
            "child_ids": ec.chunk.child_ids,
        }
        if hasattr(ec.chunk, "section_path"):
            chunk_data["section_path"] = ec.chunk.section_path
        if include_embeddings:
            chunk_data["embedding"] = ec.embedding.tolist() if hasattr(ec.embedding, "tolist") else list(ec.embedding)
        export_data["chunks"].append(chunk_data)

    # Determine output path
    if output is None:
        ext = {"json": ".json", "markdown": ".md", "csv": ".csv"}.get(format, ".json")
        output = Path(f"{doc_id}_export{ext}")

    # Write output
    try:
        if format == "json":
            with open(output, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format == "markdown":
            with open(output, "w", encoding="utf-8") as f:
                f.write(f"# {doc_id}\n\n")
                for chunk in export_data["chunks"]:
                    f.write(f"## Chunk: {chunk['chunk_id']}\n\n")
                    f.write(f"Level: {chunk['level']} | Tokens: {chunk['token_count']}\n\n")
                    f.write(chunk["content"])
                    f.write("\n\n---\n\n")
        elif format == "csv":
            import csv

            with open(output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["chunk_id", "level", "token_count", "content"],
                )
                writer.writeheader()
                for chunk in export_data["chunks"]:
                    writer.writerow({
                        "chunk_id": chunk["chunk_id"],
                        "level": chunk["level"],
                        "token_count": chunk["token_count"],
                        "content": chunk["content"][:500],  # Truncate for CSV
                    })
        else:
            print_error(f"Unknown format: {format}")
            raise typer.Exit(1)

        print_success(f"Exported {len(chunks)} chunks to {output}")

    except Exception as e:
        print_error(f"Failed to write output: {e}")
        raise typer.Exit(1)


@app.command("delete")
def delete_document(
    doc_id: str = typer.Argument(
        ...,
        help="Document identifier",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output result as JSON",
    ),
) -> None:
    """Remove document from knowledge base."""
    config = CLIConfig.load()
    store = get_store(config)

    # Check if document exists
    try:
        stats = store.get_document_stats(doc_id)
        if stats is None:
            print_error(f"Document not found: {doc_id}")
            raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to check document: {e}")
        raise typer.Exit(1)

    # Get chunk count
    chunk_count = stats.get("total_chunks", 0) if isinstance(stats, dict) else getattr(stats, "total_chunks", 0)

    # Confirm deletion
    if not force and not json_output:
        if not confirm(f"Delete document '{doc_id}' with {chunk_count} chunks?"):
            print_warning("Deletion cancelled")
            return

    # Delete
    try:
        if hasattr(store, "delete_document"):
            store.delete_document(doc_id)
        else:
            # MockChunkStore uses clear() or individual deletion
            print_warning("Delete not implemented for this store type")
            raise typer.Exit(1)

        result = {"doc_id": doc_id, "status": "deleted", "chunks_removed": chunk_count}

        if json_output:
            print_json(result)
        else:
            print_success(f"Deleted document '{doc_id}' ({chunk_count} chunks)")

    except Exception as e:
        if json_output:
            print_json({"doc_id": doc_id, "status": "error", "error": str(e)})
        else:
            print_error(f"Failed to delete: {e}")
        raise typer.Exit(1)
