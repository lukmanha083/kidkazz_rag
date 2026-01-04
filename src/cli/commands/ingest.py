"""Ingest commands for document ingestion pipeline."""

from pathlib import Path
from typing import Optional

import typer

from ..config import CLIConfig
from ..output import (
    console,
    print_chunks_preview,
    print_error,
    print_ingestion_summary,
    print_json,
    print_success,
    print_warning,
)
from ..progress import BatchTracker, IngestionTracker, ingestion_progress
from ..utils import (
    get_embedder,
    get_store,
    parse_chunk_sizes,
    resolve_doc_id,
    resolve_title,
)
from src.chunker import create_hierarchical_chunks, enrich_all_chunks

app = typer.Typer(help="Document ingestion commands")


@app.command("markdown")
def ingest_markdown(
    file: Path = typer.Argument(
        ...,
        help="Path to Markdown file",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    doc_id: Optional[str] = typer.Option(
        None,
        "--doc-id",
        "-d",
        help="Document identifier (default: filename)",
    ),
    title: Optional[str] = typer.Option(
        None,
        "--title",
        "-t",
        help="Document title (default: from H1 or filename)",
    ),
    chunk_sizes: str = typer.Option(
        "2048,512,256",
        "--chunk-sizes",
        "-c",
        help="Chunk sizes as L1,L2,overlap",
    ),
    embedder: Optional[str] = typer.Option(
        None,
        "--embedder",
        "-e",
        help="Embedder type: fastembed (local), openai (API), mock",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Embedding model name",
    ),
    store: Optional[str] = typer.Option(
        None,
        "--store",
        "-s",
        help="Storage backend (mock or helix)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show chunks without storing",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Chunk Markdown file, generate embeddings, and store in database.

    Embedder options:
        fastembed  - Local CPU-based embeddings (free, default)
        openai     - OpenAI API embeddings (requires OPENAI_API_KEY)
        mock       - Mock embeddings for testing
    """
    config = CLIConfig.load()

    # Override config with command options
    if store:
        config.store_type = store
    if model:
        config.model_name = model

    # Parse chunk sizes
    try:
        level_sizes = parse_chunk_sizes(chunk_sizes)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Read content
    try:
        content = file.read_text(encoding="utf-8")
    except Exception as e:
        print_error(f"Failed to read file: {e}")
        raise typer.Exit(1)

    # Resolve doc_id and title
    final_doc_id = resolve_doc_id(file, doc_id)
    final_title = resolve_title(file, content, title)

    if dry_run:
        # Just show what would be created
        chunks = create_hierarchical_chunks(
            content,
            doc_id=final_doc_id,
            level_sizes=level_sizes,
        )
        console.print(f"[bold]Dry run:[/bold] Would create {len(chunks)} chunks")
        console.print(f"  Document ID: {final_doc_id}")
        console.print(f"  Title: {final_title}")
        console.print(f"  Chunk sizes: {level_sizes}")
        console.print()
        print_chunks_preview(chunks)
        return

    result = {
        "doc_id": final_doc_id,
        "title": final_title,
        "source": str(file),
        "chunks": 0,
        "status": "success",
    }

    try:
        with ingestion_progress() as progress:
            tracker = IngestionTracker(progress)

            # Stage 1: Parse and chunk
            tracker.add_stage("Chunking document...", total=100)

            chunks = create_hierarchical_chunks(
                content,
                doc_id=final_doc_id,
                level_sizes=level_sizes,
            )
            metadata = enrich_all_chunks(chunks, document_id=final_doc_id)
            tracker.complete("Chunking document...")

            # Stage 2: Generate embeddings
            tracker.add_stage("Generating embeddings...", total=len(chunks))
            try:
                embedder_instance = get_embedder(config, embedder_override=embedder)
            except (ImportError, ValueError) as e:
                print_error(str(e))
                raise typer.Exit(1) from None
            embedded_chunks = embedder_instance.embed_chunks(chunks, batch_size=32)
            tracker.complete("Generating embeddings...")

            # Stage 3: Store in database
            tracker.add_stage("Storing in database...", total=100)
            store_instance = get_store(config)
            store_instance.store_document(
                final_doc_id,
                final_title,
                embedded_chunks,
                metadata,
            )
            tracker.complete("Storing in database...")

            result["chunks"] = len(chunks)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        if json_output:
            print_json(result)
        else:
            print_error(str(e))
        raise typer.Exit(1)

    if json_output:
        print_json(result)
    else:
        print_ingestion_summary(
            final_doc_id,
            final_title,
            result["chunks"],
            str(file),
        )


@app.command("batch")
def ingest_batch(
    directory: Path = typer.Argument(
        ...,
        help="Directory containing files to process",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
    pattern: str = typer.Option(
        "*.md",
        "--pattern",
        "-p",
        help="Glob pattern for files",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Search subdirectories",
    ),
    chunk_sizes: str = typer.Option(
        "2048,512,256",
        "--chunk-sizes",
        "-c",
        help="Chunk sizes as L1,L2,overlap",
    ),
    embedder: Optional[str] = typer.Option(
        None,
        "--embedder",
        "-e",
        help="Embedder type: fastembed (local), openai (API), mock",
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip files already in database",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Process multiple files in a directory."""
    config = CLIConfig.load()

    # Find files
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))

    if not files:
        if json_output:
            print_json({"files": 0, "processed": 0, "errors": 0})
        else:
            print_warning(f"No files matching '{pattern}' found in {directory}")
        return

    if not json_output:
        console.print(f"Found {len(files)} files to process")

    # Parse chunk sizes
    try:
        level_sizes = parse_chunk_sizes(chunk_sizes)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    results = []
    store_instance = get_store(config)
    try:
        embedder_instance = get_embedder(config, embedder_override=embedder)
    except (ImportError, ValueError) as e:
        print_error(str(e))
        raise typer.Exit(1) from None

    # Get existing documents if skip_existing
    existing_docs = set()
    if skip_existing:
        try:
            docs = store_instance.list_documents()
            existing_docs = {d.get("doc_id") or d.doc_id for d in docs}
        except Exception:
            pass

    with BatchTracker(len(files), "Processing files") as tracker:
        for file in files:
            doc_id = file.stem
            file_result = {
                "file": str(file),
                "doc_id": doc_id,
                "status": "success",
            }

            # Skip if exists
            if skip_existing and doc_id in existing_docs:
                file_result["status"] = "skipped"
                file_result["reason"] = "already exists"
                results.append(file_result)
                tracker.advance()
                continue

            try:
                content = file.read_text(encoding="utf-8")
                title = resolve_title(file, content)

                chunks = create_hierarchical_chunks(
                    content,
                    doc_id=doc_id,
                    level_sizes=level_sizes,
                )
                metadata = enrich_all_chunks(chunks, document_id=doc_id)
                embedded = embedder_instance.embed_chunks(chunks, batch_size=32)
                store_instance.store_document(doc_id, title, embedded, metadata)

                file_result["chunks"] = len(chunks)

            except Exception as e:
                file_result["status"] = "error"
                file_result["error"] = str(e)
                tracker.advance(error=str(e))
                results.append(file_result)
                continue

            results.append(file_result)
            tracker.advance()

    # Summary
    summary = tracker.get_summary()
    summary["results"] = results

    if json_output:
        print_json(summary)
    else:
        console.print()
        print_success(f"Processed {summary['successful']}/{summary['total']} files")
        if summary["errors"] > 0:
            print_warning(f"{summary['errors']} files failed")
            for err in summary["error_messages"][:3]:
                console.print(f"  - {err}")


@app.command("pdf")
def ingest_pdf(
    file: Path = typer.Argument(
        ...,
        help="Path to PDF file",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    doc_id: Optional[str] = typer.Option(
        None,
        "--doc-id",
        "-d",
        help="Document identifier",
    ),
    title: Optional[str] = typer.Option(
        None,
        "--title",
        "-t",
        help="Document title",
    ),
    converter: str = typer.Option(
        "marker",
        "--converter",
        help="PDF converter (marker, docling, nougat)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Save intermediate markdown",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """Convert PDF to Markdown, chunk, embed, and store.

    Note: PDF conversion requires GPU and is recommended to run via
    the Colab notebook. This command is for future local support.
    """
    # For now, PDF conversion is not implemented locally
    # Direct users to use the notebook and then ingest markdown
    print_warning(
        "PDF conversion is currently only supported via Google Colab notebook.\n"
        "Please use the notebook to convert PDF to Markdown first:\n"
        "  1. Open notebooks/pdf_to_markdown_converter.ipynb\n"
        "  2. Upload and convert your PDF\n"
        "  3. Run: kidkazz ingest markdown <output.md>"
    )

    if json_output:
        print_json({
            "status": "not_implemented",
            "message": "PDF conversion requires GPU. Use Colab notebook first.",
        })

    raise typer.Exit(1)
