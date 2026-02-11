"""Ingest commands for document ingestion pipeline."""

import json
from pathlib import Path
from typing import Optional

import typer

from ..config import CLIConfig
from ..output import (
    console,
    print_chunks_preview,
    print_error,
    print_ingestion_summary,
    print_info,
    print_json,
    print_success,
    print_warning,
)
from ..progress import BatchTracker, IngestionTracker, ingestion_progress
from ..utils import (
    get_embedder,
    get_store,
    parse_chunk_sizes,
    parse_tags,
    resolve_doc_id,
    resolve_title,
)
from src.chunker import create_hierarchical_chunks, enrich_all_chunks

app = typer.Typer(help="Document ingestion commands")


def _resolve_file_path(file: Path, config: CLIConfig) -> Path:
    """Resolve file path, checking output directory if file not found.

    Args:
        file: The file path provided by user
        config: CLI configuration with output_path

    Returns:
        Resolved absolute path to the file

    Raises:
        typer.Exit: If file not found in any location
    """
    # If absolute path or exists in current location, use as-is
    if file.is_absolute():
        if file.exists():
            return file
        print_error(f"File not found: {file}")
        raise typer.Exit(1)

    # Check current directory first
    if file.exists():
        return file.resolve()

    # Check output directory
    output_path = Path(config.output_path).expanduser()
    output_file = output_path / file
    if output_file.exists():
        return output_file.resolve()

    # Also try with just the filename in output directory
    output_file_name = output_path / file.name
    if output_file_name.exists():
        return output_file_name.resolve()

    # File not found anywhere
    print_error(
        f"File not found: {file}\n"
        f"Searched in:\n"
        f"  - Current directory: {Path.cwd()}\n"
        f"  - Output directory: {output_path}"
    )
    raise typer.Exit(1)


@app.command("markdown")
def ingest_markdown(
    file: Path = typer.Argument(
        ...,
        help="Path to Markdown file (checks output directory if not found)",
        dir_okay=False,
    ),
    title: Optional[str] = typer.Option(
        None,
        "--title",
        "-t",
        help="Document title (default: from H1 or filename)",
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        "-g",
        help="Comma-separated tags (e.g., 'inventory,accounting')",
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
        help="Embedder type: openai (API), mock",
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
    images_dir: Optional[Path] = typer.Option(
        None,
        "--images-dir",
        help="Directory with block images for multimodal embedding (e.g., ~/.kidkazz/images/<doc_id>)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Chunk Markdown file, generate embeddings, and store in database.

    Embedder options:
        openai     - OpenAI API embeddings (requires OPENAI_API_KEY, default)
        mock       - Mock embeddings for testing

    File resolution:
        If file is not found in current directory, the output directory
        (configured in .kidkazz.toml) will be checked automatically.

    Multimodal embedding:
        Use --images-dir to provide block images extracted during PDF parsing.
        Chunks containing image markers will get fused text+image embeddings
        via Cohere Embed v4 (requires cohere embedder).
    """
    config = CLIConfig.load()

    # Resolve file path (check output directory if not found)
    file = _resolve_file_path(file, config)

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
    final_doc_id = resolve_doc_id(file)
    final_title = resolve_title(file, content, title)

    # Parse tags early so they can be shown in dry-run
    tag_list = parse_tags(tags) or []

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
        if tag_list:
            console.print(f"  Tags: {', '.join(tag_list)}")
        console.print(f"  Chunk sizes: {level_sizes}")
        console.print()
        print_chunks_preview(chunks)
        return

    result = {
        "doc_id": final_doc_id,
        "title": final_title,
        "tags": tag_list,
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

            # Build image_map if images directory provided
            embed_kwargs: dict = {"batch_size": 32}
            if images_dir and images_dir.exists():
                from pathlib import Path as _Path

                image_files = {
                    f.stem: f
                    for f in images_dir.iterdir()
                    if f.suffix.lower() in (".png", ".jpg", ".jpeg")
                }
                if image_files and hasattr(embedder_instance, "embed_multimodal"):
                    embed_kwargs["image_map"] = image_files
                    print_info(
                        f"Found {len(image_files)} images for multimodal embedding"
                    )

            embedded_chunks = embedder_instance.embed_chunks(chunks, **embed_kwargs)
            tracker.complete("Generating embeddings...")

            # Stage 3: Store in database
            tracker.add_stage("Storing in database...", total=100)
            store_instance = get_store(config)

            # Check embedding dimension compatibility before storing
            if embedded_chunks and hasattr(store_instance, "check_embedding_compatibility"):
                try:
                    store_instance.check_embedding_compatibility(
                        new_dim=embedded_chunks[0].embedding_dim,
                        new_model=embedded_chunks[0].model_name,
                    )
                except RuntimeError as e:
                    print_error(str(e))
                    raise typer.Exit(1) from None

            chunk_id_map = store_instance.store_document(
                final_doc_id,
                final_title,
                embedded_chunks,
                metadata,
                tags=tag_list,
            )
            tracker.complete("Storing in database...")

            result["chunks"] = len(chunks)

            # Collect content statistics from metadata
            stats = {
                "tables": sum(1 for m in metadata if getattr(m, "has_table", False)),
                "code": sum(1 for m in metadata if getattr(m, "has_code", False)),
                "math": sum(1 for m in metadata if getattr(m, "has_math", False)),
                "lists": sum(1 for m in metadata if getattr(m, "has_list", False)),
            }
            result["content_stats"] = stats

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
        # Show content statistics
        stats = result.get("content_stats", {})
        if any(stats.values()):
            console.print("\n[bold]Content Statistics:[/bold]")
            if stats.get("tables", 0) > 0:
                console.print(f"  Tables: {stats['tables']} chunks")
            if stats.get("code", 0) > 0:
                console.print(f"  Code blocks: {stats['code']} chunks")
            if stats.get("math", 0) > 0:
                console.print(f"  Math expressions: {stats['math']} chunks")
            if stats.get("lists", 0) > 0:
                console.print(f"  Lists: {stats['lists']} chunks")


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
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        "-g",
        help="Comma-separated tags for all files (e.g., 'inventory,accounting')",
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
        help="Embedder type: openai (API), mock",
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

    # Parse tags
    tag_list = parse_tags(tags) or []

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
                store_instance.store_document(doc_id, title, embedded, metadata, tags=tag_list)

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

    Note: Use 'kidkazz inbox parse' to convert PDFs via Reducto.ai API,
    then use 'kidkazz ingest markdown' to ingest the converted markdown.
    """
    # Direct users to use Reducto.ai for PDF parsing
    print_warning(
        "Use Reducto.ai API for PDF conversion:\n"
        "  1. Place PDFs in ~/.kidkazz/inbox/\n"
        "  2. Run: kidkazz inbox parse\n"
        "  3. Run: kidkazz ingest markdown ~/.kidkazz/output/<document>.md"
    )

    if json_output:
        print_json({
            "status": "not_implemented",
            "message": "Use 'kidkazz inbox parse' for PDF conversion via Reducto.ai.",
        })

    raise typer.Exit(1)
