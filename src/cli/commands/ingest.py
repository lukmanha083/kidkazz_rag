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

# Optional concept extraction support
try:
    from src.chunker.concept_extractor import ConceptExtractor

    CONCEPT_EXTRACTION_AVAILABLE = True
except ImportError:
    ConceptExtractor = None  # type: ignore
    CONCEPT_EXTRACTION_AVAILABLE = False

# Optional table processing support
try:
    from src.chunker.table_parser import parse_markdown_table
    from src.chunker.table_summarizer import TableSummarizer
    from src.storage.table_store import TableStore

    TABLE_PROCESSING_AVAILABLE = True
except ImportError:
    parse_markdown_table = None  # type: ignore
    TableSummarizer = None  # type: ignore
    TableStore = None  # type: ignore
    TABLE_PROCESSING_AVAILABLE = False

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
    extract_concepts: bool = typer.Option(
        False,
        "--extract-concepts",
        help="Extract concepts using LLM (requires instructor)",
    ),
    concept_provider: Optional[str] = typer.Option(
        None,
        "--concept-provider",
        help="LLM provider for concept extraction",
    ),
    extract_tables: bool = typer.Option(
        False,
        "--extract-tables",
        help="Extract and summarize tables for semantic search",
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

    File resolution:
        If file is not found in current directory, the output directory
        (configured in .kidkazz.toml) will be checked automatically.
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
    final_doc_id = resolve_doc_id(file, doc_id)
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
        "concepts": 0,
        "tables": 0,
        "status": "success",
    }

    # Resolve concept extraction settings
    do_extract_concepts = extract_concepts or config.extract_concepts
    final_concept_provider = concept_provider or config.concept_provider

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
            chunk_id_map = store_instance.store_document(
                final_doc_id,
                final_title,
                embedded_chunks,
                metadata,
                tags=tag_list,
            )
            tracker.complete("Storing in database...")

            result["chunks"] = len(chunks)

            # Stage 4: Extract concepts (optional)
            if do_extract_concepts:
                if not CONCEPT_EXTRACTION_AVAILABLE:
                    print_warning(
                        "Concept extraction requires 'instructor' package. "
                        "Install with: pip install instructor"
                    )
                else:
                    tracker.add_stage("Extracting concepts...", total=100)
                    try:
                        extractor = ConceptExtractor(provider=final_concept_provider)
                        # Convert Chunk dataclasses to dicts for concept extractor
                        chunks_as_dicts = [
                            {
                                "content": c.content if hasattr(c, 'content') else c.get("content", ""),
                                "id": c.id if hasattr(c, 'id') else c.get("id", ""),
                                "section_path": c.section_path if hasattr(c, 'section_path') else c.get("section_path", []),
                            }
                            for c in chunks
                        ]
                        # Use new method that preserves per-chunk mapping
                        extracted_concepts, relations, chunk_extractions = extractor.extract_from_chunks_with_mapping(
                            chunks_as_dicts, final_title, metadata_list=metadata
                        )

                        # Store or merge concepts (cross-document linking)
                        from src.chunker.concept_extractor import slugify

                        concept_ids = {}
                        for concept in extracted_concepts:
                            concept_slug = slugify(concept.name)
                            internal_id = store_instance.store_or_merge_concept(
                                concept_id=concept_slug,
                                name=concept.name,
                                definition=concept.definition,
                                concept_type=concept.concept_type.value if hasattr(concept.concept_type, 'value') else str(concept.concept_type),
                                source_documents=[final_doc_id],
                                aliases=concept.aliases,
                            )
                            if internal_id:
                                concept_ids[concept_slug] = internal_id

                        # Store relationships between concepts
                        for relation in relations:
                            from_slug = slugify(relation.from_concept)
                            to_slug = slugify(relation.to_concept)
                            from_id = concept_ids.get(from_slug)
                            to_id = concept_ids.get(to_slug)
                            if from_id and to_id:
                                store_instance.link_concept_relates_to(
                                    from_id, to_id, relation.relation_type
                                )

                        # Create chunk-concept edges (DefinesConcept / MentionsConcept)
                        defines_count = 0
                        mentions_count = 0
                        for chunk_idx, extraction in chunk_extractions.items():
                            # Get chunk internal ID from the stored chunks
                            if chunk_idx < len(embedded_chunks):
                                chunk_str_id = embedded_chunks[chunk_idx].chunk.id
                                chunk_internal_id = chunk_id_map.get(chunk_str_id)

                                if chunk_internal_id:
                                    # Link defined concepts
                                    for defined in extraction.defined_concepts:
                                        concept_slug = slugify(defined.name)
                                        concept_internal_id = concept_ids.get(concept_slug)
                                        if concept_internal_id:
                                            store_instance.link_chunk_defines_concept(
                                                chunk_internal_id, concept_internal_id
                                            )
                                            defines_count += 1

                                    # Link mentioned concepts
                                    for mentioned_name in extraction.mentioned_concepts:
                                        concept_slug = slugify(mentioned_name)
                                        # First check concepts from this ingestion
                                        concept_internal_id = concept_ids.get(concept_slug)
                                        # If not found, check database for concepts from other documents
                                        if not concept_internal_id:
                                            existing = store_instance.get_concept_by_name(mentioned_name)
                                            if existing:
                                                concept_internal_id = existing.get("id")
                                        if concept_internal_id:
                                            store_instance.link_chunk_mentions_concept(
                                                chunk_internal_id, concept_internal_id
                                            )
                                            mentions_count += 1

                        result["concepts"] = len(extracted_concepts)
                        result["defines_edges"] = defines_count
                        result["mentions_edges"] = mentions_count
                        tracker.complete("Extracting concepts...")

                    except Exception as concept_error:
                        tracker.complete("Extracting concepts...")
                        print_warning(f"Concept extraction failed: {concept_error}")

            # Stage 5: Table processing (optional)
            if extract_tables:
                if not TABLE_PROCESSING_AVAILABLE:
                    print_warning(
                        "Table processing requires table modules. "
                        "Check installation."
                    )
                else:
                    tracker.add_stage("Processing tables...", total=100)
                    try:
                        table_summarizer = TableSummarizer(provider=final_concept_provider or "anthropic")
                        table_store = TableStore(embedder=embedder_instance)

                        table_count = 0
                        for i, (chunk, meta) in enumerate(zip(chunks, metadata, strict=False)):
                            if getattr(meta, "has_table", False):
                                # Parse table from chunk content
                                # chunk is a Chunk dataclass, access via attributes
                                chunk_content = chunk.content if hasattr(chunk, 'content') else chunk.get("content", "")
                                chunk_id = chunk.id if hasattr(chunk, 'id') else chunk.get("id", f"chunk_{i}")
                                parsed_table = parse_markdown_table(
                                    chunk_content, chunk_id
                                )
                                if parsed_table:
                                    # Generate summary
                                    summary = table_summarizer.summarize(parsed_table)
                                    # Store table
                                    table_id = table_store.store_table(
                                        parsed_table, summary, final_doc_id
                                    )
                                    table_count += 1

                        result["tables"] = table_count
                        tracker.complete("Processing tables...")

                    except Exception as table_error:
                        tracker.complete("Processing tables...")
                        print_warning(f"Table processing failed: {table_error}")

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
        help="Embedder type: fastembed (local), openai (API), mock",
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip files already in database",
    ),
    extract_concepts: bool = typer.Option(
        False,
        "--extract-concepts",
        help="Extract concepts using LLM (requires instructor)",
    ),
    concept_provider: Optional[str] = typer.Option(
        None,
        "--concept-provider",
        help="LLM provider for concept extraction",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Process multiple files in a directory."""
    config = CLIConfig.load()

    # Resolve concept extraction settings
    do_extract_concepts = extract_concepts or config.extract_concepts
    final_concept_provider = concept_provider or config.concept_provider

    # Initialize extractor if needed
    extractor = None
    if do_extract_concepts:
        if not CONCEPT_EXTRACTION_AVAILABLE:
            print_warning(
                "Concept extraction requires 'instructor' package. "
                "Install with: pip install instructor"
            )
            do_extract_concepts = False
        else:
            extractor = ConceptExtractor(provider=final_concept_provider)

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

                # Extract concepts if enabled
                if do_extract_concepts and extractor:
                    try:
                        from src.chunker.concept_extractor import slugify

                        extracted_concepts, relations = extractor.extract_from_chunks(
                            chunks, title, metadata_list=metadata
                        )

                        # Store or merge concepts (cross-document linking)
                        concept_ids = {}
                        for concept in extracted_concepts:
                            concept_slug = slugify(concept.name)
                            internal_id = store_instance.store_or_merge_concept(
                                concept_id=concept_slug,
                                name=concept.name,
                                definition=concept.definition,
                                concept_type=concept.concept_type.value if hasattr(concept.concept_type, 'value') else str(concept.concept_type),
                                source_documents=[doc_id],
                                aliases=concept.aliases,
                            )
                            if internal_id:
                                concept_ids[concept_slug] = internal_id

                        # Store relationships
                        for relation in relations:
                            from_slug = slugify(relation.from_concept)
                            to_slug = slugify(relation.to_concept)
                            from_id = concept_ids.get(from_slug)
                            to_id = concept_ids.get(to_slug)
                            if from_id and to_id:
                                store_instance.link_concept_relates_to(
                                    from_id, to_id, relation.relation_type
                                )

                        file_result["concepts"] = len(extracted_concepts)
                    except Exception as concept_error:
                        file_result["concept_error"] = str(concept_error)

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
