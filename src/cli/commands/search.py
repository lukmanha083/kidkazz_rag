"""Search commands for querying the knowledge base."""

from typing import Optional

import typer

from ..config import CLIConfig
from ..output import console, print_json, print_search_results, print_warning
from ..progress import spinner
from ..utils import get_embedder, get_reranker, get_store, parse_tags

app = typer.Typer(help="Search commands")


@app.command("semantic")
def search_semantic(
    query: str = typer.Argument(
        ...,
        help="Natural language search query",
    ),
    top_k: int = typer.Option(
        5,
        "-k",
        "--top-k",
        help="Number of results to return",
    ),
    doc_id: Optional[str] = typer.Option(
        None,
        "-d",
        "--doc-id",
        help="Filter to specific document",
    ),
    tags: Optional[str] = typer.Option(
        None,
        "-g",
        "--tags",
        help="Filter by document tags (comma-separated)",
    ),
    level: Optional[int] = typer.Option(
        None,
        "-l",
        "--level",
        help="Filter by hierarchy level (1 or 2)",
    ),
    semantic_type: Optional[str] = typer.Option(
        None,
        "-t",
        "--type",
        help="Filter by semantic type",
    ),
    threshold: float = typer.Option(
        0.0,
        "--threshold",
        help="Minimum similarity score (0.0-1.0)",
    ),
    show_context: bool = typer.Option(
        False,
        "--show-context",
        help="Include surrounding chunks",
    ),
    show_parent: bool = typer.Option(
        False,
        "--show-parent",
        help="Include parent chunk",
    ),
    compact: bool = typer.Option(
        False,
        "--compact",
        help="Show truncated snippets instead of full content",
    ),
    rerank: bool = typer.Option(
        False,
        "--rerank",
        help="Rerank results using Cohere Rerank v3.5 (auto-enabled when [reranker] enabled=true in config)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Search for chunks semantically similar to the query."""
    config = CLIConfig.load()

    # Initialize components
    embedder = get_embedder(config)
    store = get_store(config)

    # Check embedding dimension compatibility
    if hasattr(store, "check_embedding_compatibility"):
        try:
            store.check_embedding_compatibility(
                new_dim=embedder.get_embedding_dim(),
                new_model=getattr(embedder, "model_name", "unknown"),
            )
        except RuntimeError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from None

    # Parse tags
    tag_list = parse_tags(tags)

    # Determine if reranking is active (CLI flag or config)
    use_rerank = rerank or config.reranker_enabled
    reranker = get_reranker(config) if use_rerank else None

    # Generate query embedding
    with spinner("Generating query embedding"):
        if hasattr(embedder, "embed_query"):
            query_embedding = embedder.embed_query(query)
        else:
            query_embedding = embedder.embed_text(query)

    # Fetch more candidates when reranking
    fetch_k = top_k * 4 if reranker else top_k

    # Search
    results = store.search_similar(
        query_embedding=query_embedding,
        top_k=fetch_k,
        doc_id=doc_id,
        level=level,
        semantic_type=semantic_type,
        threshold=threshold,
        tags=tag_list,
    )

    # Rerank if enabled
    if reranker and results:
        with spinner("Reranking results"):
            results = reranker.rerank_chunks(query, results, top_n=top_k)

    # Format results
    formatted = []
    for ec, score in results:
        chunk = ec.chunk
        meta = getattr(chunk, "metadata", {}) or {}
        result = {
            "chunk_id": chunk.id,
            "content": chunk.content,
            "level": chunk.level,
            "semantic_type": meta.get("semantic_type") or getattr(chunk, "semantic_type", "unknown"),
            "similarity_score": round(score, 4),
            "token_count": chunk.token_count,
            "document_id": meta.get("document_id", ""),
            "header_text": meta.get("header_text", ""),
            "header_level": meta.get("header_level"),
            "topic_tags": meta.get("topic_tags", []),
            "has_table": meta.get("has_table", False),
            "has_code": meta.get("has_code", False),
            "has_math": meta.get("has_math", False),
        }

        # Add section path if available
        if hasattr(chunk, "section_path") and chunk.section_path:
            result["section_path"] = chunk.section_path

        # Get context if requested
        if show_context:
            context_chunks = store.get_context_window(chunk.id, window_size=1)
            result["context"] = [c.chunk.content for c in context_chunks if c.chunk.id != chunk.id]

        # Get parent if requested
        if show_parent and chunk.parent_id:
            parent = store.get_parent(chunk.id)
            if parent:
                result["parent"] = parent.chunk.content
                result["parent_id"] = parent.chunk.id

        formatted.append(result)

    if json_output:
        print_json(formatted)
    else:
        if not formatted:
            print_warning(f"No results found for: {query}")
        else:
            console.print(f"\n[bold]Search results for:[/bold] {query}\n")
            print_search_results(formatted, show_context=show_context, compact=compact)


@app.command("keyword")
def search_keyword(
    keyword: str = typer.Argument(
        ...,
        help="Text to search for",
    ),
    doc_id: Optional[str] = typer.Option(
        None,
        "-d",
        "--doc-id",
        help="Filter to specific document",
    ),
    case_sensitive: bool = typer.Option(
        False,
        "--case-sensitive",
        help="Enable case-sensitive matching",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Search for chunks containing a specific keyword or phrase."""
    config = CLIConfig.load()
    store = get_store(config)

    # Search
    results = store.search_keyword(
        keyword=keyword,
        doc_id=doc_id,
        case_sensitive=case_sensitive,
    )

    # Format results
    formatted = [
        {
            "chunk_id": ec.chunk.id,
            "content": ec.chunk.content,
            "level": ec.chunk.level,
            "token_count": ec.chunk.token_count,
        }
        for ec in results
    ]

    if json_output:
        print_json(formatted)
    else:
        if not formatted:
            print_warning(f"No results found for: {keyword}")
        else:
            console.print(f"\n[bold]Found {len(formatted)} results for:[/bold] {keyword}\n")
            for result in formatted[:10]:  # Limit display
                console.print(f"[cyan]{result['chunk_id']}[/cyan] (Level {result['level']})")
                preview = result["content"][:200]
                if len(result["content"]) > 200:
                    preview += "..."
                console.print(f"  {preview}\n")

            if len(formatted) > 10:
                console.print(f"[dim]... and {len(formatted) - 10} more results[/dim]")
