"""CLI commands for document summarization."""

import dataclasses
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# CPU usage threshold — abort edge creation if system CPU exceeds this
CPU_ABORT_THRESHOLD = 75.0


CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "kidkazz")

# All checkpoint phases used by the pipeline
_CHECKPOINT_PHASES = ("llm_output", "summary_progress", "concept_progress", "edge_progress")


def _checkpoint_path(doc_id: str, phase: str) -> str:
    """Path for a phase-specific checkpoint file."""
    return os.path.join(CHECKPOINT_DIR, f"{phase}_{doc_id}.json")


def _load_phase_checkpoint(doc_id: str, phase: str) -> dict:
    """Load checkpoint for a specific phase. Returns {} if none."""
    path = _checkpoint_path(doc_id, phase)
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("doc_id") == doc_id:
            return data
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return {}


def _save_phase_checkpoint(doc_id: str, phase: str, data: dict) -> None:
    """Atomic write of phase checkpoint via tmp+rename."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = _checkpoint_path(doc_id, phase)
    tmp = path + ".tmp"
    data["doc_id"] = doc_id
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)  # atomic on POSIX


def _delete_all_checkpoints(doc_id: str) -> None:
    """Remove all checkpoint files for a document after successful completion."""
    for phase in _CHECKPOINT_PHASES:
        try:
            os.remove(_checkpoint_path(doc_id, phase))
        except FileNotFoundError:
            pass


def _get_cpu_percent(interval: float = 0.5) -> Optional[float]:
    """Get current system-wide CPU usage percentage (0-100).

    Reads /proc/stat twice with a short interval to compute usage.
    Falls back to psutil if /proc/stat is unavailable (non-Linux).
    Returns None if measurement fails entirely.
    """
    # Primary: read /proc/stat (Linux, no extra deps)
    try:
        def _read_cpu_times():
            with open("/proc/stat") as f:
                line = f.readline()  # first line is aggregate "cpu ..."
            parts = line.split()
            # fields: user, nice, system, idle, iowait, irq, softirq, steal
            return [int(x) for x in parts[1:]]

        t1 = _read_cpu_times()
        time.sleep(interval)
        t2 = _read_cpu_times()

        delta_idle = (t2[3] + t2[4]) - (t1[3] + t1[4])  # idle + iowait
        delta_total = sum(t2) - sum(t1)
        if delta_total == 0:
            return 0.0
        return (1.0 - delta_idle / delta_total) * 100.0
    except (FileNotFoundError, IndexError, ValueError, OSError):
        pass

    # Fallback: psutil (cross-platform)
    try:
        import psutil
        return psutil.cpu_percent(interval=interval)
    except (ImportError, Exception):
        return None

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from src.cli.output import _html_tables_to_markdown

from src.cli.config import CLIConfig
from src.cli.output import print_error, print_json, print_success, print_warning

# Helix-DB error types for retry logic
try:
    from helix.client import HelixRequestError, HelixConnectionError
except ImportError:
    HelixRequestError = Exception  # type: ignore
    HelixConnectionError = Exception  # type: ignore

# Optional import for summarizer
try:
    from src.chunker.summarizer import (
        Concept,
        DocumentSummarizer,
        Summary,
        SummarizationStrategy,
        INSTRUCTOR_AVAILABLE,
        EXTRACTIVE_AVAILABLE,
        ASYNC_THRESHOLD,
        BATCH_THRESHOLD,
    )

    SUMMARIZER_AVAILABLE = INSTRUCTOR_AVAILABLE
except ImportError:
    Concept = None  # type: ignore
    DocumentSummarizer = None  # type: ignore
    Summary = None  # type: ignore
    SummarizationStrategy = None  # type: ignore
    SUMMARIZER_AVAILABLE = False
    EXTRACTIVE_AVAILABLE = False
    ASYNC_THRESHOLD = 50
    BATCH_THRESHOLD = 500

app = typer.Typer(help="Document summarization commands")
console = Console()


def _parse_key_points(value) -> list:
    """Safely parse key_points which may be a JSON string or already a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _extract_chapter_title_from_summary(summary: dict) -> str | None:
    """Extract chapter title from summary's key_points __title__: sentinel."""
    key_points = _parse_key_points(summary.get("key_points"))
    for kp in key_points:
        if isinstance(kp, str) and kp.startswith("__title__:"):
            return kp[len("__title__:"):]
    return None


def _get_display_key_points_from_summary(summary: dict) -> list[str]:
    """Get key_points with the title sentinel filtered out."""
    key_points = _parse_key_points(summary.get("key_points"))
    return [kp for kp in key_points if not (isinstance(kp, str) and kp.startswith("__title__:"))]


def _chapter_sort_key(summary: dict) -> tuple[int, str]:
    """Sort key for ordering chapters by source_id numeric suffix.

    Extracts the trailing number from source_id (e.g., 'inv_l1_5' -> 5).
    Falls back to alphabetical sorting if no number found.
    """
    source_id = summary.get("source_id", "")
    import re
    match = re.search(r"(\d+)$", source_id)
    if match:
        return (int(match.group(1)), source_id)
    return (999999, source_id)


def _get_store(config: CLIConfig):
    """Get storage instance."""
    from src.cli.commands.ingest import get_store
    return get_store(config)


def _get_embedder(config: CLIConfig):
    """Get embedder instance."""
    from src.cli.commands.ingest import get_embedder
    return get_embedder(config)


def _collect_edge_pairs(
    concept,
    concept_internal_id: str,
    chunk_id_cache: dict[str, str],
    edge_queue: list[tuple[str, str]],
) -> None:
    """Collect (chunk_internal_id, concept_internal_id) edge pairs from concept occurrences."""
    seen_chunk_ids: set[str] = set()
    for occ in concept.occurrences:
        # Extract chunk_id from summary_id format: "summary_{chunk_id}_{level}"
        # e.g. "summary_inv_l1_5_chapter" -> chunk_id = "inv_l1_5"
        if occ.startswith("summary_"):
            rest = occ[len("summary_"):]
            parts = rest.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ("section", "chapter", "document"):
                source_chunk_id = parts[0]
            else:
                source_chunk_id = rest
        else:
            continue

        if source_chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(source_chunk_id)

        cid = chunk_id_cache.get(source_chunk_id)
        if cid:
            edge_queue.append((cid, concept_internal_id))


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

    # Pin entire summarize command to 2 CPU cores to prevent system freeze.
    # Covers: extractive summarization (TextRank+YAKE), LLM calls,
    # summary storage, and concept storage.
    try:
        cpu_count = os.cpu_count() or 4
        if cpu_count > 2:
            os.sched_setaffinity(0, {0, 1})
            actual = os.sched_getaffinity(0)
            logger.info("CPU affinity set to cores %s (of %d total)", actual, cpu_count)
        else:
            logger.info("Only %d cores available, skipping affinity pin", cpu_count)
    except (OSError, AttributeError) as e:
        logger.warning("Could not set CPU affinity: %s", e)
    try:
        nice_val = os.nice(10)
        logger.info("Process nice level set to %d", nice_val)
    except OSError as e:
        logger.warning("Could not set nice level: %s", e)

    config = CLIConfig.load()
    store = _get_store(config)

    # Check if document exists
    docs = store.list_documents()
    doc = next((d for d in docs if d.get("doc_id") == doc_id), None)
    if not doc:
        print_error(f"Document not found: {doc_id}")
        raise typer.Exit(1)

    doc_title = doc.get("title", doc_id)

    # Check for existing summaries
    existing = store.get_document_summaries(doc_id)
    if existing and not force:
        if json_output:
            print_json({"status": "exists", "count": len(existing)})
        else:
            print_warning(
                f"Document already has {len(existing)} summaries. "
                "Use --force to regenerate."
            )
        return

    # On --force, delete old summaries (and their SummaryVector nodes) first
    if force and existing:
        if not json_output:
            console.print(f"[yellow]Deleting {len(existing)} existing summaries...[/yellow]")
        logger.info("=== PHASE: Deleting %d existing summaries for %s ===", len(existing), doc_id)
        phase_start_del = time.monotonic()
        deleted = store.delete_document_summaries(doc_id)
        elapsed_del = time.monotonic() - phase_start_del
        logger.info("=== PHASE: Deleted %d summaries in %.1fs ===", deleted, elapsed_del)
        if not json_output:
            console.print(f"  Deleted {deleted} summaries in {elapsed_del:.1f}s")
        # Brief cooldown to let Helix-DB settle
        time.sleep(3)

    # Get chunks for document
    chunks_result = store.get_document_chunks(doc_id)
    if not chunks_result:
        print_error(f"No chunks found for document: {doc_id}")
        raise typer.Exit(1)

    # Convert to dict format expected by summarizer
    chunks = []
    for ec in chunks_result:
        chunk = ec.chunk if hasattr(ec, 'chunk') else ec
        # Get metadata (may be nested in chunk.metadata or directly on chunk)
        metadata = chunk.metadata if hasattr(chunk, 'metadata') else chunk.get("metadata", {})
        chunk_dict = {
            "id": chunk.id if hasattr(chunk, 'id') else chunk.get("chunk_id"),
            "content": chunk.content if hasattr(chunk, 'content') else chunk.get("content"),
            "level": chunk.level if hasattr(chunk, 'level') else chunk.get("level"),
            "section_path": chunk.section_path if hasattr(chunk, 'section_path') else chunk.get("section_path", []),
            "source_section": chunk.source_section if hasattr(chunk, 'source_section') else chunk.get("source_section"),
            "child_ids": chunk.child_ids if hasattr(chunk, 'child_ids') else chunk.get("child_ids", []),
            # Include content flags for better summarization
            "has_table": metadata.get("has_table", False) if isinstance(metadata, dict) else getattr(metadata, "has_table", False),
            "has_code": metadata.get("has_code", False) if isinstance(metadata, dict) else getattr(metadata, "has_code", False),
            "has_math": metadata.get("has_math", False) if isinstance(metadata, dict) else getattr(metadata, "has_math", False),
        }
        chunks.append(chunk_dict)

    # Initialize summarizer
    final_provider = provider or config.summarization_provider or "openai/gpt-4o-mini"
    summarizer = DocumentSummarizer(provider=final_provider)

    # Count chunks by level
    l1_count = len([c for c in chunks if c.get("level") == 1])
    l2_count = len([c for c in chunks if c.get("level") == 2])
    api_calls = l1_count + 1  # chapters + document

    # Determine strategy based on L1 count (sections are extractive)
    strategy = summarizer.determine_strategy(l1_count)

    if not json_output:
        console.print(f"\nGenerating summaries for: [bold]{doc_title}[/bold]")
        console.print(f"Found {len(chunks)} chunks ({l2_count} sections, {l1_count} chapters)")
        console.print(f"Provider: [cyan]{final_provider}[/cyan]")

        # Show tiered strategy info
        console.print(f"  Sections: {l2_count} [green](extractive, instant)[/green]")
        console.print(f"  Chapters: {l1_count} [yellow](LLM API calls)[/yellow]")
        console.print(f"  Document: 1 [yellow](LLM API call)[/yellow]")
        console.print(f"  Total API calls: ~{api_calls}")

        if not EXTRACTIVE_AVAILABLE:
            console.print(
                "[yellow]Warning: sumy/yake not installed. "
                "Install with: pip install -e '.[extractive]'[/yellow]"
            )

        strategy_info = {
            "sequential": f"Sequential (< {ASYNC_THRESHOLD} chapters)",
            "async": f"Async with rate limiting ({ASYNC_THRESHOLD}-{BATCH_THRESHOLD} chapters)",
            "batch": f"OpenAI Batch API (> {BATCH_THRESHOLD} chapters, 50% cheaper)",
        }
        console.print(f"Strategy: [green]{strategy_info.get(strategy.value, strategy.value)}[/green]")
        console.print()

    # Progress callback for CLI
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    # =========================================================================
    # Phase 1: LLM Generation (with checkpoint — skip entirely on re-run)
    # =========================================================================
    # On --force, delete LLM checkpoint so we regenerate fresh
    if force:
        try:
            os.remove(_checkpoint_path(doc_id, "llm_output"))
        except FileNotFoundError:
            pass

    llm_ckpt = _load_phase_checkpoint(doc_id, "llm_output")
    if llm_ckpt and not force:
        # Resume: deserialize summaries + concepts from checkpoint
        logger.info("=== PHASE 1: Loading LLM output from checkpoint ===")
        if not json_output:
            console.print("[cyan]Resuming from LLM checkpoint (skipping LLM calls)[/cyan]")
        summaries = [summarizer.dict_to_summary(d) for d in llm_ckpt.get("summaries", [])]
        concepts = [Concept(**d) for d in llm_ckpt.get("concepts", [])]
        logger.info("  Loaded %d summaries, %d concepts from checkpoint", len(summaries), len(concepts))
    else:
        # Generate fresh
        logger.info("=== PHASE 1: Summarization START (sections=%d, chapters=%d) ===", l2_count, l1_count)
        phase_start = time.monotonic()
        try:
            if json_output:
                summaries, concepts = summarizer.generate_all_summaries(
                    document_id=doc_id,
                    document_title=doc_title,
                    chunks=chunks,
                    strategy=strategy,
                )
            else:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task("Generating summaries...", total=100)

                    def progress_callback(current: int, total: int, message: str):
                        pct = int((current / max(total, 1)) * 100)
                        progress.update(task, completed=pct, description=message)

                    summaries, concepts = summarizer.generate_all_summaries(
                        document_id=doc_id,
                        document_title=doc_title,
                        chunks=chunks,
                        strategy=strategy,
                        progress_callback=progress_callback,
                    )
        except Exception as e:
            print_error(f"Summarization failed: {e}")
            raise typer.Exit(1) from e

        elapsed = time.monotonic() - phase_start
        logger.info("=== PHASE 1: Summarization DONE in %.1fs (%d summaries, %d concepts) ===",
                    elapsed, len(summaries), len(concepts))

        # Save LLM output checkpoint (embeddings NOT stored — regenerated in Phase 2)
        _save_phase_checkpoint(doc_id, "llm_output", {
            "summaries": [summarizer.summary_to_dict(s) for s in summaries],
            "concepts": [dataclasses.asdict(c) for c in concepts],
        })
        logger.info("  LLM output checkpoint saved")

    # =========================================================================
    # Pre-cache IDs for Phase 2-4 (2 HTTP calls total, eliminates ~962 lookups)
    # =========================================================================
    embedder = _get_embedder(config)

    # Pre-build chunk_id -> internal_id cache (1 HTTP call instead of thousands)
    chunk_id_cache: dict[str, str] = {}
    if hasattr(store, 'build_chunk_id_cache'):
        logger.info("=== Building chunk ID cache ===")
        chunk_id_cache = store.build_chunk_id_cache(doc_id)
        logger.info("  Cached %d chunk ID mappings", len(chunk_id_cache))

    # Pre-cache document internal ID (1 HTTP call, used for document-level summary link)
    doc_internal_id: str | None = None
    if hasattr(store, '_execute_query'):
        from src.storage.queries import GetDocumentByDocId
        doc_query = GetDocumentByDocId(doc_id)
        doc_result = store._execute_query(doc_query)
        if doc_result.success and doc_result.data:
            doc_node = doc_result.data.get("node", doc_result.data)
            doc_internal_id = store._extract_node_id(doc_result.data) or (doc_node.get("id") if doc_node else None)
        logger.info("  Document internal ID: %s", doc_internal_id)

    # =========================================================================
    # Phase 2: Summary Storage (with embeddings + checkpoint)
    # =========================================================================
    summary_ckpt = _load_phase_checkpoint(doc_id, "summary_progress")
    stored_count = summary_ckpt.get("stored_count", 0)

    if stored_count > 0 and stored_count < len(summaries):
        logger.info("=== PHASE 2: Resuming summary storage from index %d/%d ===", stored_count, len(summaries))
        if not json_output:
            console.print(f"[cyan]Resuming summary storage from {stored_count}/{len(summaries)}[/cyan]")
    elif stored_count >= len(summaries):
        logger.info("=== PHASE 2: Summary storage already complete (%d/%d) ===", stored_count, len(summaries))
        if not json_output:
            console.print(f"[cyan]Summary storage already complete ({stored_count} stored)[/cyan]")
    else:
        logger.info("=== PHASE 2: Summary storage START (%d summaries) ===", len(summaries))

    phase_start = time.monotonic()

    SUMMARY_SUBPHASE_SIZE = 50       # summaries per subphase (~200 HTTP calls)
    SUMMARY_THROTTLE = 0.25          # seconds between each store call
    SUMMARY_MAX_RETRIES = 3          # retry transient errors
    SUBPHASE_COOLDOWN = 3            # seconds between subphases (keeps connection alive <5s)

    # Split remaining summaries into subphases
    remaining_indices = list(range(stored_count, len(summaries)))
    summary_subphases = [
        remaining_indices[i:i + SUMMARY_SUBPHASE_SIZE]
        for i in range(0, len(remaining_indices), SUMMARY_SUBPHASE_SIZE)
    ]

    logger.info("  %d summaries remaining, %d subphases of %d",
                len(remaining_indices), len(summary_subphases), SUMMARY_SUBPHASE_SIZE)

    consecutive_sp_failures = 0
    cpu_abort_p2 = False

    for sp_idx, subphase_indices in enumerate(summary_subphases):
        # --- Abort check: 3 consecutive subphase failures ---
        if consecutive_sp_failures >= 3:
            logger.error("  3 consecutive subphase failures — aborting summary storage")
            if not json_output:
                print_warning(
                    f"Aborted summary storage after 3 consecutive subphase failures "
                    f"({stored_count}/{len(summaries)} stored)"
                )
            break

        # --- CPU safety check before each subphase ---
        cpu_pct = _get_cpu_percent(interval=0.5)
        if cpu_pct is not None:
            logger.info("  CPU check before subphase %d/%d: %.1f%%",
                        sp_idx + 1, len(summary_subphases), cpu_pct)
            if cpu_pct > CPU_ABORT_THRESHOLD:
                logger.warning("  CPU %.1f%% > %.0f%% — waiting 10s for cooldown...",
                               cpu_pct, CPU_ABORT_THRESHOLD)
                if not json_output:
                    console.print(f"  [yellow]CPU {cpu_pct:.0f}% — pausing 10s to cool down...[/yellow]")
                time.sleep(10)
                cpu_pct = _get_cpu_percent(interval=1.0)
                if cpu_pct is not None and cpu_pct > CPU_ABORT_THRESHOLD:
                    logger.error("  CPU still %.1f%% — aborting summary storage (%d/%d stored)",
                                 cpu_pct, stored_count, len(summaries))
                    if not json_output:
                        print_warning(
                            f"CPU {cpu_pct:.0f}% still above threshold — aborting summary storage "
                            f"({stored_count}/{len(summaries)} stored). Re-run to resume."
                        )
                    cpu_abort_p2 = True
                    _save_phase_checkpoint(doc_id, "summary_progress", {"stored_count": stored_count})
                    break
                else:
                    logger.info("  CPU cooled to %.1f%% — resuming", cpu_pct or 0)

        # --- Process items in this subphase ---
        subphase_ok = True
        for i in subphase_indices:
            summary = summaries[i]
            try:
                embedding = embedder.embed_text(summary.content)
                summary.embedding = embedding

                cached_chunk_id = chunk_id_cache.get(summary.source_id) if summary.level in ("chapter", "section") else None
                cached_doc_id = doc_internal_id if summary.level == "document" else None

                last_err = None
                for attempt in range(1, SUMMARY_MAX_RETRIES + 1):
                    try:
                        store.store_summary(
                            summary,
                            doc_internal_id=cached_doc_id,
                            chunk_internal_id=cached_chunk_id,
                        )
                        last_err = None
                        break
                    except (ConnectionError, TimeoutError, OSError,
                            HelixRequestError, HelixConnectionError) as e:
                        last_err = e
                        backoff = attempt * 2
                        logger.warning(
                            "  Transient error storing %s (attempt %d/%d), retrying in %ds: %s",
                            summary.summary_id, attempt, SUMMARY_MAX_RETRIES, backoff, e,
                        )
                        time.sleep(backoff)

                if last_err is not None:
                    raise last_err

                stored_count = i + 1
                time.sleep(SUMMARY_THROTTLE)

            except Exception as e:
                subphase_ok = False
                logger.error("Failed to store summary %s: %s", summary.summary_id, e)
                if not json_output:
                    print_warning(f"Failed to store summary {summary.summary_id}: {e}")
                break  # stop subphase to keep stored_count contiguous for resume

        # --- Track success/failure ---
        if subphase_ok:
            consecutive_sp_failures = 0
        else:
            consecutive_sp_failures += 1

        # --- Checkpoint after each subphase ---
        _save_phase_checkpoint(doc_id, "summary_progress", {"stored_count": stored_count})

        # --- Log progress ---
        elapsed_so_far = time.monotonic() - phase_start
        rate = stored_count / elapsed_so_far if elapsed_so_far > 0 else 0
        logger.info(
            "  Subphase %d/%d done: %d summaries this batch, %d/%d total (%.1f/s, %.1fs) [checkpoint saved]",
            sp_idx + 1, len(summary_subphases), len(subphase_indices),
            stored_count, len(summaries), rate, elapsed_so_far,
        )
        if not json_output:
            console.print(
                f"  Summary subphase {sp_idx + 1}/{len(summary_subphases)} "
                f"({stored_count}/{len(summaries)} stored) [checkpoint saved]"
            )

        # --- Cooldown between subphases (keeps HTTP connection alive <5s) ---
        if sp_idx < len(summary_subphases) - 1:
            logger.info("  Cooling down %ds before next subphase...", SUBPHASE_COOLDOWN)
            time.sleep(SUBPHASE_COOLDOWN)

    elapsed = time.monotonic() - phase_start
    logger.info("=== PHASE 2: Summary storage DONE in %.1fs (%d/%d stored) ===",
                elapsed, stored_count, len(summaries))

    # Cooldown between summary storage and concept storage
    if concepts:
        logger.info("=== Cooldown 3s before concept storage ===")
        if not json_output:
            console.print("  [dim]Cooling down 3s before concept storage...[/dim]")
        time.sleep(3)

    # =========================================================================
    # Phase 3: Concept Storage (with checkpoint)
    # =========================================================================
    concept_ckpt = _load_phase_checkpoint(doc_id, "concept_progress")
    completed_concept_names: set[str] = set(concept_ckpt.get("completed_names", []))
    # concept_internal_ids maps concept name -> internal node ID (needed by Phase 4)
    concept_internal_ids: dict[str, str] = dict(concept_ckpt.get("concept_internal_ids", {}))

    stored_concepts = len(completed_concept_names)
    edge_queue: list[tuple[str, str]] = []  # (chunk_internal_id, concept_internal_id)
    concept_id_to_name: dict[str, str] = {}  # internal_id -> concept name (for edge checkpoint)

    if stored_concepts > 0 and stored_concepts < len(concepts):
        logger.info("=== PHASE 3: Resuming concept storage from %d/%d ===", stored_concepts, len(concepts))
        if not json_output:
            console.print(f"[cyan]Resuming concept storage ({stored_concepts}/{len(concepts)} already done)[/cyan]")
        # Rebuild edge queue from already-stored concepts
        for concept in concepts:
            if concept.name in completed_concept_names:
                cint_id = concept_internal_ids.get(concept.name)
                if cint_id:
                    concept_id_to_name[cint_id] = concept.name
                    _collect_edge_pairs(concept, cint_id, chunk_id_cache, edge_queue)
    elif stored_concepts >= len(concepts):
        logger.info("=== PHASE 3: Concept storage already complete (%d/%d) ===", stored_concepts, len(concepts))
        if not json_output:
            console.print(f"[cyan]Concept storage already complete ({stored_concepts} stored)[/cyan]")
        # Rebuild full edge queue + concept_id_to_name from checkpoint
        for concept in concepts:
            cint_id = concept_internal_ids.get(concept.name)
            if cint_id:
                concept_id_to_name[cint_id] = concept.name
                _collect_edge_pairs(concept, cint_id, chunk_id_cache, edge_queue)
    else:
        logger.info("=== PHASE 3: Concept storage START (%d concepts) ===", len(concepts))

    phase_start = time.monotonic()

    CONCEPT_SUBPHASE_SIZE = 100      # concepts per subphase (~200-300 HTTP calls)
    CONCEPT_THROTTLE = 0.30          # seconds between each store call
    CONCEPT_MAX_RETRIES = 3          # retry transient errors

    # Pre-filter to pending concepts only, then split into subphases
    pending_concepts = [c for c in concepts if c.name not in completed_concept_names]
    concept_subphases = [
        pending_concepts[i:i + CONCEPT_SUBPHASE_SIZE]
        for i in range(0, len(pending_concepts), CONCEPT_SUBPHASE_SIZE)
    ]

    logger.info("  %d concepts pending, %d subphases of %d",
                len(pending_concepts), len(concept_subphases), CONCEPT_SUBPHASE_SIZE)

    consecutive_sp_failures_p3 = 0
    cpu_abort_p3 = False

    for sp_idx, subphase_concepts in enumerate(concept_subphases):
        # --- Abort check: 3 consecutive subphase failures ---
        if consecutive_sp_failures_p3 >= 3:
            logger.error("  3 consecutive subphase failures — aborting concept storage")
            if not json_output:
                print_warning(
                    f"Aborted concept storage after 3 consecutive subphase failures "
                    f"({stored_concepts}/{len(concepts)} stored)"
                )
            break

        # --- CPU safety check before each subphase ---
        cpu_pct = _get_cpu_percent(interval=0.5)
        if cpu_pct is not None:
            logger.info("  CPU check before subphase %d/%d: %.1f%%",
                        sp_idx + 1, len(concept_subphases), cpu_pct)
            if cpu_pct > CPU_ABORT_THRESHOLD:
                logger.warning("  CPU %.1f%% > %.0f%% — waiting 10s for cooldown...",
                               cpu_pct, CPU_ABORT_THRESHOLD)
                if not json_output:
                    console.print(f"  [yellow]CPU {cpu_pct:.0f}% — pausing 10s to cool down...[/yellow]")
                time.sleep(10)
                cpu_pct = _get_cpu_percent(interval=1.0)
                if cpu_pct is not None and cpu_pct > CPU_ABORT_THRESHOLD:
                    logger.error("  CPU still %.1f%% — aborting concept storage (%d/%d stored)",
                                 cpu_pct, stored_concepts, len(concepts))
                    if not json_output:
                        print_warning(
                            f"CPU {cpu_pct:.0f}% still above threshold — aborting concept storage "
                            f"({stored_concepts}/{len(concepts)} stored). Re-run to resume."
                        )
                    cpu_abort_p3 = True
                    _save_phase_checkpoint(doc_id, "concept_progress", {
                        "completed_names": sorted(completed_concept_names),
                        "concept_internal_ids": concept_internal_ids,
                    })
                    break
                else:
                    logger.info("  CPU cooled to %.1f%% — resuming", cpu_pct or 0)

        # --- Process items in this subphase ---
        subphase_ok = True
        for concept in subphase_concepts:
            try:
                concept_internal_id = None
                last_err = None
                for attempt in range(1, CONCEPT_MAX_RETRIES + 1):
                    try:
                        concept_internal_id = store.store_or_merge_concept(
                            concept_id=concept.concept_id,
                            name=concept.name,
                            definition=concept.definition,
                            concept_type=concept.concept_type,
                            source_documents=[concept.document_id] if concept.document_id else [doc_id],
                            aliases=concept.aliases,
                        )
                        last_err = None
                        break
                    except (ConnectionError, TimeoutError, OSError,
                            HelixRequestError, HelixConnectionError) as e:
                        last_err = e
                        backoff = attempt * 2
                        logger.warning(
                            "  Transient error storing concept %s (attempt %d/%d), retrying in %ds: %s",
                            concept.name, attempt, CONCEPT_MAX_RETRIES, backoff, e,
                        )
                        time.sleep(backoff)

                if last_err is not None:
                    raise last_err

                stored_concepts += 1
                completed_concept_names.add(concept.name)

                # Collect edge pairs for Phase 4 (no HTTP calls here)
                if concept_internal_id:
                    concept_internal_ids[concept.name] = concept_internal_id
                    concept_id_to_name[concept_internal_id] = concept.name
                    _collect_edge_pairs(concept, concept_internal_id, chunk_id_cache, edge_queue)

                time.sleep(CONCEPT_THROTTLE)

            except Exception as e:
                subphase_ok = False
                logger.error("Failed to store concept %s: %s", concept.name, e)
                if not json_output:
                    print_warning(f"Failed to store concept {concept.name}: {e}")

        # --- Track success/failure ---
        if subphase_ok:
            consecutive_sp_failures_p3 = 0
        else:
            consecutive_sp_failures_p3 += 1

        # --- Checkpoint after each subphase ---
        _save_phase_checkpoint(doc_id, "concept_progress", {
            "completed_names": sorted(completed_concept_names),
            "concept_internal_ids": concept_internal_ids,
        })

        # --- Log progress ---
        elapsed_so_far = time.monotonic() - phase_start
        rate = stored_concepts / elapsed_so_far if elapsed_so_far > 0 else 0
        logger.info(
            "  Subphase %d/%d done: %d concepts this batch, %d/%d total (%.1f/s, %.1fs) [checkpoint saved]",
            sp_idx + 1, len(concept_subphases), len(subphase_concepts),
            stored_concepts, len(concepts), rate, elapsed_so_far,
        )
        if not json_output:
            console.print(
                f"  Concept subphase {sp_idx + 1}/{len(concept_subphases)} "
                f"({stored_concepts}/{len(concepts)} stored) [checkpoint saved]"
            )

        # --- Cooldown between subphases ---
        if sp_idx < len(concept_subphases) - 1:
            logger.info("  Cooling down %ds before next subphase...", SUBPHASE_COOLDOWN)
            time.sleep(SUBPHASE_COOLDOWN)

    elapsed = time.monotonic() - phase_start
    logger.info("=== PHASE 3: Concept storage DONE in %.1fs (%d/%d stored, %d edges queued) ===",
                elapsed, stored_concepts, len(concepts), len(edge_queue))

    # Cooldown before edge creation (reduced from 15s to 3s — all pauses <=3s)
    logger.info("=== Cooldown 3s before edge creation ===")
    if not json_output:
        console.print("  [dim]Cooling down 3s before edge creation...[/dim]")
    time.sleep(3)

    # =========================================================================
    # Phase 4: Edge Creation (DefinesConcept edges, with checkpoint)
    # =========================================================================
    # --- Drop existing DefinesConcept edges when --force ---
    if force and chunk_id_cache and hasattr(store, 'batch_drop_chunk_concept_edges'):
        internal_ids = list(chunk_id_cache.values())
        logger.info("=== PHASE 4: Dropping existing DefinesConcept edges (%d chunks) ===", len(internal_ids))
        if not json_output:
            console.print(f"  [dim]Dropping existing DefinesConcept edges from {len(internal_ids)} chunks...[/dim]")
        try:
            store.batch_drop_chunk_concept_edges(internal_ids)
            logger.info("  DefinesConcept edges dropped successfully")
        except Exception as e:
            logger.warning("  Failed to drop DefinesConcept edges: %s (continuing anyway)", e)
        time.sleep(3)

    # Group edges by concept_id so we can throttle by concept count.
    from collections import defaultdict

    CONCEPT_BATCH_SIZE = 50   # concepts per phase
    EDGE_BATCH_SIZE = 50      # edges per HTTP request (HelixQL FOR loop limit)
    PHASE_COOLDOWN = 3        # seconds between concept phases (reduced from 5s)

    edges_by_concept: dict[str, list[str]] = defaultdict(list)
    for chunk_id, concept_id in edge_queue:
        edges_by_concept[concept_id].append(chunk_id)

    # Load edge checkpoint — skip concepts whose edges were already created
    edge_ckpt = _load_phase_checkpoint(doc_id, "edge_progress")
    completed_edge_concepts: set[str] = set(edge_ckpt.get("completed_concepts", []))
    if completed_edge_concepts:
        before = len(edges_by_concept)
        skipped_edges = 0
        for cid in list(edges_by_concept.keys()):
            cname = concept_id_to_name.get(cid, "")
            if cname in completed_edge_concepts:
                skipped_edges += len(edges_by_concept.pop(cid))
        after = len(edges_by_concept)
        logger.info(
            "  Edge checkpoint loaded: %d concepts already done, skipping %d edges "
            "(%d → %d concepts remaining)",
            before - after, skipped_edges, before, after,
        )
        if not json_output:
            console.print(
                f"  [cyan]Resuming edge creation: {before - after} concepts already done, "
                f"{after} remaining[/cyan]"
            )

    concept_ids = list(edges_by_concept.keys())
    total_remaining_edges = sum(len(v) for v in edges_by_concept.values())
    concept_phases = [
        concept_ids[i:i + CONCEPT_BATCH_SIZE]
        for i in range(0, len(concept_ids), CONCEPT_BATCH_SIZE)
    ]

    logger.info(
        "=== PHASE 4: Edge creation START (%d edges, %d concepts, %d phases of %d concepts) ===",
        total_remaining_edges, len(concept_ids), len(concept_phases), CONCEPT_BATCH_SIZE,
    )
    phase_start = time.monotonic()
    edges_created = 0
    consecutive_phase_failures = 0
    cpu_abort = False

    for phase_idx, concept_batch in enumerate(concept_phases):
        if consecutive_phase_failures >= 3:
            logger.error("  3 consecutive subphase failures — aborting edge creation")
            if not json_output:
                print_warning(
                    f"Aborted edge creation after 3 consecutive subphase failures "
                    f"({edges_created}/{total_remaining_edges} edges created)"
                )
            break

        # --- CPU safety check before each concept phase ---
        cpu_pct = _get_cpu_percent(interval=0.5)
        if cpu_pct is not None:
            logger.info("  CPU check before subphase %d: %.1f%%", phase_idx + 1, cpu_pct)
            if cpu_pct > CPU_ABORT_THRESHOLD:
                logger.warning(
                    "  CPU %.1f%% > %.0f%% threshold — waiting 10s for cooldown...",
                    cpu_pct, CPU_ABORT_THRESHOLD,
                )
                if not json_output:
                    console.print(
                        f"  [yellow]CPU {cpu_pct:.0f}% — pausing 10s to cool down...[/yellow]"
                    )
                time.sleep(10)
                cpu_pct = _get_cpu_percent(interval=1.0)
                if cpu_pct is not None and cpu_pct > CPU_ABORT_THRESHOLD:
                    logger.error(
                        "  CPU still %.1f%% after cooldown — aborting edge creation "
                        "to prevent system freeze (%d/%d edges created)",
                        cpu_pct, edges_created, total_remaining_edges,
                    )
                    if not json_output:
                        print_warning(
                            f"CPU usage {cpu_pct:.0f}% still above {CPU_ABORT_THRESHOLD:.0f}% "
                            f"after cooldown — aborting edge creation "
                            f"({edges_created}/{total_remaining_edges} edges created). "
                            f"Run the same command again to resume."
                        )
                    cpu_abort = True
                    _save_phase_checkpoint(doc_id, "edge_progress", {
                        "completed_concepts": sorted(completed_edge_concepts),
                        "edges_created": edges_created,
                        "total_edges": total_remaining_edges,
                    })
                    break
                else:
                    logger.info("  CPU cooled to %.1f%% — resuming", cpu_pct or 0)

        # Collect all edges for this concept batch
        phase_edges = []
        for cid in concept_batch:
            for chunk_id in edges_by_concept[cid]:
                phase_edges.append((chunk_id, cid))

        # Sub-batch edges into HTTP requests (50 edges each)
        edge_sub_batches = [
            phase_edges[i:i + EDGE_BATCH_SIZE]
            for i in range(0, len(phase_edges), EDGE_BATCH_SIZE)
        ]

        phase_ok = True
        for sub_idx, sub_batch in enumerate(edge_sub_batches):
            success = False
            for attempt in range(4):
                try:
                    if hasattr(store, 'batch_link_chunk_defines_concept'):
                        count = store.batch_link_chunk_defines_concept(sub_batch)
                    else:
                        count = sum(1 for c, p in sub_batch if store.link_chunk_defines_concept(c, p))
                    edges_created += count
                    success = True
                    break
                except ConnectionRefusedError:
                    delay = 5 * (2 ** attempt)
                    logger.warning(
                        "  Subphase %d/%d sub-batch %d: connection refused, retry %d in %ds",
                        phase_idx + 1, len(concept_phases), sub_idx + 1, attempt + 1, delay,
                    )
                    time.sleep(delay)
                except (ConnectionResetError, OSError) as e:
                    if attempt < 3:
                        delay = 2 * (2 ** attempt)
                        logger.warning(
                            "  Subphase %d/%d sub-batch %d: %s, retry %d in %ds",
                            phase_idx + 1, len(concept_phases), sub_idx + 1,
                            type(e).__name__, attempt + 1, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "  Subphase %d/%d sub-batch %d: %s after %d retries",
                            phase_idx + 1, len(concept_phases), sub_idx + 1, e, attempt + 1,
                        )
                except Exception as e:
                    logger.error(
                        "  Subphase %d/%d sub-batch %d: unexpected %s: %s (no retry)",
                        phase_idx + 1, len(concept_phases), sub_idx + 1,
                        type(e).__name__, e,
                    )
                    break

            if not success:
                phase_ok = False

            # Pause between sub-batches to avoid overwhelming Helix-DB
            time.sleep(0.5)

        if phase_ok:
            consecutive_phase_failures = 0
            for cid in concept_batch:
                cname = concept_id_to_name.get(cid, "")
                if cname:
                    completed_edge_concepts.add(cname)
        else:
            consecutive_phase_failures += 1

        # Save edge checkpoint after every phase
        _save_phase_checkpoint(doc_id, "edge_progress", {
            "completed_concepts": sorted(completed_edge_concepts),
            "edges_created": edges_created,
            "total_edges": total_remaining_edges,
        })

        # Log progress after each concept phase
        elapsed_so_far = time.monotonic() - phase_start
        logger.info(
            "  Subphase %d/%d done: %d concepts, %d edges this batch, "
            "%d/%d total edges (%.1fs elapsed) [checkpoint saved]",
            phase_idx + 1, len(concept_phases), len(concept_batch),
            len(phase_edges), edges_created, total_remaining_edges, elapsed_so_far,
        )

        if not json_output:
            console.print(
                f"  Edge subphase {phase_idx + 1}/{len(concept_phases)} "
                f"({edges_created}/{total_remaining_edges} edges) [checkpoint saved]"
            )

        # Cooldown between concept phases
        if phase_idx < len(concept_phases) - 1:
            logger.info("  Cooling down %ds before next subphase...", PHASE_COOLDOWN)
            time.sleep(PHASE_COOLDOWN)

    elapsed = time.monotonic() - phase_start

    # =========================================================================
    # Phase 5: Cleanup — delete all checkpoints on full success
    # =========================================================================
    all_done = (not cpu_abort_p2 and not cpu_abort_p3 and not cpu_abort
                and stored_count >= len(summaries)
                and stored_concepts >= len(concepts)
                and consecutive_phase_failures < 3
                and edges_created >= total_remaining_edges)
    if all_done:
        _delete_all_checkpoints(doc_id)
        logger.info("=== PHASE 4: Edge creation DONE in %.1fs (%d/%d created) ===",
                     elapsed, edges_created, total_remaining_edges)
        logger.info("=== PHASE 5: All checkpoints cleared ===")
    else:
        logger.info("=== PHASE 4: Edge creation INCOMPLETE in %.1fs (%d/%d created, checkpoint saved) ===",
                     elapsed, edges_created, total_remaining_edges)
        if not json_output and not cpu_abort:
            console.print(
                f"  [yellow]Edge creation incomplete ({edges_created}/{total_remaining_edges}). "
                f"Run the same command again to resume.[/yellow]"
            )

    # Output results
    if json_output:
        print_json({
            "doc_id": doc_id,
            "title": doc_title,
            "summaries_generated": len(summaries),
            "summaries_stored": stored_count,
            "concepts_extracted": len(concepts),
            "concepts_stored": stored_concepts,
            "edges_created": edges_created,
            "levels": {
                "document": len([s for s in summaries if s.level == "document"]),
                "chapter": len([s for s in summaries if s.level == "chapter"]),
                "section": len([s for s in summaries if s.level == "section"]),
            },
            "top_concepts": [
                {"name": c.name, "type": c.concept_type, "occurrences": c.occurrence_count}
                for c in concepts[:10]  # Top 10 by occurrence
            ],
        })
    else:
        console.print()
        print_success(
            f"Generated {len(summaries)} summaries "
            f"({stored_count} stored) and extracted {len(concepts)} concepts "
            f"({stored_concepts} stored, {edges_created} edges)"
        )

        # Show breakdown
        doc_count = len([s for s in summaries if s.level == "document"])
        chap_count = len([s for s in summaries if s.level == "chapter"])
        sec_count = len([s for s in summaries if s.level == "section"])
        console.print(f"  Document: {doc_count}")
        console.print(f"  Chapters: {chap_count}")
        console.print(f"  Sections: {sec_count}")
        console.print(f"  Concepts: {len(concepts)}")

        # Show top concepts
        if concepts:
            console.print()
            console.print("[bold]Top concepts:[/bold]")
            for c in concepts[:5]:
                console.print(f"  - {c.name} ({c.concept_type}) - {c.occurrence_count} occurrences")


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

    # Sort chapters by source_id numeric suffix for consistent ordering
    if level_filter == "chapter" or level_filter is None:
        summaries = sorted(summaries, key=lambda s: (
            {"document": 0, "chapter": 1, "section": 2}.get(s.get("level", ""), 3),
            _chapter_sort_key(s) if s.get("level") == "chapter" else (0, ""),
        ))

    if json_output:
        print_json({
            "doc_id": doc_id,
            "summaries": [
                {
                    "summary_id": s.get("summary_id"),
                    "level": s.get("level"),
                    "content": s.get("content"),
                    "chapter_title": _extract_chapter_title_from_summary(s),
                    "key_points": _get_display_key_points_from_summary(s),
                    "word_count": s.get("word_count"),
                }
                for s in summaries
            ],
        })
    else:
        for summary in summaries:
            level_name = summary.get("level", "unknown").title()
            content = summary.get("content", "")
            key_points = _get_display_key_points_from_summary(summary)

            # Use chapter title if available
            chapter_title = _extract_chapter_title_from_summary(summary)
            if chapter_title and summary.get("level") == "chapter":
                panel_title = f"[bold]{chapter_title}[/bold]"
            else:
                panel_title = f"[bold]{level_name} Summary[/bold]"

            formatted = _html_tables_to_markdown(content)
            panel = Panel(
                Markdown(formatted),
                title=panel_title,
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
    threshold: float = typer.Option(
        0.3,
        "--threshold",
        "-t",
        help="Minimum similarity score (0.0 to disable)",
    ),
    rerank: bool = typer.Option(
        False,
        "--rerank",
        help="Rerank results using Cohere Rerank v3.5",
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
        kidkazz summarize search "FIFO" --level chapter --threshold 0.3
        kidkazz summarize search "cost accounting" --doc inventory_accounting
    """
    config = CLIConfig.load()
    store = _get_store(config)
    embedder = _get_embedder(config)

    # Check embedding dimension compatibility
    if hasattr(store, "check_embedding_compatibility"):
        try:
            store.check_embedding_compatibility(
                new_dim=embedder.get_embedding_dim(),
                new_model=getattr(embedder, "model_name", "unknown"),
            )
        except RuntimeError as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    # Determine if reranking is active
    use_rerank = rerank or config.reranker_enabled
    reranker = None
    if use_rerank:
        from src.cli.utils import get_reranker
        reranker = get_reranker(config)

    # Fetch more candidates when reranking
    fetch_limit = limit * 4 if reranker else limit

    # Generate query embedding
    if hasattr(embedder, "embed_query"):
        query_embedding = embedder.embed_query(query)
    else:
        query_embedding = embedder.embed_text(query)

    # Search summaries
    results = store.search_summaries(
        query_embedding=query_embedding,
        limit=fetch_limit,
        level=level,
        document_id=doc_id,
        threshold=threshold,
    )

    # Rerank if enabled
    if reranker and results:
        results = reranker.rerank_summaries(query, results, top_n=limit)

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
        raise typer.Exit(1) from e


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

        # Add chapters (sorted by source_id numeric suffix)
        chapters = sorted(
            [s for s in summaries if s.get("level") == "chapter"],
            key=_chapter_sort_key,
        )
        for chapter in chapters:
            chapter_title = _extract_chapter_title_from_summary(chapter)
            if chapter_title:
                chapter_label = f"[green]{chapter_title}[/green]\n{chapter.get('content', '')[:80]}..."
            else:
                chapter_label = f"[green]Chapter[/green]\n{chapter.get('content', '')[:80]}..."
            chapter_node = doc_node.add(chapter_label)

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
