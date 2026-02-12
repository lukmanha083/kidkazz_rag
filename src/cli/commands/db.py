"""Database management commands."""

import logging
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import typer

from ..config import CLIConfig
from ..output import (
    confirm,
    console,
    print_db_status,
    print_error,
    print_info,
    print_json,
    print_success,
    print_warning,
)
from ..utils import get_store

logger = logging.getLogger(__name__)

CONTAINER_NAME = "helix-kidkazz_rag-dev_app"

THREAD_LIMIT_VARS = [
    "HELIX_CORES_OVERRIDE=2",
    "TOKIO_WORKER_THREADS=2",
    "RAYON_NUM_THREADS=2",
]


def _find_helix_dir() -> Path:
    """Locate .helix/dev/ relative to project root."""
    # Walk up from cwd looking for .kidkazz.toml
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".kidkazz.toml").exists():
            helix_dev = parent / ".helix" / "dev"
            if helix_dev.is_dir():
                return helix_dev
            raise FileNotFoundError(
                f".helix/dev/ not found in {parent}. Run 'helix init' first."
            )
    raise FileNotFoundError(
        "Could not find project root (.kidkazz.toml not found)."
    )


def _patch_compose(compose_file: Path, port: int) -> None:
    """Patch helix-generated docker-compose.yml with port and thread-limit env vars."""
    text = compose_file.read_text()

    # Fix port mapping: replace any "XXXX:YYYY" with "{port}:{port}"
    text = re.sub(
        r'"(\d+):(\d+)"',
        f'"{port}:{port}"',
        text,
    )

    # Add thread-limiting env vars if missing
    for var in THREAD_LIMIT_VARS:
        if var not in text:
            # Insert before the "restart:" line
            text = text.replace(
                "    restart:",
                f"      - {var}\n    restart:",
            )

    compose_file.write_text(text)


def _wait_for_helix(port: int, retries: int = 10, delay: int = 2) -> bool:
    """Poll Helix-DB health endpoint until ready."""
    url = f"http://localhost:{port}/ListDocuments"
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, data=b"[{}]", method="POST")
            req.add_header("Content-Type", "application/json")
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (URLError, OSError):
            pass
        if attempt < retries:
            time.sleep(delay)
    return False

app = typer.Typer(help="Database operations")


@app.command("init")
def init_database(
    store: str = typer.Option(
        None,
        "--store",
        "-s",
        help="Storage backend (mock or helix)",
    ),
    port: int = typer.Option(
        6969,
        "--port",
        "-p",
        help="Helix-DB port",
    ),
    create_schema: bool = typer.Option(
        False,
        "--create-schema",
        help="Create database schema",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """Initialize or verify database connection."""
    config = CLIConfig.load()

    if store:
        config.store_type = store
    if port != 6969:
        config.helix_port = port

    result = {
        "store_type": config.store_type,
        "status": "unknown",
    }

    try:
        store_instance = get_store(config)

        # Try to list documents to verify connection
        docs = store_instance.list_documents()
        result["status"] = "connected"
        result["documents"] = len(docs)

        if create_schema and config.store_type == "helix":
            # Schema creation would go here
            result["schema"] = "created"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    if json_output:
        print_json(result)
    else:
        if result["status"] == "connected":
            print_success(f"Database initialized ({config.store_type})")
            console.print(f"  Documents: {result.get('documents', 0)}")
        else:
            print_error(f"Failed to connect: {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)


@app.command("status")
def database_status(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """Check database connection and statistics."""
    config = CLIConfig.load()

    status = {
        "store_type": config.store_type,
        "connected": False,
        "documents": 0,
        "chunks": 0,
    }

    if config.store_type == "helix":
        status["port"] = config.helix_port

    try:
        store = get_store(config)

        # Get document count
        docs = store.list_documents()
        status["documents"] = len(docs)
        status["connected"] = True

        # Get total chunk count
        total_chunks = 0
        for doc in docs:
            doc_id = doc.get("doc_id") if isinstance(doc, dict) else getattr(doc, "doc_id", None)
            if doc_id:
                chunks = store.get_document_chunks(doc_id)
                total_chunks += len(chunks)
        status["chunks"] = total_chunks

    except Exception as e:
        status["error"] = str(e)

    if json_output:
        print_json(status)
    else:
        print_db_status(status)


@app.command("clear")
def clear_database(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
    keep_schema: bool = typer.Option(
        False,
        "--keep-schema",
        help="Keep schema, only delete data",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output result as JSON",
    ),
) -> None:
    """Clear all data from database."""
    config = CLIConfig.load()

    # Confirm unless force
    if not force and not json_output:
        if not confirm("This will delete ALL documents and chunks. Continue?"):
            print_warning("Clear cancelled")
            return

    try:
        store = get_store(config)

        # Clear data
        if hasattr(store, "clear"):
            store.clear()
            result = {"status": "cleared", "store_type": config.store_type}
        else:
            # For stores without clear(), delete each document
            docs = store.list_documents()
            count = 0
            for doc in docs:
                doc_id = doc.get("doc_id") if isinstance(doc, dict) else getattr(doc, "doc_id", None)
                if doc_id and hasattr(store, "delete_document"):
                    store.delete_document(doc_id)
                    count += 1
            result = {"status": "cleared", "documents_removed": count}

        if json_output:
            print_json(result)
        else:
            print_success("Database cleared")

    except Exception as e:
        if json_output:
            print_json({"status": "error", "error": str(e)})
        else:
            print_error(f"Failed to clear database: {e}")
        raise typer.Exit(1)


@app.command("deploy")
def deploy_helix(
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """Deploy Helix-DB: compile queries, build image, patch config, restart."""
    config = CLIConfig.load()
    port = config.helix_port

    result: dict = {"status": "unknown"}

    # Step 1: Find .helix/dev/
    try:
        helix_dir = _find_helix_dir()
    except FileNotFoundError as e:
        if json_output:
            print_json({"status": "error", "error": str(e)})
        else:
            print_error(str(e))
        raise typer.Exit(1)

    # Step 2: Stop and remove existing container (prevents name conflict)
    if not json_output:
        console.print("[bold]Step 1/5:[/bold] Stopping existing container...")
    subprocess.run(
        ["docker", "stop", CONTAINER_NAME],
        capture_output=True,
        timeout=30,
    )
    subprocess.run(
        ["docker", "rm", CONTAINER_NAME],
        capture_output=True,
        timeout=30,
    )

    # Step 3: Run helix push dev
    if not json_output:
        console.print("[bold]Step 2/5:[/bold] Compiling queries and building image...")
    try:
        proc = subprocess.run(
            ["helix", "push", "dev"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip()
            if json_output:
                print_json({"status": "error", "step": "helix push dev", "error": err})
            else:
                print_error(f"helix push dev failed:\n{err}")
            raise typer.Exit(1)
        result["push_output"] = proc.stdout.strip()
    except FileNotFoundError:
        msg = "helix CLI not found. Install from https://install.helix-db.com"
        if json_output:
            print_json({"status": "error", "error": msg})
        else:
            print_error(msg)
        raise typer.Exit(1)

    # Step 4: Patch docker-compose.yml
    if not json_output:
        console.print("[bold]Step 3/5:[/bold] Patching docker-compose.yml...")
    compose_file = helix_dir / "docker-compose.yml"
    if not compose_file.exists():
        msg = f"docker-compose.yml not found at {compose_file}"
        if json_output:
            print_json({"status": "error", "error": msg})
        else:
            print_error(msg)
        raise typer.Exit(1)

    _patch_compose(compose_file, port)
    result["port"] = port
    result["env_vars"] = THREAD_LIMIT_VARS

    # Step 5: Stop/remove container that helix push dev started (it uses un-patched config)
    if not json_output:
        console.print("[bold]Step 4/5:[/bold] Restarting container with patched config...")
    subprocess.run(
        ["docker", "stop", CONTAINER_NAME],
        capture_output=True,
        timeout=30,
    )
    subprocess.run(
        ["docker", "rm", CONTAINER_NAME],
        capture_output=True,
        timeout=30,
    )
    proc = subprocess.run(
        ["docker", "compose", "up", "-d"],
        cwd=helix_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        if json_output:
            print_json({"status": "error", "step": "docker compose up", "error": err})
        else:
            print_error(f"docker compose up failed:\n{err}")
        raise typer.Exit(1)

    # Step 6: Wait for health check
    if not json_output:
        console.print("[bold]Step 5/5:[/bold] Waiting for Helix-DB to be ready...")
    if _wait_for_helix(port):
        result["status"] = "deployed"
        if json_output:
            print_json(result)
        else:
            print_success(f"Helix-DB deployed on port {port}")
    else:
        result["status"] = "timeout"
        if json_output:
            print_json(result)
        else:
            print_warning(f"Helix-DB started but health check timed out (port {port})")


@app.command("restart")
def restart_helix(
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """Restart Helix-DB container (no rebuild)."""
    config = CLIConfig.load()
    port = config.helix_port

    result: dict = {"status": "unknown"}

    # Find .helix/dev/
    try:
        helix_dir = _find_helix_dir()
    except FileNotFoundError as e:
        if json_output:
            print_json({"status": "error", "error": str(e)})
        else:
            print_error(str(e))
        raise typer.Exit(1)

    compose_file = helix_dir / "docker-compose.yml"
    if not compose_file.exists():
        msg = f"docker-compose.yml not found at {compose_file}"
        if json_output:
            print_json({"status": "error", "error": msg})
        else:
            print_error(msg)
        raise typer.Exit(1)

    # docker compose down + up
    if not json_output:
        console.print("[bold]Step 1/2:[/bold] Restarting container...")
    proc = subprocess.run(
        ["docker", "compose", "down"],
        cwd=helix_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        if json_output:
            print_json({"status": "error", "step": "docker compose down", "error": err})
        else:
            print_error(f"docker compose down failed:\n{err}")
        raise typer.Exit(1)

    proc = subprocess.run(
        ["docker", "compose", "up", "-d"],
        cwd=helix_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        if json_output:
            print_json({"status": "error", "step": "docker compose up", "error": err})
        else:
            print_error(f"docker compose up failed:\n{err}")
        raise typer.Exit(1)

    # Wait for health check
    if not json_output:
        console.print("[bold]Step 2/2:[/bold] Waiting for Helix-DB to be ready...")
    if _wait_for_helix(port):
        result["status"] = "restarted"
        result["port"] = port
        if json_output:
            print_json(result)
        else:
            print_success(f"Helix-DB restarted on port {port}")
    else:
        result["status"] = "timeout"
        result["port"] = port
        if json_output:
            print_json(result)
        else:
            print_warning(f"Helix-DB started but health check timed out (port {port})")


@app.command("reset")
def reset_helix(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """Stop Helix-DB, destroy container, and wipe data volume for a clean start.

    This performs a full teardown:
      1. Stop the running Helix container
      2. Remove the container
      3. Remove the Docker network
      4. Wipe the data volume (.helix/.volumes/dev/) using a Docker alpine
         container (volume is root-owned)

    After reset, run `kidkazz db deploy` to rebuild and start fresh.

    Examples:
        kidkazz db reset              # Interactive confirmation
        kidkazz db reset --force      # Skip confirmation
    """
    # Find project root / .helix/dev/
    try:
        helix_dir = _find_helix_dir()
    except FileNotFoundError as e:
        if json_output:
            print_json({"status": "error", "error": str(e)})
        else:
            print_error(str(e))
        raise typer.Exit(1)

    volumes_dir = helix_dir.parent / ".volumes" / "dev"

    # Confirm
    if not force and not json_output:
        console.print(
            "[bold red]This will destroy the Helix container and wipe ALL data.[/bold red]\n"
            f"  Container: {CONTAINER_NAME}\n"
            f"  Volume:    {volumes_dir}"
        )
        if not confirm("Proceed with full reset?"):
            print_warning("Reset cancelled")
            return

    steps_done = []

    # Step 1: docker compose down (stops container + removes container + network)
    if not json_output:
        console.print("[bold]Step 1/2:[/bold] Stopping and removing container...")
    proc = subprocess.run(
        ["docker", "compose", "down"],
        cwd=helix_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode == 0:
        steps_done.append("container_removed")
        if not json_output:
            print_info("Container and network removed")
    else:
        # Fallback: try stop + rm directly (compose file may be missing/broken)
        subprocess.run(
            ["docker", "stop", CONTAINER_NAME],
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["docker", "rm", CONTAINER_NAME],
            capture_output=True,
            timeout=30,
        )
        steps_done.append("container_removed_fallback")
        if not json_output:
            print_info("Container removed (fallback)")

    # Step 2: Wipe data volume using Docker alpine (root-owned files)
    if not json_output:
        console.print("[bold]Step 2/2:[/bold] Wiping data volume...")

    if volumes_dir.exists():
        # Volume is root-owned (created by Docker), use alpine container to delete
        abs_vol = str(volumes_dir.resolve())
        proc = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{abs_vol}:/data",
                "alpine",
                "rm", "-rf", "/data/user",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            steps_done.append("volume_wiped")
            if not json_output:
                print_info(f"Volume wiped: {volumes_dir}")
        else:
            err = proc.stderr.strip()
            if json_output:
                print_json({"status": "error", "step": "wipe_volume", "error": err})
            else:
                print_error(f"Failed to wipe volume: {err}")
            raise typer.Exit(1)
    else:
        steps_done.append("volume_not_found")
        if not json_output:
            print_info("Volume directory not found (already clean)")

    # Done
    if json_output:
        print_json({"status": "reset", "steps": steps_done})
    else:
        print_success(
            "Helix-DB reset complete. Run 'kidkazz db deploy' to start fresh."
        )


@app.command("dedup-edges")
def dedup_edges(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report duplicates without modifying anything",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """Deduplicate DefinesConcept edges across all documents.

    Scans all concepts, collects unique (chunk, concept) edge pairs,
    drops all existing DefinesConcept edges, and re-creates only unique ones.

    Examples:
        kidkazz db dedup-edges --dry-run   # Preview duplicate count
        kidkazz db dedup-edges             # Clean up duplicates
    """
    config = CLIConfig.load()
    store = get_store(config)

    # Step 1: List all concepts
    if not json_output:
        console.print("[bold]Step 1/4:[/bold] Scanning concepts and edges...")
    concepts = store.list_concepts()
    if not concepts:
        if json_output:
            print_json({"status": "no_concepts", "total_edges": 0, "duplicates": 0})
        else:
            console.print("[yellow]No concepts found[/yellow]")
        return

    # Step 2: For each concept, get definition chunks and collect unique edges
    unique_edges: set[tuple[str, str]] = set()
    total_edge_count = 0
    chunks_with_edges: set[str] = set()

    for i, concept in enumerate(concepts):
        concept_id = concept.get("concept_id")
        if not concept_id:
            continue

        internal_id = concept.get("id")
        if not internal_id:
            continue

        # Get chunks that define this concept (via edge traversal)
        try:
            def_chunks = store.get_concept_definition_chunks(concept_id)
        except Exception as e:
            logger.warning("Failed to get definition chunks for %s: %s", concept_id, e)
            continue

        for chunk in def_chunks:
            chunk_internal_id = chunk.get("id")
            if chunk_internal_id:
                total_edge_count += 1
                unique_edges.add((str(chunk_internal_id), str(internal_id)))
                chunks_with_edges.add(str(chunk_internal_id))

        # Log progress every 200 concepts
        if (i + 1) % 200 == 0:
            if not json_output:
                console.print(f"  Scanned {i + 1}/{len(concepts)} concepts...")
            time.sleep(0.5)

        # Throttle individual requests
        time.sleep(0.08)

    duplicate_count = total_edge_count - len(unique_edges)

    if not json_output:
        console.print(f"\n[bold]Edge scan results:[/bold]")
        console.print(f"  Concepts scanned: {len(concepts)}")
        console.print(f"  Total edges: {total_edge_count}")
        console.print(f"  Unique edges: {len(unique_edges)}")
        console.print(f"  Duplicates: {duplicate_count}")
        console.print(f"  Chunks with edges: {len(chunks_with_edges)}")

    if duplicate_count == 0:
        if json_output:
            print_json({
                "status": "clean",
                "concepts": len(concepts),
                "total_edges": total_edge_count,
                "unique_edges": len(unique_edges),
                "duplicates": 0,
            })
        else:
            print_success("No duplicate edges found!")
        return

    if dry_run:
        if json_output:
            print_json({
                "status": "dry_run",
                "concepts": len(concepts),
                "total_edges": total_edge_count,
                "unique_edges": len(unique_edges),
                "duplicates": duplicate_count,
                "chunks_with_edges": len(chunks_with_edges),
            })
        else:
            console.print(f"\n[yellow]Dry run — no changes made. "
                          f"Run without --dry-run to remove {duplicate_count} duplicates.[/yellow]")
        return

    # Step 3: Drop all DefinesConcept edges from affected chunks
    if not json_output:
        console.print(f"\n[bold]Step 2/4:[/bold] Dropping all DefinesConcept edges from {len(chunks_with_edges)} chunks...")

    if not hasattr(store, 'batch_drop_chunk_concept_edges'):
        msg = "Store does not support batch_drop_chunk_concept_edges"
        if json_output:
            print_json({"status": "error", "error": msg})
        else:
            print_error(msg)
        raise typer.Exit(1)

    try:
        store.batch_drop_chunk_concept_edges(list(chunks_with_edges))
    except Exception as e:
        msg = f"Failed to drop edges: {e}"
        if json_output:
            print_json({"status": "error", "error": msg})
        else:
            print_error(msg)
        raise typer.Exit(1)

    if not json_output:
        console.print("  Edges dropped successfully")

    # Brief cooldown
    time.sleep(3)

    # Step 4: Re-create only unique edges
    if not json_output:
        console.print(f"[bold]Step 3/4:[/bold] Re-creating {len(unique_edges)} unique edges...")

    edges_list = list(unique_edges)
    BATCH_SIZE = 50
    edges_created = 0

    for i in range(0, len(edges_list), BATCH_SIZE):
        batch = edges_list[i:i + BATCH_SIZE]
        try:
            if hasattr(store, 'batch_link_chunk_defines_concept'):
                count = store.batch_link_chunk_defines_concept(batch)
                edges_created += count
            else:
                for chunk_id, concept_id in batch:
                    if store.link_chunk_defines_concept(chunk_id, concept_id):
                        edges_created += 1
        except Exception as e:
            logger.error("Failed to create edge batch at offset %d: %s", i, e)
            if not json_output:
                print_warning(f"Failed batch at offset {i}: {e}")

        # Throttle
        time.sleep(0.5)

        # Progress every 10 batches
        if ((i // BATCH_SIZE) + 1) % 10 == 0 and not json_output:
            console.print(f"  Created {edges_created}/{len(unique_edges)} edges...")

    if not json_output:
        console.print(f"\n[bold]Step 4/4:[/bold] Verification")
        console.print(f"  Edges before: {total_edge_count}")
        console.print(f"  Edges after: {edges_created}")
        console.print(f"  Duplicates removed: {total_edge_count - edges_created}")

    if json_output:
        print_json({
            "status": "deduped",
            "concepts": len(concepts),
            "edges_before": total_edge_count,
            "edges_after": edges_created,
            "duplicates_removed": total_edge_count - edges_created,
        })
    else:
        print_success(
            f"Deduplication complete: {total_edge_count} -> {edges_created} edges "
            f"({total_edge_count - edges_created} duplicates removed)"
        )


@app.command("dedup-summaries")
def dedup_summaries(
    doc: str = typer.Option(
        None,
        "--doc",
        "-d",
        help="Only dedup summaries for a specific document",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report duplicates without modifying anything",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """Deduplicate summary nodes (keep newest, delete rest with full vector cleanup).

    When `kidkazz summarize generate --force` is run multiple times without
    deleting old summaries first, duplicate Summary + SummaryVector nodes
    accumulate. This command identifies duplicates by summary_id and removes
    them with full cleanup (edge + vector + node).

    Examples:
        kidkazz db dedup-summaries --dry-run          # Preview duplicate count
        kidkazz db dedup-summaries                     # Clean up all duplicates
        kidkazz db dedup-summaries --doc inventory_accounting  # Single doc
    """
    from src.storage.queries import (
        DeleteSummary,
        DropSummaryEmbeddingEdge,
        GetSummaryVector,
        DropSummaryVector,
    )

    config = CLIConfig.load()
    store = get_store(config)

    # Step 1: Fetch all summaries (or for a single doc)
    if not json_output:
        console.print("[bold]Step 1/3:[/bold] Scanning summaries...")

    if doc:
        summaries = store.get_document_summaries(doc)
    else:
        # Use ListSummarizedDocuments which returns all summaries
        from src.storage.queries import ListSummarizedDocuments
        store._ensure_connected()
        q = ListSummarizedDocuments()
        result = store._execute_query(q)
        summaries = result.data or [] if result.success else []

    if not summaries:
        if json_output:
            print_json({"status": "no_summaries", "total": 0, "duplicates": 0})
        else:
            console.print("[yellow]No summaries found[/yellow]")
        return

    # Step 2: Group by summary_id, identify duplicates
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        if isinstance(s, dict) and s.get("summary_id"):
            groups[s["summary_id"]].append(s)

    total_summaries = sum(len(v) for v in groups.values())
    unique_count = len(groups)
    duplicate_count = total_summaries - unique_count

    # Identify which nodes to delete (keep newest per group by created_at)
    to_delete: list[dict] = []
    for summary_id, copies in groups.items():
        if len(copies) <= 1:
            continue
        # Sort by created_at descending (newest first), keep the first
        copies.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        to_delete.extend(copies[1:])  # all except newest

    if not json_output:
        console.print(f"\n[bold]Summary scan results:[/bold]")
        console.print(f"  Total summary nodes: {total_summaries}")
        console.print(f"  Unique summary_ids: {unique_count}")
        console.print(f"  Duplicate nodes to delete: {len(to_delete)}")
        if doc:
            console.print(f"  Filtered to document: {doc}")

    if len(to_delete) == 0:
        if json_output:
            print_json({
                "status": "clean",
                "total": total_summaries,
                "unique": unique_count,
                "duplicates": 0,
            })
        else:
            print_success("No duplicate summaries found!")
        return

    if dry_run:
        if json_output:
            print_json({
                "status": "dry_run",
                "total": total_summaries,
                "unique": unique_count,
                "duplicates": len(to_delete),
            })
        else:
            console.print(f"\n[yellow]Dry run — no changes made. "
                          f"Run without --dry-run to remove {len(to_delete)} duplicates.[/yellow]")
        return

    # Step 3: Delete duplicates with full SummaryVector cleanup
    if not json_output:
        console.print(f"\n[bold]Step 2/3:[/bold] Deleting {len(to_delete)} duplicate summaries...")

    store._ensure_connected()
    deleted = 0
    errors = 0

    for i, summary in enumerate(to_delete):
        internal_id = summary.get("id")
        if not internal_id:
            errors += 1
            continue

        try:
            # Step A: Get linked SummaryVector
            vec_query = GetSummaryVector(internal_id)
            vec_result = store._execute_query(vec_query)
            vector_id = None
            if vec_result.success and vec_result.data:
                vector_id = store._extract_node_id(vec_result.data)

            # Step B: Drop embedding edge
            try:
                edge_query = DropSummaryEmbeddingEdge(internal_id)
                store._client.query(edge_query)
            except Exception:
                pass  # Edge may not exist

            # Step C: Drop the SummaryVector node
            if vector_id:
                try:
                    drop_vec_query = DropSummaryVector(vector_id)
                    store._client.query(drop_vec_query)
                except Exception:
                    pass  # Vector may already be gone

            # Step D: Drop the summary node
            drop_query = DeleteSummary(internal_id)
            store._client.query(drop_query)
            deleted += 1

            # Throttle
            time.sleep(0.08)

        except Exception as e:
            logger.warning("Failed to delete summary %s: %s", internal_id, e)
            errors += 1

        # Progress every 100 deletions
        if (i + 1) % 100 == 0 and not json_output:
            console.print(f"  Deleted {deleted}/{len(to_delete)}...")

    if not json_output:
        console.print(f"\n[bold]Step 3/3:[/bold] Verification")
        console.print(f"  Summaries before: {total_summaries}")
        console.print(f"  Duplicates deleted: {deleted}")
        console.print(f"  Expected remaining: {total_summaries - deleted}")
        if errors:
            console.print(f"  Errors: {errors}")

    if json_output:
        print_json({
            "status": "deduped",
            "total_before": total_summaries,
            "duplicates_deleted": deleted,
            "expected_remaining": total_summaries - deleted,
            "errors": errors,
        })
    else:
        print_success(
            f"Deduplication complete: {total_summaries} -> {total_summaries - deleted} summaries "
            f"({deleted} duplicates removed)"
        )
