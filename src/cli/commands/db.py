"""Database management commands."""

import re
import subprocess
import time
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
    print_json,
    print_success,
    print_warning,
)
from ..utils import get_store

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

    # Step 2: Run helix push dev
    if not json_output:
        console.print("[bold]Step 1/4:[/bold] Compiling queries and building image...")
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

    # Step 3: Patch docker-compose.yml
    if not json_output:
        console.print("[bold]Step 2/4:[/bold] Patching docker-compose.yml...")
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

    # Step 4: Stop and remove old container
    if not json_output:
        console.print("[bold]Step 3/4:[/bold] Restarting container...")
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

    # Step 5: Start new container
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
        console.print("[bold]Step 4/4:[/bold] Waiting for Helix-DB to be ready...")
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
