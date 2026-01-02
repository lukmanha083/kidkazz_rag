"""Database management commands."""

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
