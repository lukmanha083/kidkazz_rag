"""Main CLI entry point for KidKazz RAG."""

import logging
import os
import sys
from typing import Optional

import typer

from .commands import concepts, config_cmd, db, docs, inbox, ingest, search, summarize

# Create main app
app = typer.Typer(
    name="kidkazz",
    help="KidKazz RAG - Convert PDFs to searchable knowledge bases",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register command groups
app.add_typer(ingest.app, name="ingest", help="Ingest documents into knowledge base")
app.add_typer(search.app, name="search", help="Search the knowledge base")
app.add_typer(docs.app, name="docs", help="Manage documents")
app.add_typer(db.app, name="db", help="Database operations")
app.add_typer(config_cmd.app, name="config", help="Configuration management")
app.add_typer(inbox.app, name="inbox", help="Manage PDF inbox")
app.add_typer(concepts.app, name="concepts", help="Concept extraction and management")
app.add_typer(summarize.app, name="summarize", help="Document summarization")


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-essential output",
    ),
) -> None:
    """KidKazz RAG - Convert PDFs to searchable knowledge bases.

    Use --help on any command for more information.
    """
    # Configure logging based on verbose flag or KIDKAZZ_LOG_LEVEL env var
    env_log_level = os.getenv("KIDKAZZ_LOG_LEVEL", "").upper()
    if verbose:
        log_level = logging.DEBUG
    elif env_log_level:
        log_level = getattr(logging, env_log_level, logging.WARNING)
    else:
        log_level = logging.WARNING  # Default: only show warnings and errors

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    # Store options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


@app.command()
def version() -> None:
    """Show version information."""
    from rich.console import Console

    console = Console()
    console.print("[bold]KidKazz RAG[/bold] version 0.1.0")
    console.print("Python RAG system for PDF textbooks")


def cli() -> None:
    """Entry point for console script."""
    app()


if __name__ == "__main__":
    cli()
