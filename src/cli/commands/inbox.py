"""CLI commands for managing the PDF inbox.

This module provides commands for:
- Viewing inbox status and statistics
- Listing pending PDF files
- Clearing completed/failed entries
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.cli.config import CLIConfig
from src.pdf_inbox import PDFInboxManager, PostConversionAction

app = typer.Typer(
    name="inbox",
    help="Manage PDF inbox for conversion",
    no_args_is_help=True,
)

console = Console()


def get_inbox_manager() -> PDFInboxManager:
    """Create inbox manager from CLI config."""
    config = CLIConfig.load()

    # Map string to enum
    action_map = {
        "delete": PostConversionAction.DELETE,
        "move": PostConversionAction.MOVE,
        "keep": PostConversionAction.KEEP,
    }
    post_action = action_map.get(config.post_action, PostConversionAction.DELETE)

    return PDFInboxManager(
        inbox_path=Path(config.inbox_path).expanduser(),
        output_path=Path(config.output_path).expanduser(),
        post_action=post_action,
        recursive=config.inbox_recursive,
        processed_dir=Path(config.processed_dir).expanduser(),
    )


@app.command()
def status() -> None:
    """Show inbox status and statistics."""
    try:
        manager = get_inbox_manager()
        manager.scan()
        stats = manager.get_stats()

        if stats["total"] == 0:
            console.print("[dim]Inbox is empty[/dim]")
            console.print(f"\nInbox path: {manager.inbox_path}")
            return

        # Create status table
        table = Table(title="Inbox Status")
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")

        table.add_row("Pending", str(stats["pending"]), style="yellow")
        table.add_row("Processing", str(stats["processing"]), style="blue")
        table.add_row("Completed", str(stats["completed"]), style="green")
        table.add_row("Failed", str(stats["failed"]), style="red")
        table.add_row("Total", str(stats["total"]), style="bold")

        console.print(table)
        console.print(f"\nInbox path: {manager.inbox_path}")
        console.print(f"Post-action: {manager.post_action.value}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("list")
def list_pdfs(
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
    status_filter: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (pending, processing, completed, failed)",
    ),
) -> None:
    """List PDF files in the inbox."""
    try:
        manager = get_inbox_manager()
        pdfs = manager.scan()

        # Apply status filter
        if status_filter:
            pdfs = [p for p in pdfs if p.status.value == status_filter.lower()]

        if not pdfs:
            if json_output:
                console.print("[]")
            else:
                console.print("[dim]No PDF files found[/dim]")
            return

        if json_output:
            data = [
                {
                    "name": pdf.name,
                    "path": str(pdf.path),
                    "size": pdf.size,
                    "status": pdf.status.value,
                    "output_path": str(pdf.output_path) if pdf.output_path else None,
                    "error": pdf.error_message,
                }
                for pdf in pdfs
            ]
            console.print(json.dumps(data, indent=2))
        else:
            table = Table(title="PDF Files")
            table.add_column("Name", style="cyan")
            table.add_column("Size", justify="right")
            table.add_column("Status")

            for pdf in pdfs:
                size_str = _format_size(pdf.size)
                status_style = _get_status_style(pdf.status.value)
                table.add_row(pdf.name, size_str, f"[{status_style}]{pdf.status.value}[/{status_style}]")

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def clear(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
    failed_only: bool = typer.Option(
        False,
        "--failed",
        help="Only clear failed entries",
    ),
    completed_only: bool = typer.Option(
        False,
        "--completed",
        help="Only clear completed entries",
    ),
) -> None:
    """Clear PDF files from the inbox."""
    try:
        manager = get_inbox_manager()
        pdfs = manager.scan()

        # Filter based on options
        if failed_only:
            pdfs = [p for p in pdfs if p.is_failed]
        elif completed_only:
            pdfs = [p for p in pdfs if p.is_completed]

        if not pdfs:
            console.print("[dim]No files to clear[/dim]")
            return

        # Confirm unless --force
        if not force:
            console.print(f"This will remove {len(pdfs)} PDF file(s):")
            for pdf in pdfs[:5]:
                console.print(f"  - {pdf.name}")
            if len(pdfs) > 5:
                console.print(f"  ... and {len(pdfs) - 5} more")

            confirm = typer.confirm("Continue?")
            if not confirm:
                console.print("[yellow]Aborted[/yellow]")
                return

        # Delete files
        deleted = 0
        for pdf in pdfs:
            try:
                pdf.path.unlink()
                deleted += 1
            except Exception as e:
                console.print(f"[red]Failed to delete {pdf.name}: {e}[/red]")

        console.print(f"[green]Deleted {deleted} file(s)[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _format_size(size: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _get_status_style(status: str) -> str:
    """Get Rich style for status."""
    styles = {
        "pending": "yellow",
        "processing": "blue",
        "completed": "green",
        "failed": "red",
    }
    return styles.get(status, "white")
