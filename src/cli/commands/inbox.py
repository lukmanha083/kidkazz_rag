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
from src.pdf_inbox.cloud_sync import CloudSync

app = typer.Typer(
    name="inbox",
    help="Manage PDF inbox for conversion",
    no_args_is_help=True,
)

console = Console()


VALID_POST_ACTIONS = {"delete", "move", "keep"}
VALID_STATUSES = {"pending", "processing", "completed", "failed"}


def get_inbox_manager() -> PDFInboxManager:
    """Create inbox manager from CLI config."""
    config = CLIConfig.load()

    # Validate and map post_action to enum
    action_value = config.post_action.lower()
    if action_value not in VALID_POST_ACTIONS:
        console.print(
            f"[red]Invalid post_action '{config.post_action}'. "
            f"Valid options: {', '.join(VALID_POST_ACTIONS)}[/red]"
        )
        raise typer.Exit(1)

    action_map = {
        "delete": PostConversionAction.DELETE,
        "move": PostConversionAction.MOVE,
        "keep": PostConversionAction.KEEP,
    }
    post_action = action_map[action_value]

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
        # Validate status filter
        if status_filter:
            normalized_status = status_filter.lower()
            if normalized_status not in VALID_STATUSES:
                console.print(
                    f"[red]Invalid status '{status_filter}'. "
                    f"Valid options: {', '.join(sorted(VALID_STATUSES))}[/red]"
                )
                raise typer.Exit(1)

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
        # Check for conflicting flags
        if failed_only and completed_only:
            console.print(
                "[red]Cannot use both --failed and --completed. "
                "Choose one filter or omit both to clear all.[/red]"
            )
            raise typer.Exit(1)

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


@app.command()
def sync(
    check: bool = typer.Option(
        False,
        "--check",
        help="Check if rclone is installed",
    ),
    remotes: bool = typer.Option(
        False,
        "--remotes",
        help="List configured rclone remotes",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be synced without syncing",
    ),
    download: bool = typer.Option(
        False,
        "--download",
        help="Download from cloud instead of upload",
    ),
) -> None:
    """Sync inbox with cloud storage using rclone.

    By default, uploads local inbox PDFs to the configured remote.
    Use --download to download from cloud to local inbox.
    """
    try:
        config = CLIConfig.load()
        cloud_sync = CloudSync(
            remote_name=config.cloud_remote,
            remote_path=config.cloud_path,
        )

        # Handle --check flag
        if check:
            if cloud_sync.check_rclone_installed():
                version = cloud_sync.get_rclone_version()
                console.print(f"[green]rclone is installed: {version}[/green]")
            else:
                console.print("[red]rclone is not installed[/red]")
                console.print("Install from: https://rclone.org/install/")
                raise typer.Exit(1)
            return

        # Handle --remotes flag
        if remotes:
            remote_list = cloud_sync.list_remotes()
            if not remote_list:
                console.print("[yellow]No remotes configured[/yellow]")
                console.print("Configure a remote with: rclone config")
            else:
                console.print("[bold]Configured remotes:[/bold]")
                for remote in remote_list:
                    console.print(f"  - {remote}")
            return

        # Validate remote is configured for sync operations
        if not config.cloud_remote:
            console.print("[red]No cloud remote configured[/red]")
            console.print(
                "Configure with: kidkazz config set cloud_remote <remote_name>"
            )
            console.print("Or set environment variable: KIDKAZZ_CLOUD_REMOTE")
            raise typer.Exit(1)

        # Check rclone is installed
        if not cloud_sync.check_rclone_installed():
            console.print("[red]rclone is not installed[/red]")
            console.print("Install from: https://rclone.org/install/")
            raise typer.Exit(1)

        # Validate remote exists
        if not cloud_sync.validate_remote(config.cloud_remote):
            console.print(f"[red]Remote '{config.cloud_remote}' not found[/red]")
            console.print("Configure with: rclone config")
            console.print("Or list available remotes: kidkazz inbox sync --remotes")
            raise typer.Exit(1)

        # Get inbox path
        inbox_path = Path(config.inbox_path).expanduser()

        # Perform sync
        if download:
            console.print(
                f"[bold]Downloading from {config.cloud_remote}:{config.cloud_path}...[/bold]"
            )
            result = cloud_sync.sync_from_remote(
                local_path=inbox_path,
                dry_run=dry_run,
            )
        else:
            console.print(
                f"[bold]Uploading to {config.cloud_remote}:{config.cloud_path}...[/bold]"
            )
            result = cloud_sync.sync_to_remote(
                local_path=inbox_path,
                dry_run=dry_run,
            )

        if result.success:
            action = "Would sync" if dry_run else "Synced"
            console.print(
                f"[green]{action} {result.files_synced} file(s)[/green]"
            )
        else:
            console.print(f"[red]Sync failed: {result.error_message}[/red]")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except OSError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None


def _format_size(size: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


@app.command()
def parse(
    agentic: bool = typer.Option(
        False,
        "--agentic",
        help="Enable AI-enhanced accuracy mode (uses more credits)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be parsed without processing",
    ),
    sync_backup: bool = typer.Option(
        True,
        "--sync-backup/--no-sync-backup",
        help="Sync output to cloud backup after parsing",
    ),
) -> None:
    """Parse PDFs in inbox using Reducto.ai API.

    Requires REDUCTO_API_KEY environment variable.
    Get your API key from https://reducto.ai

    Examples:
        kidkazz inbox parse                    # Parse all PDFs
        kidkazz inbox parse --agentic          # High-accuracy mode
        kidkazz inbox parse --dry-run          # Preview only
        kidkazz inbox parse --no-sync-backup   # Skip cloud backup
    """
    try:
        from src.pdf_converter.reducto_client import (
            ReductoClient,
            ReductoConfig,
            ReductoAPIError,
        )

        config = CLIConfig.load()
        inbox_path = Path(config.inbox_path).expanduser()
        output_path = Path(config.output_path).expanduser()

        # Ensure directories exist
        inbox_path.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find PDF files (recursive if configured)
        if config.inbox_recursive:
            pdf_files = list(inbox_path.rglob("*.pdf"))
        else:
            pdf_files = list(inbox_path.glob("*.pdf"))

        if not pdf_files:
            console.print("[yellow]No PDF files found in inbox[/yellow]")
            console.print(f"Inbox path: {inbox_path}")
            return

        # Dry run mode
        if dry_run:
            console.print("[bold]Dry run - would parse these files:[/bold]")
            for pdf in pdf_files:
                size_str = _format_size(pdf.stat().st_size)
                console.print(f"  - {pdf.name} ({size_str})")
            console.print(f"\nTotal: {len(pdf_files)} file(s)")
            return

        # Initialize Reducto client
        try:
            reducto_config = ReductoConfig.from_env()
            reducto_config.agentic = agentic
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("Set your API key: export REDUCTO_API_KEY=your_key_here")
            raise typer.Exit(1) from None

        client = ReductoClient(reducto_config)

        # Parse progress callback
        def on_progress(pdf_path: Path, index: int, total: int) -> None:
            console.print(f"[blue]Parsing ({index}/{total}):[/blue] {pdf_path.name}")

        # Parse PDFs
        console.print(f"[bold]Parsing {len(pdf_files)} PDF(s) with Reducto.ai...[/bold]")
        if agentic:
            console.print("[dim]Agentic mode enabled (higher accuracy, 2x credits)[/dim]")

        try:
            results = client.parse_files(pdf_files, on_progress=on_progress)
        except ReductoAPIError as e:
            console.print(f"[red]API Error: {e}[/red]")
            raise typer.Exit(1) from None

        # Save markdown files
        saved_count = 0
        for pdf_path, markdown in results:
            output_file = output_path / f"{pdf_path.stem}.md"
            output_file.write_text(markdown, encoding="utf-8")
            saved_count += 1
            console.print(f"[green]Saved:[/green] {output_file.name}")

        console.print(f"\n[green]Successfully parsed {saved_count} file(s)[/green]")
        console.print(f"Output directory: {output_path}")

        # Sync to cloud backup
        if sync_backup and config.cloud_remote:
            try:
                output_sync = CloudSync(
                    remote_name=config.cloud_remote,
                    remote_path=f"{config.cloud_path}_output",
                )

                if output_sync.check_rclone_installed() and output_sync.validate_remote(
                    config.cloud_remote
                ):
                    console.print("\n[bold]Syncing output to cloud backup...[/bold]")
                    result = output_sync.sync_to_remote(local_path=output_path)
                    if result.success:
                        console.print(
                            f"[green]Backed up {result.files_synced} file(s)[/green]"
                        )
                    else:
                        console.print(
                            f"[yellow]Backup warning: {result.error_message}[/yellow]"
                        )
            except Exception as e:
                console.print(f"[yellow]Backup skipped: {e}[/yellow]")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None
