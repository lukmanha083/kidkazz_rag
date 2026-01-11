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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.cli.config import CLIConfig
from src.cli.utils import slugify_filename
from src.pdf_inbox import PDFInboxManager, PostConversionAction
from src.pdf_inbox.cloud_sync import CloudSync, SyncProgress

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
    show_output: bool = typer.Option(
        False,
        "--output",
        "-o",
        help="List parsed markdown files from output directory instead of inbox PDFs",
    ),
) -> None:
    """List PDF files in the inbox or parsed markdown files in output directory."""
    try:
        config = CLIConfig.load()

        # List output directory (markdown files)
        if show_output:
            output_path = Path(config.output_path).expanduser()
            output_path.mkdir(parents=True, exist_ok=True)

            md_files = list(output_path.glob("*.md"))

            if not md_files:
                if json_output:
                    console.print("[]")
                else:
                    console.print("[dim]No markdown files found[/dim]")
                return

            if json_output:
                data = [
                    {
                        "name": f.name,
                        "path": str(f),
                        "size": f.stat().st_size,
                    }
                    for f in sorted(md_files, key=lambda x: x.name)
                ]
                console.print(json.dumps(data, indent=2))
            else:
                table = Table(title="Parsed Markdown Files")
                table.add_column("Name", style="cyan")
                table.add_column("Size", justify="right")

                for md_file in sorted(md_files, key=lambda x: x.name):
                    size_str = _format_size(md_file.stat().st_size)
                    table.add_row(md_file.name, size_str)

                console.print(table)
            return

        # Default: list inbox PDFs (all are pending - parsed PDFs are moved out)
        manager = get_inbox_manager()
        pdfs = manager.scan()

        if not pdfs:
            if json_output:
                console.print("[]")
            else:
                console.print("[dim]No PDF files in inbox[/dim]")
            return

        if json_output:
            data = [
                {
                    "name": pdf.name,
                    "path": str(pdf.path),
                    "size": pdf.size,
                }
                for pdf in pdfs
            ]
            console.print(json.dumps(data, indent=2))
        else:
            table = Table(title="PDF Files in Inbox")
            table.add_column("Name", style="cyan")
            table.add_column("Size", justify="right")

            for pdf in pdfs:
                size_str = _format_size(pdf.size)
                table.add_row(pdf.name, size_str)

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

        # Perform sync with progress bar
        if download:
            console.print(
                f"[bold]Downloading from {config.cloud_remote}:{config.cloud_path}...[/bold]"
            )
            if dry_run:
                # Dry run doesn't need progress bar
                result = cloud_sync.sync_from_remote(
                    local_path=inbox_path,
                    dry_run=True,
                )
            else:
                # Count pending files first for progress bar
                pending = cloud_sync.count_pending_downloads(inbox_path)
                total_files = len(pending)

                if total_files == 0:
                    console.print("[dim]No files to download[/dim]")
                    result = cloud_sync.sync_from_remote(local_path=inbox_path)
                else:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        console=console,
                    ) as progress:
                        task = progress.add_task("Downloading...", total=total_files)

                        def update_progress(p: SyncProgress) -> None:
                            progress.update(
                                task,
                                completed=p.files_completed,
                                description=f"[cyan]{p.current_file}[/cyan]",
                            )

                        result = cloud_sync.sync_from_remote_with_progress(
                            local_path=inbox_path,
                            progress_callback=update_progress,
                        )
                        # Ensure progress bar shows final state
                        progress.update(task, completed=result.files_synced)
        else:
            console.print(
                f"[bold]Uploading to {config.cloud_remote}:{config.cloud_path}...[/bold]"
            )
            if dry_run:
                # Dry run doesn't need progress bar
                result = cloud_sync.sync_to_remote(
                    local_path=inbox_path,
                    dry_run=True,
                )
            else:
                # Count pending files first for progress bar
                pending = cloud_sync.count_pending_uploads(inbox_path)
                total_files = len(pending)

                if total_files == 0:
                    console.print("[dim]No files to upload[/dim]")
                    result = cloud_sync.sync_to_remote(local_path=inbox_path)
                else:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        console=console,
                    ) as progress:
                        task = progress.add_task("Uploading...", total=total_files)

                        def update_progress(p: SyncProgress) -> None:
                            progress.update(
                                task,
                                completed=p.files_completed,
                                description=f"[cyan]{p.current_file}[/cyan]",
                            )

                        result = cloud_sync.sync_to_remote_with_progress(
                            local_path=inbox_path,
                            progress_callback=update_progress,
                        )
                        # Ensure progress bar shows final state
                        progress.update(task, completed=result.files_synced)

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


def _get_status_style(status: str) -> str:
    """Get Rich style for status display."""
    styles = {
        "pending": "yellow",
        "processing": "blue",
        "completed": "green",
        "failed": "red",
    }
    return styles.get(status.lower(), "white")


@app.command()
def quality(
    file_path: Optional[str] = typer.Argument(
        None,
        help="Path to markdown file to check",
    ),
    all_files: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Check all markdown files in output directory",
    ),
    directory: Optional[str] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Directory containing markdown files to check",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed metrics breakdown",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        "-s",
        help="Show summary for multiple files",
    ),
    threshold: str = typer.Option(
        "normal",
        "--threshold",
        "-t",
        help="Quality threshold: strict, normal, lenient",
    ),
) -> None:
    """Check quality of parsed markdown documents.

    Checks markdown files for quality issues that could affect RAG ingestion.
    Can check a single file, all files in output directory, or a specific directory.

    Examples:
        kidkazz inbox quality document.md           # Check single file
        kidkazz inbox quality --all                 # Check all in output dir
        kidkazz inbox quality --dir /path/to/md    # Check specific directory
        kidkazz inbox quality document.md --json   # JSON output
        kidkazz inbox quality --all --summary      # Summary of all files
    """
    from src.pdf_converter.quality_checker import (
        ReductoQualityChecker,
        QualityThresholds,
    )

    try:
        # Determine threshold
        threshold_map = {
            "strict": QualityThresholds.strict(),
            "normal": QualityThresholds(),
            "lenient": QualityThresholds.lenient(),
        }
        if threshold not in threshold_map:
            console.print(f"[red]Invalid threshold: {threshold}[/red]")
            console.print("Valid options: strict, normal, lenient")
            raise typer.Exit(1)

        checker = ReductoQualityChecker(threshold_map[threshold])

        # Collect files to check
        files_to_check: list[Path] = []

        if file_path:
            path = Path(file_path).expanduser()
            if not path.exists():
                console.print(f"[red]File not found: {file_path}[/red]")
                raise typer.Exit(1)
            files_to_check.append(path)
        elif all_files or directory:
            if directory:
                check_dir = Path(directory).expanduser()
            else:
                config = CLIConfig.load()
                check_dir = Path(config.output_path).expanduser()

            if not check_dir.exists():
                console.print(f"[red]Directory not found: {check_dir}[/red]")
                raise typer.Exit(1)

            files_to_check = list(check_dir.glob("*.md"))
            if not files_to_check:
                console.print(f"[yellow]No markdown files found in {check_dir}[/yellow]")
                return
        else:
            console.print("[red]Specify a file path or use --all[/red]")
            raise typer.Exit(1)

        # Check files
        reports = []
        for md_file in files_to_check:
            report = checker.check_file(md_file)
            reports.append((md_file, report))

        # Output results
        if json_output:
            if len(reports) == 1:
                _, report = reports[0]
                console.print(json.dumps(report.to_dict(), indent=2))
            else:
                all_reports = []
                for md_file, report in reports:
                    data = report.to_dict()
                    data["file"] = str(md_file)
                    all_reports.append(data)
                console.print(json.dumps(all_reports, indent=2))
        elif summary and len(reports) > 1:
            _print_quality_summary(reports)
        else:
            for md_file, report in reports:
                _print_quality_report(md_file, report, verbose)
                if len(reports) > 1:
                    console.print()  # Add spacing between reports

        # Exit with appropriate code
        from src.pdf_converter.quality_checker import QualityStatus
        failed_count = sum(1 for _, r in reports if r.status == QualityStatus.FAIL)
        if failed_count > 0:
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _print_quality_report(file_path: Path, report, verbose: bool = False) -> None:
    """Print a human-readable quality report."""
    from src.pdf_converter.quality_checker import IssueSeverity

    # Handle both enum and string status
    status_value = report.status.value if hasattr(report.status, 'value') else report.status
    status_styles = {
        "PASS": "green",
        "FAIL": "red",
        "WARNING": "yellow",
    }
    status_style = status_styles.get(status_value, "white")

    # Header
    console.print(f"\n[bold]Quality Report: {file_path.name}[/bold]")
    console.print("━" * 50)

    # Content stats
    m = report.metrics
    console.print("[bold]Content Stats:[/bold]")
    console.print(f"  Words: {m.word_count:,} | Lines: {m.line_count:,}")
    console.print(f"  Headings: {m.heading_count} | Tables: {m.table_count}")
    console.print(f"  Code blocks: {m.code_block_count} | Lists: {m.list_count}")

    if verbose:
        console.print("\n[bold]Quality Metrics:[/bold]")
        console.print(f"  Words per page (est): {m.words_per_page:.0f}")
        console.print(f"  Special char ratio: {m.special_char_ratio:.1%}")
        console.print(f"  Empty line ratio: {m.empty_line_ratio:.1%}")
        if m.broken_table_count > 0:
            console.print(f"  [yellow]Broken tables: {m.broken_table_count}[/yellow]")

    # Issues
    if report.issues:
        console.print("\n[bold]Issues:[/bold]")
        for issue in report.issues:
            # Handle both enum and string severity
            severity_value = issue.severity.value if hasattr(issue.severity, 'value') else issue.severity
            icon = "✗" if severity_value == "error" else "⚠"
            color = "red" if severity_value == "error" else "yellow"
            console.print(f"  [{color}]{icon} {issue.message}[/{color}]")
    else:
        console.print("\n  [green]✓ No issues detected[/green]")

    # Overall score and status
    console.print(f"\n[bold]Overall Score:[/bold] {report.score}/100 [{status_style}]({status_value})[/{status_style}]")
    console.print(f"[dim]Recommendation: {report.recommendation}[/dim]")


def _print_quality_summary(reports: list) -> None:
    """Print a summary table for multiple files."""
    table = Table(title="Quality Summary")
    table.add_column("File", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Status")
    table.add_column("Issues", justify="right")

    pass_count = 0
    warn_count = 0
    fail_count = 0
    total_score = 0

    for md_file, report in reports:
        # Handle both enum and string status
        status_value = report.status.value if hasattr(report.status, 'value') else report.status
        status_style = {"PASS": "green", "FAIL": "red", "WARNING": "yellow"}.get(
            status_value, "white"
        )
        table.add_row(
            md_file.name,
            str(report.score),
            f"[{status_style}]{status_value}[/{status_style}]",
            str(len(report.issues)),
        )
        total_score += report.score
        if status_value == "PASS":
            pass_count += 1
        elif status_value == "WARNING":
            warn_count += 1
        else:
            fail_count += 1

    console.print(table)
    avg_score = total_score / len(reports) if reports else 0
    console.print(f"\n[bold]Summary:[/bold] {len(reports)} file(s) checked")
    summary_parts = [f"[green]Passed: {pass_count}[/green]"]
    if warn_count > 0:
        summary_parts.append(f"[yellow]Warnings: {warn_count}[/yellow]")
    summary_parts.append(f"[red]Failed: {fail_count}[/red]")
    console.print(f"  {' | '.join(summary_parts)}")
    console.print(f"  Average score: {avg_score:.0f}/100")


@app.command()
def parse(
    filename: Optional[str] = typer.Argument(
        None,
        help="PDF filename to parse (looks in inbox directory). If not provided, parses all PDFs in inbox.",
    ),
    agentic: bool = typer.Option(
        False,
        "--agentic",
        help="Enable AI-enhanced accuracy mode (uses more credits)",
    ),
    chunk_mode: str = typer.Option(
        "disabled",
        "--chunk-mode",
        "-c",
        help="Chunking mode: disabled (human-readable), variable (RAG default), block (citations), page, section",
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
    quality_check: bool = typer.Option(
        True,
        "--quality-check/--no-quality-check",
        help="Run quality check on parsed output (default: enabled)",
    ),
    quality_threshold: str = typer.Option(
        "normal",
        "--quality-threshold",
        "-q",
        help="Quality threshold: strict, normal, lenient",
    ),
    inbox_path_override: Optional[str] = typer.Option(
        None,
        "--inbox-path",
        help="Override inbox path from config",
    ),
    output_path_override: Optional[str] = typer.Option(
        None,
        "--output-path",
        help="Override output path from config",
    ),
) -> None:
    """Parse PDFs in inbox using Reducto.ai API.

    Requires REDUCTO_API_KEY environment variable.
    Get your API key from https://reducto.ai

    Chunk modes:
        disabled  - Human-readable continuous markdown (default)
        variable  - Adaptive chunks ~1000 chars, best for RAG
        block     - Each element as chunk, best for citations
        page      - One chunk per page
        section   - Split by document sections/headings

    Quality check is enabled by default and will block saving low-quality output.
    Use --no-quality-check to force save regardless of quality.

    Examples:
        kidkazz inbox parse document.pdf           # Parse single file from inbox
        kidkazz inbox parse                        # Parse all PDFs in inbox
        kidkazz inbox parse --chunk-mode variable  # RAG-optimized chunks
        kidkazz inbox parse --agentic -c block     # High-accuracy + citations
        kidkazz inbox parse --dry-run              # Preview only
        kidkazz inbox parse --no-quality-check     # Skip quality check
        kidkazz inbox parse --quality-threshold strict  # Stricter quality
    """
    try:
        from src.pdf_converter.reducto_client import (
            ReductoClient,
            ReductoConfig,
            ReductoAPIError,
        )
        from src.pdf_converter.quality_checker import (
            ReductoQualityChecker,
            QualityThresholds,
            QualityStatus,
        )

        config = CLIConfig.load()
        inbox_path = Path(inbox_path_override).expanduser() if inbox_path_override else Path(config.inbox_path).expanduser()
        output_path = Path(output_path_override).expanduser() if output_path_override else Path(config.output_path).expanduser()

        # Validate quality threshold
        threshold_map = {
            "strict": QualityThresholds.strict(),
            "normal": QualityThresholds(),
            "lenient": QualityThresholds.lenient(),
        }
        if quality_threshold not in threshold_map:
            console.print(f"[red]Invalid quality threshold: {quality_threshold}[/red]")
            console.print("Valid options: strict, normal, lenient")
            raise typer.Exit(1)

        checker = ReductoQualityChecker(threshold_map[quality_threshold]) if quality_check else None

        # Ensure directories exist
        inbox_path.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find PDF files
        if filename:
            # Single file mode - look for file in inbox
            pdf_path = inbox_path / filename
            if not pdf_path.exists():
                # Try adding .pdf extension if not provided
                if not filename.lower().endswith(".pdf"):
                    pdf_path = inbox_path / f"{filename}.pdf"
                if not pdf_path.exists():
                    console.print(f"[red]File not found: {filename}[/red]")
                    console.print(f"Inbox path: {inbox_path}")
                    raise typer.Exit(1)
            pdf_files = [pdf_path]
        else:
            # All files mode (recursive if configured)
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

        # Validate chunk mode
        valid_modes = ["disabled", "variable", "block", "page", "section"]
        if chunk_mode not in valid_modes:
            console.print(f"[red]Invalid chunk mode: {chunk_mode}[/red]")
            console.print(f"Valid modes: {', '.join(valid_modes)}")
            raise typer.Exit(1)

        # Initialize Reducto client
        try:
            reducto_config = ReductoConfig.from_env()
            reducto_config.agentic = agentic
            reducto_config.chunk_mode = chunk_mode
            reducto_config.raw_output = chunk_mode == "disabled"
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("Set your API key: export REDUCTO_API_KEY=your_key_here")
            raise typer.Exit(1) from None

        client = ReductoClient(reducto_config)

        # Parse PDFs with progress bar
        console.print(f"[bold]Parsing {len(pdf_files)} PDF(s) with Reducto.ai...[/bold]")
        if agentic:
            console.print("[dim]Agentic mode enabled (higher accuracy, 2x credits)[/dim]")
        if chunk_mode != "disabled":
            console.print(f"[dim]Chunk mode: {chunk_mode}[/dim]")

        results = []
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Parsing PDFs...", total=len(pdf_files))

                for pdf_file in pdf_files:
                    progress.update(task, description=f"[cyan]{pdf_file.name}[/cyan]")
                    markdown = client.parse_pdf(pdf_file)
                    results.append((pdf_file, markdown))
                    progress.advance(task)

        except ReductoAPIError as e:
            console.print(f"[red]API Error: {e}[/red]")
            raise typer.Exit(1) from None

        # Quality check and save markdown files
        saved_count = 0
        blocked_count = 0
        quality_results = []

        for pdf_path, markdown in results:
            # Slugify filename for CLI-friendly output (no spaces/special chars)
            safe_name = slugify_filename(pdf_path.stem)
            output_file = output_path / f"{safe_name}.md"

            # Run quality check if enabled
            if checker:
                report = checker.check_markdown(markdown)
                quality_results.append((pdf_path, report))

                if report.status == QualityStatus.FAIL:
                    blocked_count += 1
                    console.print(f"[red]Blocked:[/red] {pdf_path.name} (quality score: {report.score}/100)")
                    for issue in report.issues[:3]:  # Show first 3 issues
                        severity_value = issue.severity.value if hasattr(issue.severity, 'value') else issue.severity
                        icon = "✗" if severity_value == "error" else "⚠"
                        color = "red" if severity_value == "error" else "yellow"
                        console.print(f"  [{color}]{icon} {issue.message}[/{color}]")
                    continue

            # Save file (passed quality check or check disabled)
            output_file.write_text(markdown, encoding="utf-8")
            saved_count += 1

            if checker and quality_results:
                _, report = quality_results[-1]
                status_color = "green" if report.status == QualityStatus.PASS else "yellow"
                console.print(f"[green]Saved:[/green] {output_file.name} [{status_color}](quality: {report.score}/100)[/{status_color}]")
            else:
                console.print(f"[green]Saved:[/green] {output_file.name}")

        # Summary
        console.print(f"\n[green]Successfully parsed {saved_count} file(s)[/green]")
        if blocked_count > 0:
            console.print(f"[red]Blocked {blocked_count} file(s) due to quality issues[/red]")
            console.print("[dim]Use --no-quality-check to force save[/dim]")
        console.print(f"Output directory: {output_path}")

        # Apply post_action to processed PDFs
        action_value = config.post_action.lower()
        if action_value == "delete":
            deleted_count = 0
            for pdf_path, _ in results:
                try:
                    pdf_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    console.print(f"[yellow]Failed to delete {pdf_path.name}: {e}[/yellow]")
            if deleted_count > 0:
                console.print(f"[dim]Deleted {deleted_count} processed PDF(s)[/dim]")
        elif action_value == "move":
            processed_dir = Path(config.processed_dir).expanduser()
            processed_dir.mkdir(parents=True, exist_ok=True)
            moved_count = 0
            for pdf_path, _ in results:
                try:
                    dest = processed_dir / pdf_path.name
                    pdf_path.rename(dest)
                    moved_count += 1
                except Exception as e:
                    console.print(f"[yellow]Failed to move {pdf_path.name}: {e}[/yellow]")
            if moved_count > 0:
                console.print(f"[dim]Moved {moved_count} PDF(s) to {processed_dir}[/dim]")

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
