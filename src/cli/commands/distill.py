"""CLI commands for distilling textbook knowledge into agent configs."""

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from ..config import CLIConfig
from ..output import console, print_error, print_json, print_success, print_warning
from ..utils import get_store

logger = logging.getLogger(__name__)

app = typer.Typer(help="Distill textbook knowledge into specialist agent configs")

DISTILL_DIR = Path.home() / ".kidkazz" / "distilled"


@app.command("generate")
def generate(
    doc_id: str = typer.Argument(help="Document ID to distill"),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: ~/.kidkazz/distilled/<doc_id>_agent.json)",
    ),
    provider: str = typer.Option(
        "openai/gpt-4o-mini",
        "--provider",
        "-p",
        help="LLM provider for generation (e.g., openai/gpt-4o-mini)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print raw JSON to stdout",
    ),
    top_k: int = typer.Option(
        20,
        "--top-k",
        help="Number of top concepts to include in expertise",
    ),
    num_examples: int = typer.Option(
        5,
        "--examples",
        help="Number of few-shot tool usage examples to generate",
    ),
) -> None:
    """Distill a document's knowledge base into an AgentConfig JSON for agent_ex.

    Reads concept graph + chapter summaries from Helix-DB and produces a
    specialist agent configuration that agent_ex can import. Uses 2 LLM calls.
    """
    config = CLIConfig.load()
    store = get_store(config)

    try:
        from src.distiller.distiller import TextbookDistiller
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_warning("Install with: pip install 'kidkazz[concepts]'")
        raise typer.Exit(1)

    try:
        distiller = TextbookDistiller(store, provider=provider)
        result = distiller.distill(doc_id, top_k=top_k, num_examples=num_examples)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Distillation failed: {e}")
        raise typer.Exit(1)

    # Write output file
    output_path = output or (DISTILL_DIR / f"{doc_id}_agent.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    if json_output:
        print_json(result)
    else:
        _print_summary(result, output_path)


@app.command("show")
def show(
    doc_id: str = typer.Argument(help="Document ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show an existing distilled agent config."""
    path = DISTILL_DIR / f"{doc_id}_agent.json"

    if not path.exists():
        print_error(f"No distilled config found for '{doc_id}'")
        print_warning(f"Run: kidkazz distill generate {doc_id}")
        raise typer.Exit(1)

    data = json.loads(path.read_text())

    if json_output:
        print_json(data)
    else:
        _print_summary(data, path)


@app.command("list")
def list_distilled(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List all distilled agent configs."""
    if not DISTILL_DIR.exists():
        print_warning("No distilled configs found.")
        return

    configs = []
    for path in sorted(DISTILL_DIR.glob("*_agent.json")):
        try:
            data = json.loads(path.read_text())
            meta = data.get("_distill_metadata", {})
            configs.append({
                "name": data.get("name", ""),
                "source_doc_id": meta.get("source_document_id", ""),
                "total_concepts": meta.get("total_concepts", 0),
                "total_chapters": meta.get("total_chapters", 0),
                "distilled_at": meta.get("distilled_at", ""),
                "file": str(path),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    if json_output:
        print_json(configs)
        return

    if not configs:
        print_warning("No distilled configs found.")
        return

    from rich.table import Table

    table = Table(title="Distilled Agent Configs", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan")
    table.add_column("Source Doc", style="yellow")
    table.add_column("Concepts", justify="right")
    table.add_column("Chapters", justify="right")
    table.add_column("Distilled At", style="dim")

    for c in configs:
        table.add_row(
            c["name"],
            c["source_doc_id"],
            str(c["total_concepts"]),
            str(c["total_chapters"]),
            c["distilled_at"],
        )

    console.print(table)


def _print_summary(data: dict, output_path: Path) -> None:
    """Print a formatted summary of a distilled config."""
    from rich.panel import Panel
    from rich.table import Table

    meta = data.get("_distill_metadata", {})

    # Header
    console.print()
    console.print(Panel(
        f"[bold]{data.get('name', 'unknown')}[/bold]\n"
        f"{data.get('description', '')}",
        title="Distilled Agent Config",
        border_style="green",
    ))

    # Stats
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("Role", data.get("role", ""))
    table.add_row("Personality", data.get("personality", ""))
    table.add_row("Expertise", ", ".join(data.get("expertise", [])[:5]))
    table.add_row("Tools", str(len(data.get("tool_ids", []))))
    table.add_row("Examples", str(len(data.get("tool_examples", []))))
    table.add_row("Concepts", str(meta.get("total_concepts", 0)))
    table.add_row("Chapters", str(meta.get("total_chapters", 0)))
    table.add_row("Prerequisites", str(len(meta.get("prerequisite_skills", []))))
    console.print(table)

    # Concept type distribution
    type_dist = meta.get("concept_types", {})
    if type_dist:
        top_types = list(type_dist.items())[:5]
        console.print(
            f"\n[dim]Concept types: "
            + ", ".join(f"{t}: {c}" for t, c in top_types)
            + "[/dim]"
        )

    console.print()
    print_success(f"Saved to: {output_path}")
