"""CLI commands for extracting procedural skills from textbook documents."""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import typer

from ..config import CLIConfig
from ..output import console, print_error, print_json, print_success, print_warning
from ..utils import get_store

logger = logging.getLogger(__name__)

app = typer.Typer(help="Extract procedural skills from textbook documents")

SKILLS_DIR = Path.home() / ".kidkazz" / "skills"


def _safe_filename(doc_id: str) -> str:
    """Sanitize doc_id for safe use in filenames (prevent path traversal)."""
    return re.sub(r"[^\w\-]", "_", doc_id)[:100]


def _embedded_chunk_to_dict(ec) -> dict:
    """Convert an EmbeddedChunk into the dict shape skills.py expects."""
    c = ec.chunk if hasattr(ec, "chunk") else ec
    meta = c.metadata or {}
    return {
        "chunk_id": c.id,
        "content": c.content,
        "level": c.level,
        "prev_id": c.prev_id,
        "next_id": c.next_id,
        "header_text": meta.get("header_text"),
        "header_level": meta.get("header_level"),
        "semantic_type": meta.get("semantic_type"),
        "document_id": meta.get("document_id", ""),
    }


@app.command("extract")
def extract(
    doc_id: str = typer.Argument(..., help="Document ID to extract skills from"),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON path (default: ~/.kidkazz/skills/<doc_id>.json)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the JSON payload to stdout instead of saving",
    ),
    include_raw: bool = typer.Option(
        False,
        "--include-raw",
        help="Include raw concatenated markdown per skill (debugging)",
    ),
    enrich: bool = typer.Option(
        False,
        "--enrich",
        help="Call LLM to add name, goal, prerequisites, and common failures",
    ),
    provider: str = typer.Option(
        "openai/gpt-4o-mini",
        "--provider",
        "-p",
        help="LLM provider for enrichment (requires --enrich)",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Extraction profile (auto-detected from document if unset)",
    ),
) -> None:
    """Detect and assemble procedural skills from a document.

    Walks the document's chunks to find "How to" / "Steps to" / "Tutorial"
    sections, then re-parses the raw markdown to reconstruct step→code→output
    structure. Outputs JSON — no database writes.

    With --enrich, adds a single LLM call per skill to generate a canonical
    name, goal, prerequisites, and common failure modes.
    """
    from src.chunker.skills import extract_skills

    config = CLIConfig.load()
    store = get_store(config)

    # Fetch all chunks for the document
    chunks = store.get_document_chunks(doc_id)
    if not chunks:
        print_error(f"No chunks found for document '{doc_id}'")
        raise typer.Exit(1)

    chunk_dicts = [_embedded_chunk_to_dict(ec) for ec in chunks]
    skills = extract_skills(chunk_dicts)

    payload: dict = {
        "doc_id": doc_id,
        "skill_count": len(skills),
        "enriched": False,
        "skills": [],
    }

    if enrich and skills:
        _enrich_skills(payload, skills, store, doc_id, provider, profile, config)
    else:
        for s in skills:
            skill_dict = s.to_dict()
            if include_raw:
                skill_dict["raw_markdown"] = s.raw_markdown
            payload["skills"].append(skill_dict)

    if include_raw and enrich:
        # Enrichment path overwrote skills; re-attach raw markdown
        for skill_entry, source in zip(payload["skills"], skills, strict=False):
            skill_entry["raw_markdown"] = source.raw_markdown

    # Output
    if json_output:
        print_json(payload)
        return

    output_path = output or (SKILLS_DIR / f"{_safe_filename(doc_id)}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # Summary to stdout
    _print_summary(payload, output_path)


@app.command("show")
def show(
    doc_id: str = typer.Argument(..., help="Document ID"),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON"),
) -> None:
    """Show previously extracted skills for a document."""
    path = SKILLS_DIR / f"{_safe_filename(doc_id)}.json"
    if not path.exists():
        print_error(f"No skill extraction found for '{doc_id}'")
        print_warning(f"Run: kidkazz skills extract {doc_id}")
        raise typer.Exit(1)

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print_error(f"Corrupt skills file {path}: {e}")
        raise typer.Exit(1)

    if json_output:
        print_json(data)
        return

    _print_summary(data, path)


@app.command("list")
def list_extracted(
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON"),
) -> None:
    """List documents that have extracted skills on disk."""
    if not SKILLS_DIR.exists():
        print_warning("No extracted skills found.")
        return

    entries = []
    for path in sorted(SKILLS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            entries.append({
                "doc_id": data.get("doc_id", path.stem),
                "skill_count": data.get("skill_count", 0),
                "file": str(path),
            })
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue

    if json_output:
        print_json(entries)
        return

    if not entries:
        print_warning("No extracted skills found.")
        return

    from rich.table import Table

    table = Table(title="Extracted Skills", show_header=True, header_style="bold cyan")
    table.add_column("Document ID", style="cyan")
    table.add_column("Skills", justify="right")
    table.add_column("File", style="dim")
    for e in entries:
        table.add_row(e["doc_id"], str(e["skill_count"]), e["file"])
    console.print(table)


def _resolve_profile(profile_arg: Optional[str], doc_id: str, store, config) -> Optional[str]:
    """Resolve profile name: flag > document book_type > config default."""
    if profile_arg:
        return profile_arg
    docs = store.list_documents()
    doc = next((d for d in docs if d.get("doc_id") == doc_id), None)
    stored = (doc or {}).get("book_type", "")
    return stored or config.extraction_profile


def _enrich_skills(payload: dict, skills: list, store, doc_id: str,
                   provider: str, profile_arg: Optional[str], config) -> None:
    """Run LLM enrichment on the extracted skills, mutating `payload` in place."""
    try:
        from src.chunker.skill_synthesizer import SkillSynthesizer
    except ImportError as e:
        print_error(f"Missing dependency for --enrich: {e}")
        print_warning("Install with: pip install 'kidkazz[concepts]'")
        raise typer.Exit(1)

    # Resolve profile for domain-specific prompt
    profile_name = _resolve_profile(profile_arg, doc_id, store, config)
    extraction_profile = None
    if profile_name and profile_name != "general":
        try:
            from src.chunker.profiles import get_profile
            extraction_profile = get_profile(profile_name)
            logger.info("Using extraction profile: %s", profile_name)
        except ValueError:
            logger.warning("Unknown profile '%s', continuing without profile hints", profile_name)

    # Fetch known concepts once so the LLM can map prereqs to existing entries
    known_concepts: list[str] = []
    try:
        all_concepts = store.list_concepts(doc_id=doc_id)
        known_concepts = [c.get("name", "") for c in all_concepts if c.get("name")]
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not fetch known concepts: %s", e)

    synthesizer = SkillSynthesizer(provider=provider, profile=extraction_profile)
    enriched_entries = synthesizer.enrich_all(skills, known_concepts=known_concepts)

    payload["enriched"] = True
    payload["provider"] = provider
    payload["profile"] = profile_name
    payload["skills"] = enriched_entries


def _print_summary(payload: dict, output_path: Path) -> None:
    """Print a human-readable summary of extracted skills."""
    from rich.panel import Panel
    from rich.table import Table

    doc_id = payload.get("doc_id", "?")
    skills = payload.get("skills", [])

    console.print()
    console.print(Panel(
        f"[bold]{doc_id}[/bold] — {len(skills)} skill(s) extracted",
        title="Skill Extraction",
        border_style="green",
    ))

    if not skills:
        console.print("[yellow]No skills detected in this document.[/yellow]")
        console.print("[dim]Tip: skills require a 'How to' / 'Steps to' / "
                      "'Tutorial' header plus numbered steps or code fences.[/dim]")
        console.print()
        print_success(f"Saved to: {output_path}")
        return

    enriched = payload.get("enriched", False)

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    if enriched:
        table.add_column("Name", style="cyan")
        table.add_column("Difficulty", style="yellow")
    else:
        table.add_column("Anchor Header", style="cyan")
    table.add_column("Steps", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Has Code", justify="center")

    for i, skill in enumerate(skills, 1):
        boundary = skill.get("boundary", {})
        steps = skill.get("steps", [])
        meta = skill.get("metadata") or {}
        has_code = any(s.get("code_content") for s in steps)
        if enriched:
            table.add_row(
                str(i),
                meta.get("name") or boundary.get("anchor_header", "?"),
                meta.get("difficulty", "-"),
                str(len(steps)),
                str(len(boundary.get("chunk_ids", []))),
                "✓" if has_code else "",
            )
        else:
            table.add_row(
                str(i),
                boundary.get("anchor_header", "?"),
                str(len(steps)),
                str(len(boundary.get("chunk_ids", []))),
                "✓" if has_code else "",
            )

    console.print(table)

    if enriched:
        # Print per-skill enriched details
        for i, skill in enumerate(skills, 1):
            meta = skill.get("metadata")
            if not meta:
                continue
            console.print()
            console.print(f"[bold cyan]{i}. {meta.get('name', '?')}[/bold cyan]")
            if meta.get("goal"):
                console.print(f"  [dim]Goal:[/dim] {meta['goal']}")
            if meta.get("success_criteria"):
                console.print(f"  [dim]Success:[/dim] {meta['success_criteria']}")
            prereq = meta.get("prerequisite_concepts", [])
            if prereq:
                console.print(f"  [dim]Prerequisites:[/dim] {', '.join(prereq[:5])}")
            failures = meta.get("common_failures", [])
            if failures:
                console.print(f"  [dim]Common failures:[/dim] {len(failures)}")

    console.print()
    print_success(f"Saved to: {output_path}")
