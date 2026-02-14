"""Graph visualization for cross-document concept relationships."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _get_graphviz():
    """Lazy import graphviz."""
    try:
        from graphviz import Digraph
        return Digraph
    except ImportError:
        return None


# Concept type to color mapping
CONCEPT_COLORS = {
    "term": "#4A90D9",       # Blue
    "method": "#50C878",     # Green
    "principle": "#FFB347",  # Orange
    "formula": "#DA70D6",    # Orchid
    "account": "#87CEEB",    # Sky blue
}

# Document cluster colors (cycle for >2 docs)
DOC_COLORS = [
    "#FFE4E1",  # Misty rose
    "#E0F0FF",  # Light blue
    "#E8F5E9",  # Light green
    "#FFF3E0",  # Light orange
    "#F3E5F5",  # Light purple
]


def _sanitize_id(name: str) -> str:
    """Convert concept name to valid DOT node ID."""
    sanitized = re.sub(r'[^\w]', '_', name)
    return sanitized.lower()


def _short_doc_name(doc_id: str) -> str:
    """Shorten document ID for display (first 3 meaningful words)."""
    # Remove common suffixes
    name = doc_id.replace("_Z-Library", "").replace("_z-library", "")
    parts = name.split("_")
    # Take first 3 non-trivial words
    words = [p for p in parts if len(p) > 1][:3]
    return " ".join(words) if words else doc_id[:30]


def _parse_source_documents(concept: dict) -> list[str]:
    """Extract source_documents list from concept dict."""
    raw = concept.get("source_documents", "[]")
    try:
        docs = json.loads(raw) if isinstance(raw, str) else raw
        return docs if isinstance(docs, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def concepts_to_dot(
    concepts: list[dict[str, Any]],
    relations: list[tuple[str, str, str]],
    title: str = "Concept Graph",
) -> str:
    """
    Generate DOT format string from concepts and relations.

    Args:
        concepts: List of concept dicts with name, concept_type
        relations: List of (from_concept, to_concept, relation_type) tuples
        title: Graph title

    Returns:
        DOT format string
    """
    lines = [
        "digraph ConceptGraph {",
        f'    label="{title}";',
        "    labelloc=t;",
        "    fontsize=16;",
        "    rankdir=LR;",
        "    node [shape=box, style=filled, fontname=Arial];",
        "    edge [fontname=Arial, fontsize=10];",
        "",
    ]

    # Add nodes
    for concept in concepts:
        name = concept.get("name", "")
        concept_type = concept.get("concept_type", "term")
        color = CONCEPT_COLORS.get(concept_type, "#CCCCCC")

        # Parse aliases
        aliases_raw = concept.get("aliases", "[]")
        try:
            aliases = json.loads(aliases_raw) if isinstance(aliases_raw, str) else aliases_raw
        except json.JSONDecodeError:
            aliases = []

        label = name
        if aliases:
            label += f"\\n({', '.join(aliases[:2])})"

        label = label.replace('"', '\\"')
        node_id = _sanitize_id(name)

        lines.append(f'    "{node_id}" [label="{label}", fillcolor="{color}"];')

    lines.append("")

    # Add edges
    for from_name, to_name, relation_type in relations:
        from_id = _sanitize_id(from_name)
        to_id = _sanitize_id(to_name)
        lines.append(
            f'    "{from_id}" -> "{to_id}" '
            f'[label="{relation_type}", style=solid, color="#333333"];'
        )

    lines.append("")

    # Add legend
    lines.extend([
        "    subgraph cluster_legend {",
        '        label="Legend";',
        "        fontsize=12;",
        "        style=rounded;",
        "        color=gray;",
    ])

    for concept_type, color in CONCEPT_COLORS.items():
        lines.append(
            f'        legend_{concept_type} [label="{concept_type}", fillcolor="{color}", shape=box];'
        )

    lines.append("    }")
    lines.append("}")

    return "\n".join(lines)


def cross_document_graph_to_dot(
    shared_concepts: list[dict[str, Any]],
    doc_ids: list[str],
    title: str = "Cross-Document Concept Graph",
) -> str:
    """
    Generate DOT for cross-document concept visualization.

    Shows shared concepts as bridge nodes between document clusters.

    Args:
        shared_concepts: Concepts appearing in 2+ documents
        doc_ids: All unique document IDs
        title: Graph title

    Returns:
        DOT format string
    """
    lines = [
        "digraph CrossDocumentGraph {",
        f'    label="{title}";',
        "    labelloc=t;",
        "    fontsize=28;",
        "    fontname=Arial;",
        "    rankdir=LR;",
        "    overlap=false;",
        "    splines=curved;",
        "    dpi=150;",
        "    nodesep=1.0;",
        "    ranksep=2.0;",
        "    pad=0.5;",
        "    node [fontname=Arial, fontsize=16, margin=\"0.3,0.15\"];",
        "    edge [fontname=Arial, fontsize=12];",
        "",
    ]

    # Create document hub nodes (large, prominent)
    doc_color_map = {}
    for i, doc_id in enumerate(sorted(doc_ids)):
        color = DOC_COLORS[i % len(DOC_COLORS)]
        doc_color_map[doc_id] = color
        short_name = _short_doc_name(doc_id)
        node_id = _sanitize_id(doc_id)
        lines.append(
            f'    "{node_id}" [label="{short_name}", shape=ellipse, '
            f'style="filled,bold", fillcolor="{color}", fontsize=22, '
            f'penwidth=3, width=3, height=1.5];'
        )

    lines.append("")

    # Group shared concepts by type for visual clarity
    by_type: dict[str, list[dict]] = {}
    for concept in shared_concepts:
        ct = concept.get("concept_type", "term")
        by_type.setdefault(ct, []).append(concept)

    # Add concept nodes and edges to documents
    for concept_type, concepts in by_type.items():
        color = CONCEPT_COLORS.get(concept_type, "#CCCCCC")

        for concept in concepts:
            name = concept.get("name", "")
            node_id = _sanitize_id(name)
            docs = _parse_source_documents(concept)
            doc_count = len(docs)

            fontsize = 14 + (doc_count - 1) * 2
            penwidth = 2 + (doc_count - 1)

            label = name.replace('"', '\\"')
            lines.append(
                f'    "{node_id}" [label="{label}\\n[{concept_type}]", '
                f'shape=box, style="filled,rounded", fillcolor="{color}", '
                f'fontsize={fontsize}, penwidth={penwidth}];'
            )

            # Edge from each source document to this concept
            for doc_id in docs:
                doc_node = _sanitize_id(doc_id)
                doc_color = doc_color_map.get(doc_id, "#333333")
                lines.append(
                    f'    "{doc_node}" -> "{node_id}" '
                    f'[color="{doc_color}", penwidth=6.0, arrowsize=1.5];'
                )

    lines.append("")

    # Add legend
    lines.extend([
        "    subgraph cluster_legend {",
        '        label="Concept Types";',
        "        fontsize=12;",
        "        style=rounded;",
        "        color=gray;",
    ])
    for concept_type, color in CONCEPT_COLORS.items():
        lines.append(
            f'        legend_{concept_type} [label="{concept_type}", '
            f'fillcolor="{color}", shape=box, style=filled];'
        )
    lines.append("    }")
    lines.append("}")

    return "\n".join(lines)


def render_graph(
    dot_content: str,
    output_path: Path,
    output_format: str = "png",
) -> Path:
    """
    Render DOT content to image file.

    Args:
        dot_content: DOT format string
        output_path: Output file path (without extension)
        output_format: Output format (png, svg, pdf)

    Returns:
        Path to rendered file

    Raises:
        ImportError: If graphviz not installed
        RuntimeError: If rendering fails
    """
    if _get_graphviz() is None:
        raise ImportError(
            "graphviz package not installed. "
            "Install with: pip install 'kidkazz[concepts]'"
        )

    try:
        from graphviz import Source

        source = Source(dot_content)
        output_file = source.render(
            filename=str(output_path),
            format=output_format,
            cleanup=True,
        )
        return Path(output_file)

    except Exception as e:
        raise RuntimeError(f"Graph rendering failed: {e}") from e


def generate_concept_graph(
    store: Any,
    doc_id: Optional[str] = None,
    output_path: Optional[Path] = None,
    output_format: str = "png",
    title: Optional[str] = None,
    min_docs: int = 2,
) -> tuple[str, Optional[Path]]:
    """
    Generate cross-document concept graph from stored concepts.

    Shows concepts shared between documents as bridge nodes, revealing
    cross-discipline connections (e.g., accounting + warehouse management).

    When --doc-id is specified, shows all concepts for that document
    grouped by type.

    Args:
        store: HelixChunkStore instance
        doc_id: Filter to concepts from specific document
        output_path: Path to save rendered image (optional)
        output_format: Output format (dot, png, svg, pdf)
        title: Graph title
        min_docs: Minimum number of documents a concept must appear in

    Returns:
        Tuple of (DOT content, rendered file path or None)
    """
    concepts = store.list_concepts(doc_id=doc_id)

    if not concepts:
        return 'digraph Empty { label="No concepts found"; }', None

    # Collect all unique document IDs and classify concepts
    all_doc_ids: set[str] = set()
    shared_concepts = []
    single_doc_concepts: dict[str, list[dict]] = {}

    for concept in concepts:
        docs = _parse_source_documents(concept)
        for d in docs:
            all_doc_ids.add(d)

        if len(docs) >= min_docs:
            shared_concepts.append(concept)
        elif len(docs) == 1:
            single_doc_concepts.setdefault(docs[0], []).append(concept)

    if doc_id:
        # Single-document view: show all concepts for this doc
        if not title:
            title = f"Concepts from {_short_doc_name(doc_id)}"
        dot_content = concepts_to_dot(concepts, [], title)
    elif shared_concepts:
        # Cross-document view: show shared concepts as bridges
        if not title:
            title = (
                f"Cross-Document Concepts "
                f"({len(shared_concepts)} shared across {len(all_doc_ids)} documents)"
            )
        dot_content = cross_document_graph_to_dot(
            shared_concepts, sorted(all_doc_ids), title
        )
    else:
        # No shared concepts
        if not title:
            title = "Knowledge Graph (no cross-document concepts found)"
        dot_content = concepts_to_dot(concepts, [], title)

    # Render if output path specified
    rendered_path = None
    if output_path:
        if output_format == "dot":
            output_path.write_text(dot_content)
            rendered_path = output_path
        else:
            rendered_path = render_graph(dot_content, output_path.with_suffix(""), output_format)

    return dot_content, rendered_path
