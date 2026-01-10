"""Graph visualization for concept relationships."""

import json
import re
from pathlib import Path
from typing import Any, Optional

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

# Relationship type to style mapping
RELATION_STYLES = {
    "uses": {"style": "solid", "color": "#333333"},
    "requires": {"style": "dashed", "color": "#666666"},
    "calculated_from": {"style": "bold", "color": "#0066CC"},
    "component_of": {"style": "dotted", "color": "#009900"},
    "recorded_in": {"style": "solid", "color": "#990099"},
    "supersedes": {"style": "dashed", "color": "#CC0000"},
    "relates_to": {"style": "solid", "color": "#333333"},
}


def _sanitize_id(name: str) -> str:
    """Convert concept name to valid DOT node ID."""
    # Replace spaces and special chars with underscores
    sanitized = re.sub(r'[^\w]', '_', name)
    return sanitized.lower()


def concepts_to_dot(
    concepts: list[dict[str, Any]],
    relations: list[tuple[str, str, str]],  # (from_name, to_name, relation_type)
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

        # Create label with aliases (limit to 2)
        label = name
        if aliases:
            label += f"\\n({', '.join(aliases[:2])})"

        # Escape quotes in label
        label = label.replace('"', '\\"')
        node_id = _sanitize_id(name)

        lines.append(f'    "{node_id}" [label="{label}", fillcolor="{color}"];')

    lines.append("")

    # Add edges
    for from_name, to_name, relation_type in relations:
        from_id = _sanitize_id(from_name)
        to_id = _sanitize_id(to_name)
        style = RELATION_STYLES.get(relation_type, {"style": "solid", "color": "#333333"})

        lines.append(
            f'    "{from_id}" -> "{to_id}" '
            f'[label="{relation_type}", style={style["style"]}, color="{style["color"]}"];'
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
            cleanup=True,  # Remove intermediate DOT file
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
) -> tuple[str, Optional[Path]]:
    """
    Generate concept graph from stored concepts.

    Args:
        store: HelixChunkStore instance
        doc_id: Filter to concepts from specific document
        output_path: Path to save rendered image (optional)
        output_format: Output format (dot, png, svg, pdf)
        title: Graph title

    Returns:
        Tuple of (DOT content, rendered file path or None)
    """
    # Get concepts
    concepts = store.list_concepts(doc_id=doc_id)

    if not concepts:
        return 'digraph Empty { label="No concepts found"; }', None

    # Get relations by traversing each concept
    relations: list[tuple[str, str, str]] = []
    seen_relations: set[tuple[str, str]] = set()

    # Build a concept lookup by internal ID for resolving edge targets
    concept_lookup = {c.get("id"): c for c in concepts if c.get("id")}
    concept_lookup_by_slug = {c.get("concept_id"): c for c in concepts if c.get("concept_id")}

    for concept in concepts:
        concept_id = concept.get("concept_id")
        from_name = concept.get("name")

        # Try to get relations with types first
        if hasattr(store, 'get_related_concepts_with_types'):
            edges = store.get_related_concepts_with_types(concept_id)
            for edge in edges:
                # New format: {"concept": target_concept_dict, "relation_type": "uses"}
                relation_type = edge.get("relation_type", "relates_to")

                # Get target concept from 'concept' key (new format)
                # or fallback to old format for backwards compatibility
                target = edge.get("concept") or edge.get("to") or edge.get("To") or edge.get("target")
                if isinstance(target, dict):
                    to_name = target.get("name")
                elif isinstance(target, str):
                    # It's an ID - look up the concept
                    target_concept = concept_lookup.get(target) or concept_lookup_by_slug.get(target)
                    to_name = target_concept.get("name") if target_concept else None
                else:
                    to_name = None

                if from_name and to_name:
                    relation_key = (from_name, to_name)
                    if relation_key not in seen_relations:
                        seen_relations.add(relation_key)
                        relations.append((from_name, to_name, relation_type))
        else:
            # Fallback to legacy method without types
            related = store.get_related_concepts(concept_id)
            for rel_concept in related:
                to_name = rel_concept.get("name")
                relation_type = "relates_to"

                relation_key = (from_name, to_name)
                if relation_key not in seen_relations:
                    seen_relations.add(relation_key)
                    relations.append((from_name, to_name, relation_type))

    # Generate title
    if not title:
        if doc_id:
            title = f"Concepts from {doc_id}"
        else:
            title = "Knowledge Graph"

    # Generate DOT
    dot_content = concepts_to_dot(concepts, relations, title)

    # Render if output path specified and format is not DOT
    rendered_path = None
    if output_path:
        if output_format == "dot":
            # Just write DOT content
            output_path.write_text(dot_content)
            rendered_path = output_path
        else:
            rendered_path = render_graph(dot_content, output_path.with_suffix(""), output_format)

    return dot_content, rendered_path
