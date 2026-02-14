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


def _prepare_html_graph_data(
    concepts: list[dict[str, Any]],
    doc_id: Optional[str] = None,
    min_docs: int = 2,
) -> dict[str, Any]:
    """
    Transform concept data into JSON-serializable structure for vis-network.

    Args:
        concepts: List of concept dicts from store
        doc_id: If set, single-document view
        min_docs: Minimum documents for cross-doc view filtering

    Returns:
        Dict with nodes, edges, docIds, adjacency, viewMode
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    all_doc_ids: set[str] = set()
    seen_node_ids: set[str] = set()

    # Classify concepts and collect doc IDs
    for concept in concepts:
        docs = _parse_source_documents(concept)
        for d in docs:
            all_doc_ids.add(d)

    sorted_doc_ids = sorted(all_doc_ids)
    doc_color_map = {}
    for i, did in enumerate(sorted_doc_ids):
        doc_color_map[did] = DOC_COLORS[i % len(DOC_COLORS)]

    if doc_id:
        # Single-document view: concept nodes only, grouped by type
        for concept in concepts:
            node_id = _sanitize_id(concept.get("name", ""))
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)

            concept_type = concept.get("concept_type", "term")
            color = CONCEPT_COLORS.get(concept_type, "#CCCCCC")
            aliases_raw = concept.get("aliases", "[]")
            try:
                aliases = json.loads(aliases_raw) if isinstance(aliases_raw, str) else aliases_raw
            except (json.JSONDecodeError, TypeError):
                aliases = []

            nodes.append({
                "id": node_id,
                "label": concept.get("name", ""),
                "type": concept_type,
                "color": color,
                "definition": concept.get("definition", ""),
                "aliases": aliases if isinstance(aliases, list) else [],
                "docs": _parse_source_documents(concept),
                "isDoc": False,
                "size": 20,
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "docIds": sorted_doc_ids,
            "adjacency": {},
            "viewMode": "single_document",
        }

    # Cross-document view: document hubs + shared concept nodes + edges
    # Add document hub nodes
    for did in sorted_doc_ids:
        node_id = _sanitize_id(did)
        if node_id not in seen_node_ids:
            seen_node_ids.add(node_id)
            nodes.append({
                "id": node_id,
                "label": _short_doc_name(did),
                "type": "document",
                "color": doc_color_map.get(did, DOC_COLORS[0]),
                "definition": f"Document: {did}",
                "aliases": [],
                "docs": [did],
                "isDoc": True,
                "size": 40,
            })

    # Build adjacency matrix: count shared concepts between doc pairs
    adjacency: dict[str, int] = {}
    for concept in concepts:
        docs = _parse_source_documents(concept)
        if len(docs) < 2:
            continue
        # Count pairs
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                pair_key = "|".join(sorted([docs[i], docs[j]]))
                adjacency[pair_key] = adjacency.get(pair_key, 0) + 1

    # Add concept nodes and edges (all concepts, JS handles min_docs filtering)
    for concept in concepts:
        name = concept.get("name", "")
        node_id = _sanitize_id(name)
        if node_id in seen_node_ids:
            continue
        seen_node_ids.add(node_id)

        concept_type = concept.get("concept_type", "term")
        color = CONCEPT_COLORS.get(concept_type, "#CCCCCC")
        docs = _parse_source_documents(concept)
        aliases_raw = concept.get("aliases", "[]")
        try:
            aliases = json.loads(aliases_raw) if isinstance(aliases_raw, str) else aliases_raw
        except (json.JSONDecodeError, TypeError):
            aliases = []

        doc_count = len(docs)
        size = 15 + (doc_count - 1) * 5

        nodes.append({
            "id": node_id,
            "label": name,
            "type": concept_type,
            "color": color,
            "definition": concept.get("definition", ""),
            "aliases": aliases if isinstance(aliases, list) else [],
            "docs": docs,
            "isDoc": False,
            "size": size,
        })

        # Edges from each source document to this concept
        for d in docs:
            doc_node_id = _sanitize_id(d)
            edge_color = doc_color_map.get(d, "#999999")
            edges.append({
                "from": doc_node_id,
                "to": node_id,
                "color": edge_color,
                "width": 2,
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "docIds": sorted_doc_ids,
        "adjacency": adjacency,
        "viewMode": "cross_document",
    }


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; display: flex; height: 100vh; overflow: hidden; }}
  #sidebar {{ width: 320px; min-width: 320px; background: #16213e; padding: 16px; overflow-y: auto; border-right: 1px solid #0f3460; display: flex; flex-direction: column; gap: 16px; }}
  #sidebar h2 {{ color: #e94560; font-size: 16px; margin-bottom: 4px; }}
  #sidebar h3 {{ color: #0f3460; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; color: #888; }}
  #graph-container {{ flex: 1; position: relative; }}
  #network {{ width: 100%; height: 100%; }}
  #tooltip {{ position: absolute; display: none; background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 12px; max-width: 350px; font-size: 13px; pointer-events: none; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
  #tooltip .tt-name {{ color: #e94560; font-weight: bold; font-size: 15px; margin-bottom: 4px; }}
  #tooltip .tt-type {{ color: #888; font-size: 11px; text-transform: uppercase; margin-bottom: 6px; }}
  #tooltip .tt-def {{ color: #ccc; line-height: 1.4; margin-bottom: 6px; }}
  #tooltip .tt-aliases {{ color: #888; font-size: 12px; margin-bottom: 4px; }}
  #tooltip .tt-docs {{ color: #4A90D9; font-size: 12px; }}
  input[type="text"] {{ width: 100%; padding: 8px 12px; background: #1a1a2e; border: 1px solid #0f3460; border-radius: 6px; color: #e0e0e0; font-size: 14px; outline: none; }}
  input[type="text"]:focus {{ border-color: #e94560; }}
  input[type="range"] {{ width: 100%; accent-color: #e94560; }}
  .filter-group {{ background: #1a1a2e; border-radius: 8px; padding: 12px; }}
  .checkbox-row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; cursor: pointer; }}
  .checkbox-row input {{ accent-color: #e94560; }}
  .color-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
  .stats {{ font-size: 12px; color: #888; }}
  #adjacency-table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
  #adjacency-table th, #adjacency-table td {{ border: 1px solid #0f3460; padding: 4px 6px; text-align: center; }}
  #adjacency-table th {{ background: #0f3460; color: #e0e0e0; font-size: 10px; max-width: 80px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  #adjacency-table td {{ color: #ccc; }}
  #adjacency-table td.highlight {{ background: #e94560; color: #fff; font-weight: bold; }}
</style>
</head>
<body>
<div id="sidebar">
  <div>
    <h2>{title}</h2>
    <div class="stats" id="stats"></div>
  </div>

  <div class="filter-group">
    <h3>Search</h3>
    <input type="text" id="search-input" placeholder="Search concepts..." />
  </div>

  <div class="filter-group">
    <h3>Concept Types</h3>
    <div id="type-filters"></div>
  </div>

  <div class="filter-group">
    <h3>Min Documents: <span id="min-docs-label">{initial_min_docs}</span></h3>
    <input type="range" id="min-docs-slider" min="1" max="{max_docs}" value="{initial_min_docs}" />
  </div>

  <div class="filter-group" id="adjacency-section">
    <h3>Shared Concept Counts</h3>
    <div id="adjacency-container"></div>
  </div>
</div>

<div id="graph-container">
  <canvas id="network"></canvas>
  <div id="tooltip"></div>
</div>

<script>
(function() {{
  var GRAPH_DATA = {graph_data_json};
  var CONCEPT_COLORS = {concept_colors_json};
  var DOC_COLORS = {doc_colors_json};
  var allNodes = GRAPH_DATA.nodes;
  var allEdges = GRAPH_DATA.edges;
  var viewMode = GRAPH_DATA.viewMode;
  var adjacency = GRAPH_DATA.adjacency;
  var docIds = GRAPH_DATA.docIds;

  // Build vis datasets
  var visNodes = new vis.DataSet(allNodes.map(function(n) {{
    return {{
      id: n.id,
      label: n.label,
      color: {{ background: n.color, border: n.isDoc ? '#e94560' : darken(n.color, 30), highlight: {{ background: lighten(n.color, 20), border: '#e94560' }} }},
      shape: n.isDoc ? 'ellipse' : 'box',
      size: n.size,
      font: {{ color: '#e0e0e0', size: n.isDoc ? 18 : 13 }},
      borderWidth: n.isDoc ? 3 : 1,
      _data: n
    }};
  }}));

  var visEdges = new vis.DataSet(allEdges.map(function(e, i) {{
    return {{
      id: 'e' + i,
      from: e.from,
      to: e.to,
      color: {{ color: e.color, opacity: 0.6 }},
      width: e.width,
      smooth: {{ type: 'continuous' }},
      _data: e
    }};
  }}));

  var container = document.getElementById('graph-container');
  var network = new vis.Network(container, {{ nodes: visNodes, edges: visEdges }}, {{
    physics: {{
      enabled: true,
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {{ gravitationalConstant: -80, centralGravity: 0.01, springLength: 150, springConstant: 0.02, damping: 0.4 }},
      stabilization: {{ iterations: 200, fit: true }}
    }},
    interaction: {{ hover: true, tooltipDelay: 0, zoomView: true, dragView: true, navigationButtons: false }},
    nodes: {{ borderWidthSelected: 3 }},
    edges: {{ arrows: {{ to: {{ enabled: false }} }} }}
  }});

  // --- Tooltip ---
  var tooltip = document.getElementById('tooltip');
  network.on('hoverNode', function(params) {{
    var node = visNodes.get(params.node);
    if (!node || !node._data) return;
    var d = node._data;
    var html = '<div class="tt-name">' + escHtml(d.label) + '</div>';
    html += '<div class="tt-type">' + escHtml(d.type) + '</div>';
    if (d.definition) html += '<div class="tt-def">' + escHtml(d.definition) + '</div>';
    if (d.aliases && d.aliases.length) html += '<div class="tt-aliases">Aliases: ' + escHtml(d.aliases.join(', ')) + '</div>';
    if (d.docs && d.docs.length && !d.isDoc) html += '<div class="tt-docs">Documents: ' + d.docs.length + '</div>';
    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    var pos = params.event.center || {{ x: params.pointer.DOM.x, y: params.pointer.DOM.y }};
    tooltip.style.left = (pos.x + 15) + 'px';
    tooltip.style.top = (pos.y + 15) + 'px';
  }});
  network.on('blurNode', function() {{ tooltip.style.display = 'none'; }});
  network.on('dragStart', function() {{ tooltip.style.display = 'none'; }});

  // --- Click to highlight neighbors ---
  var highlightActive = false;
  network.on('click', function(params) {{
    if (params.nodes.length === 0) {{
      // Reset
      highlightActive = false;
      resetNodeOpacity();
      return;
    }}
    highlightActive = true;
    var selectedId = params.nodes[0];
    var connectedNodes = network.getConnectedNodes(selectedId);
    connectedNodes.push(selectedId);
    var connSet = new Set(connectedNodes);
    visNodes.forEach(function(node) {{
      if (connSet.has(node.id)) {{
        visNodes.update({{ id: node.id, opacity: 1.0 }});
      }} else {{
        visNodes.update({{ id: node.id, opacity: 0.15 }});
      }}
    }});
    visEdges.forEach(function(edge) {{
      var connected = connSet.has(edge.from) && connSet.has(edge.to);
      visEdges.update({{ id: edge.id, color: {{ color: edge._data.color, opacity: connected ? 0.8 : 0.05 }} }});
    }});
  }});

  function resetNodeOpacity() {{
    visNodes.forEach(function(node) {{
      visNodes.update({{ id: node.id, opacity: 1.0 }});
    }});
    visEdges.forEach(function(edge) {{
      visEdges.update({{ id: edge.id, color: {{ color: edge._data.color, opacity: 0.6 }} }});
    }});
  }}

  // --- Search ---
  var searchInput = document.getElementById('search-input');
  searchInput.addEventListener('input', function() {{ applyFilters(); }});
  searchInput.addEventListener('keydown', function(ev) {{
    if (ev.key === 'Enter') {{
      var q = searchInput.value.trim().toLowerCase();
      if (!q) return;
      // Zoom to first match
      for (var i = 0; i < allNodes.length; i++) {{
        var n = allNodes[i];
        if (matchesSearch(n, q)) {{
          network.focus(n.id, {{ scale: 1.5, animation: true }});
          network.selectNodes([n.id]);
          break;
        }}
      }}
    }}
  }});

  function matchesSearch(n, q) {{
    if (n.label.toLowerCase().indexOf(q) >= 0) return true;
    if (n.aliases) {{
      for (var i = 0; i < n.aliases.length; i++) {{
        if (n.aliases[i].toLowerCase().indexOf(q) >= 0) return true;
      }}
    }}
    return false;
  }}

  // --- Type filters ---
  var typeFiltersEl = document.getElementById('type-filters');
  var activeTypes = new Set();
  var allTypes = new Set();
  allNodes.forEach(function(n) {{ if (!n.isDoc) allTypes.add(n.type); }});
  allTypes.forEach(function(t) {{ activeTypes.add(t); }});

  allTypes.forEach(function(t) {{
    var row = document.createElement('label');
    row.className = 'checkbox-row';
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.dataset.type = t;
    cb.addEventListener('change', function() {{
      if (this.checked) activeTypes.add(t); else activeTypes.delete(t);
      applyFilters();
    }});
    var dot = document.createElement('span');
    dot.className = 'color-dot';
    dot.style.background = CONCEPT_COLORS[t] || '#CCCCCC';
    var lbl = document.createElement('span');
    lbl.textContent = t;
    row.appendChild(cb);
    row.appendChild(dot);
    row.appendChild(lbl);
    typeFiltersEl.appendChild(row);
  }});

  // --- Min docs slider ---
  var minDocsSlider = document.getElementById('min-docs-slider');
  var minDocsLabel = document.getElementById('min-docs-label');
  minDocsSlider.addEventListener('input', function() {{
    minDocsLabel.textContent = this.value;
    applyFilters();
  }});

  // --- Unified filter ---
  function applyFilters() {{
    var q = searchInput.value.trim().toLowerCase();
    var minDocs = parseInt(minDocsSlider.value, 10);
    var visibleCount = 0;
    var docCount = 0;

    visNodes.forEach(function(node) {{
      var d = node._data;
      var visible = true;

      if (d.isDoc) {{
        // Document hubs always visible in cross-doc mode
        docCount++;
      }} else {{
        // Type filter
        if (!activeTypes.has(d.type)) visible = false;
        // Min docs filter
        if (d.docs && d.docs.length < minDocs) visible = false;
        // Search filter (dims, doesn't hide)
        if (visible) visibleCount++;
      }}

      visNodes.update({{ id: node.id, hidden: !visible }});
    }});

    // Search dimming (applied on top of visibility)
    if (q) {{
      visNodes.forEach(function(node) {{
        if (node.hidden) return;
        var d = node._data;
        if (d.isDoc) return;
        var matches = matchesSearch(d, q);
        visNodes.update({{ id: node.id, opacity: matches ? 1.0 : 0.15 }});
      }});
    }} else if (!highlightActive) {{
      visNodes.forEach(function(node) {{
        if (!node.hidden) visNodes.update({{ id: node.id, opacity: 1.0 }});
      }});
    }}

    // Update edges visibility
    visEdges.forEach(function(edge) {{
      var fromNode = visNodes.get(edge.from);
      var toNode = visNodes.get(edge.to);
      var eVisible = fromNode && !fromNode.hidden && toNode && !toNode.hidden;
      visEdges.update({{ id: edge.id, hidden: !eVisible }});
    }});

    // Update stats
    document.getElementById('stats').textContent = 'Showing ' + visibleCount + ' of ' + (allNodes.length - docIds.length) + ' concepts, ' + docCount + ' documents';
  }}

  // --- Adjacency matrix ---
  if (viewMode === 'cross_document' && docIds.length > 1) {{
    var table = document.createElement('table');
    table.id = 'adjacency-table';
    // Header row
    var thead = document.createElement('tr');
    var thEmpty = document.createElement('th');
    thEmpty.textContent = '';
    thead.appendChild(thEmpty);
    for (var i = 0; i < docIds.length; i++) {{
      var th = document.createElement('th');
      th.textContent = shortDocName(docIds[i]);
      th.title = docIds[i];
      thead.appendChild(th);
    }}
    table.appendChild(thead);
    // Data rows
    for (var r = 0; r < docIds.length; r++) {{
      var tr = document.createElement('tr');
      var thRow = document.createElement('th');
      thRow.textContent = shortDocName(docIds[r]);
      thRow.title = docIds[r];
      tr.appendChild(thRow);
      for (var c = 0; c < docIds.length; c++) {{
        var td = document.createElement('td');
        if (r === c) {{
          td.textContent = '-';
        }} else {{
          var key = [docIds[r], docIds[c]].sort().join('|');
          var count = adjacency[key] || 0;
          td.textContent = count;
          if (count > 0) td.className = 'highlight';
        }}
        tr.appendChild(td);
      }}
      table.appendChild(tr);
    }}
    document.getElementById('adjacency-container').appendChild(table);
  }} else {{
    document.getElementById('adjacency-section').style.display = 'none';
  }}

  // Initial filter
  applyFilters();

  // --- Helpers ---
  function escHtml(s) {{ var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }}
  function shortDocName(id) {{
    var n = id.replace(/_Z-Library/gi, '').replace(/_z-library/gi, '');
    var parts = n.split('_').filter(function(p) {{ return p.length > 1; }});
    return parts.slice(0, 3).join(' ') || id.substring(0, 30);
  }}
  function darken(hex, pct) {{
    var r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    r = Math.max(0, Math.floor(r * (1 - pct/100))); g = Math.max(0, Math.floor(g * (1 - pct/100))); b = Math.max(0, Math.floor(b * (1 - pct/100)));
    return '#' + ((1<<24)+(r<<16)+(g<<8)+b).toString(16).slice(1);
  }}
  function lighten(hex, pct) {{
    var r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    r = Math.min(255, Math.floor(r + (255-r)*pct/100)); g = Math.min(255, Math.floor(g + (255-g)*pct/100)); b = Math.min(255, Math.floor(b + (255-b)*pct/100));
    return '#' + ((1<<24)+(r<<16)+(g<<8)+b).toString(16).slice(1);
  }}
}})();
<{script_close}>
</body>
</html>"""


def generate_html_graph(
    concepts: list[dict[str, Any]],
    output_path: Path,
    title: str = "Concept Graph",
    doc_id: Optional[str] = None,
    min_docs: int = 2,
) -> Path:
    """
    Generate interactive HTML concept graph using vis-network.

    Args:
        concepts: List of concept dicts from store
        output_path: Output HTML file path
        title: Graph title
        doc_id: If set, single-document view
        min_docs: Initial min-docs slider value

    Returns:
        Path to generated HTML file
    """
    data = _prepare_html_graph_data(concepts, doc_id=doc_id, min_docs=min_docs)

    # Calculate max docs for slider
    max_docs = 1
    for node in data["nodes"]:
        if not node.get("isDoc"):
            max_docs = max(max_docs, len(node.get("docs", [])))

    graph_data_json = json.dumps(data, ensure_ascii=False)
    # Escape </script> to prevent injection
    graph_data_json = graph_data_json.replace("</script>", "<\\/script>")

    concept_colors_json = json.dumps(CONCEPT_COLORS)
    doc_colors_json = json.dumps(DOC_COLORS)

    html = _HTML_TEMPLATE.replace("{script_close}", "/script>")
    html = html.format(
        title=title.replace('"', '&quot;').replace('<', '&lt;'),
        graph_data_json=graph_data_json,
        concept_colors_json=concept_colors_json,
        doc_colors_json=doc_colors_json,
        max_docs=max(max_docs, 2),
        initial_min_docs=min(min_docs, max(max_docs, 2)),
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path


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
        output_format: Output format (dot, png, svg, pdf, html)
        title: Graph title
        min_docs: Minimum number of documents a concept must appear in

    Returns:
        Tuple of (DOT content or HTML path string, rendered file path or None)
    """
    concepts = store.list_concepts(doc_id=doc_id)

    if not concepts:
        if output_format == "html":
            return "", None
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

    # Compute title before format dispatch
    if not title:
        if doc_id:
            title = f"Concepts from {_short_doc_name(doc_id)}"
        elif shared_concepts:
            title = (
                f"Cross-Document Concepts "
                f"({len(shared_concepts)} shared across {len(all_doc_ids)} documents)"
            )
        else:
            title = "Knowledge Graph (no cross-document concepts found)"

    # HTML format: interactive vis-network graph
    if output_format == "html":
        if output_path is None:
            return "", None
        rendered_path = generate_html_graph(
            concepts=concepts,
            output_path=output_path,
            title=title,
            doc_id=doc_id,
            min_docs=min_docs,
        )
        return str(rendered_path), rendered_path

    # DOT-based formats
    if doc_id:
        dot_content = concepts_to_dot(concepts, [], title)
    elif shared_concepts:
        dot_content = cross_document_graph_to_dot(
            shared_concepts, sorted(all_doc_ids), title
        )
    else:
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
