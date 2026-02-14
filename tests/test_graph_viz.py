"""Tests for graph visualization module."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestSanitizeId:
    """Tests for _sanitize_id function."""

    def test_sanitize_simple_name(self):
        """Simple names should be lowercased."""
        from src.cli.graph_viz import _sanitize_id

        assert _sanitize_id("FIFO") == "fifo"

    def test_sanitize_name_with_spaces(self):
        """Spaces should become underscores."""
        from src.cli.graph_viz import _sanitize_id

        assert _sanitize_id("Cost of Goods Sold") == "cost_of_goods_sold"

    def test_sanitize_name_with_special_chars(self):
        """Special characters should become underscores."""
        from src.cli.graph_viz import _sanitize_id

        assert _sanitize_id("A/B Testing") == "a_b_testing"
        assert _sanitize_id("Rate (%)") == "rate____"

    def test_sanitize_name_with_parentheses(self):
        """Parentheses should become underscores."""
        from src.cli.graph_viz import _sanitize_id

        assert _sanitize_id("FIFO (First-In)") == "fifo__first_in_"

    def test_sanitize_empty_string(self):
        """Empty string should return empty string."""
        from src.cli.graph_viz import _sanitize_id

        assert _sanitize_id("") == ""


class TestConceptsToDot:
    """Tests for concepts_to_dot function."""

    def test_generates_valid_dot_header(self):
        """Should generate proper DOT digraph header."""
        from src.cli.graph_viz import concepts_to_dot

        concepts = [{"name": "Test", "concept_type": "term", "aliases": "[]"}]
        dot = concepts_to_dot(concepts, [], "Test Graph")

        assert "digraph ConceptGraph {" in dot
        assert 'label="Test Graph"' in dot
        assert "rankdir=LR" in dot

    def test_generates_nodes_for_concepts(self):
        """Should generate node for each concept."""
        from src.cli.graph_viz import concepts_to_dot

        concepts = [
            {"name": "FIFO", "concept_type": "method", "aliases": "[]"},
            {"name": "LIFO", "concept_type": "method", "aliases": "[]"},
        ]
        dot = concepts_to_dot(concepts, [])

        assert '"fifo"' in dot
        assert '"lifo"' in dot
        assert 'label="FIFO"' in dot
        assert 'label="LIFO"' in dot

    def test_applies_color_by_concept_type(self):
        """Should apply color based on concept type."""
        from src.cli.graph_viz import concepts_to_dot, CONCEPT_COLORS

        concepts = [
            {"name": "FIFO", "concept_type": "method", "aliases": "[]"},
            {"name": "Cost", "concept_type": "formula", "aliases": "[]"},
        ]
        dot = concepts_to_dot(concepts, [])

        assert f'fillcolor="{CONCEPT_COLORS["method"]}"' in dot
        assert f'fillcolor="{CONCEPT_COLORS["formula"]}"' in dot

    def test_generates_edges_for_relations(self):
        """Should generate edges for relationships."""
        from src.cli.graph_viz import concepts_to_dot

        concepts = [
            {"name": "COGS", "concept_type": "formula", "aliases": "[]"},
            {"name": "FIFO", "concept_type": "method", "aliases": "[]"},
        ]
        relations = [("COGS", "FIFO", "uses")]
        dot = concepts_to_dot(concepts, relations)

        assert '"cogs" -> "fifo"' in dot
        assert 'label="uses"' in dot

    def test_includes_aliases_in_label(self):
        """Should include aliases in node label."""
        from src.cli.graph_viz import concepts_to_dot

        concepts = [
            {
                "name": "Cost of Goods Sold",
                "concept_type": "formula",
                "aliases": json.dumps(["COGS", "Cost of Sales"]),
            },
        ]
        dot = concepts_to_dot(concepts, [])

        assert "COGS, Cost of Sales" in dot

    def test_limits_aliases_to_two(self):
        """Should limit displayed aliases to 2."""
        from src.cli.graph_viz import concepts_to_dot

        concepts = [
            {
                "name": "Test",
                "concept_type": "term",
                "aliases": json.dumps(["A", "B", "C", "D"]),
            },
        ]
        dot = concepts_to_dot(concepts, [])

        # Should only show first 2 aliases
        assert "A, B" in dot
        assert "C" not in dot.split('"test"')[1].split(";")[0]  # Not in the node definition

    def test_empty_concepts_list(self):
        """Should handle empty concepts list."""
        from src.cli.graph_viz import concepts_to_dot

        dot = concepts_to_dot([], [])

        assert "digraph ConceptGraph {" in dot
        assert "}" in dot

    def test_includes_legend(self):
        """Should include legend subgraph."""
        from src.cli.graph_viz import concepts_to_dot

        concepts = [{"name": "Test", "concept_type": "term", "aliases": "[]"}]
        dot = concepts_to_dot(concepts, [])

        assert "subgraph cluster_legend" in dot
        assert 'label="Legend"' in dot
        assert "legend_term" in dot
        assert "legend_method" in dot

    def test_escapes_quotes_in_names(self):
        """Should escape quotes in concept names."""
        from src.cli.graph_viz import concepts_to_dot

        concepts = [
            {"name": 'Test "Quoted"', "concept_type": "term", "aliases": "[]"},
        ]
        dot = concepts_to_dot(concepts, [])

        # Should escape the quotes
        assert '\\"' in dot or "Test" in dot


class TestConceptColors:
    """Tests for CONCEPT_COLORS constant."""

    def test_has_all_concept_types(self):
        """Should have colors for all concept types."""
        from src.cli.graph_viz import CONCEPT_COLORS

        expected_types = [
            "term", "method", "principle", "formula", "account",
            "metric", "framework", "process", "model", "tool",
            "theory", "system", "practice", "parameter", "procedure", "function",
        ]
        for concept_type in expected_types:
            assert concept_type in CONCEPT_COLORS, f"Missing color for {concept_type}"

    def test_colors_are_valid_hex(self):
        """Colors should be valid hex codes."""
        from src.cli.graph_viz import CONCEPT_COLORS

        for concept_type, color in CONCEPT_COLORS.items():
            assert color.startswith("#"), f"Invalid color for {concept_type}"
            assert len(color) == 7, f"Color for {concept_type} should be #RRGGBB"


class TestRenderGraph:
    """Tests for render_graph function."""

    def test_raises_import_error_without_graphviz(self):
        """Should raise ImportError if graphviz not installed."""
        from src.cli.graph_viz import render_graph

        with patch("src.cli.graph_viz._get_graphviz", return_value=None):
            with pytest.raises(ImportError) as exc_info:
                render_graph("digraph {}", Path("/tmp/test"))

            assert "graphviz" in str(exc_info.value).lower()

    def test_render_with_graphviz(self, tmp_path):
        """Should render graph when graphviz is available."""
        import sys
        from src.cli.graph_viz import render_graph

        # Mock graphviz Source
        mock_source_instance = MagicMock()
        output_file = tmp_path / "graph.png"
        mock_source_instance.render.return_value = str(output_file)

        mock_graphviz = MagicMock()
        mock_graphviz.Source.return_value = mock_source_instance
        mock_graphviz.Digraph = MagicMock()

        with patch.dict(sys.modules, {"graphviz": mock_graphviz}):
            result = render_graph("digraph {}", tmp_path / "graph")

        mock_source_instance.render.assert_called_once()
        assert result == output_file

    def test_raises_runtime_error_on_failure(self, tmp_path):
        """Should raise RuntimeError on render failure."""
        import sys
        from src.cli.graph_viz import render_graph

        mock_source_instance = MagicMock()
        mock_source_instance.render.side_effect = Exception("Render failed")

        mock_graphviz = MagicMock()
        mock_graphviz.Source.return_value = mock_source_instance
        mock_graphviz.Digraph = MagicMock()

        with patch.dict(sys.modules, {"graphviz": mock_graphviz}):
            with pytest.raises(RuntimeError) as exc_info:
                render_graph("digraph {}", tmp_path / "graph")

            assert "Render failed" in str(exc_info.value)


class TestGenerateConceptGraph:
    """Tests for generate_concept_graph function."""

    def test_returns_empty_graph_when_no_concepts(self):
        """Should return empty graph message when no concepts."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = []

        dot, path = generate_concept_graph(mock_store)

        assert "No concepts found" in dot
        assert path is None

    def test_generates_dot_from_store_concepts_single_doc(self):
        """Should generate DOT for single-doc concepts (no cross-doc view)."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = [
            {"concept_id": "fifo", "name": "FIFO", "concept_type": "method", "aliases": "[]",
             "source_documents": '["book_a"]'},
            {"concept_id": "lifo", "name": "LIFO", "concept_type": "method", "aliases": "[]",
             "source_documents": '["book_a"]'},
        ]

        dot, path = generate_concept_graph(mock_store)

        assert "digraph" in dot
        assert "fifo" in dot.lower()
        assert "lifo" in dot.lower()
        assert path is None

    def test_cross_document_shared_concepts(self):
        """Should show shared concepts as bridges between documents."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = [
            {"concept_id": "safety-stock", "name": "Safety Stock", "concept_type": "term",
             "aliases": "[]", "source_documents": '["book_accounting", "book_warehouse"]'},
            {"concept_id": "cogs", "name": "COGS", "concept_type": "formula",
             "aliases": "[]", "source_documents": '["book_accounting"]'},
        ]

        dot, path = generate_concept_graph(mock_store)

        assert "CrossDocumentGraph" in dot
        assert "safety_stock" in dot.lower()
        # Shared concept should have edges to both documents
        assert "book_accounting" in dot.lower().replace("-", "_").replace(" ", "_")
        assert "book_warehouse" in dot.lower().replace("-", "_").replace(" ", "_")

    def test_filters_by_doc_id(self):
        """Should filter concepts by doc_id."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = []

        generate_concept_graph(mock_store, doc_id="inventory")

        mock_store.list_concepts.assert_called_once_with(doc_id="inventory")

    def test_writes_dot_file_when_format_is_dot(self, tmp_path):
        """Should write DOT file when format is 'dot'."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = [
            {"concept_id": "test", "name": "Test", "concept_type": "term", "aliases": "[]",
             "source_documents": '["doc_a"]'},
        ]

        output_path = tmp_path / "graph.dot"
        dot, path = generate_concept_graph(
            mock_store,
            output_path=output_path,
            output_format="dot",
        )

        assert path == output_path
        assert output_path.exists()
        assert "digraph" in output_path.read_text()

    def test_uses_custom_title(self):
        """Should use custom title when provided."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = [
            {"concept_id": "test", "name": "Test", "concept_type": "term", "aliases": "[]",
             "source_documents": '["doc_a"]'},
        ]

        dot, _ = generate_concept_graph(mock_store, title="Custom Title")

        assert 'label="Custom Title"' in dot

    def test_auto_title_with_doc_id(self):
        """Should auto-generate title from doc_id."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = [
            {"concept_id": "test", "name": "Test", "concept_type": "term", "aliases": "[]",
             "source_documents": '["inventory_textbook"]'},
        ]

        dot, _ = generate_concept_graph(mock_store, doc_id="inventory_textbook")

        assert "inventory" in dot.lower()

    def test_auto_title_without_doc_id(self):
        """Should use 'Knowledge Graph' as default title when no shared concepts."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = [
            {"concept_id": "test", "name": "Test", "concept_type": "term", "aliases": "[]",
             "source_documents": '["doc_a"]'},
        ]

        dot, _ = generate_concept_graph(mock_store)

        assert "Knowledge Graph" in dot


# --- HTML graph tests ---

# Shared test fixtures
def _make_concepts():
    """Create sample concepts for testing."""
    return [
        {
            "concept_id": "safety-stock",
            "name": "Safety Stock",
            "concept_type": "term",
            "definition": "Extra inventory held as buffer",
            "aliases": json.dumps(["Buffer Stock", "Reserve"]),
            "source_documents": json.dumps(["book_accounting", "book_warehouse"]),
        },
        {
            "concept_id": "fifo",
            "name": "FIFO",
            "concept_type": "method",
            "definition": "First-In, First-Out inventory method",
            "aliases": json.dumps(["First-In First-Out"]),
            "source_documents": json.dumps(["book_accounting", "book_warehouse"]),
        },
        {
            "concept_id": "cogs",
            "name": "COGS",
            "concept_type": "formula",
            "definition": "Cost of Goods Sold",
            "aliases": "[]",
            "source_documents": json.dumps(["book_accounting"]),
        },
        {
            "concept_id": "wms",
            "name": "WMS",
            "concept_type": "term",
            "definition": "Warehouse Management System",
            "aliases": "[]",
            "source_documents": json.dumps(["book_warehouse"]),
        },
    ]


class TestPrepareHtmlGraphData:
    """Tests for _prepare_html_graph_data function."""

    def test_cross_doc_view_has_document_hub_nodes(self):
        """Should create document hub nodes in cross-document view."""
        from src.cli.graph_viz import _prepare_html_graph_data

        data = _prepare_html_graph_data(_make_concepts())

        doc_nodes = [n for n in data["nodes"] if n["isDoc"]]
        assert len(doc_nodes) == 2
        labels = {n["label"] for n in doc_nodes}
        # _short_doc_name strips _Z-Library and takes first 3 words
        assert any("book" in l.lower() for l in labels)

    def test_cross_doc_view_has_concept_nodes(self):
        """Should create concept nodes."""
        from src.cli.graph_viz import _prepare_html_graph_data

        data = _prepare_html_graph_data(_make_concepts())

        concept_nodes = [n for n in data["nodes"] if not n["isDoc"]]
        names = {n["label"] for n in concept_nodes}
        assert "Safety Stock" in names
        assert "FIFO" in names
        assert "COGS" in names
        assert "WMS" in names

    def test_cross_doc_view_has_edges(self):
        """Should create edges from document hubs to concepts."""
        from src.cli.graph_viz import _prepare_html_graph_data

        data = _prepare_html_graph_data(_make_concepts())

        assert len(data["edges"]) > 0
        # Each shared concept (2 docs) should have 2 edges
        safety_stock_edges = [e for e in data["edges"] if e["to"] == "safety_stock"]
        assert len(safety_stock_edges) == 2

    def test_adjacency_matrix_counts_shared_concepts(self):
        """Should count shared concepts between document pairs."""
        from src.cli.graph_viz import _prepare_html_graph_data

        data = _prepare_html_graph_data(_make_concepts())

        adjacency = data["adjacency"]
        pair_key = "|".join(sorted(["book_accounting", "book_warehouse"]))
        # Safety Stock + FIFO = 2 shared concepts
        assert adjacency[pair_key] == 2

    def test_single_doc_view_no_edges(self):
        """Single-document view should have no edges."""
        from src.cli.graph_viz import _prepare_html_graph_data

        data = _prepare_html_graph_data(_make_concepts(), doc_id="book_accounting")

        assert data["viewMode"] == "single_document"
        assert len(data["edges"]) == 0
        assert len(data["adjacency"]) == 0

    def test_single_doc_view_has_concept_nodes_only(self):
        """Single-document view should not have document hub nodes."""
        from src.cli.graph_viz import _prepare_html_graph_data

        data = _prepare_html_graph_data(_make_concepts(), doc_id="book_accounting")

        doc_nodes = [n for n in data["nodes"] if n["isDoc"]]
        assert len(doc_nodes) == 0

    def test_cross_doc_view_mode(self):
        """Cross-document view should set viewMode correctly."""
        from src.cli.graph_viz import _prepare_html_graph_data

        data = _prepare_html_graph_data(_make_concepts())

        assert data["viewMode"] == "cross_document"

    def test_doc_ids_are_sorted(self):
        """docIds should be sorted."""
        from src.cli.graph_viz import _prepare_html_graph_data

        data = _prepare_html_graph_data(_make_concepts())

        assert data["docIds"] == sorted(data["docIds"])

    def test_no_duplicate_nodes(self):
        """Should not create duplicate node IDs."""
        from src.cli.graph_viz import _prepare_html_graph_data

        # Create concepts with potentially colliding names
        concepts = [
            {
                "name": "Test Concept",
                "concept_type": "term",
                "definition": "A test",
                "aliases": "[]",
                "source_documents": json.dumps(["doc_a"]),
            },
            {
                "name": "Test Concept",  # Duplicate
                "concept_type": "term",
                "definition": "A test duplicate",
                "aliases": "[]",
                "source_documents": json.dumps(["doc_b"]),
            },
        ]

        data = _prepare_html_graph_data(concepts)

        ids = [n["id"] for n in data["nodes"]]
        assert len(ids) == len(set(ids)), "Duplicate node IDs found"

    def test_node_colors_match_concept_type(self):
        """Concept nodes should have colors matching their type."""
        from src.cli.graph_viz import _prepare_html_graph_data, CONCEPT_COLORS

        data = _prepare_html_graph_data(_make_concepts())

        for node in data["nodes"]:
            if not node["isDoc"] and node["type"] in CONCEPT_COLORS:
                assert node["color"] == CONCEPT_COLORS[node["type"]]

    def test_shared_concept_size_larger(self):
        """Concepts in more docs should have larger size."""
        from src.cli.graph_viz import _prepare_html_graph_data

        data = _prepare_html_graph_data(_make_concepts())

        concept_nodes = {n["label"]: n for n in data["nodes"] if not n["isDoc"]}
        # Safety Stock (2 docs) should be bigger than COGS (1 doc)
        assert concept_nodes["Safety Stock"]["size"] > concept_nodes["COGS"]["size"]

    def test_empty_concepts_list(self):
        """Should handle empty concepts list."""
        from src.cli.graph_viz import _prepare_html_graph_data

        data = _prepare_html_graph_data([])

        assert data["nodes"] == []
        assert data["edges"] == []
        assert data["docIds"] == []

    def test_aliases_parsed_correctly(self):
        """Aliases should be parsed from JSON string."""
        from src.cli.graph_viz import _prepare_html_graph_data

        data = _prepare_html_graph_data(_make_concepts())

        concept_nodes = {n["label"]: n for n in data["nodes"] if not n["isDoc"]}
        assert "Buffer Stock" in concept_nodes["Safety Stock"]["aliases"]
        assert "Reserve" in concept_nodes["Safety Stock"]["aliases"]


class TestGenerateHtmlGraph:
    """Tests for generate_html_graph function."""

    def test_generates_valid_html_file(self, tmp_path):
        """Should generate a valid HTML file."""
        from src.cli.graph_viz import generate_html_graph

        output = tmp_path / "graph.html"
        result = generate_html_graph(_make_concepts(), output)

        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content

    def test_contains_vis_network_cdn(self, tmp_path):
        """Should include vis-network CDN script."""
        from src.cli.graph_viz import generate_html_graph

        output = tmp_path / "graph.html"
        generate_html_graph(_make_concepts(), output)

        content = output.read_text()
        assert "vis-network" in content
        assert "unpkg.com" in content

    def test_contains_embedded_json_data(self, tmp_path):
        """Should embed concept data as JSON."""
        from src.cli.graph_viz import generate_html_graph

        output = tmp_path / "graph.html"
        generate_html_graph(_make_concepts(), output)

        content = output.read_text()
        assert "GRAPH_DATA" in content
        assert "Safety Stock" in content
        assert "FIFO" in content

    def test_contains_search_input(self, tmp_path):
        """Should include search input element."""
        from src.cli.graph_viz import generate_html_graph

        output = tmp_path / "graph.html"
        generate_html_graph(_make_concepts(), output)

        content = output.read_text()
        assert 'id="search-input"' in content

    def test_contains_type_filters(self, tmp_path):
        """Should include type filter section."""
        from src.cli.graph_viz import generate_html_graph

        output = tmp_path / "graph.html"
        generate_html_graph(_make_concepts(), output)

        content = output.read_text()
        assert 'id="type-filters"' in content
        assert "Concept Types" in content

    def test_contains_adjacency_section(self, tmp_path):
        """Should include adjacency matrix section."""
        from src.cli.graph_viz import generate_html_graph

        output = tmp_path / "graph.html"
        generate_html_graph(_make_concepts(), output)

        content = output.read_text()
        assert 'id="adjacency-section"' in content
        assert "Shared Concept Counts" in content

    def test_contains_title(self, tmp_path):
        """Should include the custom title."""
        from src.cli.graph_viz import generate_html_graph

        output = tmp_path / "graph.html"
        generate_html_graph(_make_concepts(), output, title="My Custom Graph")

        content = output.read_text()
        assert "My Custom Graph" in content

    def test_escapes_special_chars_in_title(self, tmp_path):
        """Should escape special characters in title."""
        from src.cli.graph_viz import generate_html_graph

        output = tmp_path / "graph.html"
        generate_html_graph(_make_concepts(), output, title='Test <script>"alert"</script>')

        content = output.read_text()
        # Should not contain unescaped script tags in the title
        assert '<script>"alert"</script>' not in content.split("var GRAPH_DATA")[0]

    def test_escapes_script_close_in_json(self, tmp_path):
        """Should escape </script> in embedded JSON data."""
        from src.cli.graph_viz import generate_html_graph

        concepts = [
            {
                "name": "Test",
                "concept_type": "term",
                "definition": "Has </script> in it",
                "aliases": "[]",
                "source_documents": json.dumps(["doc_a"]),
            },
        ]

        output = tmp_path / "graph.html"
        generate_html_graph(concepts, output)

        content = output.read_text()
        # Should have exactly 2 </script> tags: CDN script + main script
        # The one from JSON data should be escaped as <\/script>
        assert content.count("</script>") == 2
        assert "<\\/script>" in content  # Escaped version in JSON

    def test_min_docs_slider_value(self, tmp_path):
        """Should set initial min-docs slider value."""
        from src.cli.graph_viz import generate_html_graph

        output = tmp_path / "graph.html"
        generate_html_graph(_make_concepts(), output, min_docs=3)

        content = output.read_text()
        assert 'value="2"' in content  # Clamped to max_docs (2)

    def test_contains_stats_element(self, tmp_path):
        """Should include stats display element."""
        from src.cli.graph_viz import generate_html_graph

        output = tmp_path / "graph.html"
        generate_html_graph(_make_concepts(), output)

        content = output.read_text()
        assert 'id="stats"' in content

    def test_contains_zoom_controls(self, tmp_path):
        """Should include zoom +/- and fit buttons."""
        from src.cli.graph_viz import generate_html_graph

        output = tmp_path / "graph.html"
        generate_html_graph(_make_concepts(), output)

        content = output.read_text()
        assert 'id="zoom-controls"' in content
        assert 'id="zoom-in"' in content
        assert 'id="zoom-out"' in content
        assert 'id="zoom-fit"' in content


class TestGenerateConceptGraphHtmlFormat:
    """Integration tests for generate_concept_graph with html format."""

    def test_generates_html_file(self, tmp_path):
        """Should generate HTML file when format is 'html'."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = _make_concepts()

        output_path = tmp_path / "graph.html"
        content, rendered_path = generate_concept_graph(
            mock_store,
            output_path=output_path,
            output_format="html",
        )

        assert rendered_path == output_path
        assert output_path.exists()
        html = output_path.read_text()
        assert "<!DOCTYPE html>" in html
        assert "vis-network" in html

    def test_returns_empty_for_no_concepts(self):
        """Should return empty string for HTML format with no concepts."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = []

        content, path = generate_concept_graph(
            mock_store,
            output_format="html",
        )

        assert content == ""
        assert path is None

    def test_returns_none_without_output_path(self):
        """Should return None path when no output path specified for HTML."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = _make_concepts()

        content, path = generate_concept_graph(
            mock_store,
            output_format="html",
        )

        assert path is None

    def test_uses_custom_title(self, tmp_path):
        """Should pass custom title to HTML generator."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = _make_concepts()

        output_path = tmp_path / "graph.html"
        generate_concept_graph(
            mock_store,
            output_path=output_path,
            output_format="html",
            title="Custom HTML Title",
        )

        html = output_path.read_text()
        assert "Custom HTML Title" in html

    def test_auto_title_cross_document(self, tmp_path):
        """Should auto-generate cross-document title."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = _make_concepts()

        output_path = tmp_path / "graph.html"
        generate_concept_graph(
            mock_store,
            output_path=output_path,
            output_format="html",
        )

        html = output_path.read_text()
        assert "Cross-Document" in html

    def test_single_doc_view(self, tmp_path):
        """Should generate single-document view when doc_id specified."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = _make_concepts()

        output_path = tmp_path / "graph.html"
        generate_concept_graph(
            mock_store,
            doc_id="book_accounting",
            output_path=output_path,
            output_format="html",
        )

        html = output_path.read_text()
        assert "single_document" in html
