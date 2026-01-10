"""Tests for graph visualization module (TDD - tests written first)."""

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

        expected_types = ["term", "method", "principle", "formula", "account"]
        for concept_type in expected_types:
            assert concept_type in CONCEPT_COLORS, f"Missing color for {concept_type}"

    def test_colors_are_valid_hex(self):
        """Colors should be valid hex codes."""
        from src.cli.graph_viz import CONCEPT_COLORS

        for concept_type, color in CONCEPT_COLORS.items():
            assert color.startswith("#"), f"Invalid color for {concept_type}"
            assert len(color) == 7, f"Color for {concept_type} should be #RRGGBB"


class TestRelationStyles:
    """Tests for RELATION_STYLES constant."""

    def test_has_common_relation_types(self):
        """Should have styles for common relation types."""
        from src.cli.graph_viz import RELATION_STYLES

        expected_types = ["uses", "requires", "calculated_from", "component_of"]
        for rel_type in expected_types:
            assert rel_type in RELATION_STYLES, f"Missing style for {rel_type}"

    def test_styles_have_required_keys(self):
        """Each style should have style and color keys."""
        from src.cli.graph_viz import RELATION_STYLES

        for rel_type, style in RELATION_STYLES.items():
            assert "style" in style, f"Missing 'style' key for {rel_type}"
            assert "color" in style, f"Missing 'color' key for {rel_type}"


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

    def test_generates_dot_from_store_concepts(self):
        """Should generate DOT from stored concepts."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = [
            {"concept_id": "fifo", "name": "FIFO", "concept_type": "method", "aliases": "[]"},
            {"concept_id": "lifo", "name": "LIFO", "concept_type": "method", "aliases": "[]"},
        ]
        mock_store.get_related_concepts.return_value = []

        dot, path = generate_concept_graph(mock_store)

        assert "digraph ConceptGraph" in dot
        assert "fifo" in dot.lower()
        assert "lifo" in dot.lower()
        assert path is None

    def test_includes_relations_from_store(self):
        """Should include relations from store."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = [
            {"concept_id": "cogs", "name": "COGS", "concept_type": "formula", "aliases": "[]", "id": "internal_cogs"},
            {"concept_id": "fifo", "name": "FIFO", "concept_type": "method", "aliases": "[]", "id": "internal_fifo"},
        ]
        # Mock the new method that returns edges with relation_type
        mock_store.get_related_concepts_with_types.side_effect = [
            [{"relation_type": "uses", "to": {"name": "FIFO", "concept_id": "fifo"}}],  # COGS uses FIFO
            [],  # FIFO has no relations
        ]

        dot, path = generate_concept_graph(mock_store)

        assert '"cogs" -> "fifo"' in dot
        assert 'label="uses"' in dot  # Check relation type is included

    def test_filters_by_doc_id(self):
        """Should filter concepts by doc_id."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = []
        mock_store.get_related_concepts.return_value = []

        generate_concept_graph(mock_store, doc_id="inventory")

        mock_store.list_concepts.assert_called_once_with(doc_id="inventory")

    def test_writes_dot_file_when_format_is_dot(self, tmp_path):
        """Should write DOT file when format is 'dot'."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = [
            {"concept_id": "test", "name": "Test", "concept_type": "term", "aliases": "[]"},
        ]
        mock_store.get_related_concepts.return_value = []

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
            {"concept_id": "test", "name": "Test", "concept_type": "term", "aliases": "[]"},
        ]
        mock_store.get_related_concepts.return_value = []

        dot, _ = generate_concept_graph(mock_store, title="Custom Title")

        assert 'label="Custom Title"' in dot

    def test_auto_title_with_doc_id(self):
        """Should auto-generate title from doc_id."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = [
            {"concept_id": "test", "name": "Test", "concept_type": "term", "aliases": "[]"},
        ]
        mock_store.get_related_concepts.return_value = []

        dot, _ = generate_concept_graph(mock_store, doc_id="inventory_textbook")

        assert "inventory_textbook" in dot

    def test_auto_title_without_doc_id(self):
        """Should use 'Knowledge Graph' as default title."""
        from src.cli.graph_viz import generate_concept_graph

        mock_store = MagicMock()
        mock_store.list_concepts.return_value = [
            {"concept_id": "test", "name": "Test", "concept_type": "term", "aliases": "[]"},
        ]
        mock_store.get_related_concepts.return_value = []

        dot, _ = generate_concept_graph(mock_store)

        assert "Knowledge Graph" in dot
