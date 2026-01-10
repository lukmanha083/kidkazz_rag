"""Tests for concepts graph and export CLI commands (TDD - tests written first)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_store():
    """Create a mock store with test concepts."""
    store = MagicMock()
    store.list_concepts.return_value = [
        {
            "concept_id": "fifo",
            "name": "FIFO",
            "concept_type": "method",
            "definition": "First-In, First-Out",
            "aliases": "[]",
        },
        {
            "concept_id": "lifo",
            "name": "LIFO",
            "concept_type": "method",
            "definition": "Last-In, First-Out",
            "aliases": "[]",
        },
    ]
    store.get_related_concepts.return_value = []
    return store


class TestGraphCommand:
    """Tests for 'kidkazz concepts graph' command."""

    @patch("src.cli.commands.concepts.get_store")
    def test_graph_outputs_dot_to_stdout(self, mock_get_store, mock_store):
        """Should output DOT format to stdout by default."""
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["concepts", "graph"])

        assert result.exit_code == 0
        assert "digraph" in result.output
        assert "fifo" in result.output.lower()
        assert "lifo" in result.output.lower()

    @patch("src.cli.commands.concepts.get_store")
    def test_graph_with_doc_id_filter(self, mock_get_store, mock_store):
        """Should filter by doc_id."""
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["concepts", "graph", "inventory"])

        assert result.exit_code == 0
        mock_store.list_concepts.assert_called_with(doc_id="inventory")

    @patch("src.cli.commands.concepts.get_store")
    def test_graph_saves_dot_file(self, mock_get_store, mock_store, tmp_path):
        """Should save DOT file with --output and --format dot."""
        mock_get_store.return_value = mock_store
        output_file = tmp_path / "graph.dot"

        result = runner.invoke(app, [
            "concepts", "graph",
            "--output", str(output_file),
            "--format", "dot",
        ])

        assert result.exit_code == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "digraph" in content

    @patch("src.cli.commands.concepts.get_store")
    def test_graph_with_custom_title(self, mock_get_store, mock_store):
        """Should use custom title."""
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, [
            "concepts", "graph",
            "--title", "My Custom Graph",
        ])

        assert result.exit_code == 0
        assert "My Custom Graph" in result.output

    @patch("src.cli.commands.concepts.get_store")
    def test_graph_shows_empty_message_when_no_concepts(self, mock_get_store):
        """Should handle no concepts gracefully."""
        mock_store = MagicMock()
        mock_store.list_concepts.return_value = []
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["concepts", "graph"])

        assert result.exit_code == 0
        assert "No concepts found" in result.output

    @patch("src.cli.commands.concepts.get_store")
    @patch("src.cli.graph_viz.render_graph")
    def test_graph_renders_png(self, mock_render, mock_get_store, mock_store, tmp_path):
        """Should render PNG when format is png."""
        mock_get_store.return_value = mock_store
        output_file = tmp_path / "graph.png"
        mock_render.return_value = output_file

        result = runner.invoke(app, [
            "concepts", "graph",
            "--output", str(tmp_path / "graph"),
            "--format", "png",
        ])

        assert result.exit_code == 0
        mock_render.assert_called_once()

    @patch("src.cli.commands.concepts.get_store")
    def test_graph_handles_import_error(self, mock_get_store, mock_store, tmp_path):
        """Should show helpful message when graphviz not installed."""
        mock_get_store.return_value = mock_store

        with patch("src.cli.graph_viz.render_graph") as mock_render:
            mock_render.side_effect = ImportError("graphviz not installed")

            result = runner.invoke(app, [
                "concepts", "graph",
                "--output", str(tmp_path / "graph"),
                "--format", "png",
            ])

        assert result.exit_code == 1
        # Should show helpful message about installing graphviz


class TestExportCommand:
    """Tests for 'kidkazz concepts export' command."""

    @patch("src.cli.commands.concepts.get_store")
    def test_export_json(self, mock_get_store, mock_store, tmp_path):
        """Should export concepts as JSON."""
        mock_get_store.return_value = mock_store
        output_file = tmp_path / "concepts.json"

        result = runner.invoke(app, [
            "concepts", "export",
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert len(data) == 2
        assert data[0]["name"] == "FIFO"

    @patch("src.cli.commands.concepts.get_store")
    def test_export_csv(self, mock_get_store, mock_store, tmp_path):
        """Should export concepts as CSV."""
        mock_get_store.return_value = mock_store
        output_file = tmp_path / "concepts.csv"

        result = runner.invoke(app, [
            "concepts", "export",
            "--output", str(output_file),
            "--format", "csv",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text()
        assert "concept_id" in content  # Header
        assert "FIFO" in content
        assert "LIFO" in content

    @patch("src.cli.commands.concepts.get_store")
    def test_export_with_doc_id_filter(self, mock_get_store, mock_store, tmp_path):
        """Should filter by doc_id."""
        mock_get_store.return_value = mock_store
        output_file = tmp_path / "concepts.json"

        result = runner.invoke(app, [
            "concepts", "export",
            "--output", str(output_file),
            "--doc-id", "inventory",
        ])

        assert result.exit_code == 0
        mock_store.list_concepts.assert_called_with(doc_id="inventory")

    @patch("src.cli.commands.concepts.get_store")
    def test_export_warns_when_no_concepts(self, mock_get_store, tmp_path):
        """Should warn when no concepts to export."""
        mock_store = MagicMock()
        mock_store.list_concepts.return_value = []
        mock_get_store.return_value = mock_store
        output_file = tmp_path / "concepts.json"

        result = runner.invoke(app, [
            "concepts", "export",
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert "No concepts to export" in result.output

    @patch("src.cli.commands.concepts.get_store")
    def test_export_unknown_format_error(self, mock_get_store, mock_store, tmp_path):
        """Should error on unknown format."""
        mock_get_store.return_value = mock_store
        output_file = tmp_path / "concepts.xml"

        result = runner.invoke(app, [
            "concepts", "export",
            "--output", str(output_file),
            "--format", "xml",
        ])

        assert result.exit_code == 1
        assert "Unknown format" in result.output

    @patch("src.cli.commands.concepts.get_store")
    def test_export_shows_success_message(self, mock_get_store, mock_store, tmp_path):
        """Should show success message with count."""
        mock_get_store.return_value = mock_store
        output_file = tmp_path / "concepts.json"

        result = runner.invoke(app, [
            "concepts", "export",
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert "Exported 2 concepts" in result.output
