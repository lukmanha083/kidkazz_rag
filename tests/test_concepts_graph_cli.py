"""Tests for concepts graph and export CLI commands."""

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
            "source_documents": '["doc_a", "doc_b"]',
        },
        {
            "concept_id": "lifo",
            "name": "LIFO",
            "concept_type": "method",
            "definition": "Last-In, First-Out",
            "aliases": "[]",
            "source_documents": '["doc_a", "doc_b"]',
        },
    ]
    store.get_related_concepts.return_value = []
    return store


class TestGraphCommand:
    """Tests for 'kidkazz concepts graph' command."""

    @patch("src.cli.commands.concepts.get_store")
    @patch("src.cli.graph_viz.generate_html_graph")
    def test_graph_generates_html(self, mock_gen_html, mock_get_store, mock_store, tmp_path):
        """Should generate HTML graph and print success message."""
        mock_get_store.return_value = mock_store
        output_file = tmp_path / "graph.html"
        mock_gen_html.return_value = output_file

        result = runner.invoke(app, [
            "concepts", "graph",
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert "Graph saved to" in result.output
        mock_gen_html.assert_called_once()

    @patch("src.cli.commands.concepts.get_store")
    @patch("src.cli.graph_viz.generate_html_graph")
    def test_graph_with_doc_id_filter(self, mock_gen_html, mock_get_store, mock_store, tmp_path):
        """Should filter by doc_id."""
        mock_get_store.return_value = mock_store
        mock_gen_html.return_value = tmp_path / "graph.html"

        result = runner.invoke(app, ["concepts", "graph", "inventory"])

        assert result.exit_code == 0
        mock_store.list_concepts.assert_called_with(doc_id="inventory")

    @patch("src.cli.commands.concepts.get_store")
    @patch("src.cli.graph_viz.generate_html_graph")
    def test_graph_default_output_path(self, mock_gen_html, mock_get_store, mock_store):
        """Should default to concept_graph.html when no --output given."""
        mock_get_store.return_value = mock_store
        mock_gen_html.return_value = Path("concept_graph.html")

        result = runner.invoke(app, ["concepts", "graph"])

        assert result.exit_code == 0
        # Verify generate_html_graph was called with the default path
        call_kwargs = mock_gen_html.call_args
        assert call_kwargs[1]["output_path"] == Path("concept_graph.html") or \
               call_kwargs.kwargs.get("output_path") == Path("concept_graph.html")

    @patch("src.cli.commands.concepts.get_store")
    @patch("src.cli.graph_viz.generate_html_graph")
    def test_graph_with_custom_title(self, mock_gen_html, mock_get_store, mock_store, tmp_path):
        """Should pass custom title to generate_html_graph."""
        mock_get_store.return_value = mock_store
        mock_gen_html.return_value = tmp_path / "graph.html"

        result = runner.invoke(app, [
            "concepts", "graph",
            "--output", str(tmp_path / "graph.html"),
            "--title", "My Custom Graph",
        ])

        assert result.exit_code == 0
        # Verify title was passed through
        call_kwargs = mock_gen_html.call_args
        assert call_kwargs[1].get("title") == "My Custom Graph" or \
               call_kwargs.kwargs.get("title") == "My Custom Graph"

    @patch("src.cli.commands.concepts.get_store")
    def test_graph_shows_error_when_no_concepts(self, mock_get_store):
        """Should show error and exit 1 when no concepts found."""
        mock_store = MagicMock()
        mock_store.list_concepts.return_value = []
        mock_get_store.return_value = mock_store

        result = runner.invoke(app, ["concepts", "graph"])

        assert result.exit_code == 1
        assert "No concepts found" in result.output


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
