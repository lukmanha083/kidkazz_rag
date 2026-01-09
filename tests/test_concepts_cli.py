"""Tests for concepts CLI commands (TDD - tests written first)."""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner


runner = CliRunner()


# ============================================================================
# Tests for concepts list command
# ============================================================================


class TestConceptsListCommand:
    """Tests for 'kidkazz concepts list' command."""

    def test_list_no_concepts(self):
        """Should show warning when no concepts found."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.list_concepts.return_value = []
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["list"])

            assert result.exit_code == 0
            assert "No concepts found" in result.output or "0" in result.output

    def test_list_with_concepts(self):
        """Should display concepts in table format."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.list_concepts.return_value = [
                {
                    "concept_id": "fifo",
                    "name": "FIFO",
                    "concept_type": "method",
                    "definition": "First-In, First-Out inventory method.",
                    "aliases": '["First-In First-Out"]',
                },
                {
                    "concept_id": "lifo",
                    "name": "LIFO",
                    "concept_type": "method",
                    "definition": "Last-In, First-Out inventory method.",
                    "aliases": "[]",
                },
            ]
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["list"])

            assert result.exit_code == 0
            assert "FIFO" in result.output
            assert "LIFO" in result.output

    def test_list_with_doc_filter(self):
        """Should filter concepts by document."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.list_concepts.return_value = [
                {
                    "concept_id": "cogs",
                    "name": "COGS",
                    "concept_type": "formula",
                    "definition": "Cost of Goods Sold.",
                    "aliases": "[]",
                },
            ]
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["list", "--doc-id", "inventory_textbook"])

            assert result.exit_code == 0
            mock_store.list_concepts.assert_called_with(doc_id="inventory_textbook")

    def test_list_with_type_filter(self):
        """Should filter concepts by type."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.list_concepts.return_value = [
                {
                    "concept_id": "fifo",
                    "name": "FIFO",
                    "concept_type": "method",
                    "definition": "First-In, First-Out.",
                    "aliases": "[]",
                },
                {
                    "concept_id": "cogs",
                    "name": "COGS",
                    "concept_type": "formula",
                    "definition": "Cost of Goods Sold.",
                    "aliases": "[]",
                },
            ]
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["list", "--type", "method"])

            assert result.exit_code == 0
            assert "FIFO" in result.output
            # Formula concepts should be filtered out
            assert "COGS" not in result.output or "method" in result.output

    def test_list_json_output(self):
        """Should output JSON when --json flag used."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.list_concepts.return_value = [
                {
                    "concept_id": "fifo",
                    "name": "FIFO",
                    "concept_type": "method",
                    "definition": "First-In, First-Out.",
                    "aliases": "[]",
                },
            ]
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["list", "--json"])

            assert result.exit_code == 0
            # Output should be valid JSON
            try:
                data = json.loads(result.output)
                assert isinstance(data, list)
            except json.JSONDecodeError:
                pytest.fail("Output is not valid JSON")


# ============================================================================
# Tests for concepts show command
# ============================================================================


class TestConceptsShowCommand:
    """Tests for 'kidkazz concepts show' command."""

    def test_show_existing_concept(self):
        """Should display concept details."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_concept_by_name.return_value = {
                "concept_id": "fifo",
                "name": "FIFO",
                "concept_type": "method",
                "definition": "First-In, First-Out inventory valuation method.",
                "aliases": '["First-In First-Out"]',
            }
            mock_store.get_concept_definition_chunks.return_value = []
            mock_store.get_related_concepts.return_value = []
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["show", "FIFO"])

            assert result.exit_code == 0
            assert "FIFO" in result.output
            assert "method" in result.output

    def test_show_nonexistent_concept(self):
        """Should show error for non-existent concept."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_concept_by_name.return_value = None
            mock_store.get_concept.return_value = None
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["show", "NonExistent"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_show_with_citations(self):
        """Should display citations when available."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_concept_by_name.return_value = {
                "concept_id": "fifo",
                "name": "FIFO",
                "concept_type": "method",
                "definition": "First-In, First-Out.",
                "aliases": "[]",
            }
            mock_store.get_concept_definition_chunks.return_value = [
                {
                    "chunk_id": "chunk_1",
                    "document_id": "inventory_textbook",
                    "section_path": '["Chapter 5", "Inventory Methods"]',
                },
            ]
            mock_store.get_related_concepts.return_value = []
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["show", "FIFO"])

            assert result.exit_code == 0
            assert "inventory_textbook" in result.output or "Sources" in result.output

    def test_show_json_output(self):
        """Should output JSON when --json flag used."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_concept_by_name.return_value = {
                "concept_id": "fifo",
                "name": "FIFO",
                "concept_type": "method",
                "definition": "First-In, First-Out.",
                "aliases": "[]",
            }
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["show", "FIFO", "--json"])

            assert result.exit_code == 0
            try:
                data = json.loads(result.output)
                assert isinstance(data, dict)
            except json.JSONDecodeError:
                pytest.fail("Output is not valid JSON")


# ============================================================================
# Tests for concepts search command
# ============================================================================


class TestConceptsSearchCommand:
    """Tests for 'kidkazz concepts search' command."""

    def test_search_with_matches(self):
        """Should return matching concepts."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.list_concepts.return_value = [
                {
                    "concept_id": "fifo",
                    "name": "FIFO",
                    "concept_type": "method",
                    "definition": "First-In, First-Out inventory method.",
                    "aliases": '["First-In First-Out"]',
                },
                {
                    "concept_id": "lifo",
                    "name": "LIFO",
                    "concept_type": "method",
                    "definition": "Last-In, First-Out inventory method.",
                    "aliases": "[]",
                },
            ]
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["search", "inventory"])

            assert result.exit_code == 0
            assert "FIFO" in result.output or "LIFO" in result.output

    def test_search_no_matches(self):
        """Should show warning when no matches found."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.list_concepts.return_value = [
                {
                    "concept_id": "fifo",
                    "name": "FIFO",
                    "concept_type": "method",
                    "definition": "First-In, First-Out.",
                    "aliases": "[]",
                },
            ]
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["search", "nonexistent_term"])

            assert result.exit_code == 0
            assert "No concepts found" in result.output or "0" in result.output or result.output.strip() == ""

    def test_search_json_output(self):
        """Should output JSON when --json flag used."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.list_concepts.return_value = [
                {
                    "concept_id": "fifo",
                    "name": "FIFO",
                    "concept_type": "method",
                    "definition": "First-In, First-Out inventory method.",
                    "aliases": "[]",
                },
            ]
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["search", "inventory", "--json"])

            assert result.exit_code == 0


# ============================================================================
# Tests for concepts related command
# ============================================================================


class TestConceptsRelatedCommand:
    """Tests for 'kidkazz concepts related' command."""

    def test_related_with_concepts(self):
        """Should show related concepts."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_concept_by_name.return_value = {
                "concept_id": "cogs",
                "name": "COGS",
                "concept_type": "formula",
                "definition": "Cost of Goods Sold.",
                "aliases": "[]",
            }
            mock_store.get_related_concepts.return_value = [
                {
                    "concept_id": "fifo",
                    "name": "FIFO",
                    "concept_type": "method",
                    "definition": "First-In, First-Out.",
                    "aliases": "[]",
                },
            ]
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["related", "COGS"])

            assert result.exit_code == 0
            assert "FIFO" in result.output

    def test_related_no_concepts(self):
        """Should show warning when no related concepts."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_concept_by_name.return_value = {
                "concept_id": "isolated",
                "name": "Isolated Concept",
                "concept_type": "term",
                "definition": "An isolated concept.",
                "aliases": "[]",
            }
            mock_store.get_related_concepts.return_value = []
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["related", "Isolated Concept"])

            assert result.exit_code == 0
            assert "No related" in result.output or "0" in result.output

    def test_related_nonexistent_concept(self):
        """Should show error for non-existent concept."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_concept_by_name.return_value = None
            mock_store.get_concept.return_value = None
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["related", "NonExistent"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_related_json_output(self):
        """Should output JSON when --json flag used."""
        from src.cli.commands.concepts import app

        with patch("src.cli.commands.concepts.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_concept_by_name.return_value = {
                "concept_id": "cogs",
                "name": "COGS",
                "concept_type": "formula",
                "definition": "Cost of Goods Sold.",
                "aliases": "[]",
            }
            mock_store.get_related_concepts.return_value = [
                {
                    "concept_id": "fifo",
                    "name": "FIFO",
                    "concept_type": "method",
                    "definition": "First-In, First-Out.",
                    "aliases": "[]",
                },
            ]
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["related", "COGS", "--json"])

            assert result.exit_code == 0
            try:
                data = json.loads(result.output)
                assert isinstance(data, list)
            except json.JSONDecodeError:
                pytest.fail("Output is not valid JSON")
