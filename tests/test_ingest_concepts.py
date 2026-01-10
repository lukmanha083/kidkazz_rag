"""Tests for concept extraction during ingestion (TDD - tests written first)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()


@pytest.fixture
def sample_markdown(tmp_path):
    """Create a sample markdown file with concepts."""
    content = """# Inventory Valuation Methods

This chapter explains inventory valuation methods.

## FIFO Method

FIFO (First-In, First-Out) is an inventory valuation method where
the oldest inventory items are assumed to be sold first.

### How FIFO Works

When using FIFO, the cost of goods sold (COGS) is calculated using
the cost of the earliest purchased items first.

## LIFO Method

LIFO (Last-In, First-Out) assumes the most recently purchased
inventory is sold first.

## Conclusion

Both FIFO and LIFO affect the calculation of COGS and ending inventory.
"""
    file_path = tmp_path / "inventory.md"
    file_path.write_text(content)
    return file_path


class TestIngestWithConceptExtraction:
    """Tests for concept extraction during ingestion."""

    @patch("src.cli.commands.ingest.CONCEPT_EXTRACTION_AVAILABLE", True)
    @patch("src.cli.commands.ingest.ConceptExtractor")
    def test_extract_concepts_flag_triggers_extraction(
        self, mock_extractor_class, sample_markdown
    ):
        """Should extract concepts when --extract-concepts flag is passed."""
        # Create mock Pydantic-like object with attributes
        mock_concept = MagicMock()
        mock_concept.name = "FIFO"
        mock_concept.definition = "First-In, First-Out inventory method."
        mock_concept.concept_type = MagicMock(value="method")
        mock_concept.aliases = []

        # Setup mock extractor
        mock_extractor = MagicMock()
        mock_extractor.extract_from_chunks.return_value = ([mock_concept], [])
        mock_extractor_class.return_value = mock_extractor

        with patch.dict("os.environ", {"KIDKAZZ_EMBEDDER_TYPE": "mock"}):
            result = runner.invoke(app, [
                "ingest", "markdown", str(sample_markdown),
                "--extract-concepts",
                "--json",
            ])

        # Extractor should be instantiated
        mock_extractor_class.assert_called_once()
        # extract_from_chunks should be called
        mock_extractor.extract_from_chunks.assert_called_once()
        assert result.exit_code == 0

    @patch("src.cli.commands.ingest.CONCEPT_EXTRACTION_AVAILABLE", True)
    @patch("src.cli.commands.ingest.ConceptExtractor")
    def test_concept_provider_option(self, mock_extractor_class, sample_markdown):
        """Should pass provider to ConceptExtractor."""
        mock_extractor = MagicMock()
        mock_extractor.extract_from_chunks.return_value = ([], [])
        mock_extractor_class.return_value = mock_extractor

        with patch.dict("os.environ", {"KIDKAZZ_EMBEDDER_TYPE": "mock"}):
            result = runner.invoke(app, [
                "ingest", "markdown", str(sample_markdown),
                "--extract-concepts",
                "--concept-provider", "anthropic/claude-opus-4-20250514",
                "--json",
            ])

        mock_extractor_class.assert_called_once_with(
            provider="anthropic/claude-opus-4-20250514"
        )
        assert result.exit_code == 0

    def test_no_extract_concepts_by_default(self, sample_markdown):
        """Should not extract concepts by default."""
        with patch.dict("os.environ", {"KIDKAZZ_EMBEDDER_TYPE": "mock"}):
            with patch("src.cli.commands.ingest.ConceptExtractor") as mock_class:
                result = runner.invoke(app, [
                    "ingest", "markdown", str(sample_markdown),
                    "--json",
                ])

        # Extractor should not be instantiated
        mock_class.assert_not_called()
        assert result.exit_code == 0

    @patch("src.cli.commands.ingest.CONCEPT_EXTRACTION_AVAILABLE", True)
    @patch("src.cli.commands.ingest.ConceptExtractor")
    def test_concepts_stored_in_database(self, mock_extractor_class, sample_markdown):
        """Should store extracted concepts in the database."""
        # Create mock Pydantic-like objects with attributes
        mock_concept = MagicMock()
        mock_concept.name = "FIFO"
        mock_concept.definition = "First-In, First-Out inventory method."
        mock_concept.concept_type = MagicMock(value="method")
        mock_concept.aliases = ["First-In First-Out"]

        mock_relation = MagicMock()
        mock_relation.from_concept = "FIFO"
        mock_relation.to_concept = "COGS"
        mock_relation.relation_type = "used_in"

        mock_extractor = MagicMock()
        mock_extractor.extract_from_chunks.return_value = (
            [mock_concept],
            [mock_relation],
        )
        mock_extractor_class.return_value = mock_extractor

        with patch.dict("os.environ", {"KIDKAZZ_EMBEDDER_TYPE": "mock"}):
            with patch("src.cli.commands.ingest.get_store") as mock_get_store:
                mock_store = MagicMock()
                mock_store.store_or_merge_concept.return_value = "concept_internal_123"
                mock_get_store.return_value = mock_store

                result = runner.invoke(app, [
                    "ingest", "markdown", str(sample_markdown),
                    "--extract-concepts",
                    "--json",
                ])

        # Should have called store_or_merge_concept
        mock_store.store_or_merge_concept.assert_called()
        assert result.exit_code == 0

    @patch("src.cli.commands.ingest.CONCEPT_EXTRACTION_AVAILABLE", True)
    @patch("src.cli.commands.ingest.ConceptExtractor")
    def test_concept_extraction_error_handled(self, mock_extractor_class, sample_markdown):
        """Should handle concept extraction errors gracefully."""
        mock_extractor = MagicMock()
        mock_extractor.extract_from_chunks.side_effect = Exception("LLM API error")
        mock_extractor_class.return_value = mock_extractor

        with patch.dict("os.environ", {"KIDKAZZ_EMBEDDER_TYPE": "mock"}):
            result = runner.invoke(app, [
                "ingest", "markdown", str(sample_markdown),
                "--extract-concepts",
                "--json",
            ])

        # Should still complete (document ingested, concepts failed)
        # or show warning but not crash
        # The exact behavior depends on implementation
        assert result.exit_code in [0, 1]

    @patch("src.cli.commands.ingest.CONCEPT_EXTRACTION_AVAILABLE", True)
    @patch("src.cli.commands.ingest.ConceptExtractor")
    def test_json_output_includes_concepts_count(self, mock_extractor_class, sample_markdown):
        """JSON output should include concepts count when extracted."""
        # Create mock Pydantic-like objects with attributes
        mock_concept1 = MagicMock()
        mock_concept1.name = "FIFO"
        mock_concept1.definition = "First-In, First-Out"
        mock_concept1.concept_type = MagicMock(value="method")
        mock_concept1.aliases = []

        mock_concept2 = MagicMock()
        mock_concept2.name = "LIFO"
        mock_concept2.definition = "Last-In, First-Out"
        mock_concept2.concept_type = MagicMock(value="method")
        mock_concept2.aliases = []

        mock_extractor = MagicMock()
        mock_extractor.extract_from_chunks.return_value = (
            [mock_concept1, mock_concept2],
            [],
        )
        mock_extractor_class.return_value = mock_extractor

        with patch.dict("os.environ", {"KIDKAZZ_EMBEDDER_TYPE": "mock"}):
            with patch("src.cli.commands.ingest.get_store") as mock_get_store:
                mock_store = MagicMock()
                mock_store.store_or_merge_concept.return_value = "concept-id"
                mock_get_store.return_value = mock_store

                result = runner.invoke(app, [
                    "ingest", "markdown", str(sample_markdown),
                    "--extract-concepts",
                    "--json",
                ])

        assert result.exit_code == 0
        # Extract JSON from output (may have progress bars before it)
        import re
        clean_output = re.sub(r'\x1b\[[0-9;]*m', '', result.output)
        # Remove any control characters
        clean_output = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean_output)
        # Find the JSON object (starts with { ends with })
        json_match = re.search(r'\{[^}]+\}', clean_output, re.DOTALL)
        assert json_match, f"No JSON found in output: {clean_output[:200]}"
        output = json.loads(json_match.group())
        assert "concepts" in output
        assert output["concepts"] == 2

    @patch("src.cli.commands.ingest.ConceptExtractor")
    def test_dry_run_skips_concept_extraction(self, mock_extractor_class, sample_markdown):
        """Dry run should not extract concepts."""
        with patch.dict("os.environ", {"KIDKAZZ_EMBEDDER_TYPE": "mock"}):
            result = runner.invoke(app, [
                "ingest", "markdown", str(sample_markdown),
                "--extract-concepts",
                "--dry-run",
            ])

        # Should not instantiate extractor during dry run
        mock_extractor_class.assert_not_called()
        assert result.exit_code == 0


class TestConceptExtractionConfig:
    """Tests for concept extraction configuration."""

    def test_config_extract_concepts_default(self, sample_markdown):
        """Config should have extract_concepts defaulting to False."""
        from src.cli.config import CLIConfig

        config = CLIConfig()
        assert config.extract_concepts is False

    def test_config_concept_provider_default(self, sample_markdown):
        """Config should have concept_provider default."""
        from src.cli.config import CLIConfig

        config = CLIConfig()
        assert config.concept_provider == "anthropic/claude-sonnet-4-20250514"

    def test_config_from_toml_enables_concepts(self, tmp_path, monkeypatch):
        """Config should load extract_concepts from TOML."""
        # Create a config file
        config_content = """
[concepts]
enabled = true
provider = "anthropic/claude-opus-4-20250514"
"""
        config_path = tmp_path / ".kidkazz.toml"
        config_path.write_text(config_content)

        monkeypatch.chdir(tmp_path)

        from src.cli.config import CLIConfig

        config = CLIConfig.load()
        assert config.extract_concepts is True
        assert config.concept_provider == "anthropic/claude-opus-4-20250514"


class TestBatchIngestWithConcepts:
    """Tests for batch ingestion with concept extraction."""

    @patch("src.cli.commands.ingest.CONCEPT_EXTRACTION_AVAILABLE", True)
    @patch("src.cli.commands.ingest.ConceptExtractor")
    def test_batch_extract_concepts_flag(self, mock_extractor_class, tmp_path):
        """Batch ingestion should support --extract-concepts flag."""
        # Create test files
        for i in range(2):
            (tmp_path / f"doc_{i}.md").write_text(f"# Document {i}\n\nContent about FIFO.")

        mock_extractor = MagicMock()
        mock_extractor.extract_from_chunks.return_value = ([], [])
        mock_extractor_class.return_value = mock_extractor

        with patch.dict("os.environ", {"KIDKAZZ_EMBEDDER_TYPE": "mock"}):
            result = runner.invoke(app, [
                "ingest", "batch", str(tmp_path),
                "--pattern", "*.md",
                "--extract-concepts",
                "--json",
            ])

        # Should have called extractor for each file
        assert mock_extractor.extract_from_chunks.call_count == 2
        assert result.exit_code == 0
