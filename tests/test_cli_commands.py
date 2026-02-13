"""Tests for CLI commands."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()


class TestMainApp:
    """Tests for main CLI app."""

    def test_help_shows_commands(self):
        """Test --help shows available commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ingest" in result.output
        assert "search" in result.output
        assert "docs" in result.output
        assert "db" in result.output
        assert "config" in result.output

    def test_version_command(self):
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "KidKazz" in result.output

    def test_no_args_shows_help(self):
        """Test no args shows help."""
        result = runner.invoke(app)
        # Typer exits with code 0 or 2 for help display
        assert result.exit_code in [0, 2]
        # Should show help since no_args_is_help=True


class TestIngestCommands:
    """Tests for ingest commands."""

    def test_ingest_help(self):
        """Test ingest --help."""
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "markdown" in result.output
        assert "batch" in result.output

    def test_ingest_markdown_missing_file(self):
        """Test ingest markdown with missing file."""
        result = runner.invoke(app, ["ingest", "markdown", "nonexistent.md"])
        assert result.exit_code != 0

    def test_ingest_markdown_dry_run(self, tmp_path):
        """Test ingest markdown dry run."""
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test\n\nThis is test content.")

        result = runner.invoke(app, [
            "ingest", "markdown", str(test_file),
            "--dry-run"
        ])
        assert result.exit_code == 0
        assert "Dry run" in result.output


class TestSearchCommands:
    """Tests for search commands."""

    def test_search_help(self):
        """Test search --help."""
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        assert "semantic" in result.output
        assert "keyword" in result.output

    def test_search_semantic_empty_store(self):
        """Test semantic search on empty store."""
        with patch.dict(os.environ, {
            "KIDKAZZ_EMBEDDER_TYPE": "mock",
            "KIDKAZZ_STORE_TYPE": "mock",
        }):
            result = runner.invoke(app, [
                "search", "semantic", "test query",
                "--json"
            ])
            # Should return empty results, not error
            assert result.exit_code == 0
            assert "[]" in result.output or "No results" in result.output

    def test_search_keyword_empty_store(self):
        """Test keyword search on empty store."""
        with patch.dict(os.environ, {
            "KIDKAZZ_EMBEDDER_TYPE": "mock",
            "KIDKAZZ_STORE_TYPE": "mock",
        }):
            result = runner.invoke(app, [
                "search", "keyword", "test",
                "--json"
            ])
            assert result.exit_code == 0
            assert "[]" in result.output or "No results" in result.output


class TestDocsCommands:
    """Tests for docs commands."""

    def test_docs_help(self):
        """Test docs --help."""
        result = runner.invoke(app, ["docs", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "stats" in result.output
        assert "export" in result.output
        assert "delete" in result.output

    def test_docs_list_empty(self):
        """Test docs list on empty store."""
        with patch.dict(os.environ, {
            "KIDKAZZ_EMBEDDER_TYPE": "mock",
            "KIDKAZZ_STORE_TYPE": "mock",
        }):
            result = runner.invoke(app, ["docs", "list", "--json"])
            assert result.exit_code == 0
            assert "[]" in result.output or "No documents" in result.output

    def test_docs_stats_nonexistent(self):
        """Test docs stats for nonexistent document."""
        result = runner.invoke(app, ["docs", "stats", "nonexistent_doc"])
        assert result.exit_code != 0


class TestDbCommands:
    """Tests for db commands."""

    def test_db_help(self):
        """Test db --help."""
        result = runner.invoke(app, ["db", "--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "status" in result.output
        assert "clear" in result.output

    def test_db_status(self):
        """Test db status."""
        result = runner.invoke(app, ["db", "status", "--json"])
        assert result.exit_code == 0
        assert "connected" in result.output.lower() or "store_type" in result.output

    def test_db_init(self):
        """Test db init."""
        result = runner.invoke(app, ["db", "init", "--json"])
        assert result.exit_code == 0


class TestConfigCommands:
    """Tests for config commands."""

    def test_config_help(self):
        """Test config --help."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output
        assert "set" in result.output
        assert "reset" in result.output

    def test_config_show(self):
        """Test config show."""
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "store_type" in result.output or "storage" in result.output.lower()

    def test_config_show_json(self):
        """Test config show --json."""
        result = runner.invoke(app, ["config", "show", "--json"])
        assert result.exit_code == 0
        assert "{" in result.output  # JSON output

    def test_config_set_invalid_key(self):
        """Test config set with invalid key."""
        result = runner.invoke(app, ["config", "set", "invalid_key", "value"])
        assert result.exit_code != 0
        assert "Invalid" in result.output or "invalid" in result.output.lower()


class TestJsonOutput:
    """Tests for JSON output mode."""

    def test_docs_list_json(self):
        """Test docs list with JSON output."""
        result = runner.invoke(app, ["docs", "list", "--json"])
        assert result.exit_code == 0
        # Should be valid JSON (starts with [ or {)
        output = result.output.strip()
        assert output.startswith("[") or output.startswith("{")

    def test_db_status_json(self):
        """Test db status with JSON output."""
        result = runner.invoke(app, ["db", "status", "--json"])
        assert result.exit_code == 0
        output = result.output.strip()
        assert output.startswith("{")

    def test_config_show_json(self):
        """Test config show with JSON output."""
        result = runner.invoke(app, ["config", "show", "--json"])
        assert result.exit_code == 0
        output = result.output.strip()
        assert output.startswith("{")
