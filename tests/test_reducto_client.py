"""Tests for Reducto.ai client.

This module tests the Reducto API client for PDF parsing.
Tests are written BEFORE implementation following TDD principles.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestReductoConfig:
    """Tests for ReductoConfig dataclass."""

    def test_create_config_with_api_key(self):
        """Test creating config with API key."""
        from src.pdf_converter.reducto_client import ReductoConfig

        config = ReductoConfig(api_key="test_api_key")

        assert config.api_key == "test_api_key"

    def test_config_default_values(self):
        """Test config has sensible defaults."""
        from src.pdf_converter.reducto_client import ReductoConfig

        config = ReductoConfig(api_key="test_key")

        assert config.agentic is False
        assert config.chunk_mode == "variable"
        assert config.table_format == "md"

    def test_config_custom_values(self):
        """Test config with custom values."""
        from src.pdf_converter.reducto_client import ReductoConfig

        config = ReductoConfig(
            api_key="test_key",
            agentic=True,
            chunk_mode="section",
            table_format="html",
        )

        assert config.agentic is True
        assert config.chunk_mode == "section"
        assert config.table_format == "html"

    def test_from_env_success(self, monkeypatch):
        """Test creating config from environment variable."""
        from src.pdf_converter.reducto_client import ReductoConfig

        monkeypatch.setenv("REDUCTO_API_KEY", "env_api_key")

        config = ReductoConfig.from_env()

        assert config.api_key == "env_api_key"

    def test_from_env_missing_key(self, monkeypatch):
        """Test error when API key not in environment."""
        from src.pdf_converter.reducto_client import ReductoConfig

        monkeypatch.delenv("REDUCTO_API_KEY", raising=False)

        with pytest.raises(ValueError, match="REDUCTO_API_KEY"):
            ReductoConfig.from_env()

    def test_from_env_with_custom_env_var(self, monkeypatch):
        """Test creating config from custom environment variable name."""
        from src.pdf_converter.reducto_client import ReductoConfig

        monkeypatch.setenv("MY_REDUCTO_KEY", "custom_key")

        config = ReductoConfig.from_env(env_var="MY_REDUCTO_KEY")

        assert config.api_key == "custom_key"


class TestReductoClient:
    """Tests for ReductoClient class."""

    def test_create_client(self):
        """Test creating a client with config."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        assert client.config == config

    def test_client_lazy_loads_sdk(self):
        """Test SDK client is lazy loaded."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        # Client should not be initialized yet
        assert client._client is None

    @patch("reducto.Reducto")
    def test_client_initializes_on_access(self, mock_reducto):
        """Test SDK client initializes when accessed."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        # Access the client property
        _ = client.client

        mock_reducto.assert_called_once_with(api_key="test_key")

    @patch("reducto.Reducto")
    def test_parse_pdf_success(self, mock_reducto, tmp_path):
        """Test successful PDF parsing."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        # Create mock response
        mock_chunk1 = MagicMock()
        mock_chunk1.content = "# Document Title"
        mock_chunk2 = MagicMock()
        mock_chunk2.content = "This is the content of the document."

        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk1, mock_chunk2]

        mock_reducto.return_value.parse.run.return_value = mock_response

        # Create test PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        result = client.parse_pdf(pdf_path)

        assert "# Document Title" in result
        assert "This is the content of the document." in result

    @patch("reducto.Reducto")
    def test_parse_pdf_with_agentic_mode(self, mock_reducto, tmp_path):
        """Test PDF parsing with agentic mode enabled."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        mock_chunk = MagicMock()
        mock_chunk.content = "Content"
        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]
        mock_reducto.return_value.parse.run.return_value = mock_response

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf")

        config = ReductoConfig(api_key="test_key", agentic=True)
        client = ReductoClient(config)
        client.parse_pdf(pdf_path)

        # Verify agentic was passed to API
        call_kwargs = mock_reducto.return_value.parse.run.call_args[1]
        assert call_kwargs["enhance"]["agentic"] is True

    @patch("reducto.Reducto")
    def test_parse_pdf_with_custom_chunk_mode(self, mock_reducto, tmp_path):
        """Test PDF parsing with custom chunk mode."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        mock_chunk = MagicMock()
        mock_chunk.content = "Content"
        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]
        mock_reducto.return_value.parse.run.return_value = mock_response

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf")

        config = ReductoConfig(api_key="test_key", chunk_mode="section")
        client = ReductoClient(config)
        client.parse_pdf(pdf_path)

        # Verify chunk_mode was passed to API
        call_kwargs = mock_reducto.return_value.parse.run.call_args[1]
        assert call_kwargs["retrieval"]["chunking"]["chunk_mode"] == "section"

    @patch("reducto.Reducto")
    def test_parse_pdf_api_error(self, mock_reducto, tmp_path):
        """Test handling of API errors."""
        from src.pdf_converter.reducto_client import (
            ReductoClient,
            ReductoConfig,
            ReductoAPIError,
        )

        mock_reducto.return_value.parse.run.side_effect = Exception("API Error")

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf")

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        with pytest.raises(ReductoAPIError, match="API Error"):
            client.parse_pdf(pdf_path)

    def test_parse_pdf_file_not_found(self):
        """Test error when PDF file doesn't exist."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        with pytest.raises(FileNotFoundError):
            client.parse_pdf(Path("/nonexistent/file.pdf"))


class TestReductoClientBatch:
    """Tests for batch processing functionality."""

    @patch("reducto.Reducto")
    def test_parse_directory(self, mock_reducto, tmp_path):
        """Test parsing all PDFs in a directory."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        # Create mock response
        mock_chunk = MagicMock()
        mock_chunk.content = "Parsed content"
        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]
        mock_reducto.return_value.parse.run.return_value = mock_response

        # Create test PDFs
        (tmp_path / "doc1.pdf").write_bytes(b"%PDF-1.4 doc1")
        (tmp_path / "doc2.pdf").write_bytes(b"%PDF-1.4 doc2")
        (tmp_path / "not_pdf.txt").write_text("not a pdf")

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        results = client.parse_directory(tmp_path)

        assert len(results) == 2
        assert all(isinstance(r[0], Path) for r in results)
        assert all(isinstance(r[1], str) for r in results)

    @patch("reducto.Reducto")
    def test_parse_directory_empty(self, mock_reducto, tmp_path):
        """Test parsing empty directory."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        results = client.parse_directory(tmp_path)

        assert results == []

    @patch("reducto.Reducto")
    def test_parse_directory_with_callback(self, mock_reducto, tmp_path):
        """Test parsing with progress callback."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        mock_chunk = MagicMock()
        mock_chunk.content = "Content"
        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]
        mock_reducto.return_value.parse.run.return_value = mock_response

        (tmp_path / "doc1.pdf").write_bytes(b"%PDF-1.4 doc1")
        (tmp_path / "doc2.pdf").write_bytes(b"%PDF-1.4 doc2")

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        progress_calls = []

        def on_progress(pdf_path, index, total):
            progress_calls.append((pdf_path.name, index, total))

        client.parse_directory(tmp_path, on_progress=on_progress)

        assert len(progress_calls) == 2
        assert progress_calls[0][1] == 1
        assert progress_calls[0][2] == 2


class TestReductoParseResult:
    """Tests for ParseResult dataclass."""

    def test_create_parse_result(self):
        """Test creating a parse result."""
        from src.pdf_converter.reducto_client import ParseResult

        result = ParseResult(
            source_path=Path("/path/to/doc.pdf"),
            markdown="# Title\n\nContent",
            success=True,
        )

        assert result.source_path == Path("/path/to/doc.pdf")
        assert result.markdown == "# Title\n\nContent"
        assert result.success is True
        assert result.error_message is None

    def test_create_failed_result(self):
        """Test creating a failed parse result."""
        from src.pdf_converter.reducto_client import ParseResult

        result = ParseResult(
            source_path=Path("/path/to/doc.pdf"),
            markdown="",
            success=False,
            error_message="API rate limit exceeded",
        )

        assert result.success is False
        assert result.error_message == "API rate limit exceeded"

    def test_result_output_filename(self):
        """Test generating output filename from result."""
        from src.pdf_converter.reducto_client import ParseResult

        result = ParseResult(
            source_path=Path("/path/to/my document.pdf"),
            markdown="Content",
            success=True,
        )

        assert result.output_filename == "my document.md"

    def test_save_markdown(self, tmp_path):
        """Test saving markdown to file."""
        from src.pdf_converter.reducto_client import ParseResult

        result = ParseResult(
            source_path=Path("/path/to/doc.pdf"),
            markdown="# Title\n\nContent here",
            success=True,
        )

        output_path = result.save(tmp_path)

        assert output_path.exists()
        assert output_path.name == "doc.md"
        assert output_path.read_text() == "# Title\n\nContent here"
