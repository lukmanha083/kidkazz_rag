"""Tests for Reducto.ai client.

This module tests the Reducto API client for PDF parsing.
Tests are written BEFORE implementation following TDD principles.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import src.pdf_converter.reducto_client as reducto_module


@pytest.fixture
def mock_reducto_sdk():
    """Fixture to mock Reducto SDK components."""
    mock_reducto_class = MagicMock()
    mock_retrieval_class = MagicMock()
    mock_chunking_class = MagicMock()

    # Store original values
    original_reducto = reducto_module.Reducto
    original_retrieval = reducto_module.Retrieval
    original_chunking = reducto_module.Chunking

    # Set mock values
    reducto_module.Reducto = mock_reducto_class
    reducto_module.Retrieval = mock_retrieval_class
    reducto_module.Chunking = mock_chunking_class

    yield {
        "Reducto": mock_reducto_class,
        "Retrieval": mock_retrieval_class,
        "Chunking": mock_chunking_class,
    }

    # Restore original values
    reducto_module.Reducto = original_reducto
    reducto_module.Retrieval = original_retrieval
    reducto_module.Chunking = original_chunking


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

    def test_client_initializes_on_access(self, mock_reducto_sdk):
        """Test SDK client initializes when accessed."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        # Access the client property
        _ = client.client

        mock_reducto_sdk["Reducto"].assert_called_once_with(api_key="test_key")

    def test_parse_pdf_success(self, mock_reducto_sdk, tmp_path):
        """Test successful PDF parsing."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        # Create mock response
        mock_chunk1 = MagicMock()
        mock_chunk1.content = "# Document Title"
        mock_chunk2 = MagicMock()
        mock_chunk2.content = "This is the content of the document."

        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk1, mock_chunk2]

        mock_upload = MagicMock()
        mock_upload.file_id = "test_file_id"
        mock_reducto_sdk["Reducto"].return_value.upload.return_value = mock_upload
        mock_reducto_sdk["Reducto"].return_value.parse.run.return_value = mock_response

        # Create test PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        result = client.parse_pdf(pdf_path)

        assert "# Document Title" in result
        assert "This is the content of the document." in result

    def test_parse_pdf_with_custom_chunk_mode(self, mock_reducto_sdk, tmp_path):
        """Test PDF parsing with custom chunk mode uses Retrieval and Chunking."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        mock_chunk = MagicMock()
        mock_chunk.content = "Content"
        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]

        mock_upload = MagicMock()
        mock_upload.file_id = "test_file_id"
        mock_reducto_sdk["Reducto"].return_value.upload.return_value = mock_upload
        mock_reducto_sdk["Reducto"].return_value.parse.run.return_value = mock_response

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf")

        config = ReductoConfig(api_key="test_key", chunk_mode="section")
        client = ReductoClient(config)
        client.parse_pdf(pdf_path)

        # Verify Chunking was called with correct chunk_mode
        mock_reducto_sdk["Chunking"].assert_called_once_with(chunk_mode="section")
        # Verify Retrieval was called with Chunking result
        mock_reducto_sdk["Retrieval"].assert_called_once()

    def test_parse_pdf_disabled_chunk_mode(self, mock_reducto_sdk, tmp_path):
        """Test PDF parsing with disabled chunk mode skips retrieval."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        mock_chunk = MagicMock()
        mock_chunk.content = "Content"
        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]

        mock_upload = MagicMock()
        mock_upload.file_id = "test_file_id"
        mock_reducto_sdk["Reducto"].return_value.upload.return_value = mock_upload
        mock_reducto_sdk["Reducto"].return_value.parse.run.return_value = mock_response

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf")

        config = ReductoConfig(api_key="test_key", chunk_mode="disabled")
        client = ReductoClient(config)
        client.parse_pdf(pdf_path)

        # Verify Chunking and Retrieval were NOT called when disabled
        mock_reducto_sdk["Chunking"].assert_not_called()
        mock_reducto_sdk["Retrieval"].assert_not_called()

        # Verify parse.run was called without retrieval param
        call_kwargs = mock_reducto_sdk["Reducto"].return_value.parse.run.call_args[1]
        assert "retrieval" not in call_kwargs

    def test_parse_pdf_api_error(self, mock_reducto_sdk, tmp_path):
        """Test handling of API errors."""
        from src.pdf_converter.reducto_client import (
            ReductoClient,
            ReductoConfig,
            ReductoAPIError,
        )

        mock_upload = MagicMock()
        mock_upload.file_id = "test_file_id"
        mock_reducto_sdk["Reducto"].return_value.upload.return_value = mock_upload
        mock_reducto_sdk["Reducto"].return_value.parse.run.side_effect = Exception("API Error")

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

    def test_parse_directory(self, mock_reducto_sdk, tmp_path):
        """Test parsing all PDFs in a directory."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        # Create mock response
        mock_chunk = MagicMock()
        mock_chunk.content = "Parsed content"
        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]

        mock_upload = MagicMock()
        mock_upload.file_id = "test_file_id"
        mock_reducto_sdk["Reducto"].return_value.upload.return_value = mock_upload
        mock_reducto_sdk["Reducto"].return_value.parse.run.return_value = mock_response

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

    def test_parse_directory_empty(self, tmp_path):
        """Test parsing empty directory."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        results = client.parse_directory(tmp_path)

        assert results == []

    def test_parse_directory_with_callback(self, mock_reducto_sdk, tmp_path):
        """Test parsing with progress callback."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        mock_chunk = MagicMock()
        mock_chunk.content = "Content"
        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]

        mock_upload = MagicMock()
        mock_upload.file_id = "test_file_id"
        mock_reducto_sdk["Reducto"].return_value.upload.return_value = mock_upload
        mock_reducto_sdk["Reducto"].return_value.parse.run.return_value = mock_response

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

    def test_parse_directory_recursive(self, mock_reducto_sdk, tmp_path):
        """Test parsing PDFs recursively in subdirectories."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        mock_chunk = MagicMock()
        mock_chunk.content = "Content"
        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]

        mock_upload = MagicMock()
        mock_upload.file_id = "test_file_id"
        mock_reducto_sdk["Reducto"].return_value.upload.return_value = mock_upload
        mock_reducto_sdk["Reducto"].return_value.parse.run.return_value = mock_response

        # Create test PDFs in subdirectories
        (tmp_path / "doc1.pdf").write_bytes(b"%PDF-1.4 doc1")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "doc2.pdf").write_bytes(b"%PDF-1.4 doc2")

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        # Non-recursive should only find 1 file
        results_non_recursive = client.parse_directory(tmp_path, recursive=False)
        assert len(results_non_recursive) == 1

        # Recursive should find 2 files
        results_recursive = client.parse_directory(tmp_path, recursive=True)
        assert len(results_recursive) == 2

    def test_parse_files(self, mock_reducto_sdk, tmp_path):
        """Test parsing a list of PDF files."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        mock_chunk = MagicMock()
        mock_chunk.content = "Parsed content"
        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]

        mock_upload = MagicMock()
        mock_upload.file_id = "test_file_id"
        mock_reducto_sdk["Reducto"].return_value.upload.return_value = mock_upload
        mock_reducto_sdk["Reducto"].return_value.parse.run.return_value = mock_response

        # Create test PDFs
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.write_bytes(b"%PDF-1.4 doc1")
        pdf2.write_bytes(b"%PDF-1.4 doc2")

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        # Parse specific files
        results = client.parse_files([pdf1, pdf2])

        assert len(results) == 2
        assert results[0][0] == pdf1
        assert results[1][0] == pdf2

    def test_parse_files_with_callback(self, mock_reducto_sdk, tmp_path):
        """Test parse_files with progress callback."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        mock_chunk = MagicMock()
        mock_chunk.content = "Content"
        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]

        mock_upload = MagicMock()
        mock_upload.file_id = "test_file_id"
        mock_reducto_sdk["Reducto"].return_value.upload.return_value = mock_upload
        mock_reducto_sdk["Reducto"].return_value.parse.run.return_value = mock_response

        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.write_bytes(b"%PDF-1.4 doc1")
        pdf2.write_bytes(b"%PDF-1.4 doc2")

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        progress_calls = []

        def on_progress(pdf_path, index, total):
            progress_calls.append((pdf_path.name, index, total))

        client.parse_files([pdf1, pdf2], on_progress=on_progress)

        assert len(progress_calls) == 2
        assert progress_calls[0] == ("doc1.pdf", 1, 2)
        assert progress_calls[1] == ("doc2.pdf", 2, 2)


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

        # Filenames are slugified for CLI-friendly output
        assert result.output_filename == "my_document.md"

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


class TestHeaderReconstruction:
    """Tests for header reconstruction from block metadata."""

    def test_process_chunk_without_blocks(self):
        """Test processing chunk without block metadata returns content as-is."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        mock_chunk = MagicMock()
        mock_chunk.content = "Plain text content"
        mock_chunk.blocks = None

        result = client._process_chunk_with_headers(mock_chunk)
        assert result == "Plain text content"

    def test_process_chunk_with_header_block(self):
        """Test processing chunk with Header block type reconstructs markdown header."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        # Create mock block with Header type
        mock_block = MagicMock()
        mock_block.type = "Header"
        mock_block.content = "Introduction"
        mock_block.level = None  # No explicit level

        mock_chunk = MagicMock()
        mock_chunk.content = "Introduction"
        mock_chunk.blocks = [mock_block]

        result = client._process_chunk_with_headers(mock_chunk)
        # Should have markdown header syntax
        assert result.startswith("#")
        assert "Introduction" in result

    def test_process_chunk_with_header_and_text_blocks(self):
        """Test processing chunk with mixed Header and Text blocks."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        # Create mock blocks
        header_block = MagicMock()
        header_block.type = "Header"
        header_block.content = "Chapter One"
        header_block.level = None

        text_block = MagicMock()
        text_block.type = "Text"
        text_block.content = "This is the chapter content."

        mock_chunk = MagicMock()
        mock_chunk.content = "Chapter One\n\nThis is the chapter content."
        mock_chunk.blocks = [header_block, text_block]

        result = client._process_chunk_with_headers(mock_chunk)

        # Should contain markdown header for Chapter One (h1 due to "chapter" keyword)
        assert "# Chapter One" in result
        assert "This is the chapter content." in result

    def test_process_chunk_with_explicit_header_level(self):
        """Test header block with explicit level attribute."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        mock_block = MagicMock()
        mock_block.type = "Header"
        mock_block.content = "Subsection Title"
        mock_block.level = 3  # Explicit h3

        mock_chunk = MagicMock()
        mock_chunk.blocks = [mock_block]

        result = client._process_chunk_with_headers(mock_chunk)
        assert result == "### Subsection Title"

    def test_process_chunk_lowercase_header_type(self):
        """Test header type matching is case-insensitive."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        mock_block = MagicMock()
        mock_block.type = "header"  # lowercase
        mock_block.content = "Title"
        mock_block.level = 2

        mock_chunk = MagicMock()
        mock_chunk.blocks = [mock_block]

        result = client._process_chunk_with_headers(mock_chunk)
        assert result == "## Title"

    def test_infer_header_level_all_caps_short(self):
        """Test all caps short content infers h1."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        level = client._infer_header_level("INTRODUCTION")
        assert level == 1

    def test_infer_header_level_chapter_keyword(self):
        """Test chapter keyword infers h1."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        level = client._infer_header_level("Chapter 1: Getting Started")
        assert level == 1

    def test_infer_header_level_section_keyword(self):
        """Test section keyword infers h1."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        level = client._infer_header_level("Section 2.1: Overview")
        assert level == 1

    def test_infer_header_level_medium_content(self):
        """Test medium length content infers h2."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        level = client._infer_header_level("Understanding the Basics of Inventory")
        assert level == 2

    def test_infer_header_level_long_content(self):
        """Test long content infers h3."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        long_title = "This is a very long subsection title that describes in detail what this particular section covers and includes multiple topics"
        level = client._infer_header_level(long_title)
        assert level == 3

    def test_chunks_to_markdown_with_headers(self):
        """Test _chunks_to_markdown reconstructs headers from blocks."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        # Create chunk with header block
        header_block = MagicMock()
        header_block.type = "Header"
        header_block.content = "Document Title"
        header_block.level = 1

        text_block = MagicMock()
        text_block.type = "Text"
        text_block.content = "Some paragraph text here."

        mock_chunk = MagicMock()
        mock_chunk.blocks = [header_block, text_block]

        result = client._chunks_to_markdown([mock_chunk])

        assert "# Document Title" in result
        assert "Some paragraph text here." in result

    def test_chunks_to_markdown_fallback_to_content(self):
        """Test _chunks_to_markdown falls back to chunk.content when no blocks."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        mock_chunk = MagicMock()
        mock_chunk.content = "Fallback content"
        mock_chunk.blocks = None

        result = client._chunks_to_markdown([mock_chunk])

        assert result == "Fallback content"

    def test_process_chunk_empty_blocks(self):
        """Test processing chunk with empty blocks list falls back to content."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        mock_chunk = MagicMock()
        mock_chunk.content = "Fallback content"
        mock_chunk.blocks = []

        result = client._process_chunk_with_headers(mock_chunk)
        assert result == "Fallback content"

    def test_process_chunk_skips_empty_block_content(self):
        """Test blocks with empty content are skipped."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        empty_block = MagicMock()
        empty_block.type = "Text"
        empty_block.content = ""

        text_block = MagicMock()
        text_block.type = "Text"
        text_block.content = "Valid content"

        mock_chunk = MagicMock()
        mock_chunk.content = "Fallback"
        mock_chunk.blocks = [empty_block, text_block]

        result = client._process_chunk_with_headers(mock_chunk)
        assert result == "Valid content"
        assert "Fallback" not in result


class TestBlockMetadataExtraction:
    """Tests for extracting and preserving block metadata from Reducto (TDD)."""

    def test_extract_block_metadata_from_header_block(self):
        """Should extract header metadata from a Header block."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        mock_block = MagicMock()
        mock_block.type = "Header"
        mock_block.content = "Chapter 1: Introduction"
        mock_block.level = 1

        mock_chunk = MagicMock()
        mock_chunk.blocks = [mock_block]

        result = client.extract_block_metadata(mock_chunk)

        assert result["header_text"] == "Chapter 1: Introduction"
        assert result["header_level"] == 1
        assert result["block_type"] == "Header"

    def test_extract_block_metadata_from_text_block(self):
        """Should extract metadata from a Text block."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        mock_block = MagicMock()
        mock_block.type = "Text"
        mock_block.content = "This is paragraph text."

        mock_chunk = MagicMock()
        mock_chunk.blocks = [mock_block]

        result = client.extract_block_metadata(mock_chunk)

        assert result["header_text"] is None
        assert result["header_level"] is None
        assert result["block_type"] == "Text"

    def test_extract_block_metadata_infers_header_level(self):
        """Should infer header level when not explicitly provided."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        mock_block = MagicMock()
        mock_block.type = "Header"
        mock_block.content = "INTRODUCTION"  # All caps, should infer level 1
        mock_block.level = None

        mock_chunk = MagicMock()
        mock_chunk.blocks = [mock_block]

        result = client.extract_block_metadata(mock_chunk)

        assert result["header_text"] == "INTRODUCTION"
        assert result["header_level"] == 1  # Inferred from all caps
        assert result["block_type"] == "Header"

    def test_extract_block_metadata_no_blocks(self):
        """Should return empty metadata when chunk has no blocks."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        mock_chunk = MagicMock()
        mock_chunk.blocks = None

        result = client.extract_block_metadata(mock_chunk)

        assert result["header_text"] is None
        assert result["header_level"] is None
        assert result["block_type"] is None

    def test_extract_block_metadata_uses_first_block_type(self):
        """Should use the first block's type as the primary block_type."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        header_block = MagicMock()
        header_block.type = "Header"
        header_block.content = "Section Title"
        header_block.level = 2

        text_block = MagicMock()
        text_block.type = "Text"
        text_block.content = "Some text follows."

        mock_chunk = MagicMock()
        mock_chunk.blocks = [header_block, text_block]

        result = client.extract_block_metadata(mock_chunk)

        # Should use the header block's metadata
        assert result["header_text"] == "Section Title"
        assert result["header_level"] == 2
        assert result["block_type"] == "Header"

    def test_extract_block_metadata_table_block(self):
        """Should identify Table block type."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        mock_block = MagicMock()
        mock_block.type = "Table"
        mock_block.content = "| A | B |\n|---|---|\n| 1 | 2 |"

        mock_chunk = MagicMock()
        mock_chunk.blocks = [mock_block]

        result = client.extract_block_metadata(mock_chunk)

        assert result["header_text"] is None
        assert result["header_level"] is None
        assert result["block_type"] == "Table"

    def test_chunks_to_markdown_with_metadata_returns_list(self):
        """Should return list of (markdown, metadata) tuples."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key")
        client = ReductoClient(config)

        header_block = MagicMock()
        header_block.type = "Header"
        header_block.content = "Chapter 1"
        header_block.level = 1

        text_block = MagicMock()
        text_block.type = "Text"
        text_block.content = "Content here."

        chunk1 = MagicMock()
        chunk1.blocks = [header_block]

        chunk2 = MagicMock()
        chunk2.blocks = [text_block]

        result = client.chunks_to_markdown_with_metadata([chunk1, chunk2])

        assert len(result) == 2
        assert isinstance(result, list)
        # First item should have header metadata
        assert result[0]["markdown"] == "# Chapter 1"
        assert result[0]["metadata"]["header_text"] == "Chapter 1"
        assert result[0]["metadata"]["header_level"] == 1
        assert result[0]["metadata"]["block_type"] == "Header"
        # Second item should have text metadata
        assert result[1]["markdown"] == "Content here."
        assert result[1]["metadata"]["header_text"] is None
        assert result[1]["metadata"]["block_type"] == "Text"


class TestResponseToMarkdownHeaderExtraction:
    """Tests for header extraction in _response_to_markdown for all chunk modes."""

    def test_chunk_mode_block_extracts_headers(self):
        """Verify block chunk mode extracts headers from block metadata."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        # Simulate block mode (raw_output=False is set by CLI for block/variable/page/section)
        config = ReductoConfig(api_key="test_key", chunk_mode="block", raw_output=False)
        client = ReductoClient(config)

        # Create mock response with header block
        header_block = MagicMock()
        header_block.type = "Header"
        header_block.content = "Chapter 1: Introduction"
        header_block.level = 1

        text_block = MagicMock()
        text_block.type = "Text"
        text_block.content = "This is the chapter content."

        mock_chunk = MagicMock()
        mock_chunk.content = "Chapter 1: Introduction\n\nThis is the chapter content."
        mock_chunk.blocks = [header_block, text_block]

        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]

        result = client._response_to_markdown(mock_response)

        # Should contain markdown header syntax
        assert "# Chapter 1: Introduction" in result
        assert "This is the chapter content." in result

    def test_chunk_mode_variable_extracts_headers(self):
        """Verify variable chunk mode also extracts headers."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key", chunk_mode="variable", raw_output=False)
        client = ReductoClient(config)

        header_block = MagicMock()
        header_block.type = "Header"
        header_block.content = "Getting Started"
        header_block.level = 2

        mock_chunk = MagicMock()
        mock_chunk.content = "Getting Started"
        mock_chunk.blocks = [header_block]

        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]

        result = client._response_to_markdown(mock_response)

        assert "## Getting Started" in result

    def test_chunk_mode_page_extracts_headers(self):
        """Verify page chunk mode also extracts headers."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key", chunk_mode="page", raw_output=False)
        client = ReductoClient(config)

        header_block = MagicMock()
        header_block.type = "Header"
        header_block.content = "Page Title"
        header_block.level = 1

        mock_chunk = MagicMock()
        mock_chunk.content = "Page Title"
        mock_chunk.blocks = [header_block]

        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]

        result = client._response_to_markdown(mock_response)

        assert "# Page Title" in result

    def test_chunk_mode_section_extracts_headers(self):
        """Verify section chunk mode also extracts headers."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key", chunk_mode="section", raw_output=False)
        client = ReductoClient(config)

        header_block = MagicMock()
        header_block.type = "Header"
        header_block.content = "Section Heading"
        header_block.level = 2

        mock_chunk = MagicMock()
        mock_chunk.content = "Section Heading"
        mock_chunk.blocks = [header_block]

        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]

        result = client._response_to_markdown(mock_response)

        assert "## Section Heading" in result

    def test_multiple_chunks_with_headers_separated_by_dividers(self):
        """Verify multiple chunks are separated by --- and headers are extracted."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key", raw_output=False)
        client = ReductoClient(config)

        # First chunk with header
        header_block1 = MagicMock()
        header_block1.type = "Header"
        header_block1.content = "Section A"
        header_block1.level = 1

        chunk1 = MagicMock()
        chunk1.content = "Section A"
        chunk1.blocks = [header_block1]

        # Second chunk with header
        header_block2 = MagicMock()
        header_block2.type = "Header"
        header_block2.content = "Section B"
        header_block2.level = 1

        chunk2 = MagicMock()
        chunk2.content = "Section B"
        chunk2.blocks = [header_block2]

        mock_response = MagicMock()
        mock_response.result.chunks = [chunk1, chunk2]

        result = client._response_to_markdown(mock_response)

        # Both headers should be extracted
        assert "# Section A" in result
        assert "# Section B" in result
        # Chunks should be separated by ---
        assert "---" in result

    def test_blocks_as_dictionaries(self):
        """Verify header extraction works when blocks are dictionaries (real Reducto API format)."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key", chunk_mode="block", raw_output=False)
        client = ReductoClient(config)

        # Simulate real Reducto API response where blocks are dictionaries
        mock_chunk = MagicMock()
        mock_chunk.content = "Chapter 1\n\nSome content here."
        mock_chunk.blocks = [
            {"type": "Header", "content": "Chapter 1", "level": 1},
            {"type": "Text", "content": "Some content here."},
        ]

        mock_response = MagicMock()
        mock_response.result.chunks = [mock_chunk]

        result = client._response_to_markdown(mock_response)

        # Should extract header from dictionary block
        assert "# Chapter 1" in result
        assert "Some content here." in result

    def test_raw_output_true_uses_continuous_markdown(self):
        """Verify raw_output=True uses _chunks_to_markdown without dividers."""
        from src.pdf_converter.reducto_client import ReductoClient, ReductoConfig

        config = ReductoConfig(api_key="test_key", raw_output=True)
        client = ReductoClient(config)

        header_block = MagicMock()
        header_block.type = "Header"
        header_block.content = "Title"
        header_block.level = 1

        text_block = MagicMock()
        text_block.type = "Text"
        text_block.content = "Content"

        chunk1 = MagicMock()
        chunk1.blocks = [header_block]

        chunk2 = MagicMock()
        chunk2.blocks = [text_block]

        mock_response = MagicMock()
        mock_response.result.chunks = [chunk1, chunk2]

        result = client._response_to_markdown(mock_response)

        assert "# Title" in result
        assert "Content" in result
        # Should NOT have --- separators in raw mode
        assert "---" not in result
