"""Tests for CLI utility functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli.utils import (
    format_file_size,
    get_project_config_path,
    get_user_config_path,
    parse_chunk_sizes,
    parse_tags,
    resolve_doc_id,
    resolve_title,
    validate_file_exists,
)


class TestParseChunkSizes:
    """Tests for chunk size parsing."""

    def test_valid_sizes(self):
        """Test parsing valid chunk sizes."""
        result = parse_chunk_sizes("2048,512,256")
        assert result == (2048, 512, 256)

    def test_different_values(self):
        """Test parsing different values."""
        result = parse_chunk_sizes("1024,256,128")
        assert result == (1024, 256, 128)

    def test_invalid_format_too_few(self):
        """Test error on too few values."""
        with pytest.raises(ValueError, match="must have 3 values"):
            parse_chunk_sizes("2048,512")

    def test_invalid_format_too_many(self):
        """Test error on too many values."""
        with pytest.raises(ValueError, match="must have 3 values"):
            parse_chunk_sizes("2048,512,256,128")

    def test_invalid_non_integer(self):
        """Test error on non-integer values."""
        with pytest.raises(ValueError, match="must be integers"):
            parse_chunk_sizes("2048,abc,256")


class TestResolveDocId:
    """Tests for document ID resolution."""

    def test_explicit_doc_id(self):
        """Test explicit doc_id takes precedence."""
        result = resolve_doc_id(Path("file.md"), "custom_id")
        assert result == "custom_id"

    def test_doc_id_from_filename(self):
        """Test doc_id from filename stem."""
        result = resolve_doc_id(Path("/path/to/document.md"))
        assert result == "document"

    def test_doc_id_from_complex_path(self):
        """Test doc_id from complex path."""
        result = resolve_doc_id(Path("/some/deep/path/my_file.md"))
        assert result == "my_file"


class TestResolveTitle:
    """Tests for title resolution."""

    def test_explicit_title(self):
        """Test explicit title takes precedence."""
        result = resolve_title(Path("file.md"), "# Content", "Custom Title")
        assert result == "Custom Title"

    def test_title_from_h1(self):
        """Test title from markdown H1."""
        content = "# My Document Title\n\nSome content here."
        result = resolve_title(Path("file.md"), content)
        assert result == "My Document Title"

    def test_title_from_h1_with_spaces(self):
        """Test title extraction handles whitespace."""
        content = "#   Spaced Title   \n\nContent"
        result = resolve_title(Path("file.md"), content)
        assert result == "Spaced Title"

    def test_title_from_filename_no_h1(self):
        """Test fallback to filename when no H1."""
        content = "No heading here, just content."
        result = resolve_title(Path("my_document.md"), content)
        assert result == "my_document"

    def test_title_from_filename_empty_content(self):
        """Test fallback with no content."""
        result = resolve_title(Path("document.md"))
        assert result == "document"


class TestFormatFileSize:
    """Tests for file size formatting."""

    def test_bytes(self):
        """Test formatting bytes."""
        assert format_file_size(500) == "500.0 B"

    def test_kilobytes(self):
        """Test formatting kilobytes."""
        assert format_file_size(2048) == "2.0 KB"

    def test_megabytes(self):
        """Test formatting megabytes."""
        assert format_file_size(1048576) == "1.0 MB"

    def test_gigabytes(self):
        """Test formatting gigabytes."""
        assert format_file_size(1073741824) == "1.0 GB"


class TestValidateFileExists:
    """Tests for file validation."""

    def test_existing_file(self, tmp_path):
        """Test validation of existing file."""
        test_file = tmp_path / "test.md"
        test_file.write_text("content")

        result = validate_file_exists(test_file)
        assert result == test_file

    def test_nonexistent_file(self, tmp_path):
        """Test error on nonexistent file."""
        with pytest.raises(FileNotFoundError):
            validate_file_exists(tmp_path / "nonexistent.md")

    def test_directory_not_file(self, tmp_path):
        """Test error when path is directory."""
        with pytest.raises(ValueError, match="Not a file"):
            validate_file_exists(tmp_path)

    def test_wrong_extension(self, tmp_path):
        """Test error on wrong extension."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with pytest.raises(ValueError, match="Expected .md"):
            validate_file_exists(test_file, extension="md")

    def test_correct_extension(self, tmp_path):
        """Test validation with correct extension."""
        test_file = tmp_path / "test.md"
        test_file.write_text("content")

        result = validate_file_exists(test_file, extension="md")
        assert result == test_file


class TestConfigPaths:
    """Tests for configuration path functions."""

    def test_project_config_path(self):
        """Test project config path."""
        path = get_project_config_path()
        assert path.name == ".kidkazz.toml"
        assert path.parent == Path.cwd()

    def test_user_config_path(self):
        """Test user config path."""
        path = get_user_config_path()
        assert path.name == "config.toml"
        assert "kidkazz" in str(path)
        assert ".config" in str(path) or "config" in str(path).lower()


class TestParseTags:
    """Tests for tag parsing."""

    def test_parses_comma_separated_tags(self):
        """Test parsing comma-separated tags."""
        result = parse_tags("inventory,accounting")
        assert result == ["inventory", "accounting"]

    def test_strips_whitespace(self):
        """Test whitespace is stripped from tags."""
        result = parse_tags("  inventory  ,  accounting  ")
        assert result == ["inventory", "accounting"]

    def test_returns_none_for_empty_string(self):
        """Test returns None for empty string."""
        result = parse_tags("")
        assert result is None

    def test_returns_none_for_none_input(self):
        """Test returns None for None input."""
        result = parse_tags(None)
        assert result is None

    def test_filters_empty_tags_from_consecutive_commas(self):
        """Test empty tags from consecutive commas are filtered."""
        result = parse_tags("inventory,,accounting")
        assert result == ["inventory", "accounting"]

    def test_filters_whitespace_only_tags(self):
        """Test whitespace-only tags are filtered."""
        result = parse_tags("inventory,   ,accounting")
        assert result == ["inventory", "accounting"]

    def test_returns_none_for_all_empty_tags(self):
        """Test returns None when all tags are empty."""
        result = parse_tags(",,,  ,")
        assert result is None

    def test_single_tag(self):
        """Test parsing single tag."""
        result = parse_tags("inventory")
        assert result == ["inventory"]

    def test_handles_leading_trailing_commas(self):
        """Test handles leading/trailing commas."""
        result = parse_tags(",inventory,accounting,")
        assert result == ["inventory", "accounting"]
