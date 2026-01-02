"""Tests for PDF conversion logic."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.pdf_converter.converter import (
    validate_pdf_path,
    validate_tool,
    find_output_file,
    convert_pdf,
    ConversionResult,
    SUPPORTED_TOOLS,
    TOOL_META,
)


class TestValidatePdfPath:
    """Tests for validate_pdf_path function."""

    def test_valid_pdf_path(self, single_pdf_path):
        """Should validate existing PDF file."""
        valid, error = validate_pdf_path(str(single_pdf_path[0]))
        assert valid is True
        assert error == ""

    def test_nonexistent_path(self):
        """Should reject non-existent path."""
        valid, error = validate_pdf_path("/nonexistent/file.pdf")
        assert valid is False
        assert "not found" in error

    def test_directory_path(self, temp_dir):
        """Should reject directory path."""
        valid, error = validate_pdf_path(str(temp_dir))
        assert valid is False
        assert "not a file" in error

    def test_non_pdf_file(self, temp_dir):
        """Should reject non-PDF files."""
        txt_file = temp_dir / "document.txt"
        txt_file.write_text("not a pdf")
        valid, error = validate_pdf_path(str(txt_file))
        assert valid is False
        assert "not a PDF" in error


class TestValidateTool:
    """Tests for validate_tool function."""

    def test_valid_tools(self):
        """Should accept all supported tools."""
        for tool in SUPPORTED_TOOLS:
            valid, error = validate_tool(tool)
            assert valid is True
            assert error == ""

    def test_invalid_tool(self):
        """Should reject unsupported tools."""
        valid, error = validate_tool("unknown_tool")
        assert valid is False
        assert "Unsupported tool" in error

    def test_case_sensitive(self):
        """Tool names should be case-sensitive."""
        valid, error = validate_tool("MARKER")
        assert valid is False


class TestFindOutputFile:
    """Tests for find_output_file function."""

    def test_finds_docling_output(self, temp_dir):
        """Should find docling-style output."""
        output_file = temp_dir / "document_docling.md"
        output_file.write_text("content")

        result = find_output_file(str(temp_dir), "document", "docling")
        assert result is not None
        assert result.name == "document_docling.md"

    def test_finds_marker_output(self, temp_dir):
        """Should find marker-style output."""
        output_file = temp_dir / "document.md"
        output_file.write_text("content")

        result = find_output_file(str(temp_dir), "document", "marker")
        assert result is not None
        assert result.name == "document.md"

    def test_finds_nougat_mmd_output(self, temp_dir):
        """Should find nougat .mmd output."""
        output_file = temp_dir / "document.mmd"
        output_file.write_text("content")

        result = find_output_file(str(temp_dir), "document", "nougat")
        assert result is not None
        assert result.name == "document.mmd"

    def test_finds_nested_output(self, temp_dir):
        """Should find output in subdirectory."""
        subdir = temp_dir / "document"
        subdir.mkdir()
        output_file = subdir / "document.md"
        output_file.write_text("content")

        result = find_output_file(str(temp_dir), "document", "marker")
        assert result is not None

    def test_returns_none_when_not_found(self, temp_dir):
        """Should return None when no output found."""
        result = find_output_file(str(temp_dir), "missing", "marker")
        assert result is None

    def test_finds_most_recent_fallback(self, temp_dir):
        """Should fall back to most recent .md file."""
        import time

        old_file = temp_dir / "old.md"
        old_file.write_text("old")
        time.sleep(0.1)

        new_file = temp_dir / "new.md"
        new_file.write_text("new")

        result = find_output_file(str(temp_dir), "different_name", "marker")
        assert result is not None
        assert result.name == "new.md"


class TestConvertPdf:
    """Tests for convert_pdf function."""

    def test_invalid_pdf_returns_failure(self):
        """Should return failure for invalid PDF path."""
        result = convert_pdf("/nonexistent/file.pdf", "/output", "marker")
        assert result.success is False
        assert "not found" in result.error_message

    def test_invalid_tool_returns_failure(self, single_pdf_path):
        """Should return failure for invalid tool."""
        result = convert_pdf(str(single_pdf_path[0]), "/output", "unknown")
        assert result.success is False
        assert "Unsupported tool" in result.error_message

    def test_dry_run_validates_only(self, single_pdf_path, temp_dir):
        """Dry run should validate but not convert."""
        result = convert_pdf(
            str(single_pdf_path[0]),
            str(temp_dir),
            "marker",
            dry_run=True
        )
        assert result.success is True
        assert result.output_path is None
        assert "Dry run" in result.error_message

    def test_creates_output_directory(self, single_pdf_path, temp_dir):
        """Should create output directory if needed."""
        output_dir = temp_dir / "new_output"
        result = convert_pdf(
            str(single_pdf_path[0]),
            str(output_dir),
            "marker",
            dry_run=True
        )
        assert output_dir.exists()

    def test_returns_conversion_result(self, single_pdf_path, temp_dir):
        """Should return ConversionResult object."""
        result = convert_pdf(
            str(single_pdf_path[0]),
            str(temp_dir),
            "marker",
            dry_run=True
        )
        assert isinstance(result, ConversionResult)
        assert result.tool_used == "marker"


class TestConvertPdfWithMocking:
    """Tests for convert_pdf with mocked external tools."""

    @patch("subprocess.run")
    def test_marker_command_structure(self, mock_run, single_pdf_path, temp_dir):
        """Should call marker with correct arguments."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        # Create expected output file
        output_file = temp_dir / "single.md"
        output_file.write_text("converted content")

        result = convert_pdf(
            str(single_pdf_path[0]),
            str(temp_dir),
            "marker"
        )

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "marker_single" in call_args
        assert str(single_pdf_path[0]) in call_args
        assert "--output_dir" in call_args

    @patch("subprocess.run")
    def test_nougat_command_structure(self, mock_run, single_pdf_path, temp_dir):
        """Should call nougat with correct arguments."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        # Create expected output file
        output_file = temp_dir / "single.md"
        output_file.write_text("converted content")

        result = convert_pdf(
            str(single_pdf_path[0]),
            str(temp_dir),
            "nougat"
        )

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "nougat" in call_args
        assert str(single_pdf_path[0]) in call_args
        assert "-o" in call_args
        assert "--no-skipping" in call_args

    @patch("subprocess.run")
    def test_handles_command_failure(self, mock_run, single_pdf_path, temp_dir):
        """Should handle command failure gracefully."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Error: conversion failed"
        )

        result = convert_pdf(
            str(single_pdf_path[0]),
            str(temp_dir),
            "marker"
        )

        assert result.success is False
        assert "Command failed" in result.error_message

    @patch("subprocess.run")
    def test_handles_missing_tool(self, mock_run, single_pdf_path, temp_dir):
        """Should handle missing tool (FileNotFoundError)."""
        mock_run.side_effect = FileNotFoundError("marker_single not found")

        result = convert_pdf(
            str(single_pdf_path[0]),
            str(temp_dir),
            "marker"
        )

        assert result.success is False
        assert "not installed" in result.error_message

    @patch("subprocess.run")
    def test_handles_timeout(self, mock_run, single_pdf_path, temp_dir):
        """Should handle conversion timeout."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 3600)

        result = convert_pdf(
            str(single_pdf_path[0]),
            str(temp_dir),
            "marker"
        )

        assert result.success is False
        assert "timed out" in result.error_message


class TestToolMeta:
    """Tests for tool metadata."""

    def test_all_supported_tools_have_meta(self):
        """All supported tools should have metadata."""
        for tool in SUPPORTED_TOOLS:
            assert tool in TOOL_META

    def test_meta_has_required_fields(self):
        """Each tool's metadata should have name, description, time."""
        for tool, meta in TOOL_META.items():
            assert len(meta) == 3
            name, desc, time_est = meta
            assert isinstance(name, str)
            assert isinstance(desc, str)
            assert isinstance(time_est, str)
