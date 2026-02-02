"""PDF to Markdown conversion logic."""

import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


# Tool metadata: (display_name, description, estimated_time)
TOOL_META = {
    "docling": ("Docling", "Mixed content, good tables", "~2-3 min/100pg"),
    "marker": ("Marker", "Fast, clean layouts", "~1-2 min/100pg"),
    "nougat": ("Nougat", "Academic, math-heavy", "~5-10 min/100pg"),
}

SUPPORTED_TOOLS = {"marker", "docling", "nougat"}


@dataclass
class ConversionResult:
    """Result of a PDF conversion."""
    success: bool
    output_path: Optional[Path]
    tool_used: str
    error_message: Optional[str] = None


def validate_pdf_path(pdf_path: str) -> tuple[bool, str]:
    """
    Validate that a PDF path exists and is readable.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(pdf_path)

    if not path.exists():
        return False, f"PDF file not found: {pdf_path}"

    if not path.is_file():
        return False, f"Path is not a file: {pdf_path}"

    if path.suffix.lower() != ".pdf":
        return False, f"File is not a PDF: {pdf_path}"

    return True, ""


def validate_tool(tool: str) -> tuple[bool, str]:
    """
    Validate that a tool name is supported.

    Args:
        tool: Tool name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if tool not in SUPPORTED_TOOLS:
        return False, f"Unsupported tool: {tool}. Must be one of: {SUPPORTED_TOOLS}"
    return True, ""


def find_output_file(output_dir: str, pdf_stem: str, tool: str) -> Optional[Path]:
    """
    Find the output markdown file after conversion.

    Args:
        output_dir: Directory where output was written
        pdf_stem: Original PDF filename without extension
        tool: Tool that was used for conversion

    Returns:
        Path to output file, or None if not found
    """
    output_path = Path(output_dir)

    # Tool-specific patterns
    patterns = {
        "docling": [f"{pdf_stem}_docling.md", f"{pdf_stem}.md"],
        "marker": [f"{pdf_stem}.md", f"{pdf_stem}/*.md"],
        "nougat": [f"{pdf_stem}.md", f"{pdf_stem}.mmd"],
    }

    # Try direct matches first
    for pattern in patterns.get(tool, [f"{pdf_stem}.md"]):
        matches = list(output_path.glob(pattern))
        if matches:
            return matches[0]

    # Fallback: find any .md file
    md_files = list(output_path.rglob("*.md"))
    if md_files:
        return max(md_files, key=lambda p: p.stat().st_mtime)

    # Check for .mmd files (Nougat)
    mmd_files = list(output_path.rglob("*.mmd"))
    if mmd_files:
        return mmd_files[0]

    return None


def convert_pdf(
    pdf_path: str,
    output_dir: str,
    tool: str,
    dry_run: bool = False
) -> ConversionResult:
    """
    Convert a PDF to markdown using the specified tool.

    Note: This function is designed for testing. For production use,
    use Reducto.ai via 'kidkazz inbox parse'. This provides a subprocess-based
    alternative for local/CI testing.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory for output files
        tool: Tool to use ("marker", "docling", or "nougat")
        dry_run: If True, validate inputs but don't actually convert

    Returns:
        ConversionResult with success status and output path
    """
    # Validate inputs
    valid, error = validate_pdf_path(pdf_path)
    if not valid:
        return ConversionResult(
            success=False,
            output_path=None,
            tool_used=tool,
            error_message=error
        )

    valid, error = validate_tool(tool)
    if not valid:
        return ConversionResult(
            success=False,
            output_path=None,
            tool_used=tool,
            error_message=error
        )

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if dry_run:
        return ConversionResult(
            success=True,
            output_path=None,
            tool_used=tool,
            error_message="Dry run - no conversion performed"
        )

    # Build command based on tool
    pdf_stem = Path(pdf_path).stem

    commands = {
        "marker": ["marker_single", pdf_path, "--output_dir", output_dir],
        "nougat": ["nougat", pdf_path, "-o", output_dir, "--no-skipping"],
        "docling": None,  # Docling uses Python API, not CLI
    }

    if tool == "docling":
        # Docling conversion via Python
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            markdown = result.document.export_to_markdown()

            output_name = f"{pdf_stem}_docling.md"
            output_path = Path(output_dir) / output_name
            output_path.write_text(markdown, encoding="utf-8")

            return ConversionResult(
                success=True,
                output_path=output_path,
                tool_used=tool
            )
        except ImportError:
            return ConversionResult(
                success=False,
                output_path=None,
                tool_used=tool,
                error_message="Docling not installed. Run: pip install docling"
            )
        except Exception as e:
            return ConversionResult(
                success=False,
                output_path=None,
                tool_used=tool,
                error_message=f"Docling conversion failed: {e!s}"
            )
    else:
        # CLI-based conversion
        cmd = commands.get(tool)
        if not cmd:
            return ConversionResult(
                success=False,
                output_path=None,
                tool_used=tool,
                error_message=f"No command defined for tool: {tool}"
            )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode != 0:
                return ConversionResult(
                    success=False,
                    output_path=None,
                    tool_used=tool,
                    error_message=f"Command failed: {result.stderr}"
                )

            # Find output file
            output_file = find_output_file(output_dir, pdf_stem, tool)

            if not output_file:
                return ConversionResult(
                    success=False,
                    output_path=None,
                    tool_used=tool,
                    error_message="Conversion completed but output file not found"
                )

            # Rename .mmd to .md for Nougat
            if output_file.suffix == ".mmd":
                new_path = output_file.with_suffix(".md")
                output_file.rename(new_path)
                output_file = new_path

            return ConversionResult(
                success=True,
                output_path=output_file,
                tool_used=tool
            )

        except FileNotFoundError:
            return ConversionResult(
                success=False,
                output_path=None,
                tool_used=tool,
                error_message=f"{tool} not installed. Run: pip install {tool}-pdf"
            )
        except subprocess.TimeoutExpired:
            return ConversionResult(
                success=False,
                output_path=None,
                tool_used=tool,
                error_message="Conversion timed out after 1 hour"
            )
        except Exception as e:
            return ConversionResult(
                success=False,
                output_path=None,
                tool_used=tool,
                error_message=f"Conversion failed: {e!s}"
            )
