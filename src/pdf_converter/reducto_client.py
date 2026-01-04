"""Reducto.ai client for PDF parsing.

This module provides a client for the Reducto.ai API to convert
PDF documents to markdown format.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import os

from dotenv import load_dotenv

# Load .env file for API keys
load_dotenv()


class ReductoAPIError(Exception):
    """Exception raised when Reducto API call fails."""

    pass


@dataclass
class ReductoConfig:
    """Configuration for Reducto.ai API client.

    Attributes:
        api_key: Reducto API key for authentication.
        agentic: Enable AI-enhanced accuracy mode (uses more credits).
        chunk_mode: How to chunk output (variable, section, page, block).
        table_format: Table output format (md, html, json, csv).
    """

    api_key: str
    agentic: bool = False
    chunk_mode: str = "variable"
    table_format: str = "md"

    @classmethod
    def from_env(cls, env_var: str = "REDUCTO_API_KEY") -> "ReductoConfig":
        """Create config from environment variable.

        Args:
            env_var: Name of environment variable containing API key.

        Returns:
            ReductoConfig with API key from environment.

        Raises:
            ValueError: If environment variable is not set.
        """
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(
                f"{env_var} not set. Get your API key from https://reducto.ai"
            )
        return cls(api_key=api_key)


@dataclass
class ParseResult:
    """Result of parsing a PDF document.

    Attributes:
        source_path: Path to the original PDF file.
        markdown: Parsed markdown content.
        success: Whether parsing was successful.
        error_message: Error message if parsing failed.
    """

    source_path: Path
    markdown: str
    success: bool
    error_message: Optional[str] = None

    @property
    def output_filename(self) -> str:
        """Generate output filename from source path."""
        return self.source_path.stem + ".md"

    def save(self, output_dir: Path) -> Path:
        """Save markdown to output directory.

        Args:
            output_dir: Directory to save markdown file.

        Returns:
            Path to saved file.
        """
        output_path = output_dir / self.output_filename
        output_path.write_text(self.markdown, encoding="utf-8")
        return output_path


class ReductoClient:
    """Client for Reducto.ai PDF parsing API.

    This client wraps the Reducto Python SDK to provide
    a simple interface for parsing PDF documents.

    Example:
        >>> config = ReductoConfig.from_env()
        >>> client = ReductoClient(config)
        >>> markdown = client.parse_pdf(Path("document.pdf"))
    """

    def __init__(self, config: ReductoConfig) -> None:
        """Initialize client with configuration.

        Args:
            config: ReductoConfig with API key and settings.
        """
        self.config = config
        self._client = None

    @property
    def client(self):
        """Lazy-load the Reducto SDK client."""
        if self._client is None:
            from reducto import Reducto

            self._client = Reducto(api_key=self.config.api_key)
        return self._client

    def parse_pdf(self, pdf_path: Path) -> str:
        """Parse a PDF file and return markdown content.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Markdown content from the parsed PDF.

        Raises:
            FileNotFoundError: If PDF file doesn't exist.
            ReductoAPIError: If API call fails.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            response = self.client.parse.run(
                input=str(pdf_path),
                enhance={"agentic": self.config.agentic},
                retrieval={"chunking": {"chunk_mode": self.config.chunk_mode}},
                formatting={"table_output_format": self.config.table_format},
            )
            return self._response_to_markdown(response)
        except Exception as e:
            raise ReductoAPIError(str(e)) from e

    def _response_to_markdown(self, response) -> str:
        """Convert Reducto API response to markdown string.

        Args:
            response: Response from Reducto parse API.

        Returns:
            Markdown content as string.
        """
        chunks = response.result.chunks
        return "\n\n".join(chunk.content for chunk in chunks)

    def parse_files(
        self,
        pdf_files: list[Path],
        on_progress: Optional[Callable[[Path, int, int], None]] = None,
    ) -> list[tuple[Path, str]]:
        """Parse a list of PDF files.

        Args:
            pdf_files: List of PDF file paths to parse.
            on_progress: Optional callback for progress updates.
                Called with (pdf_path, current_index, total_count).

        Returns:
            List of tuples (pdf_path, markdown_content).
        """
        results = []
        total = len(pdf_files)

        for i, pdf_file in enumerate(pdf_files, 1):
            if on_progress:
                on_progress(pdf_file, i, total)

            markdown = self.parse_pdf(pdf_file)
            results.append((pdf_file, markdown))

        return results

    def parse_directory(
        self,
        inbox_path: Path,
        on_progress: Optional[Callable[[Path, int, int], None]] = None,
        recursive: bool = False,
    ) -> list[tuple[Path, str]]:
        """Parse all PDF files in a directory.

        Args:
            inbox_path: Directory containing PDF files.
            on_progress: Optional callback for progress updates.
                Called with (pdf_path, current_index, total_count).
            recursive: If True, search subdirectories recursively.

        Returns:
            List of tuples (pdf_path, markdown_content).
        """
        if recursive:
            pdf_files = list(inbox_path.rglob("*.pdf"))
        else:
            pdf_files = list(inbox_path.glob("*.pdf"))

        return self.parse_files(pdf_files, on_progress)
