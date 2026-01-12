"""Reducto.ai client for PDF parsing.

This module provides a client for the Reducto.ai API to convert
PDF documents to markdown format.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

# Lazy imports for Reducto SDK (only when actually used)
Reducto = None
Retrieval = None
Chunking = None


def _ensure_reducto_imports():
    """Lazily import Reducto SDK components."""
    global Reducto, Retrieval, Chunking
    if Reducto is None:
        from reducto import Reducto as _Reducto
        from reducto.types import Retrieval as _Retrieval
        from reducto.types.shared import Chunking as _Chunking
        Reducto = _Reducto
        Retrieval = _Retrieval
        Chunking = _Chunking


def _slugify_filename(filename: str) -> str:
    """Convert filename to CLI-friendly format (no spaces/special chars)."""
    slug = re.sub(r'[^\w\-]', '_', filename)
    slug = re.sub(r'_+', '_', slug)
    return slug.strip('_') or 'document'

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
        raw_output: Output continuous markdown instead of chunked format.
    """

    api_key: str
    agentic: bool = False
    chunk_mode: str = "variable"
    table_format: str = "md"
    raw_output: bool = True  # Default to human-readable output

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
        """Generate CLI-friendly output filename from source path."""
        return _slugify_filename(self.source_path.stem) + ".md"

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
            _ensure_reducto_imports()
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

        logger.info(
            "Parsing PDF: %s (chunk_mode=%s, agentic=%s)",
            pdf_path.name,
            self.config.chunk_mode,
            self.config.agentic,
        )

        try:
            _ensure_reducto_imports()

            # Upload local file first, then parse using the file_id
            logger.debug("Uploading PDF to Reducto API...")
            upload_response = self.client.upload(file=pdf_path)

            # Extract file_id from upload response
            file_id = upload_response.file_id
            logger.debug("Upload complete: file_id=%s", file_id)

            # Build parse kwargs
            parse_kwargs: dict = {"input": file_id}

            # Add chunking options if not disabled
            if self.config.chunk_mode != "disabled":
                parse_kwargs["retrieval"] = Retrieval(
                    chunking=Chunking(chunk_mode=self.config.chunk_mode)
                )

            logger.debug("Calling Reducto parse API...")
            response = self.client.parse.run(**parse_kwargs)
            logger.debug("Parse API call complete")

            return self._response_to_markdown(response)
        except Exception as e:
            logger.error("Reducto API error: %s", str(e))
            raise ReductoAPIError(str(e)) from e

    def _response_to_markdown(self, response) -> str:
        """Convert Reducto API response to markdown string.

        Args:
            response: Response from Reducto parse API.

        Returns:
            Markdown content as string.
        """
        result = response.result

        # Handle different response formats
        if hasattr(result, "chunks") and result.chunks:
            # Log response structure for debugging
            self._log_response_structure(result.chunks)

            if self.config.raw_output:
                # Join chunks into continuous readable markdown
                return self._chunks_to_markdown(result.chunks)
            else:
                # Keep chunked format for embedding pipeline
                # Still process headers from block metadata
                processed = [
                    self._process_chunk_with_headers(chunk)
                    for chunk in result.chunks
                ]
                return "\n\n---\n\n".join(p for p in processed if p)
        elif hasattr(result, "content"):
            return result.content
        elif hasattr(result, "markdown"):
            return result.markdown
        elif hasattr(result, "text"):
            return result.text
        elif hasattr(result, "url"):
            # Result is a URL - fetch the JSON content from S3
            import httpx

            url = result.url
            logger.debug("Fetching result from URL: %s", url[:100] if url else "None")

            try:
                # Add timeout to prevent hanging
                resp = httpx.get(url, timeout=60.0)
                resp.raise_for_status()  # Raise exception for 4xx/5xx status codes
            except httpx.HTTPStatusError as e:
                logger.error("HTTP error fetching URL: %s", e)
                raise ReductoAPIError(f"Failed to fetch result from URL: {e}") from e
            except httpx.RequestError as e:
                logger.error("Request error fetching URL: %s", e)
                raise ReductoAPIError(f"Failed to fetch result from URL: {e}") from e

            # Parse the JSON response which contains chunks
            try:
                data = resp.json()
                logger.debug("URL response keys: %s", list(data.keys()) if isinstance(data, dict) else type(data))

                # Extract chunks from the JSON response
                if isinstance(data, dict) and "chunks" in data:
                    chunks = data["chunks"]
                    logger.info("Fetched %d chunks from URL", len(chunks))

                    # Log and process chunks (they're dicts, not objects)
                    self._log_response_structure_from_dicts(chunks)

                    if self.config.raw_output:
                        return self._chunks_to_markdown_from_dicts(chunks)
                    else:
                        processed = [
                            self._process_chunk_with_headers_from_dict(chunk)
                            for chunk in chunks
                        ]
                        return "\n\n---\n\n".join(p for p in processed if p)
                else:
                    # Fallback to raw text if no chunks found
                    logger.warning("URL response has no chunks, using raw text")
                    return resp.text
            except ValueError as e:
                # JSON parsing failed
                logger.warning("Failed to parse URL response as JSON: %s", e)
                return resp.text
        else:
            # Try to get any text-like attribute
            for attr in ["output", "data", "body"]:
                if hasattr(result, attr):
                    return str(getattr(result, attr))
            # Last resort - convert to string
            return str(result)

    def _log_response_structure(self, chunks) -> None:
        """Log Reducto API response structure for debugging.

        Args:
            chunks: List of chunk objects from Reducto API response.
        """
        chunk_count = len(chunks)
        total_blocks = 0
        block_types: dict[str, int] = {}
        header_count = 0

        for chunk in chunks:
            blocks = getattr(chunk, "blocks", None) or []
            total_blocks += len(blocks)

            for block in blocks:
                # Handle both dict and object blocks
                if isinstance(block, dict):
                    block_type = block.get("type", "Unknown")
                else:
                    block_type = getattr(block, "type", "Unknown")

                block_types[block_type] = block_types.get(block_type, 0) + 1
                if block_type and block_type.lower() == "header":
                    header_count += 1

        # INFO level: summary stats
        logger.info(
            "Reducto API response: chunks=%d, blocks=%d, headers=%d",
            chunk_count,
            total_blocks,
            header_count,
        )

        # DEBUG level: detailed block type breakdown
        if block_types:
            logger.debug("Block types: %s", block_types)

        # DEBUG level: sample first chunk structure
        if chunks and logger.isEnabledFor(logging.DEBUG):
            first_chunk = chunks[0]
            first_blocks = getattr(first_chunk, "blocks", None) or []
            if first_blocks:
                sample_block = first_blocks[0]
                if isinstance(sample_block, dict):
                    logger.debug("Sample block (dict): keys=%s", list(sample_block.keys()))
                else:
                    logger.debug(
                        "Sample block (object): attrs=%s",
                        [a for a in dir(sample_block) if not a.startswith("_")],
                    )

    def _log_response_structure_from_dicts(self, chunks: list[dict]) -> None:
        """Log response structure when chunks are dictionaries (from URL response)."""
        chunk_count = len(chunks)
        total_blocks = 0
        block_types: dict[str, int] = {}
        header_count = 0

        for chunk in chunks:
            blocks = chunk.get("blocks", []) or []
            total_blocks += len(blocks)

            for block in blocks:
                block_type = block.get("type", "Unknown")
                block_types[block_type] = block_types.get(block_type, 0) + 1
                if block_type and block_type.lower() == "header":
                    header_count += 1

        logger.info(
            "Reducto API response: chunks=%d, blocks=%d, headers=%d",
            chunk_count,
            total_blocks,
            header_count,
        )

        if block_types:
            logger.debug("Block types: %s", block_types)

    def _chunks_to_markdown_from_dicts(self, chunks: list[dict]) -> str:
        """Convert dict chunks to continuous readable markdown."""
        parts = []
        for chunk in chunks:
            chunk_content = self._process_chunk_with_headers_from_dict(chunk)
            if chunk_content:
                parts.append(chunk_content)
        return "\n\n".join(parts)

    def _process_chunk_with_headers_from_dict(self, chunk: dict) -> str:
        """Process a dict chunk, reconstructing markdown headers from block metadata."""
        blocks = chunk.get("blocks", []) or []

        if not blocks:
            content = chunk.get("content", "")
            return content.strip() if content else ""

        processed_parts = []
        for block in blocks:
            block_type = block.get("type")
            block_content = block.get("content", "")

            if not block_content:
                continue

            block_content = block_content.strip()
            if not block_content:
                continue

            if block_type and block_type.lower() == "header":
                block_level = block.get("level")
                if block_level is None:
                    block_level = self._infer_header_level(block_content)
                prefix = "#" * block_level
                processed_parts.append(f"{prefix} {block_content}")
            else:
                processed_parts.append(block_content)

        if processed_parts:
            return "\n\n".join(processed_parts)

        content = chunk.get("content", "")
        return content.strip() if content else ""

    def _chunks_to_markdown(self, chunks) -> str:
        """Convert chunks to continuous readable markdown.

        Merges chunks intelligently, reconstructing header syntax from
        block metadata and creating a smooth document flow.
        """
        parts = []
        for chunk in chunks:
            chunk_content = self._process_chunk_with_headers(chunk)
            if chunk_content:
                parts.append(chunk_content)

        # Join with double newlines to create continuous flow
        return "\n\n".join(parts)

    def _process_chunk_with_headers(self, chunk) -> str:
        """Process a single chunk, reconstructing markdown headers from block metadata.

        Args:
            chunk: A Reducto chunk object with content and optional blocks.

        Returns:
            Processed markdown string with headers properly formatted.
        """
        # Check if chunk has block-level metadata
        blocks = getattr(chunk, "blocks", None)
        if not blocks:
            # No block metadata - return content as-is
            return chunk.content.strip() if hasattr(chunk, "content") else ""

        # Process blocks to reconstruct markdown
        processed_parts = []
        for block in blocks:
            # Blocks can be dicts (from Reducto API) or objects (from mocks)
            if isinstance(block, dict):
                block_type = block.get("type")
                block_content = block.get("content", "")
                block_level = block.get("level")
            else:
                block_type = getattr(block, "type", None)
                block_content = getattr(block, "content", "")
                block_level = getattr(block, "level", None)

            if not block_content:
                continue

            block_content = block_content.strip()
            if not block_content:
                continue

            # Reconstruct markdown headers from Header blocks
            if block_type and block_type.lower() == "header":
                # Try to get header level if available, default to h2
                if block_level is None:
                    # Infer level from content characteristics
                    block_level = self._infer_header_level(block_content)
                prefix = "#" * block_level
                processed_parts.append(f"{prefix} {block_content}")
            else:
                # Non-header content - use as-is
                processed_parts.append(block_content)

        if processed_parts:
            return "\n\n".join(processed_parts)

        # Fallback to chunk.content if no blocks processed
        return chunk.content.strip() if hasattr(chunk, "content") else ""

    def _infer_header_level(self, content: str) -> int:
        """Infer markdown header level from content characteristics.

        This is a heuristic for headers where Reducto doesn't provide an
        explicit level. Only called for content already identified as a header.

        Args:
            content: Header text content (from a Reducto Header block).

        Returns:
            Header level 1, 2, or 3 based on heuristics:
            - 1: All caps short text, or contains chapter/part/section keywords
            - 2: Medium length headers (default for most section headers)
            - 3: Longer headers (subsections)
        """
        # Very short, all caps = likely h1 (document/chapter title)
        if len(content) < 50 and content.isupper():
            return 1

        # Short content with chapter/part keywords = h1
        lower = content.lower()
        if any(kw in lower for kw in ["chapter", "part", "section"]) and len(content) < 80:
            return 1

        # Medium length = h2 (default for most section headers)
        if len(content) < 100:
            return 2

        # Longer headers = h3 (subsections)
        return 3

    def extract_block_metadata(self, chunk) -> dict:
        """Extract header and block metadata from a Reducto chunk.

        Args:
            chunk: A Reducto chunk object with optional blocks array.

        Returns:
            Dictionary with header_text, header_level, and block_type.
        """
        blocks = getattr(chunk, "blocks", None)
        if not blocks:
            return {
                "header_text": None,
                "header_level": None,
                "block_type": None,
            }

        # Helper to get block attributes (blocks can be dicts or objects)
        def get_block_attr(blk, attr, default=None):
            if isinstance(blk, dict):
                return blk.get(attr, default)
            return getattr(blk, attr, default)

        # Find the first non-empty block to determine primary block type
        first_block = None
        for block in blocks:
            block_content = get_block_attr(block, "content", "")
            if block_content and block_content.strip():
                first_block = block
                break

        if not first_block:
            return {
                "header_text": None,
                "header_level": None,
                "block_type": None,
            }

        block_type = get_block_attr(first_block, "type")
        block_content = get_block_attr(first_block, "content", "").strip()

        # Check if this is a header block
        if block_type and block_type.lower() == "header":
            level = get_block_attr(first_block, "level")
            if level is None:
                level = self._infer_header_level(block_content)
            return {
                "header_text": block_content,
                "header_level": level,
                "block_type": block_type,
            }

        # Non-header block
        return {
            "header_text": None,
            "header_level": None,
            "block_type": block_type,
        }

    def chunks_to_markdown_with_metadata(self, chunks) -> list[dict]:
        """Convert chunks to markdown with preserved block metadata.

        Args:
            chunks: List of Reducto chunk objects.

        Returns:
            List of dictionaries with 'markdown' and 'metadata' keys.
        """
        results = []
        for chunk in chunks:
            # Get markdown content
            markdown = self._process_chunk_with_headers(chunk)
            if not markdown:
                continue

            # Extract block metadata
            metadata = self.extract_block_metadata(chunk)

            results.append({
                "markdown": markdown,
                "metadata": metadata,
            })

        return results

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
