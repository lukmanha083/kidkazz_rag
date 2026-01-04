"""Utility functions for CLI commands."""

from pathlib import Path
from typing import Any, Optional, Tuple

from .config import CLIConfig


def parse_chunk_sizes(sizes_str: str) -> Tuple[int, int, int]:
    """Parse chunk sizes string into tuple.

    Args:
        sizes_str: Comma-separated sizes (e.g., "2048,512,256")

    Returns:
        Tuple of (level_1_size, level_2_size, overlap)

    Raises:
        ValueError: If format is invalid
    """
    parts = sizes_str.split(",")
    if len(parts) != 3:
        raise ValueError("chunk_sizes must have 3 values: L1,L2,overlap")

    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        raise ValueError("chunk_sizes must be integers")


def get_store(config: CLIConfig) -> Any:
    """Get storage instance based on configuration.

    Args:
        config: CLI configuration

    Returns:
        ChunkStore instance (MockChunkStore or HelixChunkStore)
    """
    if config.store_type == "helix":
        from src.storage import HelixChunkStore, HelixConfig

        helix_config = HelixConfig(
            local=config.helix_local,
            port=config.helix_port,
        )
        return HelixChunkStore(config=helix_config)
    else:
        from src.storage import MockChunkStore

        return MockChunkStore()


def get_embedder(config: CLIConfig) -> Any:
    """Get embedder instance based on configuration.

    Args:
        config: CLI configuration

    Returns:
        Embedder instance (MockEmbedder or ChunkEmbedder)
    """
    if config.embedder_type == "fastembed":
        try:
            from src.chunker import ChunkEmbedder

            return ChunkEmbedder(model_name=config.model_name)
        except ImportError:
            # Fall back to mock if fastembed not installed
            from src.chunker.embedder import MockEmbedder

            return MockEmbedder()
    else:
        from src.chunker.embedder import MockEmbedder

        return MockEmbedder()


def resolve_doc_id(file_path: Path, doc_id: Optional[str] = None) -> str:
    """Resolve document ID from file path or explicit value.

    Args:
        file_path: Path to source file
        doc_id: Explicit document ID (optional)

    Returns:
        Document ID string
    """
    if doc_id:
        return doc_id
    return file_path.stem


def resolve_title(
    file_path: Path,
    content: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Resolve document title from content or file path.

    Args:
        file_path: Path to source file
        content: File content (for extracting H1)
        title: Explicit title (optional)

    Returns:
        Document title string
    """
    if title:
        return title

    # Try to extract from markdown H1
    if content:
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()

    return file_path.stem


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def validate_file_exists(path: Path, extension: Optional[str] = None) -> Path:
    """Validate that a file exists and optionally check extension.

    Args:
        path: Path to validate
        extension: Expected extension (without dot)

    Returns:
        Validated path

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If extension doesn't match
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    if extension and path.suffix.lower() != f".{extension.lower()}":
        raise ValueError(f"Expected .{extension} file, got {path.suffix}")

    return path


def get_project_config_path() -> Path:
    """Get path to project configuration file."""
    return Path.cwd() / ".kidkazz.toml"


def get_user_config_path() -> Path:
    """Get path to user configuration file."""
    return Path.home() / ".config" / "kidkazz" / "config.toml"


def ensure_config_dir() -> Path:
    """Ensure user config directory exists."""
    config_dir = Path.home() / ".config" / "kidkazz"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def parse_tags(tags: Optional[str]) -> Optional[list[str]]:
    """Parse comma-separated tags string into list.

    Handles edge cases like empty strings, consecutive commas, and whitespace.

    Args:
        tags: Comma-separated tags string (e.g., "inventory,accounting")

    Returns:
        List of non-empty stripped tags, or None if no valid tags
    """
    if not tags:
        return None

    # Split, strip, and filter empty strings
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    return tag_list if tag_list else None
