"""Configuration for MCP server."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional
import os

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass
class MCPServerConfig:
    """Configuration for the KidKazz RAG MCP server.

    Attributes:
        store_type: Storage backend - "mock" for testing, "helix" for production
        helix_port: Port for Helix-DB connection (default: 6969)
        helix_local: Whether to connect to local Helix-DB instance
        embedder_type: Embedder backend - "mock", "fastembed", or "openai"
        model_name: Name of embedding model
        cache_dir: Optional cache directory for embeddings
        log_level: Logging level (default: INFO)
    """

    store_type: Literal["mock", "helix"] = "mock"
    helix_port: int = 6969
    helix_local: bool = True
    embedder_type: Literal["mock", "fastembed", "openai"] = "fastembed"
    model_name: str = "BAAI/bge-small-en-v1.5"
    cache_dir: Optional[Path] = None
    log_level: str = "INFO"

    def create_store(self) -> Any:
        """Create storage instance based on configuration.

        Returns:
            MockChunkStore or HelixChunkStore instance

        Raises:
            ImportError: If required dependencies are not installed
        """
        if self.store_type == "mock":
            from src.storage import MockChunkStore

            return MockChunkStore()
        else:
            from src.storage import HelixChunkStore, HelixConfig

            return HelixChunkStore(
                HelixConfig(
                    local=self.helix_local,
                    port=self.helix_port,
                )
            )

    def create_embedder(self) -> Any:
        """Create embedder instance based on configuration.

        Returns:
            MockEmbedder, ChunkEmbedder, or OpenAIEmbedder instance

        Raises:
            ImportError: If required dependencies are not installed
            ValueError: If OpenAI API key is not set
        """
        if self.embedder_type == "mock":
            from src.chunker import MockEmbedder

            return MockEmbedder(model_name=self.model_name)
        elif self.embedder_type == "openai":
            from src.chunker import OpenAIEmbedder

            return OpenAIEmbedder(model_name=self.model_name)
        else:
            from src.chunker import ChunkEmbedder

            return ChunkEmbedder(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )

    @classmethod
    def from_env(cls) -> "MCPServerConfig":
        """Load configuration from .kidkazz.toml file.

        Searches for config in:
            1. Current directory (.kidkazz.toml)
            2. User config (~/.config/kidkazz/config.toml)

        Only API keys should be in .env file.
        All other config comes from .kidkazz.toml.

        Returns:
            MCPServerConfig instance with values from config file
        """
        # Search for config file
        config_paths = [
            Path.cwd() / ".kidkazz.toml",
            Path.home() / ".config" / "kidkazz" / "config.toml",
        ]

        config_data: dict[str, Any] = {}
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path, "rb") as f:
                    config_data = tomllib.load(f)
                break

        # Extract settings from TOML structure
        storage = config_data.get("storage", {})
        embeddings = config_data.get("embeddings", {})

        return cls(
            store_type=storage.get("store_type", "mock"),  # type: ignore
            helix_port=storage.get("helix_port", 6969),
            helix_local=storage.get("helix_local", True),
            embedder_type=embeddings.get("embedder_type", "fastembed"),  # type: ignore
            model_name=embeddings.get("model_name", "BAAI/bge-small-en-v1.5"),
            cache_dir=None,
            log_level=os.getenv("KIDKAZZ_LOG_LEVEL", "INFO"),
        )


class LazyEmbedder:
    """Lazy-loaded embedder to avoid slow startup.

    The embedder is only initialized when first used, which helps
    avoid slow server startup times due to model loading.
    """

    def __init__(self, config: MCPServerConfig):
        """Initialize lazy embedder with configuration.

        Args:
            config: MCP server configuration
        """
        self._config = config
        self._embedder: Any = None

    def embed_text(self, text: str) -> list[float]:
        """Embed text into a vector.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if self._embedder is None:
            self._embedder = self._config.create_embedder()
        return self._embedder.embed_text(text)

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._config.model_name


class ServerState:
    """Server state container with lazy-loaded components.

    Provides lazy initialization of storage and embedder to avoid
    slow startup times.
    """

    def __init__(self, config: Optional[MCPServerConfig] = None):
        """Initialize server state.

        Args:
            config: Optional configuration (uses from_env() if not provided)
        """
        self._config = config or MCPServerConfig.from_env()
        self._store: Any = None
        self._embedder: Optional[LazyEmbedder] = None

    @property
    def config(self) -> MCPServerConfig:
        """Get server configuration."""
        return self._config

    @property
    def store(self) -> Any:
        """Get storage instance (lazy-loaded)."""
        if self._store is None:
            self._store = self._config.create_store()
        return self._store

    @property
    def embedder(self) -> LazyEmbedder:
        """Get embedder instance (lazy-loaded)."""
        if self._embedder is None:
            self._embedder = LazyEmbedder(self._config)
        return self._embedder
