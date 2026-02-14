"""Tests for MCP server configuration."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.mcp_server.config import LazyEmbedder, MCPServerConfig, ServerState


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MCPServerConfig()
        assert config.store_type == "mock"
        assert config.helix_port == 6969
        assert config.helix_local is True
        assert config.embedder_type == "openai"
        assert config.model_name == "text-embedding-3-small"
        assert config.cache_dir is None
        assert config.log_level == "INFO"
        assert config.transport == "stdio"
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.api_key is None

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = MCPServerConfig(
            store_type="helix",
            helix_port=7070,
            helix_local=False,
            embedder_type="mock",
            model_name="custom-model",
            cache_dir=Path("/tmp/cache"),
            log_level="DEBUG",
        )
        assert config.store_type == "helix"
        assert config.helix_port == 7070
        assert config.helix_local is False
        assert config.embedder_type == "mock"
        assert config.model_name == "custom-model"
        assert config.cache_dir == Path("/tmp/cache")
        assert config.log_level == "DEBUG"

    def test_from_env_defaults(self):
        """Test from_env with no config file returns defaults."""
        with patch("src.mcp_server.config.Path.cwd", return_value=Path("/nonexistent")), \
             patch("src.mcp_server.config.Path.home", return_value=Path("/nonexistent")), \
             patch.dict(os.environ, {}, clear=True):
            config = MCPServerConfig.from_env()
            assert config.store_type == "mock"
            assert config.embedder_type == "openai"

    def test_from_env_reads_toml(self, tmp_path):
        """Test from_env reads .kidkazz.toml config file."""
        toml_content = b"""
[storage]
store_type = "helix"
helix_port = 8080
helix_local = false

[embeddings]
embedder_type = "mock"
model_name = "test-model"
"""
        (tmp_path / ".kidkazz.toml").write_bytes(toml_content)
        with patch("src.mcp_server.config.Path.cwd", return_value=tmp_path), \
             patch.dict(os.environ, {"KIDKAZZ_LOG_LEVEL": "WARNING"}, clear=True):
            config = MCPServerConfig.from_env()
            assert config.store_type == "helix"
            assert config.helix_port == 8080
            assert config.helix_local is False
            assert config.embedder_type == "mock"
            assert config.model_name == "test-model"
            assert config.log_level == "WARNING"

    def test_create_mock_store(self):
        """Test create_store returns MockChunkStore for mock type."""
        config = MCPServerConfig(store_type="mock")
        store = config.create_store()
        from src.storage import MockChunkStore

        assert isinstance(store, MockChunkStore)

    def test_create_mock_embedder(self):
        """Test create_embedder returns MockEmbedder for mock type."""
        config = MCPServerConfig(embedder_type="mock")
        embedder = config.create_embedder()
        from src.chunker import MockEmbedder

        assert isinstance(embedder, MockEmbedder)
        assert embedder.model_name == config.model_name

    def test_transport_from_env(self):
        """Test transport config from environment variables."""
        env = {
            "KIDKAZZ_TRANSPORT": "streamable-http",
            "KIDKAZZ_HOST": "127.0.0.1",
            "KIDKAZZ_PORT": "9090",
            "KIDKAZZ_API_KEY": "test-secret-key",
        }
        with patch("src.mcp_server.config.Path.cwd", return_value=Path("/nonexistent")), \
             patch("src.mcp_server.config.Path.home", return_value=Path("/nonexistent")), \
             patch.dict(os.environ, env, clear=True):
            config = MCPServerConfig.from_env()
            assert config.transport == "streamable-http"
            assert config.host == "127.0.0.1"
            assert config.port == 9090
            assert config.api_key == "test-secret-key"

    def test_transport_from_toml(self, tmp_path):
        """Test transport config from [server] section in .kidkazz.toml."""
        toml_content = b"""
[server]
transport = "streamable-http"
host = "0.0.0.0"
port = 8888
api_key = "toml-key"
"""
        (tmp_path / ".kidkazz.toml").write_bytes(toml_content)
        with patch("src.mcp_server.config.Path.cwd", return_value=tmp_path), \
             patch.dict(os.environ, {}, clear=True):
            config = MCPServerConfig.from_env()
            assert config.transport == "streamable-http"
            assert config.host == "0.0.0.0"
            assert config.port == 8888
            assert config.api_key == "toml-key"

    def test_env_overrides_toml_transport(self, tmp_path):
        """Test env vars override TOML [server] settings."""
        toml_content = b"""
[server]
transport = "stdio"
port = 8888
"""
        (tmp_path / ".kidkazz.toml").write_bytes(toml_content)
        env = {"KIDKAZZ_TRANSPORT": "streamable-http", "KIDKAZZ_PORT": "9999"}
        with patch("src.mcp_server.config.Path.cwd", return_value=tmp_path), \
             patch.dict(os.environ, env, clear=True):
            config = MCPServerConfig.from_env()
            assert config.transport == "streamable-http"
            assert config.port == 9999


class TestLazyEmbedder:
    """Tests for LazyEmbedder class."""

    def test_lazy_initialization(self):
        """Test embedder is not initialized until first use."""
        config = MCPServerConfig(embedder_type="mock")
        lazy = LazyEmbedder(config)
        assert lazy._embedder is None

    def test_embed_text_initializes(self):
        """Test embed_text initializes embedder on first call."""
        config = MCPServerConfig(embedder_type="mock")
        lazy = LazyEmbedder(config)
        result = lazy.embed_text("test text")
        assert lazy._embedder is not None
        assert isinstance(result, list)
        assert len(result) > 0

    def test_embed_text_reuses_embedder(self):
        """Test subsequent calls reuse the same embedder."""
        config = MCPServerConfig(embedder_type="mock")
        lazy = LazyEmbedder(config)
        lazy.embed_text("first")
        embedder_ref = lazy._embedder
        lazy.embed_text("second")
        assert lazy._embedder is embedder_ref

    def test_model_name_property(self):
        """Test model_name returns config model name."""
        config = MCPServerConfig(model_name="test-model")
        lazy = LazyEmbedder(config)
        assert lazy.model_name == "test-model"


class TestServerState:
    """Tests for ServerState class."""

    def test_default_config(self):
        """Test ServerState uses from_env if no config provided."""
        with patch("src.mcp_server.config.Path.cwd", return_value=Path("/nonexistent")), \
             patch("src.mcp_server.config.Path.home", return_value=Path("/nonexistent")), \
             patch.dict(os.environ, {}, clear=True):
            state = ServerState()
            assert state.config.store_type == "mock"

    def test_custom_config(self):
        """Test ServerState uses provided config."""
        config = MCPServerConfig(store_type="mock", embedder_type="mock")
        state = ServerState(config)
        assert state.config is config

    def test_lazy_store(self):
        """Test store is lazy-loaded."""
        config = MCPServerConfig(store_type="mock")
        state = ServerState(config)
        assert state._store is None
        store = state.store
        assert state._store is not None
        assert store is state.store  # Same instance

    def test_lazy_embedder(self):
        """Test embedder is lazy-loaded."""
        config = MCPServerConfig(embedder_type="mock")
        state = ServerState(config)
        assert state._embedder is None
        embedder = state.embedder
        assert state._embedder is not None
        assert isinstance(embedder, LazyEmbedder)
        assert embedder is state.embedder  # Same instance
