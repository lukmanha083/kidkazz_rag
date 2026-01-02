"""Tests for CLI configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cli.config import CLIConfig


class TestCLIConfigDefaults:
    """Tests for default configuration values."""

    def test_default_store_type(self):
        """Test default store type is mock."""
        config = CLIConfig()
        assert config.store_type == "mock"

    def test_default_embedder_type(self):
        """Test default embedder type is fastembed."""
        config = CLIConfig()
        assert config.embedder_type == "fastembed"

    def test_default_model_name(self):
        """Test default model name."""
        config = CLIConfig()
        assert config.model_name == "BAAI/bge-small-en-v1.5"

    def test_default_chunk_sizes(self):
        """Test default chunk sizes."""
        config = CLIConfig()
        assert config.level_1_size == 2048
        assert config.level_2_size == 512
        assert config.overlap == 256

    def test_default_helix_port(self):
        """Test default helix port."""
        config = CLIConfig()
        assert config.helix_port == 6969

    def test_default_output_format(self):
        """Test default output format."""
        config = CLIConfig()
        assert config.output_format == "human"
        assert config.color is True


class TestCLIConfigEnvironment:
    """Tests for environment variable loading."""

    def test_load_store_type_from_env(self):
        """Test loading store_type from environment."""
        with patch.dict(os.environ, {"KIDKAZZ_STORE_TYPE": "helix"}):
            config = CLIConfig.load()
            assert config.store_type == "helix"
            assert config.get_source("store_type") == "environment (KIDKAZZ_STORE_TYPE)"

    def test_load_helix_port_from_env(self):
        """Test loading helix_port from environment."""
        with patch.dict(os.environ, {"KIDKAZZ_HELIX_PORT": "7070"}):
            config = CLIConfig.load()
            assert config.helix_port == 7070

    def test_load_model_name_from_env(self):
        """Test loading model_name from environment."""
        with patch.dict(os.environ, {"KIDKAZZ_MODEL_NAME": "custom-model"}):
            config = CLIConfig.load()
            assert config.model_name == "custom-model"

    def test_load_embedder_type_from_env(self):
        """Test loading embedder_type from environment."""
        with patch.dict(os.environ, {"KIDKAZZ_EMBEDDER_TYPE": "mock"}):
            config = CLIConfig.load()
            assert config.embedder_type == "mock"

    def test_invalid_env_value_ignored(self):
        """Test that invalid environment values are ignored."""
        with patch.dict(os.environ, {"KIDKAZZ_HELIX_PORT": "not_a_number"}):
            config = CLIConfig.load()
            assert config.helix_port == 6969  # Default


class TestCLIConfigToDict:
    """Tests for config serialization."""

    def test_to_dict_structure(self):
        """Test to_dict returns correct structure."""
        config = CLIConfig()
        data = config.to_dict()

        assert "storage" in data
        assert "embeddings" in data
        assert "chunking" in data
        assert "output" in data

    def test_to_dict_storage_values(self):
        """Test storage values in dict."""
        config = CLIConfig()
        config.store_type = "helix"
        config.helix_port = 8080

        data = config.to_dict()
        assert data["storage"]["store_type"] == "helix"
        assert data["storage"]["helix_port"] == 8080

    def test_to_dict_embeddings_values(self):
        """Test embeddings values in dict."""
        config = CLIConfig()
        config.model_name = "test-model"

        data = config.to_dict()
        assert data["embeddings"]["model_name"] == "test-model"


class TestCLIConfigSetValue:
    """Tests for setting configuration values."""

    def test_set_valid_string_value(self):
        """Test setting valid string value."""
        config = CLIConfig()
        assert config.set_value("store_type", "helix")
        assert config.store_type == "helix"

    def test_set_valid_int_value(self):
        """Test setting valid int value."""
        config = CLIConfig()
        assert config.set_value("helix_port", "9000")
        assert config.helix_port == 9000

    def test_set_valid_bool_value(self):
        """Test setting valid bool value."""
        config = CLIConfig()
        assert config.set_value("color", "false")
        assert config.color is False

    def test_set_invalid_key(self):
        """Test setting invalid key returns False."""
        config = CLIConfig()
        assert config.set_value("invalid_key", "value") is False

    def test_set_chunk_sizes(self):
        """Test setting chunk size values."""
        config = CLIConfig()
        assert config.set_value("level_1_size", "1024")
        assert config.level_1_size == 1024


class TestCLIConfigGetSource:
    """Tests for source tracking."""

    def test_get_source_default(self):
        """Test get_source returns 'default' for unset values."""
        config = CLIConfig()
        assert config.get_source("store_type") == "default"

    def test_get_source_environment(self):
        """Test get_source returns environment source."""
        with patch.dict(os.environ, {"KIDKAZZ_STORE_TYPE": "helix"}):
            config = CLIConfig.load()
            assert "environment" in config.get_source("store_type")


class TestCLIConfigGetDefaults:
    """Tests for getting default configuration."""

    def test_get_defaults_returns_config(self):
        """Test get_defaults returns CLIConfig instance."""
        defaults = CLIConfig.get_defaults()
        assert isinstance(defaults, CLIConfig)

    def test_get_defaults_has_default_values(self):
        """Test defaults have expected values."""
        defaults = CLIConfig.get_defaults()
        assert defaults.store_type == "mock"
        assert defaults.embedder_type == "fastembed"
