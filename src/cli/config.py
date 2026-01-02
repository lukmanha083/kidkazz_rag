"""Configuration management for KidKazz CLI.

Supports layered configuration from multiple sources:
1. Command-line arguments (highest priority)
2. Environment variables (KIDKAZZ_*)
3. Project config (.kidkazz.toml)
4. User config (~/.config/kidkazz/config.toml)
5. Default values (lowest priority)
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Use tomllib for Python 3.11+, tomli for earlier versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore

try:
    import tomli_w
except ImportError:
    tomli_w = None  # type: ignore


@dataclass
class CLIConfig:
    """CLI configuration with layered sources."""

    # Storage settings
    store_type: str = "mock"
    helix_port: int = 6969
    helix_local: bool = True

    # Embedding settings
    embedder_type: str = "fastembed"
    model_name: str = "BAAI/bge-small-en-v1.5"

    # Chunking settings
    level_1_size: int = 2048
    level_2_size: int = 512
    overlap: int = 256

    # Output settings
    output_format: str = "human"
    color: bool = True

    # Source tracking (which config file each setting came from)
    _sources: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "CLIConfig":
        """Load configuration from all sources."""
        config = cls()

        # Load from user config
        user_config_path = Path.home() / ".config" / "kidkazz" / "config.toml"
        if user_config_path.exists():
            config._load_from_file(user_config_path, "user")

        # Load from project config
        project_config_path = Path.cwd() / ".kidkazz.toml"
        if project_config_path.exists():
            config._load_from_file(project_config_path, "project")

        # Load from environment (highest priority for file-based)
        config._load_from_env()

        return config

    def _load_from_file(self, path: Path, source: str) -> None:
        """Load settings from TOML file."""
        if tomllib is None:
            return

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            self._apply_settings(data, source)
        except Exception:
            pass  # Ignore malformed config files

    def _apply_settings(self, data: dict[str, Any], source: str) -> None:
        """Apply settings from a dictionary."""
        # Storage settings
        if "storage" in data:
            storage = data["storage"]
            if "store_type" in storage:
                self.store_type = storage["store_type"]
                self._sources["store_type"] = source
            if "helix_port" in storage:
                self.helix_port = int(storage["helix_port"])
                self._sources["helix_port"] = source
            if "helix_local" in storage:
                self.helix_local = bool(storage["helix_local"])
                self._sources["helix_local"] = source

        # Embedding settings
        if "embeddings" in data:
            embeddings = data["embeddings"]
            if "embedder_type" in embeddings:
                self.embedder_type = embeddings["embedder_type"]
                self._sources["embedder_type"] = source
            if "model_name" in embeddings:
                self.model_name = embeddings["model_name"]
                self._sources["model_name"] = source

        # Chunking settings
        if "chunking" in data:
            chunking = data["chunking"]
            if "level_1_size" in chunking:
                self.level_1_size = int(chunking["level_1_size"])
                self._sources["level_1_size"] = source
            if "level_2_size" in chunking:
                self.level_2_size = int(chunking["level_2_size"])
                self._sources["level_2_size"] = source
            if "overlap" in chunking:
                self.overlap = int(chunking["overlap"])
                self._sources["overlap"] = source

        # Output settings
        if "output" in data:
            output = data["output"]
            if "format" in output:
                self.output_format = output["format"]
                self._sources["output_format"] = source
            if "color" in output:
                self.color = bool(output["color"])
                self._sources["color"] = source

    def _load_from_env(self) -> None:
        """Load settings from environment variables."""
        env_mappings: dict[str, tuple[str, type]] = {
            "KIDKAZZ_STORE_TYPE": ("store_type", str),
            "KIDKAZZ_HELIX_PORT": ("helix_port", int),
            "KIDKAZZ_HELIX_LOCAL": ("helix_local", lambda x: x.lower() in ("true", "1", "yes")),
            "KIDKAZZ_EMBEDDER_TYPE": ("embedder_type", str),
            "KIDKAZZ_MODEL_NAME": ("model_name", str),
            "KIDKAZZ_LEVEL_1_SIZE": ("level_1_size", int),
            "KIDKAZZ_LEVEL_2_SIZE": ("level_2_size", int),
            "KIDKAZZ_OVERLAP": ("overlap", int),
            "KIDKAZZ_OUTPUT_FORMAT": ("output_format", str),
            "KIDKAZZ_COLOR": ("color", lambda x: x.lower() in ("true", "1", "yes")),
        }

        for env_var, (attr, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    setattr(self, attr, converter(value))
                    self._sources[attr] = f"environment ({env_var})"
                except (ValueError, TypeError):
                    pass  # Ignore invalid values

    def get_source(self, key: str) -> str:
        """Get the source of a configuration value."""
        return self._sources.get(key, "default")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for saving."""
        return {
            "storage": {
                "store_type": self.store_type,
                "helix_port": self.helix_port,
                "helix_local": self.helix_local,
            },
            "embeddings": {
                "embedder_type": self.embedder_type,
                "model_name": self.model_name,
            },
            "chunking": {
                "level_1_size": self.level_1_size,
                "level_2_size": self.level_2_size,
                "overlap": self.overlap,
            },
            "output": {
                "format": self.output_format,
                "color": self.color,
            },
        }

    def save_project(self, path: Optional[Path] = None) -> bool:
        """Save configuration to project file."""
        if tomli_w is None:
            return False

        path = path or Path.cwd() / ".kidkazz.toml"
        return self._save_to_file(path)

    def save_user(self) -> bool:
        """Save configuration to user file."""
        if tomli_w is None:
            return False

        path = Path.home() / ".config" / "kidkazz" / "config.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        return self._save_to_file(path)

    def _save_to_file(self, path: Path) -> bool:
        """Save settings to TOML file."""
        if tomli_w is None:
            return False

        try:
            with open(path, "wb") as f:
                tomli_w.dump(self.to_dict(), f)
            return True
        except Exception:
            return False

    def set_value(self, key: str, value: Any) -> bool:
        """Set a configuration value by key."""
        key_mapping = {
            "store_type": ("store_type", str),
            "helix_port": ("helix_port", int),
            "helix_local": ("helix_local", lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x)),
            "embedder_type": ("embedder_type", str),
            "model_name": ("model_name", str),
            "level_1_size": ("level_1_size", int),
            "level_2_size": ("level_2_size", int),
            "overlap": ("overlap", int),
            "output_format": ("output_format", str),
            "color": ("color", lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x)),
        }

        if key not in key_mapping:
            return False

        attr, converter = key_mapping[key]
        try:
            setattr(self, attr, converter(value))
            return True
        except (ValueError, TypeError):
            return False

    @classmethod
    def get_defaults(cls) -> "CLIConfig":
        """Get a config with all default values."""
        return cls()
