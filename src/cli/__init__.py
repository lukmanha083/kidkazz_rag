"""KidKazz RAG CLI.

A command-line interface for managing PDF-to-knowledge-base workflows.

Usage:
    kidkazz --help              # Show all commands
    kidkazz ingest markdown     # Ingest a markdown file
    kidkazz search semantic     # Search with natural language
    kidkazz docs list           # List all documents

Configuration:
    Project config: .kidkazz.toml
    User config: ~/.config/kidkazz/config.toml
    Environment: KIDKAZZ_* variables
"""

from .main import app, cli

__all__ = [
    "app",
    "cli",
]
