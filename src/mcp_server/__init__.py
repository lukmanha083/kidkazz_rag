"""KidKazz RAG MCP Server.

This module provides an MCP (Model Context Protocol) server that exposes
the KidKazz RAG knowledge base to Claude Code and other MCP-compatible clients.

Usage:
    # Run as module
    python -m src.mcp_server

    # Or use the installed command
    kidkazz-mcp

Configuration:
    All settings are read from .kidkazz.toml file.
    Only API keys should be stored in .env file.

    See .kidkazz.toml for available settings:
        [storage] - store_type, helix_port, helix_local
        [embeddings] - embedder_type, model_name
"""

from .config import LazyEmbedder, MCPServerConfig, ServerState
from .server import create_server, run_server

# Entry point for console script
from .__main__ import main

__all__ = [
    "LazyEmbedder",
    "MCPServerConfig",
    "ServerState",
    "create_server",
    "main",
    "run_server",
]
