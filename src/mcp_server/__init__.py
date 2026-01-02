"""KidKazz RAG MCP Server.

This module provides an MCP (Model Context Protocol) server that exposes
the KidKazz RAG knowledge base to Claude Code and other MCP-compatible clients.

Usage:
    # Run as module
    python -m src.mcp_server

    # Or use the installed command
    kidkazz-mcp

Configuration via environment variables:
    KIDKAZZ_STORE_TYPE: "mock" or "helix" (default: "mock")
    KIDKAZZ_HELIX_PORT: Port number (default: 6969)
    KIDKAZZ_EMBEDDER_TYPE: "mock" or "fastembed" (default: "fastembed")
    KIDKAZZ_MODEL_NAME: Embedding model name (default: "BAAI/bge-small-en-v1.5")
    KIDKAZZ_LOG_LEVEL: Logging level (default: "INFO")
"""

from .config import LazyEmbedder, MCPServerConfig, ServerState
from .server import create_server, run_server

# Entry point for console script
from .__main__ import main

__all__ = [
    "MCPServerConfig",
    "ServerState",
    "LazyEmbedder",
    "create_server",
    "run_server",
    "main",
]
