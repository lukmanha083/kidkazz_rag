"""FastMCP server definition for KidKazz RAG."""

import logging
from typing import Any, Optional

from .config import MCPServerConfig, ServerState
from .resources import register_resources
from .tools import register_tools

logger = logging.getLogger(__name__)


def create_server(
    config: Optional[MCPServerConfig] = None,
    state: Optional[ServerState] = None,
) -> Any:
    """Create and configure the MCP server.

    Args:
        config: Optional server configuration (uses from_env() if not provided)
        state: Optional server state (creates from config if not provided)

    Returns:
        Configured FastMCP server instance with tools and resources registered
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "MCP package not installed. Install with: pip install 'kidkazz-rag[mcp]'"
        ) from e

    # Create server state if not provided
    if state is None:
        if config is None:
            config = MCPServerConfig.from_env()
        state = ServerState(config)

    logger.info(
        f"Creating MCP server with store_type={state.config.store_type}, "
        f"embedder_type={state.config.embedder_type}"
    )

    # Create FastMCP server
    mcp = FastMCP(
        name="kidkazz-rag",
    )

    # Register tools and resources
    register_tools(mcp, state)
    register_resources(mcp, state)

    logger.info("MCP server created successfully")
    return mcp


def run_server(config: Optional[MCPServerConfig] = None) -> None:
    """Run the MCP server with stdio transport.

    Args:
        config: Optional server configuration (uses from_env() if not provided)
    """
    mcp = create_server(config)
    logger.info("Starting MCP server with stdio transport")
    mcp.run()
