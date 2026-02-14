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

    # Build FastMCP kwargs — add HTTP params only for non-stdio transport
    fastmcp_kwargs: dict[str, Any] = {"name": "kidkazz-rag"}
    if state.config.transport != "stdio":
        from mcp.server.transport_security import TransportSecuritySettings

        fastmcp_kwargs.update(
            host=state.config.host,
            port=state.config.port,
            # Disable DNS rebinding protection — Fly.io proxy forwards
            # requests with its own Host header, not localhost
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=False,
            ),
        )

    mcp = FastMCP(**fastmcp_kwargs)

    # Register /health endpoint for HTTP transport (Fly.io health checks)
    if state.config.transport != "stdio":
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        @mcp.custom_route("/health", methods=["GET"])
        async def health_check(request: Request) -> JSONResponse:
            return JSONResponse({"status": "ok"})

    # Register tools and resources
    register_tools(mcp, state)
    register_resources(mcp, state)

    logger.info("MCP server created successfully")
    return mcp


def run_server(config: Optional[MCPServerConfig] = None) -> None:
    """Run the MCP server.

    Supports stdio (default) and streamable-http transports.
    When api_key is set with HTTP transport, wraps the app with
    ApiKeyMiddleware for Bearer token authentication.

    Args:
        config: Optional server configuration (uses from_env() if not provided)
    """
    if config is None:
        config = MCPServerConfig.from_env()

    mcp = create_server(config)

    if config.transport == "stdio":
        logger.info("Starting MCP server with stdio transport")
        mcp.run()
    elif config.api_key:
        # HTTP with auth — wrap the Starlette app with middleware
        _run_http_with_auth(mcp, config)
    else:
        # HTTP without auth
        logger.info(
            f"Starting MCP server with streamable-http on "
            f"{config.host}:{config.port}"
        )
        mcp.run(transport="streamable-http")


def _run_http_with_auth(mcp: Any, config: MCPServerConfig) -> None:
    """Run HTTP transport with API key authentication middleware.

    Gets the underlying Starlette app from FastMCP and wraps it with
    ApiKeyMiddleware, then serves via uvicorn directly.
    """
    import uvicorn

    from .auth import ApiKeyMiddleware

    app = mcp.streamable_http_app()
    app = ApiKeyMiddleware(app, config.api_key)

    logger.info(
        f"Starting MCP server with streamable-http on "
        f"{config.host}:{config.port} (auth enabled)"
    )

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="warning",
    )
