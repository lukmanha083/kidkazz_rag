"""Entry point for running MCP server via python -m src.mcp_server."""

import logging
import sys

from .config import MCPServerConfig
from .server import run_server


def main() -> None:
    """Main entry point for the MCP server.

    Configures logging to stderr (critical for stdio transport)
    and starts the server.
    """
    # Load configuration from environment
    config = MCPServerConfig.from_env()

    # Configure logging to stderr (NEVER stdout for stdio transport)
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting KidKazz RAG MCP server")
    logger.info(f"Store type: {config.store_type}")
    logger.info(f"Embedder type: {config.embedder_type}")
    logger.info(f"Model name: {config.model_name}")

    # Run the server
    run_server(config)


if __name__ == "__main__":
    main()
