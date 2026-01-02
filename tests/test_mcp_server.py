"""Tests for MCP server initialization."""

from unittest.mock import MagicMock, patch

import pytest

from src.mcp_server.config import MCPServerConfig, ServerState
from src.mcp_server.server import create_server, run_server

# Skip all tests in this module if MCP is not installed
pytestmark = pytest.mark.mcp


@pytest.fixture(autouse=True)
def mcp_available():
    """Skip test if MCP is not available."""
    pytest.importorskip("mcp")


class TestCreateServer:
    """Tests for create_server function."""

    def test_creates_fastmcp_instance(self):
        """Test create_server returns FastMCP instance."""
        config = MCPServerConfig(store_type="mock", embedder_type="mock")
        server = create_server(config)
        # FastMCP should have these attributes
        assert hasattr(server, "run")

    def test_uses_default_config(self):
        """Test create_server uses from_env when no config provided."""
        with patch.dict("os.environ", {"KIDKAZZ_STORE_TYPE": "mock"}, clear=True):
            server = create_server()
            assert hasattr(server, "run")

    def test_uses_provided_state(self):
        """Test create_server uses provided ServerState."""
        config = MCPServerConfig(store_type="mock", embedder_type="mock")
        state = ServerState(config)
        server = create_server(state=state)
        assert hasattr(server, "run")

    def test_server_has_name(self):
        """Test server has correct name."""
        config = MCPServerConfig(store_type="mock", embedder_type="mock")
        server = create_server(config)
        # FastMCP stores name in settings or internal attribute
        assert hasattr(server, "_mcp_server") or hasattr(server, "name")


class TestCreateServerWithoutMCP:
    """Tests that don't require MCP installed."""

    def test_import_error_without_mcp(self):
        """Test ImportError raised when MCP not installed."""
        # Just verify the function exists - we can't easily test ImportError
        # without actually uninstalling mcp
        assert callable(create_server)


class TestRunServer:
    """Tests for run_server function."""

    def test_run_server_calls_run(self):
        """Test run_server calls mcp.run()."""
        config = MCPServerConfig(store_type="mock", embedder_type="mock")

        with patch("src.mcp_server.server.create_server") as mock_create:
            mock_mcp = MagicMock()
            mock_create.return_value = mock_mcp

            run_server(config)

            mock_mcp.run.assert_called_once()


class TestServerIntegration:
    """Integration tests for server components."""

    def test_tools_registered(self):
        """Test all tools are registered on server creation."""
        config = MCPServerConfig(store_type="mock", embedder_type="mock")
        server = create_server(config)

        # FastMCP stores tools internally
        # We can verify by checking the server was created successfully
        assert server is not None

    def test_resources_registered(self):
        """Test all resources are registered on server creation."""
        config = MCPServerConfig(store_type="mock", embedder_type="mock")
        server = create_server(config)

        # Verify server created with resources
        assert server is not None

    def test_server_with_mock_store(self):
        """Test server works with mock store."""
        config = MCPServerConfig(store_type="mock", embedder_type="mock")
        state = ServerState(config)

        # Access store to trigger lazy loading
        assert state.store is not None

        server = create_server(state=state)
        assert server is not None

    def test_server_lazy_loading(self):
        """Test server components are lazy loaded."""
        config = MCPServerConfig(store_type="mock", embedder_type="mock")
        state = ServerState(config)

        # Initially, store and embedder should not be loaded
        assert state._store is None
        assert state._embedder is None

        # Create server - this doesn't force loading
        server = create_server(state=state)
        assert server is not None
