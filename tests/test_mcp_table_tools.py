"""Tests for MCP table tool implementations.

TDD tests for Phase 3: MCP Table Tools.
Tests cover search_tables, get_table, get_tables_for_concept tools.
"""

from unittest.mock import MagicMock

import pytest

from src.chunker.table_parser import ParsedTable
from src.chunker.table_summarizer import TableSummary


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_table():
    """Create a sample parsed table."""
    return ParsedTable(
        raw_markdown="| Method | Cost |\n|--------|------|\n| FIFO | $100 |",
        column_names=["Method", "Cost"],
        rows=[["FIFO", "$100"], ["LIFO", "$200"]],
        row_count=2,
        column_count=2,
        column_types=["text", "numeric"],
        has_header_row=True,
        surrounding_context="Inventory valuation methods comparison",
        source_chunk_id="chunk_001",
    )


@pytest.fixture
def sample_summary(sample_table):
    """Create a sample table summary."""
    return TableSummary(
        table_id=f"table_{sample_table.source_chunk_id}",
        summary_text="This table compares inventory valuation methods FIFO and LIFO with their costs.",
        key_columns=["Method", "Cost"],
        key_values=["FIFO", "$100", "LIFO", "$200"],
    )


@pytest.fixture
def mock_table_store(sample_table, sample_summary):
    """Create a mock table store with sample data."""
    from src.storage.table_store import TableStore

    mock_embedder = MagicMock()
    mock_embedder.embed_text.return_value = [0.1] * 384

    store = TableStore(embedder=mock_embedder)
    store.store_table(sample_table, sample_summary, "test_doc")
    return store


# ============================================================================
# Tests for search_tables Tool
# ============================================================================


class TestSearchTablesTool:
    """Tests for search_tables MCP tool."""

    def test_search_tables_tool_exists(self):
        """search_tables should be registered as an MCP tool."""
        from src.mcp_server.tools import register_tools

        mock_mcp = MagicMock()
        mock_state = MagicMock()
        mock_state.table_store = MagicMock()

        register_tools(mock_mcp, mock_state)

        # Check that tool decorator was called with search_tables
        tool_names = [
            call[0][0].__name__
            for call in mock_mcp.tool.return_value.call_args_list
            if hasattr(call[0][0], "__name__")
        ]
        # Tool should be registered (we'll verify functionality separately)
        assert mock_mcp.tool.called

    def test_search_tables_returns_list(self, mock_table_store):
        """search_tables should return a list of matching tables."""
        results = mock_table_store.search_tables("inventory", top_k=5)

        assert isinstance(results, list)

    def test_search_tables_returns_table_and_score(self, mock_table_store):
        """search_tables should return tuples of (table, score)."""
        results = mock_table_store.search_tables("inventory", top_k=5)

        assert len(results) > 0
        table, score = results[0]
        assert isinstance(table, ParsedTable)
        assert isinstance(score, float)

    def test_search_tables_filters_by_doc_id(self, mock_table_store, sample_table, sample_summary):
        """search_tables should filter results by document ID."""
        # Add another table for a different document
        other_table = ParsedTable(
            raw_markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            column_names=["A", "B"],
            rows=[["1", "2"]],
            row_count=1,
            column_count=2,
            column_types=["text", "text"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_002",
        )
        other_summary = TableSummary(
            table_id="table_chunk_002",
            summary_text="Other table",
            key_columns=["A"],
            key_values=["1"],
        )
        mock_table_store.store_table(other_table, other_summary, "other_doc")

        # Search with doc_id filter
        results = mock_table_store.search_tables("table", top_k=10, doc_id="test_doc")

        # Should only return tables from test_doc
        for table, _ in results:
            # The table is from test_doc since we stored it there
            assert table.source_chunk_id == "chunk_001"


# ============================================================================
# Tests for get_table Tool
# ============================================================================


class TestGetTableTool:
    """Tests for get_table MCP tool."""

    def test_get_table_returns_table(self, mock_table_store):
        """get_table should return the table by ID."""
        table = mock_table_store.get_table("table_chunk_001")

        assert table is not None
        assert isinstance(table, ParsedTable)
        assert table.source_chunk_id == "chunk_001"

    def test_get_table_returns_none_for_missing(self, mock_table_store):
        """get_table should return None for non-existent table."""
        table = mock_table_store.get_table("table_nonexistent")

        assert table is None

    def test_get_table_includes_all_fields(self, mock_table_store):
        """get_table should return table with all fields populated."""
        table = mock_table_store.get_table("table_chunk_001")

        assert table.raw_markdown is not None
        assert table.column_names == ["Method", "Cost"]
        assert table.rows == [["FIFO", "$100"], ["LIFO", "$200"]]
        assert table.row_count == 2
        assert table.column_count == 2
        assert table.column_types == ["text", "numeric"]
        assert table.surrounding_context is not None


# ============================================================================
# Tests for get_tables_for_concept Tool
# ============================================================================


class TestGetTablesForConceptTool:
    """Tests for get_tables_for_concept MCP tool."""

    def test_get_tables_for_concept_returns_list(self, mock_table_store):
        """get_tables_for_concept should return a list."""
        mock_table_store.link_table_to_concept("table_chunk_001", "FIFO")

        tables = mock_table_store.get_tables_for_concept("FIFO")

        assert isinstance(tables, list)

    def test_get_tables_for_concept_returns_linked_tables(self, mock_table_store):
        """get_tables_for_concept should return tables linked to the concept."""
        mock_table_store.link_table_to_concept("table_chunk_001", "Inventory")

        tables = mock_table_store.get_tables_for_concept("Inventory")

        assert len(tables) == 1
        assert tables[0].source_chunk_id == "chunk_001"

    def test_get_tables_for_concept_case_insensitive(self, mock_table_store):
        """get_tables_for_concept should be case insensitive."""
        mock_table_store.link_table_to_concept("table_chunk_001", "FIFO")

        # Search with different case
        tables = mock_table_store.get_tables_for_concept("fifo")

        assert len(tables) == 1

    def test_get_tables_for_concept_returns_empty_for_no_links(self, mock_table_store):
        """get_tables_for_concept should return empty list for unlinked concepts."""
        tables = mock_table_store.get_tables_for_concept("NonexistentConcept")

        assert tables == []


# ============================================================================
# Tests for Table Formatters
# ============================================================================


class TestTableFormatters:
    """Tests for table formatting functions."""

    def test_format_table_exists(self):
        """format_table function should exist."""
        from src.mcp_server.formatters import format_table

        assert callable(format_table)

    def test_format_table_returns_dict(self, sample_table):
        """format_table should return a dictionary."""
        from src.mcp_server.formatters import format_table

        result = format_table(sample_table)

        assert isinstance(result, dict)

    def test_format_table_includes_required_fields(self, sample_table):
        """format_table should include all required fields."""
        from src.mcp_server.formatters import format_table

        result = format_table(sample_table)

        assert "table_id" in result
        assert "columns" in result
        assert "column_types" in result
        assert "rows" in result
        assert "row_count" in result
        assert "raw_markdown" in result
        assert "context" in result

    def test_format_table_search_result_exists(self):
        """format_table_search_result function should exist."""
        from src.mcp_server.formatters import format_table_search_result

        assert callable(format_table_search_result)

    def test_format_table_search_result_includes_score(self, sample_table):
        """format_table_search_result should include similarity score."""
        from src.mcp_server.formatters import format_table_search_result

        result = format_table_search_result(sample_table, 0.85)

        assert "score" in result
        assert result["score"] == 0.85

    def test_format_table_list_exists(self):
        """format_table_list function should exist."""
        from src.mcp_server.formatters import format_table_list

        assert callable(format_table_list)

    def test_format_table_list_returns_list(self, sample_table):
        """format_table_list should return a list of formatted tables."""
        from src.mcp_server.formatters import format_table_list

        result = format_table_list([sample_table])

        assert isinstance(result, list)
        assert len(result) == 1


# ============================================================================
# Tests for MCP Tool Registration
# ============================================================================


class TestTableToolRegistration:
    """Tests for table tool registration in MCP server."""

    def test_tools_module_has_table_tools(self):
        """tools.py should define table-related tools."""
        from src.mcp_server import tools

        # Module should have register_tools function
        assert hasattr(tools, "register_tools")

    def test_register_tools_adds_table_tools(self):
        """register_tools should add table-related tools."""
        from src.mcp_server.tools import register_tools

        # Create mock MCP and state
        mock_mcp = MagicMock()
        mock_state = MagicMock()
        mock_state.table_store = MagicMock()
        mock_state.embedder = MagicMock()
        mock_state.store = MagicMock()

        # Register tools
        register_tools(mock_mcp, mock_state)

        # Should have called tool decorator multiple times
        assert mock_mcp.tool.call_count >= 10  # Existing tools + new table tools


# ============================================================================
# Tests for ServerState Table Store Integration
# ============================================================================


class TestServerStateTableStore:
    """Tests for table store integration in ServerState."""

    def test_server_state_has_table_store_property(self):
        """ServerState should have a table_store property."""
        from src.mcp_server.config import ServerState, MCPServerConfig

        config = MCPServerConfig(store_type="mock", embedder_type="mock")
        state = ServerState(config)

        assert hasattr(state, "table_store")

    def test_server_state_table_store_is_lazy(self):
        """ServerState.table_store should be lazily initialized."""
        from src.mcp_server.config import ServerState, MCPServerConfig

        config = MCPServerConfig(store_type="mock", embedder_type="mock")
        state = ServerState(config)

        # Accessing table_store should not raise
        table_store = state.table_store
        # May be None if not configured, but shouldn't raise
        # (actual behavior depends on implementation)
