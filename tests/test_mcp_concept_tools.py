"""Tests for MCP concept tools (TDD - tests written first)."""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_state():
    """Create a mock ServerState with store."""
    state = MagicMock()
    state.store = MagicMock()
    return state


@pytest.fixture
def sample_concepts():
    """Sample concept data for tests."""
    return [
        {
            "concept_id": "cost-of-goods-sold",
            "name": "Cost of Goods Sold",
            "definition": "The direct costs attributable to goods sold.",
            "concept_type": "formula",
            "source_documents": '["inventory_textbook"]',
            "aliases": '["COGS", "cost of sales"]',
        },
        {
            "concept_id": "fifo",
            "name": "FIFO",
            "definition": "First-In, First-Out inventory method.",
            "concept_type": "method",
            "source_documents": '["inventory_textbook"]',
            "aliases": '["First-In First-Out"]',
        },
        {
            "concept_id": "lifo",
            "name": "LIFO",
            "definition": "Last-In, First-Out inventory method.",
            "concept_type": "method",
            "source_documents": '["inventory_textbook"]',
            "aliases": '[]',
        },
    ]


# ============================================================================
# Tests for format_concept
# ============================================================================


class TestFormatConcept:
    """Tests for format_concept formatter."""

    def test_formats_concept_dict(self):
        """Should format concept dictionary."""
        from src.mcp_server.formatters import format_concept

        concept = {
            "concept_id": "fifo",
            "name": "FIFO",
            "definition": "First-In, First-Out.",
            "concept_type": "method",
            "source_documents": '["doc1"]',
            "aliases": '["First-In First-Out"]',
        }

        result = format_concept(concept)

        assert result["concept_id"] == "fifo"
        assert result["name"] == "FIFO"
        assert result["definition"] == "First-In, First-Out."
        assert result["concept_type"] == "method"
        assert result["aliases"] == ["First-In First-Out"]
        assert result["source_documents"] == ["doc1"]

    def test_handles_empty_aliases(self):
        """Should handle empty aliases array."""
        from src.mcp_server.formatters import format_concept

        concept = {
            "concept_id": "test",
            "name": "Test",
            "definition": "Test def.",
            "concept_type": "term",
            "source_documents": "[]",
            "aliases": "[]",
        }

        result = format_concept(concept)

        assert result["aliases"] == []

    def test_handles_missing_fields(self):
        """Should handle missing fields gracefully."""
        from src.mcp_server.formatters import format_concept

        concept = {"name": "Partial"}

        result = format_concept(concept)

        assert result["name"] == "Partial"
        assert result["concept_id"] == ""


class TestFormatConceptList:
    """Tests for format_concept_list formatter."""

    def test_formats_list_of_concepts(self, sample_concepts):
        """Should format a list of concepts."""
        from src.mcp_server.formatters import format_concept_list

        result = format_concept_list(sample_concepts)

        assert len(result) == 3
        assert result[0]["name"] == "Cost of Goods Sold"
        assert result[1]["name"] == "FIFO"

    def test_returns_empty_for_empty_list(self):
        """Should return empty list for empty input."""
        from src.mcp_server.formatters import format_concept_list

        result = format_concept_list([])

        assert result == []


class TestFormatConceptWithContext:
    """Tests for format_concept_with_context formatter."""

    def test_formats_concept_with_citations_and_related(self):
        """Should format concept with full context."""
        from src.mcp_server.formatters import format_concept_with_context

        concept = {
            "concept_id": "cogs",
            "name": "COGS",
            "definition": "Cost of Goods Sold.",
            "concept_type": "formula",
            "aliases": "[]",
            "source_documents": "[]",
        }
        chunks = [
            {
                "document_id": "inventory_textbook",
                "section_path": '["Chapter 5", "COGS"]',
                "content": "Definition of COGS here with more content...",
                "chunk_id": "chunk_1",
            },
        ]
        related = [
            {
                "concept_id": "fifo",
                "name": "FIFO",
                "definition": "First-In, First-Out.",
                "concept_type": "method",
                "aliases": "[]",
                "source_documents": "[]",
            },
        ]

        result = format_concept_with_context(concept, chunks, related)

        assert "concept" in result
        assert result["concept"]["name"] == "COGS"
        assert "citations" in result
        assert len(result["citations"]) == 1
        assert result["citations"][0]["document_id"] == "inventory_textbook"
        assert "related_concepts" in result
        assert len(result["related_concepts"]) == 1


# ============================================================================
# Tests for list_concepts tool
# ============================================================================


class TestListConceptsTool:
    """Tests for list_concepts MCP tool."""

    def test_returns_all_concepts(self, mock_state, sample_concepts):
        """Should return all concepts when no filters."""
        mock_state.store.list_concepts.return_value = sample_concepts

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["list_concepts"]()

        assert len(result) == 3
        mock_state.store.list_concepts.assert_called_with(doc_id=None)

    def test_filters_by_doc_id(self, mock_state, sample_concepts):
        """Should filter concepts by document."""
        mock_state.store.list_concepts.return_value = sample_concepts[:1]

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["list_concepts"](doc_id="inventory_textbook")

        mock_state.store.list_concepts.assert_called_with(doc_id="inventory_textbook")

    def test_filters_by_concept_type(self, mock_state, sample_concepts):
        """Should filter concepts by type."""
        mock_state.store.list_concepts.return_value = sample_concepts

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["list_concepts"](concept_type="method")

        # Should filter to only "method" types
        assert all(c["concept_type"] == "method" for c in result)


# ============================================================================
# Tests for get_concept tool
# ============================================================================


class TestGetConceptTool:
    """Tests for get_concept MCP tool."""

    def test_returns_concept_by_name(self, mock_state, sample_concepts):
        """Should return concept by exact name."""
        mock_state.store.get_concept_by_name.return_value = sample_concepts[0]

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["get_concept"](name="Cost of Goods Sold")

        assert result is not None
        assert result["name"] == "Cost of Goods Sold"

    def test_returns_none_for_nonexistent(self, mock_state):
        """Should return None for non-existent concept."""
        mock_state.store.get_concept_by_name.return_value = None
        mock_state.store.get_concept.return_value = None
        mock_state.store.list_concepts.return_value = []

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["get_concept"](name="NonExistent")

        assert result is None

    def test_finds_concept_by_alias(self, mock_state, sample_concepts):
        """Should find concept by alias."""
        mock_state.store.get_concept_by_name.return_value = None
        mock_state.store.get_concept.return_value = None
        mock_state.store.list_concepts.return_value = sample_concepts

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["get_concept"](name="COGS")

        assert result is not None
        assert result["name"] == "Cost of Goods Sold"


# ============================================================================
# Tests for get_concept_with_citations tool
# ============================================================================


class TestGetConceptWithCitationsTool:
    """Tests for get_concept_with_citations MCP tool."""

    def test_returns_concept_with_citations(self, mock_state, sample_concepts):
        """Should return concept with source citations."""
        mock_state.store.get_concept_by_name.return_value = sample_concepts[0]
        mock_state.store.get_concept_definition_chunks.return_value = [
            {
                "document_id": "inventory_textbook",
                "section_path": '["Chapter 5"]',
                "content": "COGS is defined as...",
                "chunk_id": "chunk_1",
            }
        ]
        mock_state.store.get_related_concepts.return_value = []

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["get_concept_with_citations"](name="Cost of Goods Sold")

        assert result is not None
        assert "concept" in result
        assert "citations" in result
        assert len(result["citations"]) == 1

    def test_returns_none_for_nonexistent(self, mock_state):
        """Should return None for non-existent concept."""
        mock_state.store.get_concept_by_name.return_value = None
        mock_state.store.get_concept.return_value = None

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["get_concept_with_citations"](name="NonExistent")

        assert result is None


# ============================================================================
# Tests for get_related_concepts tool
# ============================================================================


class TestGetRelatedConceptsTool:
    """Tests for get_related_concepts MCP tool."""

    def test_returns_related_concepts(self, mock_state, sample_concepts):
        """Should return related concepts."""
        mock_state.store.get_concept_by_name.return_value = sample_concepts[0]
        mock_state.store.get_related_concepts.return_value = sample_concepts[1:2]

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["get_related_concepts"](name="Cost of Goods Sold")

        assert len(result) == 1
        assert result[0]["name"] == "FIFO"

    def test_returns_empty_for_nonexistent(self, mock_state):
        """Should return empty list for non-existent concept."""
        mock_state.store.get_concept_by_name.return_value = None
        mock_state.store.get_concept.return_value = None

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["get_related_concepts"](name="NonExistent")

        assert result == []


# ============================================================================
# Tests for search_concepts tool
# ============================================================================


class TestSearchConceptsTool:
    """Tests for search_concepts MCP tool."""

    def test_finds_exact_name_match(self, mock_state, sample_concepts):
        """Should rank exact name matches highest."""
        mock_state.store.list_concepts.return_value = sample_concepts

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["search_concepts"](query="FIFO")

        assert len(result) > 0
        assert result[0]["name"] == "FIFO"

    def test_finds_partial_name_match(self, mock_state, sample_concepts):
        """Should find partial name matches."""
        mock_state.store.list_concepts.return_value = sample_concepts

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["search_concepts"](query="inventory")

        # Should match concepts with "inventory" in definition
        assert len(result) > 0

    def test_finds_alias_match(self, mock_state, sample_concepts):
        """Should find concepts by alias."""
        mock_state.store.list_concepts.return_value = sample_concepts

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["search_concepts"](query="cost of sales")

        # Should find COGS by its alias
        assert len(result) > 0
        assert any(c["name"] == "Cost of Goods Sold" for c in result)

    def test_respects_top_k_limit(self, mock_state, sample_concepts):
        """Should respect top_k limit."""
        mock_state.store.list_concepts.return_value = sample_concepts

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["search_concepts"](query="method", top_k=1)

        assert len(result) <= 1


# ============================================================================
# Tests for explain_concept_cross_document tool
# ============================================================================


class TestExplainConceptCrossDocumentTool:
    """Tests for explain_concept_cross_document MCP tool."""

    def test_returns_comprehensive_context(self, mock_state, sample_concepts):
        """Should return concept with cross-document context."""
        mock_state.store.get_concept_by_name.return_value = sample_concepts[0]
        mock_state.store.get_concept_definition_chunks.return_value = [
            {
                "document_id": "inventory_textbook",
                "section_path": '["Chapter 5"]',
                "content": "COGS is the direct cost...",
            }
        ]
        mock_state.store.get_related_concepts.return_value = [sample_concepts[1]]

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["explain_concept_cross_document"](name="Cost of Goods Sold")

        assert "concept" in result
        assert "citations" in result
        assert "related_concepts_with_sources" in result

    def test_returns_error_for_nonexistent(self, mock_state):
        """Should return error for non-existent concept."""
        mock_state.store.get_concept_by_name.return_value = None
        mock_state.store.get_concept.return_value = None

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["explain_concept_cross_document"](name="NonExistent")

        assert "error" in result


# ============================================================================
# Tests for SCHEMA_INFO update
# ============================================================================


class TestSchemaInfo:
    """Tests for concept tools in SCHEMA_INFO."""

    def test_schema_includes_concept_tools(self):
        """SCHEMA_INFO should include concept tools."""
        from src.mcp_server.resources import SCHEMA_INFO

        tool_names = [t["name"] for t in SCHEMA_INFO.get("tools", [])]

        assert "list_concepts" in tool_names
        assert "get_concept" in tool_names
        assert "search_concepts" in tool_names

    def test_schema_includes_graph_tool(self):
        """SCHEMA_INFO should include graph visualization tool."""
        from src.mcp_server.resources import SCHEMA_INFO

        tool_names = [t["name"] for t in SCHEMA_INFO.get("tools", [])]

        assert "get_concept_graph_dot" in tool_names


# ============================================================================
# Tests for get_concept_graph_dot tool
# ============================================================================


class TestGetConceptGraphDotTool:
    """Tests for get_concept_graph_dot MCP tool."""

    def test_returns_dot_format(self, mock_state, sample_concepts):
        """Should return DOT format string."""
        mock_state.store.list_concepts.return_value = sample_concepts
        mock_state.store.get_related_concepts.return_value = []

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["get_concept_graph_dot"]()

        assert "digraph ConceptGraph" in result
        assert "fifo" in result.lower()
        assert "cost_of_goods_sold" in result.lower()

    def test_filters_by_doc_id(self, mock_state, sample_concepts):
        """Should filter concepts by doc_id."""
        mock_state.store.list_concepts.return_value = sample_concepts[:1]
        mock_state.store.get_related_concepts.return_value = []

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        tools["get_concept_graph_dot"](doc_id="inventory")

        mock_state.store.list_concepts.assert_called_with(doc_id="inventory")

    def test_returns_empty_graph_message(self, mock_state):
        """Should return empty message when no concepts."""
        mock_state.store.list_concepts.return_value = []

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["get_concept_graph_dot"]()

        assert "No concepts found" in result

    def test_includes_relations(self, mock_state, sample_concepts):
        """Should include relations in graph."""
        mock_state.store.list_concepts.return_value = sample_concepts
        mock_state.store.get_related_concepts.side_effect = [
            [{"concept_id": "fifo", "name": "FIFO"}],  # COGS relates to FIFO
            [],  # FIFO has no relations
            [],  # LIFO has no relations
        ]

        from src.mcp_server.tools import register_tools

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_state)

        result = tools["get_concept_graph_dot"]()

        # Should have edge from COGS to FIFO
        assert '"cost_of_goods_sold" -> "fifo"' in result
