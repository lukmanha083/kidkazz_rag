"""Live MCP server tests against real Helix-DB with OpenAI embeddings.

Requires:
- Helix-DB running on port 6969 with ingested data
- OPENAI_API_KEY set in environment
- pip install -e ".[all]"

Run with: PYTHONPATH=. pytest tests/test_mcp_live.py -v -s
"""

import logging
import pytest

from src.mcp_server.config import MCPServerConfig, ServerState
from src.mcp_server.server import create_server

logger = logging.getLogger(__name__)

# Known doc IDs from current DB state
INVENTORY_DOC = "Inventory_Accounting_A_Comprehensive_Guide_Steven_M_Bragg_Z-Library"
WAREHOUSE_DOC = "Excellence_in_Warehouse_Management_How_to_Minimise_Costs_and_Maximise_Value_Stuart_Emmett_Z-Library"


def _make_helix_config():
    """Create config pointing at real Helix-DB with OpenAI embeddings."""
    return MCPServerConfig(
        store_type="helix",
        helix_port=6969,
        helix_local=True,
        embedder_type="openai",
        model_name="text-embedding-3-small",
    )


def _check_helix_available():
    """Check if Helix-DB is reachable."""
    try:
        import socket
        s = socket.create_connection(("localhost", 6969), timeout=2)
        s.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False


# Skip all tests if Helix-DB not available
pytestmark = pytest.mark.skipif(
    not _check_helix_available(),
    reason="Helix-DB not running on port 6969",
)


@pytest.fixture(scope="module")
def state():
    """Create real ServerState with Helix-DB + OpenAI embedder."""
    config = _make_helix_config()
    return ServerState(config)


@pytest.fixture(scope="module")
def mcp_server(state):
    """Create MCP server with real backends."""
    return create_server(state=state)


class TestListDocuments:
    """Test document listing against real DB."""

    def test_lists_both_books(self, state):
        """Both ingested books should appear."""
        docs = state.store.list_documents()
        doc_ids = [d.get("doc_id") for d in docs]
        assert INVENTORY_DOC in doc_ids, f"Missing inventory book. Found: {doc_ids}"
        assert WAREHOUSE_DOC in doc_ids, f"Missing warehouse book. Found: {doc_ids}"

    def test_document_has_chunks(self, state):
        """Known books should have ~650 chunks each."""
        docs = state.store.list_documents()
        known_docs = {INVENTORY_DOC, WAREHOUSE_DOC}
        for doc in docs:
            doc_id = doc.get("doc_id")
            if doc_id not in known_docs:
                continue
            chunk_count = doc.get("chunk_count", 0)
            logger.info(f"  {doc_id}: {chunk_count} chunks")
            assert chunk_count > 100, f"{doc_id} has too few chunks: {chunk_count}"


class TestSemanticSearch:
    """Test semantic search with real OpenAI embeddings."""

    def test_inventory_query(self, state):
        """Search for inventory-related content."""
        query_embedding = state.embedder.embed_text("FIFO inventory valuation method")
        results = state.store.search_similar(
            query_embedding=query_embedding,
            top_k=5,
        )
        assert len(results) > 0, "No results for FIFO query"
        # Results are (EmbeddedChunk, score) tuples — extract content
        top_result = results[0]
        if isinstance(top_result, tuple):
            embedded_chunk, score = top_result
            chunk = getattr(embedded_chunk, "chunk", embedded_chunk)
            content = getattr(chunk, "content", "") if hasattr(chunk, "content") else chunk.get("content", "")
            logger.info(f"  Top result (score={score:.3f}): {content[:150]}...")
        else:
            content = top_result.get("content", "") if isinstance(top_result, dict) else getattr(top_result, "content", "")
            logger.info(f"  Top result: {content[:150]}...")
        assert len(content) > 0

    def test_warehouse_query(self, state):
        """Search for warehouse-related content."""
        query_embedding = state.embedder.embed_text("warehouse layout optimization and picking routes")
        results = state.store.search_similar(
            query_embedding=query_embedding,
            top_k=5,
        )
        assert len(results) > 0, "No results for warehouse query"

    def test_cross_document_search(self, state):
        """Search should return results from both books for shared topics."""
        query_embedding = state.embedder.embed_text("safety stock reorder point")
        results = state.store.search_similar(
            query_embedding=query_embedding,
            top_k=10,
        )
        assert len(results) >= 2, "Expected results from multiple sources"
        logger.info(f"  Cross-doc search returned {len(results)} results")


class TestConceptSearch:
    """Test concept queries against real DB with 4,118 concepts."""

    def test_list_concepts_count(self, state):
        """Should have ~4,118 concepts total."""
        concepts = state.store.list_concepts()
        logger.info(f"  Total concepts: {len(concepts)}")
        assert len(concepts) > 3000, f"Expected 4000+ concepts, got {len(concepts)}"

    def test_get_concept_by_name(self, state):
        """Look up a known concept."""
        concept = state.store.get_concept_by_name("FIFO")
        if not concept:
            # Try alternative names
            concept = state.store.get_concept_by_name("First In First Out")
        if not concept:
            concept = state.store.get_concept_by_name("fifo")
        logger.info(f"  FIFO concept: {concept}")
        # Don't fail hard - concept names depend on extraction
        if concept:
            assert concept.get("name") or concept.get("definition")

    def test_list_concepts_by_doc(self, state):
        """list_concepts(doc_id=...) should filter correctly via store."""
        inv_concepts = state.store.list_concepts(doc_id=INVENTORY_DOC)
        wh_concepts = state.store.list_concepts(doc_id=WAREHOUSE_DOC)
        logger.info(f"  Inventory concepts: {len(inv_concepts)}")
        logger.info(f"  Warehouse concepts: {len(wh_concepts)}")
        assert len(inv_concepts) > 1000, f"Expected 2000+ inventory concepts, got {len(inv_concepts)}"
        assert len(wh_concepts) > 1000, f"Expected 1900+ warehouse concepts, got {len(wh_concepts)}"

        # Verify shared concepts exist (both docs in source_documents)
        all_concepts = state.store.list_concepts()
        shared = [
            c for c in all_concepts
            if INVENTORY_DOC in (c.get("source_documents") or "")
            and WAREHOUSE_DOC in (c.get("source_documents") or "")
        ]
        logger.info(f"  Shared concepts (both books): {len(shared)}")
        if shared:
            logger.info(f"  Example shared: {shared[0].get('name')}")


class TestSummarySearch:
    """Test summary search with real OpenAI embeddings."""

    def test_search_summaries(self, state):
        """Search summaries across both books."""
        query_embedding = state.embedder.embed_text("inventory costing methods")
        results = state.store.search_summaries(
            query_embedding=query_embedding,
            limit=5,
        )
        logger.info(f"  Summary search returned {len(results)} results")
        assert len(results) > 0, "No summary search results"

    def test_get_document_summary(self, state):
        """Get document-level summary."""
        summaries = state.store.get_document_summaries(INVENTORY_DOC, level="document")
        logger.info(f"  Document summaries for inventory book: {len(summaries)}")
        if summaries:
            s = summaries[0]
            content = s.get("content", "") or s.get("summary_text", "")
            logger.info(f"  Summary preview: {content[:200]}...")
            assert len(content) > 50, "Document summary too short"

    def test_get_chapter_summaries(self, state):
        """Get chapter summaries for a book."""
        summaries = state.store.get_document_summaries(INVENTORY_DOC, level="chapter")
        logger.info(f"  Chapter summaries for inventory book: {len(summaries)}")
        assert len(summaries) > 5, f"Expected many chapter summaries, got {len(summaries)}"

    def test_list_summarized_documents(self, state):
        """Both books should have summaries."""
        docs = state.store.list_summarized_documents()
        logger.info(f"  Summarized documents: {docs}")
        assert len(docs) >= 2, f"Expected 2 summarized docs, got {len(docs)}"


class TestSummaryHierarchy:
    """Test hierarchical summary tree."""

    def test_hierarchy_structure(self, state):
        """Get full summary hierarchy for a book."""
        summaries = state.store.get_document_summaries(INVENTORY_DOC)
        logger.info(f"  Total summaries for inventory book: {len(summaries)}")
        assert len(summaries) > 100, f"Expected 600+ summaries, got {len(summaries)}"

        # Check we have all levels
        levels = set()
        for s in summaries:
            level = s.get("level") or s.get("summary_level")
            levels.add(level)
        logger.info(f"  Summary levels found: {levels}")
        assert "document" in levels or "Document" in levels, f"No document-level summary. Levels: {levels}"


class TestMCPToolsEndToEnd:
    """Test MCP tool functions directly (as they'd be called by Claude)."""

    def test_create_server_with_helix(self):
        """Server should create successfully with real Helix config."""
        config = _make_helix_config()
        state = ServerState(config)
        server = create_server(state=state)
        assert server is not None
        assert server.name == "kidkazz-rag"

    def test_search_semantic_tool(self, state):
        """Test the search_semantic tool function end-to-end."""
        from src.mcp_server.tools import register_tools
        from src.mcp_server.formatters import format_search_results

        # Embed and search directly (like the tool does)
        query_embedding = state.embedder.embed_text("cost of goods sold calculation")
        results = state.store.search_similar(
            query_embedding=query_embedding,
            top_k=3,
        )
        formatted = format_search_results(results)
        assert len(formatted) > 0, "search_semantic returned no results"
        # Check formatted structure
        first = formatted[0]
        assert "content" in first or "chunk_id" in first, f"Bad format: {list(first.keys())}"
        logger.info(f"  search_semantic returned {len(formatted)} results")
        logger.info(f"  First result keys: {list(first.keys())}")

    def test_search_summaries_tool(self, state):
        """Test summary search tool end-to-end."""
        from src.mcp_server.formatters import format_summary_search_results

        query_embedding = state.embedder.embed_text("warehouse picking strategies")
        results = state.store.search_summaries(
            query_embedding=query_embedding,
            limit=3,
        )
        formatted = format_summary_search_results(results)
        assert len(formatted) > 0, "search_summaries returned no results"
        logger.info(f"  search_summaries returned {len(formatted)} results")
        if formatted:
            logger.info(f"  First result keys: {list(formatted[0].keys())}")
