"""Tests for table methods in MockChunkStore and HelixChunkStore.

Tests cover the table storage interface that's now integrated into
both MockChunkStore and HelixChunkStore.
"""

import pytest

from src.chunker.table_parser import ParsedTable
from src.chunker.table_summarizer import TableSummary
from src.storage import MockChunkStore


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_table():
    """Create a sample parsed table."""
    return ParsedTable(
        raw_markdown="| Method | Cost |\n|--------|------|\n| FIFO | $100 |\n| LIFO | $200 |",
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
        embedding=[0.1] * 384,  # Include embedding for search tests
    )


@pytest.fixture
def mock_store():
    """Create a MockChunkStore instance."""
    return MockChunkStore()


@pytest.fixture
def populated_store(mock_store, sample_table, sample_summary):
    """Create a MockChunkStore with a sample table stored."""
    mock_store.store_table(sample_table, sample_summary, "doc_001")
    return mock_store


# ============================================================================
# Tests for MockChunkStore Table Methods
# ============================================================================


class TestMockStoreTable:
    """Tests for store_table method."""

    def test_store_table_returns_table_id(self, mock_store, sample_table, sample_summary):
        """store_table should return the table ID."""
        result = mock_store.store_table(sample_table, sample_summary, "doc_001")

        assert result == sample_summary.table_id

    def test_store_table_stores_data(self, mock_store, sample_table, sample_summary):
        """store_table should store the table data."""
        mock_store.store_table(sample_table, sample_summary, "doc_001")

        assert sample_summary.table_id in mock_store._tables

    def test_store_table_stores_embedding(self, mock_store, sample_table, sample_summary):
        """store_table should store the embedding."""
        mock_store.store_table(sample_table, sample_summary, "doc_001")

        assert sample_summary.table_id in mock_store._table_embeddings
        assert mock_store._table_embeddings[sample_summary.table_id] == sample_summary.embedding


class TestMockGetTable:
    """Tests for get_table method."""

    def test_get_table_returns_none_for_missing(self, mock_store):
        """get_table should return None for non-existent table."""
        result = mock_store.get_table("nonexistent_table")

        assert result is None

    def test_get_table_returns_stored_table(self, populated_store, sample_table, sample_summary):
        """get_table should return the stored table."""
        result = populated_store.get_table(sample_summary.table_id)

        assert result is not None
        assert isinstance(result, ParsedTable)
        assert result.raw_markdown == sample_table.raw_markdown
        assert result.column_names == sample_table.column_names

    def test_get_table_with_summary_returns_tuple(self, populated_store, sample_summary):
        """get_table_with_summary should return tuple of (table, summary)."""
        result = populated_store.get_table_with_summary(sample_summary.table_id)

        assert result is not None
        table, summary = result
        assert isinstance(table, ParsedTable)
        assert isinstance(summary, TableSummary)
        assert summary.table_id == sample_summary.table_id


class TestMockSearchTables:
    """Tests for search_tables method."""

    def test_search_tables_returns_list(self, populated_store):
        """search_tables should return a list."""
        query_embedding = [0.1] * 384
        results = populated_store.search_tables(query_embedding, top_k=5)

        assert isinstance(results, list)

    def test_search_tables_returns_tuples(self, populated_store):
        """search_tables should return list of (table, score) tuples."""
        query_embedding = [0.1] * 384
        results = populated_store.search_tables(query_embedding, top_k=5)

        assert len(results) > 0
        table, score = results[0]
        assert isinstance(table, ParsedTable)
        assert isinstance(score, float)

    def test_search_tables_filters_by_doc_id(self, mock_store, sample_table, sample_summary):
        """search_tables should filter by document ID."""
        # Store tables in different documents
        mock_store.store_table(sample_table, sample_summary, "doc_001")

        table2 = ParsedTable(
            raw_markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            column_names=["A", "B"],
            rows=[["1", "2"]],
            row_count=1,
            column_count=2,
            column_types=["numeric", "numeric"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_002",
        )
        summary2 = TableSummary(
            table_id="table_chunk_002",
            summary_text="Simple numeric table",
            key_columns=["A"],
            key_values=["1"],
            embedding=[0.2] * 384,
        )
        mock_store.store_table(table2, summary2, "doc_002")

        # Search with doc_id filter
        query_embedding = [0.1] * 384
        results = mock_store.search_tables(query_embedding, top_k=5, doc_id="doc_001")

        # Should only return table from doc_001
        assert len(results) == 1
        table, _ = results[0]
        assert table.source_chunk_id == "chunk_001"


class TestMockListTables:
    """Tests for list_tables method."""

    def test_list_tables_returns_empty_initially(self, mock_store):
        """list_tables should return empty list initially."""
        result = mock_store.list_tables()

        assert result == []

    def test_list_tables_returns_stored_tables(self, populated_store, sample_summary):
        """list_tables should return all stored tables."""
        result = populated_store.list_tables()

        assert len(result) == 1
        assert result[0]["table_id"] == sample_summary.table_id

    def test_list_tables_filters_by_doc_id(self, mock_store, sample_table, sample_summary):
        """list_tables should filter by document ID."""
        mock_store.store_table(sample_table, sample_summary, "doc_001")

        table2 = ParsedTable(
            raw_markdown="| X | Y |\n|---|---|\n| 1 | 2 |",
            column_names=["X", "Y"],
            rows=[["1", "2"]],
            row_count=1,
            column_count=2,
            column_types=["numeric", "numeric"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_003",
        )
        summary2 = TableSummary(
            table_id="table_chunk_003",
            summary_text="Another table",
            key_columns=["X"],
            key_values=["1"],
        )
        mock_store.store_table(table2, summary2, "doc_002")

        result = mock_store.list_tables(doc_id="doc_001")

        assert len(result) == 1
        assert result[0]["table_id"] == sample_summary.table_id


class TestMockTableConceptLinking:
    """Tests for table-concept linking methods."""

    def test_link_table_to_concept_returns_true(self, populated_store, sample_summary):
        """link_table_to_concept should return True for valid table."""
        result = populated_store.link_table_to_concept(sample_summary.table_id, "concept_001")

        assert result is True

    def test_link_table_to_concept_returns_false_for_missing(self, mock_store):
        """link_table_to_concept should return False for non-existent table."""
        result = mock_store.link_table_to_concept("nonexistent", "concept_001")

        assert result is False

    def test_get_tables_for_concept_returns_linked_tables(self, populated_store, sample_summary):
        """get_tables_for_concept should return linked tables."""
        populated_store.link_table_to_concept(sample_summary.table_id, "concept_001")

        results = populated_store.get_tables_for_concept("concept_001")

        assert len(results) == 1
        assert results[0].source_chunk_id == "chunk_001"

    def test_get_tables_for_concept_returns_empty_for_no_links(self, populated_store):
        """get_tables_for_concept should return empty list for no links."""
        results = populated_store.get_tables_for_concept("nonexistent_concept")

        assert results == []


class TestMockDeleteTable:
    """Tests for delete_table method."""

    def test_delete_table_returns_true_for_existing(self, populated_store, sample_summary):
        """delete_table should return True for existing table."""
        result = populated_store.delete_table(sample_summary.table_id)

        assert result is True

    def test_delete_table_returns_false_for_missing(self, mock_store):
        """delete_table should return False for non-existent table."""
        result = mock_store.delete_table("nonexistent")

        assert result is False

    def test_delete_table_removes_from_storage(self, populated_store, sample_summary):
        """delete_table should remove table from storage."""
        populated_store.delete_table(sample_summary.table_id)

        assert populated_store.get_table(sample_summary.table_id) is None

    def test_delete_table_removes_embeddings(self, populated_store, sample_summary):
        """delete_table should remove embeddings."""
        populated_store.delete_table(sample_summary.table_id)

        assert sample_summary.table_id not in populated_store._table_embeddings


class TestMockClearTables:
    """Tests for clear method including tables."""

    def test_clear_removes_tables(self, populated_store, sample_summary):
        """clear should remove all tables."""
        populated_store.clear()

        assert len(populated_store._tables) == 0
        assert len(populated_store._table_embeddings) == 0
        assert len(populated_store._table_concept_links) == 0


# ============================================================================
# Tests for Graph Traversal Methods
# ============================================================================


class TestMockGetTableForChunk:
    """Tests for get_table_for_chunk method."""

    def test_get_table_for_chunk_returns_none_for_no_table(self, mock_store):
        """get_table_for_chunk should return None when chunk has no table."""
        result = mock_store.get_table_for_chunk("nonexistent_chunk")

        assert result is None

    def test_get_table_for_chunk_returns_table(self, populated_store, sample_table):
        """get_table_for_chunk should return table for source chunk."""
        result = populated_store.get_table_for_chunk(sample_table.source_chunk_id)

        assert result is not None
        assert isinstance(result, ParsedTable)
        assert result.source_chunk_id == sample_table.source_chunk_id

    def test_get_table_for_chunk_returns_none_for_wrong_chunk(self, populated_store):
        """get_table_for_chunk should return None for chunk without table."""
        result = populated_store.get_table_for_chunk("other_chunk_id")

        assert result is None


class TestMockGetTablesViaDocumentEdge:
    """Tests for get_tables_via_document_edge method."""

    def test_get_tables_via_document_edge_returns_empty_for_no_tables(self, mock_store):
        """get_tables_via_document_edge should return empty list for doc with no tables."""
        result = mock_store.get_tables_via_document_edge("nonexistent_doc")

        assert result == []

    def test_get_tables_via_document_edge_returns_tables(self, populated_store, sample_table):
        """get_tables_via_document_edge should return tables for document."""
        result = populated_store.get_tables_via_document_edge("doc_001")

        assert len(result) == 1
        assert result[0].source_chunk_id == sample_table.source_chunk_id


class TestMockGetConceptsForTable:
    """Tests for get_concepts_for_table method."""

    def test_get_concepts_for_table_returns_empty_for_no_links(self, populated_store, sample_summary):
        """get_concepts_for_table should return empty list for unlinked table."""
        result = populated_store.get_concepts_for_table(sample_summary.table_id)

        assert result == []

    def test_get_concepts_for_table_returns_linked_concepts(self, populated_store, sample_summary):
        """get_concepts_for_table should return linked concept IDs."""
        # Link table to concepts
        populated_store.link_table_to_concept(sample_summary.table_id, "concept_001")
        populated_store.link_table_to_concept(sample_summary.table_id, "concept_002")

        result = populated_store.get_concepts_for_table(sample_summary.table_id)

        assert len(result) == 2
        concept_ids = {c["concept_id"] for c in result}
        assert concept_ids == {"concept_001", "concept_002"}

    def test_get_concepts_for_table_returns_empty_for_missing_table(self, mock_store):
        """get_concepts_for_table should return empty list for missing table."""
        result = mock_store.get_concepts_for_table("nonexistent_table")

        assert result == []
