"""Tests for table storage module.

TDD tests for multi-vector table storage and retrieval.
"""

import pytest
from unittest.mock import MagicMock, patch


# ============================================================================
# Tests for TableStore Class
# ============================================================================


class TestTableStoreClass:
    """Tests for TableStore class structure."""

    def test_table_store_exists(self):
        """TableStore should be importable."""
        from src.storage.table_store import TableStore

        assert TableStore is not None

    def test_table_store_accepts_embedder(self):
        """TableStore should accept embedder parameter."""
        from src.storage.table_store import TableStore

        mock_embedder = MagicMock()
        store = TableStore(embedder=mock_embedder)

        assert store.embedder is mock_embedder

    def test_table_store_has_store_table_method(self):
        """TableStore should have store_table method."""
        from src.storage.table_store import TableStore

        mock_embedder = MagicMock()
        store = TableStore(embedder=mock_embedder)

        assert hasattr(store, "store_table")
        assert callable(store.store_table)

    def test_table_store_has_search_tables_method(self):
        """TableStore should have search_tables method."""
        from src.storage.table_store import TableStore

        mock_embedder = MagicMock()
        store = TableStore(embedder=mock_embedder)

        assert hasattr(store, "search_tables")
        assert callable(store.search_tables)

    def test_table_store_has_get_table_method(self):
        """TableStore should have get_table method."""
        from src.storage.table_store import TableStore

        mock_embedder = MagicMock()
        store = TableStore(embedder=mock_embedder)

        assert hasattr(store, "get_table")
        assert callable(store.get_table)


# ============================================================================
# Tests for Storing Tables
# ============================================================================


class TestStoreTable:
    """Tests for store_table method."""

    def test_store_table_returns_table_id(self):
        """store_table should return the stored table ID."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummary
        from src.storage.table_store import TableStore

        table = ParsedTable(
            raw_markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            column_names=["A", "B"],
            rows=[["1", "2"]],
            row_count=1,
            column_count=2,
            column_types=["text", "text"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        summary = TableSummary(
            table_id="table_chunk_001",
            summary_text="A simple table",
            key_columns=["A"],
            key_values=["1"],
        )

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384

        store = TableStore(embedder=mock_embedder)
        table_id = store.store_table(table, summary, "doc_001")

        assert table_id is not None
        assert "chunk_001" in table_id

    def test_store_table_embeds_summary(self):
        """store_table should embed the summary text."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummary
        from src.storage.table_store import TableStore

        table = ParsedTable(
            raw_markdown="| A |\n|---|\n| 1 |",
            column_names=["A"],
            rows=[["1"]],
            row_count=1,
            column_count=1,
            column_types=["text"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        summary = TableSummary(
            table_id="table_chunk_001",
            summary_text="Summary to embed",
            key_columns=["A"],
            key_values=["1"],
        )

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384

        store = TableStore(embedder=mock_embedder)
        store.store_table(table, summary, "doc_001")

        mock_embedder.embed_text.assert_called_once_with("Summary to embed")

    def test_store_table_stores_all_fields(self):
        """store_table should store all table fields."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummary
        from src.storage.table_store import TableStore

        table = ParsedTable(
            raw_markdown="| Method | Cost |\n|---|---|\n| FIFO | $500 |",
            column_names=["Method", "Cost"],
            rows=[["FIFO", "$500"]],
            row_count=1,
            column_count=2,
            column_types=["text", "numeric"],
            has_header_row=True,
            surrounding_context="Inventory context",
            source_chunk_id="chunk_001",
        )

        summary = TableSummary(
            table_id="table_chunk_001",
            summary_text="Inventory methods comparison",
            key_columns=["Method", "Cost"],
            key_values=["FIFO", "$500"],
        )

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384

        store = TableStore(embedder=mock_embedder)
        table_id = store.store_table(table, summary, "doc_001")

        # Retrieve and verify
        retrieved = store.get_table(table_id)
        assert retrieved is not None
        assert retrieved.column_names == ["Method", "Cost"]
        assert retrieved.row_count == 1


# ============================================================================
# Tests for Searching Tables
# ============================================================================


class TestSearchTables:
    """Tests for search_tables method."""

    def test_search_tables_returns_list(self):
        """search_tables should return a list of results."""
        from src.storage.table_store import TableStore

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384

        store = TableStore(embedder=mock_embedder)
        results = store.search_tables("inventory methods", top_k=5)

        assert isinstance(results, list)

    def test_search_tables_embeds_query(self):
        """search_tables should embed the query when tables exist."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummary
        from src.storage.table_store import TableStore

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384

        store = TableStore(embedder=mock_embedder)

        # Store a table first so search will embed the query
        table = ParsedTable(
            raw_markdown="| A |\n|---|\n| 1 |",
            column_names=["A"],
            rows=[["1"]],
            row_count=1,
            column_count=1,
            column_types=["text"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )
        summary = TableSummary(
            table_id="table_chunk_001",
            summary_text="Test",
            key_columns=["A"],
            key_values=["1"],
        )
        store.store_table(table, summary, "doc_001")

        # Reset mock to clear store_table calls
        mock_embedder.reset_mock()
        mock_embedder.embed_text.return_value = [0.1] * 384

        store.search_tables("my query", top_k=5)

        mock_embedder.embed_text.assert_called_with("my query")

    def test_search_tables_returns_matching_tables(self):
        """search_tables should return tables matching the query."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummary
        from src.storage.table_store import TableStore

        # Create and store a table
        table = ParsedTable(
            raw_markdown="| Method | Cost |\n|---|---|\n| FIFO | $500 |",
            column_names=["Method", "Cost"],
            rows=[["FIFO", "$500"]],
            row_count=1,
            column_count=2,
            column_types=["text", "numeric"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        summary = TableSummary(
            table_id="table_chunk_001",
            summary_text="FIFO inventory costing method",
            key_columns=["Method"],
            key_values=["FIFO"],
        )

        mock_embedder = MagicMock()
        # Return similar embeddings for matching content
        mock_embedder.embed_text.return_value = [0.1] * 384

        store = TableStore(embedder=mock_embedder)
        store.store_table(table, summary, "doc_001")

        results = store.search_tables("FIFO method", top_k=5)

        assert len(results) > 0

    def test_search_tables_returns_scores(self):
        """search_tables should return similarity scores."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummary
        from src.storage.table_store import TableStore

        table = ParsedTable(
            raw_markdown="| A |\n|---|\n| 1 |",
            column_names=["A"],
            rows=[["1"]],
            row_count=1,
            column_count=1,
            column_types=["text"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        summary = TableSummary(
            table_id="table_chunk_001",
            summary_text="Test table",
            key_columns=["A"],
            key_values=["1"],
        )

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384

        store = TableStore(embedder=mock_embedder)
        store.store_table(table, summary, "doc_001")

        results = store.search_tables("test", top_k=5)

        # Results should be tuples of (table, score)
        if results:
            table_result, score = results[0]
            assert isinstance(score, float)


# ============================================================================
# Tests for Getting Tables
# ============================================================================


class TestGetTable:
    """Tests for get_table method."""

    def test_get_table_returns_none_for_missing(self):
        """get_table should return None for missing table."""
        from src.storage.table_store import TableStore

        mock_embedder = MagicMock()
        store = TableStore(embedder=mock_embedder)

        result = store.get_table("nonexistent_id")

        assert result is None

    def test_get_table_returns_stored_table(self):
        """get_table should return previously stored table."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummary
        from src.storage.table_store import TableStore

        table = ParsedTable(
            raw_markdown="| Name | Value |\n|---|---|\n| Test | 123 |",
            column_names=["Name", "Value"],
            rows=[["Test", "123"]],
            row_count=1,
            column_count=2,
            column_types=["text", "numeric"],
            has_header_row=True,
            surrounding_context="Test context",
            source_chunk_id="chunk_test",
        )

        summary = TableSummary(
            table_id="table_chunk_test",
            summary_text="Test table summary",
            key_columns=["Name"],
            key_values=["Test"],
        )

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384

        store = TableStore(embedder=mock_embedder)
        table_id = store.store_table(table, summary, "doc_001")

        result = store.get_table(table_id)

        assert result is not None
        assert result.column_names == ["Name", "Value"]
        assert result.rows == [["Test", "123"]]


# ============================================================================
# Tests for Table-Concept Linking
# ============================================================================


class TestTableConceptLinking:
    """Tests for linking tables to concepts."""

    def test_link_table_to_concept_method_exists(self):
        """TableStore should have link_table_to_concept method."""
        from src.storage.table_store import TableStore

        mock_embedder = MagicMock()
        store = TableStore(embedder=mock_embedder)

        assert hasattr(store, "link_table_to_concept")
        assert callable(store.link_table_to_concept)

    def test_get_tables_for_concept_method_exists(self):
        """TableStore should have get_tables_for_concept method."""
        from src.storage.table_store import TableStore

        mock_embedder = MagicMock()
        store = TableStore(embedder=mock_embedder)

        assert hasattr(store, "get_tables_for_concept")
        assert callable(store.get_tables_for_concept)

    def test_get_tables_for_concept_returns_linked_tables(self):
        """get_tables_for_concept should return tables linked to concept."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummary
        from src.storage.table_store import TableStore

        table = ParsedTable(
            raw_markdown="| Method | Cost |\n|---|---|\n| FIFO | $500 |",
            column_names=["Method", "Cost"],
            rows=[["FIFO", "$500"]],
            row_count=1,
            column_count=2,
            column_types=["text", "numeric"],
            has_header_row=True,
            surrounding_context="",
            source_chunk_id="chunk_001",
        )

        summary = TableSummary(
            table_id="table_chunk_001",
            summary_text="Inventory methods",
            key_columns=["Method"],
            key_values=["FIFO"],
        )

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384

        store = TableStore(embedder=mock_embedder)
        table_id = store.store_table(table, summary, "doc_001")

        # Link table to concept
        store.link_table_to_concept(table_id, "FIFO")

        # Get tables for concept
        tables = store.get_tables_for_concept("FIFO")

        assert len(tables) > 0


# ============================================================================
# Tests for Listing Tables
# ============================================================================


class TestListTables:
    """Tests for listing all tables."""

    def test_list_tables_method_exists(self):
        """TableStore should have list_tables method."""
        from src.storage.table_store import TableStore

        mock_embedder = MagicMock()
        store = TableStore(embedder=mock_embedder)

        assert hasattr(store, "list_tables")
        assert callable(store.list_tables)

    def test_list_tables_returns_empty_initially(self):
        """list_tables should return empty list initially."""
        from src.storage.table_store import TableStore

        mock_embedder = MagicMock()
        store = TableStore(embedder=mock_embedder)

        result = store.list_tables()

        assert result == []

    def test_list_tables_returns_stored_tables(self):
        """list_tables should return all stored tables."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummary
        from src.storage.table_store import TableStore

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384

        store = TableStore(embedder=mock_embedder)

        # Store multiple tables
        for i in range(3):
            table = ParsedTable(
                raw_markdown=f"| Col{i} |\n|---|\n| {i} |",
                column_names=[f"Col{i}"],
                rows=[[str(i)]],
                row_count=1,
                column_count=1,
                column_types=["text"],
                has_header_row=True,
                surrounding_context="",
                source_chunk_id=f"chunk_{i}",
            )
            summary = TableSummary(
                table_id=f"table_chunk_{i}",
                summary_text=f"Table {i}",
                key_columns=[f"Col{i}"],
                key_values=[str(i)],
            )
            store.store_table(table, summary, "doc_001")

        result = store.list_tables()

        assert len(result) == 3

    def test_list_tables_filters_by_doc_id(self):
        """list_tables should filter by document ID."""
        from src.chunker.table_parser import ParsedTable
        from src.chunker.table_summarizer import TableSummary
        from src.storage.table_store import TableStore

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384

        store = TableStore(embedder=mock_embedder)

        # Store tables in different documents
        for doc_id in ["doc_1", "doc_2"]:
            table = ParsedTable(
                raw_markdown="| A |\n|---|\n| 1 |",
                column_names=["A"],
                rows=[["1"]],
                row_count=1,
                column_count=1,
                column_types=["text"],
                has_header_row=True,
                surrounding_context="",
                source_chunk_id=f"chunk_{doc_id}",
            )
            summary = TableSummary(
                table_id=f"table_chunk_{doc_id}",
                summary_text="Test",
                key_columns=["A"],
                key_values=["1"],
            )
            store.store_table(table, summary, doc_id)

        # Filter by doc_id
        result = store.list_tables(doc_id="doc_1")

        assert len(result) == 1
